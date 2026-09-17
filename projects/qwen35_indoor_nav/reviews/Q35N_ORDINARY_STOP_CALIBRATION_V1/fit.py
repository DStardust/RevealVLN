"""Fit one STOP offset on the frozen FIT set only; no automatic DEV run."""
import collections
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
CASE=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v1'
s=importlib.util.spec_from_file_location('calibration_counterfactual',HERE/'counterfactual.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    assert not (HERE/'FIT_RESULT.json').exists(),'ALREADY_FITTED'
    locks=read(CASE/'SOURCE_LOCK.json')['files']
    for p in (HERE/'counterfactual.py',HERE/'fit.py',HERE/'SPEC_ZH.md'):
        assert sha(p)==locks[str(p)],p
    p=read(CASE/'PROTOCOL.json');sel=read(CASE/'SELECTION.json');r=read(CASE/'run_001/RESULT.json')
    assert p['split']==sel['split']=='FIT' and not sel['dev_or_confirm_used']
    assert r['status']=='COMPLETE' and r['completed']==r['planned']==64 and r['trace_audit_passed']
    assert r['selected_batch_size']==1 and r['checkpoint_sha256']==p['checkpoint_sha256']
    split=read(LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001/SPLIT.json')
    assert set(p['houses'])<=set(split['FIT']) and set(p['houses']).isdisjoint(split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
    with gzip.open(p['gt_path']) as f:gt=json.load(f)
    episodes={};decisions=collections.defaultdict(list)
    for lane in sorted((CASE/'run_001/lanes').glob('lane_*')):
        for file in lane.glob('episode_*.json'):
            e=read(file);assert e['index'] not in episodes;episodes[e['index']]=e
        for line in (lane/'POLICY_STEPS.jsonl').read_text().splitlines():
            d=json.loads(line);decisions[d['index']].append(d)
    assert sorted(episodes)==list(range(64))
    candidates=[];before=None
    for bias in c.GRID:
        rows=[c.prefix(episodes[i],decisions[i],gt[str(episodes[i]['episode_id'])]['locations'],bias) for i in range(64)]
        if bias==0:
            before=rows
            for i,row in enumerate(rows):
                for key in ('success','spl','ndtw','steps','navigation_error_m','path_length_m'):
                    assert math.isclose(row[key],episodes[i][key],rel_tol=1e-9,abs_tol=1e-9),(i,key)
            baseline=c.stats(rows)
            for key in ('sr','spl','ndtw'):assert math.isclose(baseline[key],r[key],abs_tol=1e-9),key
        candidates.append(dict(bias=bias,metrics=c.stats(rows),comparison=c.assess(before,rows),episodes=rows))
    selected=c.select(candidates)
    result=dict(status='COMPLETE',unix=time.time(),selected_bias=selected['bias'] if selected else None,
         fit_gate_passed=selected is not None,candidates=candidates,baseline=candidates[0]['metrics'],
         fixed_checkpoint_sha256=p['checkpoint_sha256'],fit_source_lock_sha256=sha(CASE/'SOURCE_LOCK.json'),
         fit_result_sha256=sha(CASE/'run_001/RESULT.json'),fit_input_sha256=sha(CASE/'EPISODES_PRIVILEGED.json'),
         dev_used_for_parameter_selection=False,scientific_gain_verified=False,automatic_dev_run=False,
         interpretation='exact early-STOP prefixes of FIT rollouts; not new simulated rollouts and not validation')
    save(HERE/'FIT_RESULT.json',result)
    save(HERE/'CALIBRATION.json',dict(status='FROZEN_FIT_PARAMETER' if selected else 'FIT_GATE_FAILED_NO_PARAMETER',
         bias=selected['bias'] if selected else None,checkpoint_sha256=p['checkpoint_sha256'],
         grid=list(c.GRID),fit_result_sha256=sha(HERE/'FIT_RESULT.json'),rule='logits[3]+=bias; argmax first-index tie',
         policy_inputs=['four_model_logits'],privileged_policy_inputs=[],dev_parameter_tuning_allowed=False))
    print(json.dumps(dict(status=result['status'],selected_bias=result['selected_bias'],fit_gate_passed=result['fit_gate_passed'],
         baseline=result['baseline'],selected={k:selected[k] for k in ('bias','metrics','comparison')} if selected else None),ensure_ascii=False))


if __name__=='__main__':main()
