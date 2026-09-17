"""No launch until FIT admits one frozen scalar; no DEV parameter selection."""
import collections
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
BASE=HERE.parent/'ordinary_expanded_dev_after_single_v1'
FIT=HERE.parent/'ordinary_stop_calibration_fit_v2'
REVIEW=LINE/'reviews/Q35N_ORDINARY_STOP_CALIBRATION_V1'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    assert not (HERE/'SOURCE_LOCK.json').exists() and not (HERE/'run_001').exists()
    calibration=read(REVIEW/'CALIBRATION.json');fr=read(REVIEW/'FIT_RESULT.json')
    assert calibration['status']=='FROZEN_FIT_PARAMETER' and fr['fit_gate_passed'],'FIT_GATE_REQUIRED'
    assert sha(REVIEW/'FIT_RESULT.json')==calibration['fit_result_sha256']
    assert calibration['bias']==fr['selected_bias'] and 0<calibration['bias']<=2
    assert not fr['dev_used_for_parameter_selection'] and not calibration['dev_parameter_tuning_allowed']
    outputs=[]
    for script in (HERE/'tests.py',REVIEW/'test_counterfactual.py',REVIEW/'test_final_gate.py'):
        r=subprocess.run([str(PY),'-I','-S','-B',str(script)],capture_output=True,text=True,timeout=60)
        assert r.returncode==0,r.stderr;outputs.append(dict(script=str(script),output=r.stdout+r.stderr))
    script=REVIEW/'test_metric_v2.py'
    r=subprocess.run([str(LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'),'-I','-B',str(script)],capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stderr;outputs.append(dict(script=str(script),output=r.stdout+r.stderr))
    p=read(BASE/'PROTOCOL.json');assert p['checkpoint_sha256']==calibration['checkpoint_sha256']
    p.update(id='Q35N_ORDINARY_STOP_CALIBRATED_DEV_V1',checkpoint_role='base4000_fit_calibrated_stop',
             stop_logit_bias=calibration['bias'],calibration_sha256=sha(REVIEW/'CALIBRATION.json'),
             controller_enabled=True,controller_kind='single_FIT_fitted_STOP_logit_offset',visual_stall_guard_enabled=False,
             main_criterion='SR delta >0, SPL delta >=0, nDTW delta >=-0.01 vs pure4000; no DEV threshold tuning')
    save(HERE/'PROTOCOL.json',p)
    for name in ('EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json'):
        with (HERE/name).open('xb') as f:f.write((BASE/name).read_bytes())
        assert sha(HERE/name)==sha(BASE/name)
    save(HERE/'CPU_TEST_RESULT.json',dict(passed=True,tests=outputs,only_one_frozen_bias=True))
    files=dict(read(FIT/'SOURCE_LOCK.json')['files'])
    for path in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+list(REVIEW.glob('*.py'))+[REVIEW/'CALIBRATION.json',REVIEW/'FIT_RESULT.json',REVIEW/'FIT_METRIC_REVISION_SEAL.json',REVIEW/'METRIC_REVISION_V2.md',BASE/'run_001/RESULT.json']:
        files[str(path)]=sha(path)
    for path,digest in files.items():assert Path(path).resolve().is_relative_to(ROOT) and sha(path)==digest,path
    save(HERE/'SOURCE_LOCK.json',dict(files=files,scope='fixed FIT bias on original 100 DEV; no parameter search',training_updates=0))
    # Only now inspect the DEV response to this one immutable parameter. A bad
    # prediction is still retained and does not cancel the registered real run.
    s=importlib.util.spec_from_file_location('frozen_stop_prefix',REVIEW/'counterfactual_v2.py')
    c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
    with gzip.open(p['gt_path']) as f:gt=json.load(f)
    episodes={};decisions=collections.defaultdict(list);trace_hashes={}
    for lane in sorted((BASE/'run_001/lanes').glob('lane_*')):
        for file in lane.glob('episode_*.json'):
            e=read(file);episodes[e['index']]=e;trace_hashes[str(file)]=sha(file)
        file=lane/'POLICY_STEPS.jsonl';trace_hashes[str(file)]=sha(file)
        for line in file.read_text().splitlines():
            d=json.loads(line);decisions[d['index']].append(d)
    assert sorted(episodes)==list(range(100))
    before=[];after=[]
    for i in range(100):
        e=episodes[i];ref=gt[str(e['episode_id'])]['locations']
        b=c.prefix(e,decisions[i],ref,0);a=c.prefix(e,decisions[i],ref,p['stop_logit_bias'])
        for key in ('success','spl','ndtw'):assert math.isclose(b[key],e[key],abs_tol=1e-9),(i,key)
        before.append(b);after.append(a)
    comparison=c.assess(before,after);comparison.pop('fit_gate')
    prediction=dict(status='EXACT_PREFIX_PREDICTION_NOT_LIVE_EVALUATION',bias=p['stop_logit_bias'],
         calibration_sha256=p['calibration_sha256'],protocol_sha256=sha(HERE/'PROTOCOL.json'),
         before=c.stats(before),predicted=c.stats(after),comparison=comparison,episodes=after,
         trace_hashes=trace_hashes,additional_biases_evaluated_on_dev=0,real_run_still_required=True,
         cancel_real_run_based_on_prediction=False,scientific_gain_verified=False)
    save(REVIEW/'DEV_PREDICTION.json',prediction)
    save(HERE/'MAIN_REVIEW.json',dict(status='PASS_FOR_ONE_FIXED_DEV_RUN',source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),
         calibration_sha256=p['calibration_sha256'],prediction_sha256=sha(REVIEW/'DEV_PREDICTION.json'),
         gpu=1,requires_empty_gpu=True,automatic_retry=False,borrowed_holders_allowed=False,unix=time.time()))
    print(json.dumps(dict(status='FROZEN',bias=p['stop_logit_bias'],real_run_required=True,only_one_dev_parameter=True)))


if __name__=='__main__':main()
