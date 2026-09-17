"""Post-run stdlib audit: seals, exact decision schedules, weight denominators."""
import hashlib
import json
import math
from pathlib import Path
OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def main():
    verified={}
    for run in ['v1','efficiency_run_v1','action_balance_v1']:
        directory=LINE/'sft_acceptance'/run
        manifest=(directory/'SHA256SUMS').read_text().splitlines()
        for entry in manifest:
            expected,rel=entry.split('  ',1);path=(directory/rel).resolve()
            assert path.is_relative_to(directory.resolve())
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,(run,rel)
        verified[run]=len(manifest)
    a=LINE/'sft_acceptance/efficiency_run_v1';b=LINE/'sft_acceptance/action_balance_v1'
    assert (a/'SUBSET.json').read_bytes()==(b/'SUBSET.json').read_bytes()
    before_a=json.loads((a/'rank0_EVAL_before.json').read_text())
    before_b=json.loads((b/'rank0_EVAL_before.json').read_text())
    assert before_a==before_b
    rank_steps=[];all_steps=[]
    for rank in range(2):
        x=rows(a/f'rank{rank}_train_steps.jsonl');y=rows(b/f'rank{rank}_train_steps.jsonl')
        key=lambda row:tuple(row[k] for k in ['update','row','epoch','t','target'])
        assert list(map(key,x))==list(map(key,y))
        rank_steps.append(len(y));all_steps.extend(y)
    weights=json.loads((b/'WEIGHTS.json').read_text())['values']
    events=[x for x in rows(b/'rank0_events.jsonl') if x['event']=='update']
    assert len(events)==200
    for e in events:
        ss=[s for s in all_steps if s['update']==e['update']]
        assert len(ss)==e['decisions']
        assert all(s['weight']==weights[s['target']] for s in ss)
        denom=sum(s['weight'] for s in ss)
        assert math.isclose(denom,e['global_weight_sum'],rel_tol=1e-12)
        raw=sum(s['ce'] for s in ss)/len(ss)
        weighted=sum(s['ce']*s['weight'] for s in ss)/denom
        assert math.isclose(raw,e['CE'],rel_tol=1e-10)
        assert math.isclose(weighted,e['weighted_CE'],rel_tol=1e-6)
    execution=json.loads((b/'EXECUTION.json').read_text())
    assert execution['returncodes']==[0,0] and execution['all_leased_holders_restored']
    result=dict(pass_audit=True,seals_verified=verified,initial_evaluation_exact=True,
        subset_bytes_exact=True,rank_training_step_sequences_exact=True,rank_steps=rank_steps,
        updates_with_weight_denominator_and_loss_verified=200,GPU_holders_restored=True,
        scope='integrity and implementation audit, not scientific validation')
    with (OUT/'DELIVERY_AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
