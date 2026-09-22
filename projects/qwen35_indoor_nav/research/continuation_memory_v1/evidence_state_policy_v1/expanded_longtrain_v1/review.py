"""Recompute sealed groups, keep planned missing slots, report paired house/seed effects."""
from collections import Counter
import csv
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
evaluator=local_module("evaluate_continuations");registry,admitted=evaluator.registry,evaluator.admitted
from evaluator_v16 import legacy,evaluate
from select_action import select

def metric(rows):
    n=len(rows);counts=Counter(r['status'] for r in rows)
    u=sum(counts[k] for k in ('UNKNOWN','ERROR','NOT_RUN','NOT_COLLECTED'))
    return dict(planned=n,passed=counts['PASS'],unknown=u,counts=dict(counts),
        lower=counts['PASS']/n,upper=(counts['PASS']+u)/n,cost=sum(r['cost'] for r in rows)/n,
        collisions=sum(r.get('collisions',0)>0 for r in rows),exhausted=sum(r.get('exhausted',False) for r in rows))

def main(run):
    import numpy as np
    cfg=config(run);reg=registry(run);groups=admitted(run,reg);families={f['family_id']:f for f in read(run/'DATA.json')['raw_families']};rows=[];arrays={};audits=[]
    for session in set(groups.values()):
        for p in session.glob('GROUP_*.json'):
            row=read(p)
            if groups.get(row['condition'])==session:audits.extend(row['audits'])
    if len(audits)!=0 or any(not a['input_prefix_matched'] or not a['action_prefix_matched'] or a['argmax_flip_count'] for a in audits):raise ValueError('PREFIX_AUDIT')
    for slot in reg['slots']:
        cond=reg['conditions'][slot['condition']];session=groups.get(slot['condition'])
        row=dict(slot,**cond,status='NOT_RUN' if cond['available'] else 'NOT_COLLECTED',cost=1.)
        if session:
            folder=session/'rollouts'/f"{slot['rank']:04d}";saved=read(folder/'ROLLOUT.json')['task_result'];f=families[cond['family_id']]
            trace=read(folder/'TRACE_PRIVILEGED.json');cut=len(f['histories'][cond['history_id']])
            for obs in trace['observations']:
                for kind,shape,dtype in (('rgb',(224,224,3),np.uint8),('semantic',(224,224),np.uint32)):
                    p=session/'content'/(obs[kind+'_hash']+'.'+kind+'.npy')
                    if p not in arrays:
                        a=np.load(p,allow_pickle=False)
                        if a.shape!=shape or a.dtype!=dtype or hashlib.sha256(a.tobytes()).hexdigest()!=obs[kind+'_hash']:raise ValueError('AUTONOMOUS_ARRAY_CHANGED')
                        pixels=None
                        if kind=='semantic':
                            ids,counts=np.unique(a,return_counts=True);pixels={str(int(i)):int(n) for i,n in zip(ids,counts)}
                        arrays[p]=pixels
                    if kind=='semantic' and arrays[p]!=obs['pixels']:raise ValueError('AUTONOMOUS_PIXEL_COUNTS_CHANGED')
            result=evaluate(legacy.Compiler(**f['compiler']),trace,cond['task_id'],cut)
            if any(saved[k]!=v for k,v in result.items()):raise ValueError('LABEL_RECOMPUTATION')
            steps=c.records(folder/'POLICY_STEPS.jsonl')
            if len(steps)!=saved['autonomous_decisions']:raise ValueError('MISSING_STEPS')
            for t,step in enumerate(steps):
                if select(step['logits'],step['method_logits'])['executed_action']!=step['executed_action']:raise ValueError('SELECTOR_CHANGED')
                if c.ACTIONS['FLRS'.index(trace['actions'][cut+t])]!=step['executed_action']:raise ValueError('ACTION_LOG_MISMATCH')
            row.update(status=result['safe_v16_label'],legacy_label=result['legacy_v15_label'],cost=result['budget_penalized_cost'],
                decisions=result['total_decisions'],collisions=result['collisions'],exhausted=result['budget_exhausted'],
                ready_at_stop=result['ready_at_active_stop'],path=str(folder.relative_to(LINE)))
        rows.append(row)
    metrics=[]
    for endpoint in ('main','control'):
        for house in [None]+cfg['houses']:
            selected=[r for r in rows if r['endpoint']==endpoint and (house is None or r['house']==house)]
            metrics.append(dict(endpoint=endpoint,house=house,arm='EXPANDED',seed=1209,**metric(selected)))
    result=dict(status='COMPLETE_AVAILABLE_DEVELOPMENT_EVALUATION' if len(groups)==sum(x['available'] for x in reg['conditions']) else 'INCOMPLETE',
        step=cfg['evaluated_step'],complete=len(groups),planned=len(rows),metrics=metrics,
        missing=[r['rank'] for r in rows if r['status'] not in ('PASS','FAIL')],
        base_updates=0,paired_comparisons=0,adopted=False,raw_arrays_verified=len(arrays),
        limitation='Single seed 1209, 20 registered checkpoints on four exposed development houses; not a new OLD control, blind test, or natural VLN-CE result.')
    immutable(run/'ROLLOUTS.json',rows);immutable(run/'RESULT.json',result)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    if not (run/'ROLLOUTS.csv').exists():
        with (run/'ROLLOUTS.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (run/'REPORT_ZH.md').write_text(result['status']+'\n'+str(result)+'\n')

if __name__=='__main__':main(Path(sys.argv[1]))
