"""Final full-denominator report, independent trace audit and house uncertainty."""
import collections
import csv
import gzip
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

HERE=Path(__file__).resolve().parent;RUN=HERE/'run_001'
s=importlib.util.spec_from_file_location('common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def read(path):return json.loads(path.read_text())
def records(path):
    with path.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def exact_ndtw_check(positions,reference):
    unique=[x for i,x in enumerate(positions) if i==0 or x!=positions[i-1]]
    cost=[[math.inf]*(len(reference)+1) for _ in range(len(unique)+1)];cost[0][0]=0.
    for i,a in enumerate(unique,1):
        for j,b in enumerate(reference,1):cost[i][j]=math.dist(a,b)+min(cost[i-1][j],cost[i][j-1],cost[i-1][j-1])
    return math.exp(-cost[-1][-1]/(3*len(reference)))


def stats(rows):
    return dict(n=len(rows),successes=int(sum(x['success'] for x in rows)),**{
        output:statistics.mean(x[key] for x in rows) if rows else None for output,key in
        [('sr','success'),('spl','spl'),('ne_m','navigation_error_m'),('osr','oracle_success'),('ndtw','ndtw'),('sdtw','sdtw')]})


def main():
    c.verify_lock();p=read(HERE/'PROTOCOL.json');launch=read(RUN/'LAUNCH_RESULT.json')
    scheduled=read(HERE/'EPISODES_PRIVILEGED.json');gt=json.load(gzip.open(p['gt_path']))
    rows=[];audited=0
    for lane in range(p['lanes']):
        folder=RUN/'lanes'/f'lane_{lane:02d}'
        completed=[read(x) for x in sorted(folder.glob('episode_*.json'))];lookup={x['index']:x for x in completed}
        for e in completed:
            truth=scheduled[e['index']];assert e['episode_id']==truth['episode_id'] and e['house']==truth['scene_id'].split('/')[-2]
            assert len(e['positions'])==len(e['distances'])==e['steps']+1 and 1<=e['steps']<=500
            assert e['stopped'] or e['steps']==500
            success=float(e['stopped'] and e['distances'][-1]<3.)
            path=sum(math.dist(a,b) for a,b in zip(e['positions'],e['positions'][1:]))
            spl=success*e['distances'][0]/max(e['distances'][0],path)
            assert success==e['success'] and e['navigation_error_m']==e['distances'][-1]
            assert e['oracle_success']==float(min(e['distances'])<3.)
            assert math.isclose(path,e['path_length_m'],rel_tol=1e-5,abs_tol=1e-5)
            assert math.isclose(spl,e['spl'],rel_tol=1e-5,abs_tol=1e-5)
            ndtw=exact_ndtw_check(e['positions'],gt[str(e['episode_id'])]['locations'])
            assert math.isclose(ndtw,e['ndtw'],rel_tol=1e-9,abs_tol=1e-9)
            assert math.isclose(success*ndtw,e['sdtw'],rel_tol=1e-9,abs_tol=1e-9)
        counters=collections.defaultdict(collections.Counter);last_steps=collections.defaultdict(int)
        if (folder/'POLICY_STEPS.jsonl').exists() and (folder/'STEPS_PRIVILEGED.jsonl').exists():
            for policy,step in itertools.zip_longest(records(folder/'POLICY_STEPS.jsonl'),records(folder/'STEPS_PRIVILEGED.jsonl')):
                if policy is None or step is None:
                    assert launch['status']!='COMPLETE';continue
                assert policy['index']==step['index'] and policy['step']==step['step'] and policy['action']==step['action']
                index=step['index'];st=step['step'];assert st==last_steps[index]+1;last_steps[index]=st
                assert policy['images']==min(2,st) and policy['executed_history']==min(8,st-1)
                counters[index][step['action']]+=1;counters[index]['collisions']+=int(step['collided'])
                if index in lookup:
                    e=lookup[index];assert step['position']==e['positions'][st] and step['distance_to_goal']==e['distances'][st]
                    assert step['action']!='STOP' or st==e['steps']
                audited+=1
        for index,e in lookup.items():
            assert last_steps[index]==e['steps']
            assert all(counters[index][a]==n for a,n in e['action_counts'].items())
            assert counters[index]['collisions']==e['collisions']
        rows.extend(completed)
    assert len(set(x['index'] for x in rows))==len(rows)
    rows.sort(key=lambda x:x['index']);full=launch['status']=='COMPLETE' and len(rows)==p['episode_count']
    if full:
        assert [x['index'] for x in rows]==list(range(p['episode_count']))
        inference=read(RUN/'INFERENCE_RESULT.json')
        assert inference['completed']==p['episode_count'] and inference['trainable_unchanged'] and inference['optimizer_updates']==0
        assert audited==inference['total_actions']==sum(x['steps'] for x in rows)
    assert full, 'INCOMPLETE_DEV_CASE_NO_ADMITTED_METRICS'
    overall=stats(rows);by_house={h:stats([x for x in rows if x['house']==h]) for h in p['houses']}
    result=dict(status='COMPLETE',unix=time.time(),checkpoint_updates=p['checkpoint_updates'],
        checkpoint_sha256=p['checkpoint_sha256'],benchmark='R2R-CE train held-out INTERNAL_DEV diagnostic',
        planned=p['episode_count'],completed=len(rows),missing=0,**overall,house_count=len(p['houses']),by_house=by_house,
        stopped=sum(int(x['stopped']) for x in rows),failure_categories=dict(collections.Counter(x['failure_category'] for x in rows)),
        collisions=sum(x['collisions'] for x in rows),environment_actions=sum(x['steps'] for x in rows),
        trace_audit_passed=True,audited_actions=audited,source_lock_verified=True,
        selected_batch_size=read(RUN/'BATCH_PARITY_GATE.json').get('selected_batch_size'),
        optimizer_updates=0,scientific_gain_verified=False,wall_seconds=launch['wall_seconds'],
        training_processes_signaled=[],checkpoint_role=p['checkpoint_role'],run_dir=str(RUN))
    fields=['index','episode_id','trajectory_id','house','success','spl','navigation_error_m','oracle_success','ndtw','sdtw','steps','stopped','collisions','failure_category']
    with (RUN/'episodes.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:x[k] for k in fields} for x in rows)
    c.write(RUN/'RESULT.json',result,True)
    c.write(RUN/'PROGRESS.json',dict(status='COMPLETE',unix=time.time(),completed=len(rows),total=p['episode_count']))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
