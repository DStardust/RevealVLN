"""Materialize identical ordinary-only decision order for the two history arms."""
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import time
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
BASE=HERE.parent/'ordinary_expanded_v1'
SNAP=LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def save(name,value):
    with (HERE/name).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=BASE/'SAMPLE_INDEX.jsonl'
    assert sha(source)=='632e69f94bec413acbf1c8cab3b138f94769b7399a8d42990dd053ec50f22e82'
    assert sha(SNAP/'TRAINING_INDEX.jsonl')=='cde540543491579d644ffeca8f42127511c45876a3a34830f15f27f35758d7ec'
    rows=[json.loads(x) for x in (SNAP/'TRAINING_INDEX.jsonl').read_text().splitlines()]
    split=json.loads((SNAP/'SPLIT.json').read_text())
    fit=set(split['FIT'])
    assert len(fit)==51 and fit.isdisjoint(split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
    assert len(rows)==37114 and all(r['split']=='FIT' and r['scene_group'] in fit for r in rows)
    total=2650347;count=384000;seed=1209
    order=random.Random(seed).sample(range(total),count)
    wanted=set(order);samples={}
    with source.open() as stream:
        for i,line in enumerate(stream):
            if i in wanted:samples[i]=json.loads(line)
    assert i+1==total and len(samples)==count
    sources=collections.Counter();houses=collections.Counter();actions=collections.Counter();records=set()
    for sample_id in order:
        record,t,target,weight,old_est=samples[sample_id]
        row=rows[record]
        assert 0<=t<row['decisions'] and target in range(4) and weight in (1.,3.2)
        assert row['scene_group'] in fit
        sources[row['source']]+=1;houses[row['scene_group']]+=1;actions[target]+=1;records.add(record)
    plan=[[] for _ in range(3)]
    for update in range(4000):
        group=order[update*96:(update+1)*96]
        for rank in range(3):plan[rank].append(group[rank*32:(rank+1)*32])
    merged=[idx for update in range(4000) for rank in range(3) for idx in plan[rank][update]]
    assert merged==order and all(len(b)==32 for rank in plan for b in rank)
    save('DECISION_ORDER.json',order)
    save('PLAN.json',plan)
    with (HERE/'SELECTED_SAMPLES.jsonl').open('x') as stream:
        for sample_id in order:
            stream.write(json.dumps(dict(source_sample_id=sample_id,entry=samples[sample_id]))+'\n')
    result=dict(status='CPU_PAIRED_ORDER_READY',unix=time.time(),runtime_allowed=False,
        arms=['control_recent2','treatment_prefix8'],common_seed=seed,
        original_index_sha256=sha(source),ordinary_snapshot_sha256=sha(SNAP/'TRAINING_INDEX.jsonl'),
        ordinary_only=True,special_or_recovery_decisions=0,fit_houses=len(houses),
        planned_updates_per_arm=4000,planned_decisions_per_arm=count,unique_decisions_per_arm=count,
        global_batch=96,rank_batch=32,microbatch=None,gradient_accumulation=None,
        learning_rate=None,optimizer_moment_resume=None,wall_budget_seconds=None,
        sources=dict(sources),house_decisions=dict(houses),action_counts=dict(actions),instruction_records=len(records),
        order_sha256=sha(HERE/'DECISION_ORDER.json'),plan_sha256=sha(HERE/'PLAN.json'),
        selected_samples_sha256=sha(HERE/'SELECTED_SAMPLES.jsonl'),
        new_training_updates=0,new_gpu_launches=0,paired_navigation_result=None)
    save('PREPARATION_RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='house_decisions'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

