"""Recompute sealed groups, keep planned missing slots, report paired house/seed effects."""
from collections import Counter
import csv
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from evaluate_continuations import registry,admitted
from evaluator_v16 import legacy,evaluate
from select_action import select

def metric(rows):
    n=len(rows);counts=Counter(r['status'] for r in rows)
    u=sum(counts[k] for k in ('UNKNOWN','ERROR','NOT_RUN','NOT_COLLECTED'))
    return dict(planned=n,passed=counts['PASS'],unknown=u,counts=dict(counts),
        lower=counts['PASS']/n,upper=(counts['PASS']+u)/n,cost=sum(r['cost'] for r in rows)/n,
        collisions=sum(r.get('collisions',0)>0 for r in rows),exhausted=sum(r.get('exhausted',False) for r in rows))

def paired(rows):
    keys=sorted({(r['condition'],r['seed']) for r in rows});indexed={(r['condition'],r['seed'],r['arm']):r for r in rows}
    wins=[];losses=[];ties=[];unknown=[]
    for condition,seed in keys:
        a=indexed[condition,seed,'DIRECT'];b=indexed[condition,seed,'MONOTONIC'];key=[condition,seed]
        if a['status'] not in ('PASS','FAIL') or b['status'] not in ('PASS','FAIL'):unknown.append(key)
        elif a['status']==b['status']:ties.append(key)
        elif b['status']=='PASS':wins.append(key)
        else:losses.append(key)
    a=metric([r for r in rows if r['arm']=='DIRECT']);b=metric([r for r in rows if r['arm']=='MONOTONIC'])
    return dict(n=len(keys),wins=wins,losses=losses,ties=ties,unidentified=unknown,
        delta=b['lower']-a['lower'],delta_lower=b['lower']-a['upper'],delta_upper=b['upper']-a['lower'],
        note='Identification bounds, not confidence intervals. Variants, histories and seeds within a house are correlated.')

def main(run):
    import numpy as np
    cfg=config(run);reg=registry(run);groups=admitted(run,reg);families={f['family_id']:f for f in read(run/'DATA.json')['raw_families']};rows=[];arrays={};audits=[]
    for session in set(groups.values()):
        for p in session.glob('GROUP_*.json'):
            row=read(p)
            if groups.get(row['condition'])==session:audits.extend(row['audits'])
    if len(audits)!=len(groups)*len(cfg['seeds']) or any(not a['input_prefix_matched'] or not a['action_prefix_matched'] or a['argmax_flip_count'] for a in audits):raise ValueError('PREFIX_AUDIT')
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
    metrics=[];pairs=[]
    for endpoint in ('main','control'):
        endpoint_rows=[r for r in rows if r['endpoint']==endpoint]
        for house in [None]+cfg['houses']:
            for seed in [None]+cfg['seeds']:
                selected=[r for r in endpoint_rows if (house is None or r['house']==house) and (seed is None or r['seed']==seed)]
                pairs.append(dict(endpoint=endpoint,house=house,seed=seed,**paired(selected)))
                for arm in cfg['arms']:metrics.append(dict(endpoint=endpoint,house=house,seed=seed,arm=arm,**metric([r for r in selected if r['arm']==arm])))
    strata=[]
    for endpoint in ('main','control'):
        for stratum in ('terminal_present','terminal_absent'):
            for history in ('seen','missing'):
                for arm in cfg['arms']:
                    selected=[r for r in rows if r['endpoint']==endpoint and r['stratum']==stratum and r['history_id'].startswith(history) and r['arm']==arm]
                    strata.append(dict(endpoint=endpoint,stratum=stratum,history=history,arm=arm,**metric(selected)))
    main=next(p for p in pairs if p['endpoint']=='main' and p['house'] is None and p['seed'] is None)
    positive_houses=sum(p['delta_lower']>0 for p in pairs if p['endpoint']=='main' and p['house'] is not None and p['seed'] is None)
    positive_seeds=sum(p['delta_lower']>0 for p in pairs if p['endpoint']=='main' and p['house'] is None and p['seed'] is not None)
    complete=len(groups)==len(reg['conditions']);signal=complete and main['delta_lower']>0 and positive_houses>=3 and positive_seeds>=2
    result=dict(status='VALID_COMPLETE_CONTROLLED_HOLDOUT' if complete else 'INCOMPLETE_PLANNED_HOLDOUT',complete=len(groups)*len(reg['models']),planned=len(rows),
        groups=len(groups),metrics=metrics,pairs=pairs,strata=strata,positive_houses=positive_houses,positive_seeds=positive_seeds,
        supports_limited_new_house_signal=signal,method_adopted=False,new_updates=0,raw_arrays_verified=len(arrays),
        prefix=dict(comparisons=len(audits),bitwise_equal=sum(a['logits_bitwise_equal'] for a in audits),max_logit_delta=max((a['max_logit_delta'] for a in audits),default=None)),
        limitation='Four new memory-evaluation houses, shared instruction templates and stationary SEE2 task. Base-training exposure unresolved; no natural VLN-CE, deployment or paper acceptance claim.')
    immutable(run/'ROLLOUTS.json',rows);immutable(run/'RESULT.json',result)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (run/'ROLLOUTS.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=[result['status'],'',f"完整续接 {result['complete']}/{result['planned']}；完整六模型组 {len(groups)}/{len(reg['conditions'])}。新增训练更新 0。",
        '', '|终点|方法|PASS/N|未知/未采集/未运行|成本|','|---|---|---:|---:|---:|']
    for m in metrics:
        if m['house'] is None and m['seed'] is None:lines.append(f"|{m['endpoint']}|{m['arm']}|{m['passed']}/{m['planned']}|{m['unknown']}|{m['cost']:.4f}|")
    lines.extend(['',f"MONOTONIC−DIRECT 主终点差值：{main['delta']:.2%}；未知分配识别界 [{main['delta_lower']:.2%}, {main['delta_upper']:.2%}]。这不是置信区间。",
        f"方向为正的房屋 {positive_houses}/4、种子 {positive_seeds}/3；有限新屋信号：{signal}。",'',
        '既有开发屋结果用于选择本次候选，未重命名为盲测。此次权重、采样顺序与终点在新屋模型分数产生前冻结。',
        '父族内历史、终点变体、同屋和多个种子有相关性，不能将1536次执行当成1536个独立泛化样本。',
        '本轮不自动采用、不继续训练；普通VLN-CE收益与算法新颖性仍需另外证据。',result['limitation']])
    (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main(Path(sys.argv[1]))
