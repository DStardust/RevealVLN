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
        collisions=sum(r.get('collisions',0)>0 for r in rows),exhausted=sum(r.get('exhausted',False) for r in rows),premature_stops=sum(r.get('premature_stop',False) for r in rows))

def paired(rows):
    keys=sorted({(r['condition'],r['seed']) for r in rows});indexed={(r['condition'],r['seed'],r['arm']):r for r in rows}
    wins=[];losses=[];ties=[];unknown=[]
    for condition,seed in keys:
        a=indexed[condition,seed,'ORIGINAL'];b=indexed[condition,seed,'REPAIR'];key=[condition,seed]
        if a['status'] not in ('PASS','FAIL') or b['status'] not in ('PASS','FAIL'):unknown.append(key)
        elif a['status']==b['status']:ties.append(key)
        elif b['status']=='PASS':wins.append(key)
        else:losses.append(key)
    a=metric([r for r in rows if r['arm']=='ORIGINAL']);b=metric([r for r in rows if r['arm']=='REPAIR'])
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
                ready_at_stop=result['ready_at_active_stop'],stopped=result['stopped'],premature_stop=bool(result['stopped'] and not result['event_order_satisfied']),path=str(folder.relative_to(LINE)))
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
    complete=len(groups)==len(reg['conditions']);signal=complete and main['delta_lower']>0 and positive_seeds>=2
    missing=paired([r for r in rows if r['endpoint']=='main' and r['history_id'].startswith('missing')]);signal=signal and missing['delta_lower']>0
    result=dict(status='VALID_COMPLETE_DEV_ACTION_PRESERVATION' if complete else 'INCOMPLETE_DEV_ACTION_PRESERVATION',complete=len(groups)*len(reg['models']),planned=len(rows),
        groups=len(groups),missing_history_pair=missing,metrics=metrics,pairs=pairs,strata=strata,positive_houses=positive_houses,positive_seeds=positive_seeds,
        supports_dev_repair_signal=signal,method_adopted=False,new_updates=600,event_probe_updates=0,base_updates=0,raw_arrays_verified=len(arrays),
        prefix=dict(comparisons=len(audits),bitwise_equal=sum(a['logits_bitwise_equal'] for a in audits),max_logit_delta=max((a['max_logit_delta'] for a in audits),default=None)),
        limitation='Exposed DEV1 house; same MONOTONIC architecture and original trajectory pool, Only200event-MLP updates from trained ORIGINAL; original action/recurrent/state-action/prior parameters frozen. Stationary SEE2, no independent test or natural VLN-CE claim.')
    result['retained_original_successes']=sum(r['endpoint']=='main' and r['arm']=='ORIGINAL' and r['status']=='PASS' for r in rows)-len(main['losses'])
    result['lost_original_successes']=len(main['losses']);result['new_successes']=len(main['wins'])
    immutable(run/'ROLLOUTS.json',rows);immutable(run/'RESULT.json',result)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (run/'ROLLOUTS.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=[result['status'],'',f"完整续接 {result['complete']}/{result['planned']}；完整六模型组 {len(groups)}/{len(reg['conditions'])}。本轮3个修复模型各200次事件MLP更新，共600次；3个ORIGINAL直接复用且本轮0更新，全部同进程重新评测；底模更新0。",
        '', '|终点|方法|PASS/N|未知|成本|碰撞轨迹|耗尽|未满足STOP|','|---|---|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        if m['house'] is None and m['seed'] is None:lines.append(f"|{m['endpoint']}|{m['arm']}|{m['passed']}/{m['planned']}|{m['unknown']}|{m['cost']:.4f}|{m['collisions']}|{m['exhausted']}|{m['premature_stops']}|")
    lines.extend(['',f"REPAIR−ORIGINAL 主终点差值：{main['delta']:.2%}；未知分配识别界 [{main['delta_lower']:.2%}, {main['delta_upper']:.2%}]。这不是置信区间。",
        f"方向为正的房屋 {positive_houses}/1、种子 {positive_seeds}/3；原DEV局部修复信号：{signal}。",'',
        'REPAIR从各seed已训练ORIGINAL出发，仅更新事件MLP200步；其余权重冻结。原动作、状态、KL及普通动作损失保留，事件BCE采用之前冻结的去重屋/角色等权方案。ORIGINAL本轮0更新，作为冻结参考重新运行；本轮衡量整项有界修复，不宣称对新监督做了等计算量方法比较。',
        '父族内历史、终点变体、同屋和多个种子有相关性，不能将768次执行当成768个独立泛化样本。',
        f"缺失历史主任务配对差：{missing['delta']:.2%}，识别界[{missing['delta_lower']:.2%},{missing['delta_upper']:.2%}]；固定200步事件复核另报，未重复留一屋筛选，不将读出指标当闭环收益。",'本轮不自动采用、不追加训练；普通VLN-CE收益与算法新颖性仍需另外证据。',result['limitation']])
    (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main(Path(sys.argv[1]))
