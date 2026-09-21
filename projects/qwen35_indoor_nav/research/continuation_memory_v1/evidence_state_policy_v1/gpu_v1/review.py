"""Complete planned denominators, independent raw-label recomputation and paired effects."""
from collections import Counter, defaultdict
import csv
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from evaluate_continuations import registry, admitted
from evaluator_v16 import legacy, evaluate
from select_action import select

def main(run):
    cfg=config(run);reg=registry(run);groups=admitted(run,reg);data=read(run/'DATA.json');raw={f['family_id']:f for f in data['raw_families']};rows=[]
    for slot in reg['slots']:
        cond=reg['conditions'][slot['condition']];session=groups.get(slot['condition']);row=dict(slot,**cond,status='NOT_RUN',label='UNKNOWN',cost=1.)
        if session:
            folder=session/'rollouts'/f"{slot['rank']:04d}";saved=read(folder/'ROLLOUT.json')['task_result'];family=raw[cond['family_id']]
            trace=read(folder/'TRACE_PRIVILEGED.json');result=evaluate(legacy.Compiler(**family['compiler']),trace,cond['task_id'],len(family['histories'][cond['history_id']]))
            for key,value in result.items():
                if saved[key]!=value:raise ValueError('LABEL_RECOMPUTATION_DIFFERS')
            steps=c.records(folder/'POLICY_STEPS.jsonl');cut=len(family['histories'][cond['history_id']])
            if len(steps)!=saved['autonomous_decisions']:raise ValueError('MISSING_DECISIONS')
            for t,s in enumerate(steps):
                if select(s['logits'],s['method_logits'])['executed_action']!=s['executed_action']:raise ValueError('SELECTOR_CHANGED')
                if c.ACTIONS['FLRS'.index(trace['actions'][cut+t])]!=s['executed_action']:raise ValueError('EXECUTED_ACTION_CHANGED')
            row.update(status=result['safe_v16_label'],label=result['safe_v16_label'],cost=result['budget_penalized_cost'],
                collisions=result['collisions'],decisions=result['total_decisions'],exhausted=result['budget_exhausted'],ready_at_stop=result['ready_at_active_stop'])
        rows.append(row)
    immutable(run/'ROLLOUTS.json',rows)
    aggregates=[]
    for endpoint in ('main','control'):
        for arm in cfg['arms']:
            selected=[r for r in rows if r['endpoint']==endpoint and r['arm']==arm];count=Counter(r['status'] for r in selected)
            aggregates.append(dict(endpoint=endpoint,arm=arm,planned=len(selected),counts=count,success_lower=count['PASS']/len(selected),
                success_upper=(count['PASS']+count['UNKNOWN']+count['ERROR']+count['NOT_RUN'])/len(selected),mean_penalized_cost=sum(x['cost'] for x in selected)/len(selected)))
    pairs=[]
    for seed in cfg['seeds']:
        for endpoint in ('main','control'):
            selected=[r for r in rows if r['seed']==seed and r['endpoint']==endpoint]
            for baseline in ('DIRECT','MONOTONIC'):
                wins=[];losses=[];ties=[];unknown=[]
                for condition in sorted({r['condition'] for r in selected}):
                    ours=next(r for r in selected if r['condition']==condition and r['arm']=='REVISE')
                    base=next(r for r in selected if r['condition']==condition and r['arm']==baseline)
                    if ours['status'] not in ('PASS','FAIL') or base['status'] not in ('PASS','FAIL'):unknown.append(condition)
                    elif (ours['label']=='PASS')==(base['label']=='PASS'):ties.append(condition)
                    elif ours['label']=='PASS':wins.append(condition)
                    else:losses.append(condition)
                n=len(wins)+len(losses)+len(ties)+len(unknown)
                pairs.append(dict(seed=seed,endpoint=endpoint,baseline=baseline,wins=wins,losses=losses,ties=ties,unknown=unknown,
                    delta_lower=(len(wins)-len(losses)-len(unknown))/n,delta_upper=(len(wins)-len(losses)+len(unknown))/n))
    complete=len(groups)==len(reg['conditions'])
    result=dict(status='VALID_COMPLETE_DEVELOPMENT_PILOT' if complete else 'INCOMPLETE_DEVELOPMENT_PILOT',planned=len(reg['slots']),complete=len(groups)*9,identified=sum(r['status'] in ('PASS','FAIL') for r in rows),
        groups=len(groups),metrics=aggregates,pairs=pairs,method_adopted=False,independent_DEV_houses=1,
        interpretation='Controlled stationary SEE2 tasks in an exposed DEV house. No ordinary VLN-CE or paper generalization claim.')
    strata=[]
    for stratum in ('terminal_present','terminal_absent'):
        for endpoint in ('main','control'):
            for history_group in ('all','seen','missing'):
                for seed in [None]+cfg['seeds']:
                    for arm in cfg['arms']:
                        selected=[r for r in rows if r['stratum']==stratum and r['endpoint']==endpoint and r['arm']==arm
                                  and (seed is None or r['seed']==seed) and (history_group=='all' or r['history_id'].startswith(history_group))]
                        n=len(selected);counts=Counter(r['status'] for r in selected)
                        strata.append(dict(stratum=stratum,endpoint=endpoint,history=history_group,seed=seed,arm=arm,
                            planned=n,passed=counts['PASS'],unknown=sum(counts[k] for k in ('UNKNOWN','ERROR','NOT_RUN')),
                            success=counts['PASS']/n,penalized_cost=sum(r['cost'] for r in selected)/n,
                            collision_episodes=sum(r.get('collisions',0)>0 for r in selected),budget_exhausted=sum(r.get('exhausted',False) for r in selected)))
    result['stratified_metrics']=strata
    result['primary_comparison']='REVISE minus DIRECT, shared CPU-audited data; main task split by takeover coverage and missing/seen history'
    result['data_increment_causal_effect_identified']=False
    result['parent_dependence']='Variants, histories and seeds share eight exposed DEV parent families in one house; not independent samples.'
    result['training_intervention']='Shared event/state supervision and state action input; DIRECT vs MONOTONIC vs learned REVISE history transition; full KL unchanged'
    result['old_rollouts_reused']=False
    immutable(run/'RESULT.json',result)
    lines=['# DIRECT / MONOTONIC / REVISE 匹配对照结果','',result['status'],f"完整配对组 {len(groups)}/{len(reg['conditions'])}；续接 {result['complete']}/{len(reg['slots'])}。",'',
           '|终点|方法|PASS/计划|未知|预算惩罚成本|','|---|---|---:|---:|---:|']
    for r in aggregates:
        counts=r['counts'];unknown=sum(counts.get(k,0) for k in ('UNKNOWN','ERROR','NOT_RUN'))
        lines.append(f"|{r['endpoint']}|{r['arm']}|{counts.get('PASS',0)}/{r['planned']}|{unknown}|{r['mean_penalized_cost']:.4f}|")
    lines += ['','|接管条件|历史|方法|主任务PASS/N|成本|','|---|---|---|---:|---:|']
    for r in strata:
        if r['endpoint']=='main' and r['seed'] is None:
            lines.append(f"|{r['stratum']}|{r['history']}|{r['arm']}|{r['passed']}/{r['planned']}|{r['penalized_cost']:.4f}|")
    lines += ['',result['interpretation'],'','DIAG_*.json 是缓存教师路径诊断，CALIBRATION.json 是自主轨迹上的只读校准统计，均不替代闭环；未排除完整动作历史捷径。全部模型固定 final1200，没有按 DEV 选 checkpoint。']
    path=run/'REPORT_ZH.md'
    with path.open('x') as stream:stream.write('\n'.join(lines)+'\n')

if __name__=='__main__':main(Path(sys.argv[1]))
