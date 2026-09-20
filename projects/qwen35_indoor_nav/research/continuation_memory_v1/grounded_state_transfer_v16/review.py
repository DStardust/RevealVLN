"""Recompute planned denominators and label identification bounds, never select a run by score."""
import csv
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
from evaluator_v16 import legacy,evaluate
from select_action import select

def timing_summary(rows,key):
    values=sorted(r[key] for r in rows if key in r)
    if not values:return dict(N=0,p50=None,p95=None)
    def quantile(q):
        position=(len(values)-1)*q;lo=int(position);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(position-lo)
    return dict(N=len(values),p50=quantile(.5),p95=quantile(.95),scope='shared GPU, includes possible contention; not exclusive deployment latency')

def rates(rows):
    n=len(rows);counts=Counter(r['safe_v16_label'] for r in rows)
    p=counts['PASS'];u=counts['UNKNOWN']+counts['ERROR']+counts['NOT_RUN']
    return dict(N=n,P=p,F=counts['FAIL'],U=counts['UNKNOWN'],ERROR=counts['ERROR'],NOT_RUN=counts['NOT_RUN'],
        pass_fraction=p/n if n else None,identification_bounds=[p/n,(p+u)/n] if n else None,
        budget_penalized_cost=sum(r.get('budget_penalized_cost',1.) for r in rows)/n if n else None,
        collision_trajectories=sum((r.get('collisions') or 0)>0 for r in rows),
        budget_exhausted=sum(bool(r.get('budget_exhausted')) for r in rows),
        wrong_stop=sum(r.get('stopped',False) and not r.get('ready_at_active_stop',False) for r in rows))

def difference(a,b):
    return dict(delta=a['pass_fraction']-b['pass_fraction'],identification_bounds=[a['identification_bounds'][0]-b['identification_bounds'][1],a['identification_bounds'][1]-b['identification_bounds'][0]],
                interval_kind='worst-case missing/unknown-label allocation, not a confidence interval')

def audit_rollout(folder,family,condition,slot):
    result=read(folder/'ROLLOUT.json');trace=read(folder/'TRACE_PRIVILEGED.json');steps=c.records(folder/'POLICY_STEPS.jsonl')
    if result['policy_sha256']!=sha(folder/'POLICY_STEPS.jsonl') or result['trace_sha256']!=sha(folder/'TRACE_PRIVILEGED.json'):raise ValueError('ROLLOUT_HASH')
    history=family['histories'][condition['history_id']];cut=len(history)
    if trace['actions'][:cut]!=history or len(trace['actions'])!=cut+len(steps) or len(trace['actions'])>500:raise ValueError('REVIEW_ACTION_BUDGET')
    if trace['actions'][-1]!='S' and len(trace['actions'])!=500:raise ValueError('EARLY_INCOMPLETE_EPISODE')
    for t,row in enumerate(steps,cut):
        expected=select(row['logits'],row['method_logits'])
        if row['decision']!=t or any(row[k]!=v for k,v in expected.items()):raise ValueError('SELECTOR_RECOMPUTE')
        if c.ACTIONS['FLRS'.index(trace['actions'][t])]!=row['executed_action']:raise ValueError('EXECUTED_ACTION_TRANSPORT')
        raw=row['raw']
        if raw['rgb_sha256']!=[o['rgb_hash'] for o in trace['observations'][max(0,t-1):t+1]]:raise ValueError('RGB_WINDOW_TRANSPORT')
        if raw['executed_history']!=[c.ACTIONS['FLR'.index(a)] for a in trace['actions'][max(0,t-8):t]]:raise ValueError('ACTION_HISTORY_TRANSPORT')
    compiler=legacy.Compiler(**family['compiler']);labels=evaluate(compiler,trace,condition['task_id'],cut)
    if any(result['task_result'][k]!=v for k,v in labels.items()):raise ValueError('CHECKER_RECOMPUTE_MISMATCH')
    return dict(**slot,**condition,**labels,status='COMPLETE',attempt=str(folder.parent.parent),complete=True,
                policy_sha256=sha(folder/'POLICY_STEPS.jsonl'),trace_sha256=sha(folder/'TRACE_PRIVILEGED.json'),
                native_stop_method_continue=sum(r['native_stop_method_continue'] for r in steps),
                repeated_inputs=len(steps)-len({r['raw']['input_key'] for r in steps}),
                mean_native_margin=statistics.mean(r['native_margin'] for r in steps),
                mean_method_margin=statistics.mean(r['method_margin'] for r in steps),
                timing={k:timing_summary(steps,k) for k in ('preprocess_seconds','inference_seconds','controller_seconds')})

def collect_report(run):
    counts=Counter();traces=decisions=arrays=0;accepted=[]
    for p in run.glob('proposals/*/FAILURE.json'):counts[read(p)['error']]+=1
    for p in run.glob('proposals/*/*.json'):
        row=read(p)
        if 'actions' in row and 'observations' in row:traces+=1;decisions+=len(row['actions'])
        if p.name=='FAMILY.json':accepted.append(dict(family_id=row['family_id'],house=row['house'],split=row['split']))
    return dict(accepted=accepted,rejection_counts=dict(counts),recorded_physical_traces=traces,
                actual_environment_decisions=decisions,raw_arrays=len(list((run/'content').glob('*.npy'))),
                attempted_proposals=len(list((run/'proposals').glob('*'))))

def main(run):
    config=read(run/'PROTOCOL.json');resources=[r for p in (HERE/'runs').glob('*/RESOURCE_SESSIONS.jsonl') for r in c.records(p)]
    collection=collect_report(run)
    if not (run/'DATA.json').exists():
        result=dict(status='DATA_COLLECTION_INCOMPLETE',planned_main_per_arm=192,planned_rollouts=720,
            completed_models=0,completed_rollouts=0,paired_effect=None,adopted=False,collection=collection,
            gpu_session_hours=sum(r['wall_seconds'] for r in resources)/3600,
            method_effect='UNKNOWN; no V16 arm has been trained or evaluated',test_started=False)
        write(run/'REVIEW.json',result)
        report=f"# DATA_COLLECTION_INCOMPLETE\n\n完整自主续接：0/720；训练模型：0/9；方法差值 UNKNOWN，未采用。\n\n本次采集目录实得 {len(collection['accepted'])} 个合格族；尚未满足 26 族注册规模。拒绝原因：`{json.dumps(collection['rejection_counts'],ensure_ascii=False)}`。\n\n累计已结束 GPU 会话 {result['gpu_session_hours']:.4f} 小时；尚未执行 B1/B2/Ours 比较。CPU 契约通过不等于方法收益。旧 V15 结果未修改。\n"
        (run/'REPORT_ZH.md').write_text(report)
        return result
    from evaluate_continuations import registry,admitted
    reg=registry(run);groups=admitted(run,reg);data=read(run/'DATA.json');families={f['family_id']:f for f in data['raw_families']}
    rows=[]
    for slot in reg['slots']:
        condition=reg['conditions'][slot['condition']];session=groups.get(slot['condition'])
        if session:
            rows.append(audit_rollout(session/'rollouts'/f'{slot["rank"]:04d}',families[condition['family_id']],condition,slot))
        else:
            attempts=list((run/'evaluate').glob(f'session_*/rollouts/{slot["rank"]:04d}'))
            rows.append(dict(**slot,**condition,status='UNSEALED_ATTEMPT' if attempts else 'NOT_RUN',complete=False,
                safe_v16_label='ERROR' if attempts else 'NOT_RUN',legacy_v15_label='UNKNOWN',attempts=list(map(str,attempts)),
                budget_penalized_cost=1.))
    mainrows=[r for r in rows if r['endpoint']=='main'];controls=[r for r in rows if r['endpoint']=='control']
    arms={a:rates([r for r in mainrows if r['arm']==a]) for a in config['arms']}
    houses={h:{a:rates([r for r in mainrows if r['house']==h and r['arm']==a]) for a in config['arms']} for h in sorted({r['house'] for r in mainrows})}
    seeds={str(s):{a:rates([r for r in mainrows if r['seed']==s and r['arm']==a]) for a in config['arms']} for s in config['seeds']}
    by_family={f:{a:rates([r for r in mainrows if r['family_id']==f and r['arm']==a]) for a in config['arms']} for f in sorted({r['family_id'] for r in mainrows})}
    delta=difference(arms['Ours'],arms['B2']);complete=all(r['complete'] for r in rows)
    mechanisms=read(run/'MEMORY_INTERVENTIONS.json') if (run/'MEMORY_INTERVENTIONS.json').exists() else dict(status='NOT_RUN',supports_history_specific_effect=False)
    criteria=dict(complete=complete,delta_at_least_10pp=delta['delta']>=.10,positive_under_worst_unknown=delta['identification_bounds'][0]>0,
        positive_seeds=sum(v['Ours']['pass_fraction']>v['B2']['pass_fraction'] for v in seeds.values())>=2,
        positive_houses=sum(v['Ours']['pass_fraction']>v['B2']['pass_fraction'] for v in houses.values())>=3,
        not_dominated_by_B1=arms['Ours']['pass_fraction']>=arms['B1']['pass_fraction'],
        registered_mechanism_support=mechanisms.get('supports_history_specific_effect') is True)
    if all(criteria.values()):decision='CONTINUE_OURS_WITHIN_REGISTERED_SCOPE'
    elif not complete:decision='INCOMPLETE_EFFECT_UNKNOWN'
    elif max(v['P'] for v in arms.values())==0:decision='COMMON_LEARNING_PATH_NOT_RECOVERED'
    elif arms['B2']['pass_fraction']>arms['B1']['pass_fraction']:decision='PRIORITIZE_B2; NOT_DEPLOYMENT_ACCEPTANCE'
    else:decision='PRIORITIZE_B1; NOT_DEPLOYMENT_ACCEPTANCE'
    pairs=[]
    for seed in config['seeds']:
        for condition in range(len(reg['conditions'])):
            if reg['conditions'][condition]['endpoint']!='main':continue
            subset={r['arm']:r for r in mainrows if r['condition']==condition and r['seed']==seed}
            for baseline in ('B1','B2'):
                a,b=subset['Ours'],subset[baseline]
                known=a['safe_v16_label'] in ('PASS','FAIL') and b['safe_v16_label'] in ('PASS','FAIL')
                pairs.append(dict(condition=condition,seed=seed,baseline=baseline,known=known,
                    difference=int(a['safe_v16_label']=='PASS')-int(b['safe_v16_label']=='PASS') if known else None))
    result=dict(status=decision,planned_rollouts=720,completed_rollouts=sum(r['complete'] for r in rows),complete_groups=len(groups),
        primary=arms,delta_ours_minus_b2=delta,by_house=houses,by_seed=seeds,by_family=by_family,paired_conditions=pairs,
        controls={a:rates([r for r in controls if r['arm']==a]) for a in config['arms']},criteria=criteria,
        scientific_generalization='4 TEST houses; correlated seeds, routes and task variants. Identification bounds are not confidence intervals.',
        adopted=False,r2r_full_sr_available=False,new_architecture_evidence=False,
        gpu_session_hours=sum(r['wall_seconds'] for r in resources)/3600)
    write(run/'REVIEW.json',result)
    parity=[dict(session=p.parent.name,**r) for p in (run/'evaluate').glob('session_*/CACHE_LIVE_PARITY.jsonl') for r in c.records(p)]
    write(run/'CACHE_LIVE_PARITY.json',dict(N=len(parity),native_argmax_flips=sum(r['native_flip'] for r in parity),
        method_argmax_flips=sum(r['method_flip'] for r in parity),max_feature_delta=max((r['feature_delta'] for r in parity),default=None),
        max_native_delta=max((r['native_delta'] for r in parity),default=None),max_method_delta=max((r['method_delta'] for r in parity),default=None),
        scope='Only actually encountered cached inputs; no claim of universal FLA repair.',rows=parity))
    keys=sorted({k for row in rows for k in row})
    with (run/'ROLLOUTS.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    (run/'REPORT_ZH.md').write_text(f"# {decision}\n\n完整续接 {result['completed_rollouts']}/720；完整九模型条件组 {len(groups)}/80。主任务每臂 N=192，task_T 单列。\n\n主终点：\n```json\n{json.dumps(arms,ensure_ascii=False,indent=2)}\n```\n\nOurs−B2：{delta['delta']:.4f}；未知标签最坏分配差值界：{delta['identification_bounds']}，不是置信区间。\n\n判定：{decision}。未自动采用或部署；无完整 R2R SR，无新架构收益证据。4 个 TEST 屋不足以支持广泛统计泛化；共同数据、STOP 选择和评测修复不归因于 Ours。\n")
    return result

if __name__=='__main__':main(Path(sys.argv[1]))
