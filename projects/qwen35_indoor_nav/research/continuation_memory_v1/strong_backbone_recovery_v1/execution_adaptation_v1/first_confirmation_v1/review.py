"""Separate the 289-route follow-up from the 80-route discovery block."""
import csv
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys

HERE=Path(__file__).resolve().parent;ADAPT=HERE.parent
sys.path.insert(0,str(ADAPT.parent))
import common as u
spec=importlib.util.spec_from_file_location('frozen_scope_review',ADAPT/'scope_validation_v1/review.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
records=old.records


def bootstrap(rows,left,right,seeds,repetitions=5000):
    by_house={}
    for r in rows.values():
        diffs=[]
        for seed in seeds:
            a=right.format(seed=seed);b=left.format(seed=seed)
            diffs.append(r['outcomes'][a]['success']-r['outcomes'][b]['success'])
        by_house.setdefault(r['house'],[]).append(statistics.mean(diffs))
    houses=sorted(by_house);rng=random.Random(1209);values=[]
    for _ in range(repetitions):
        sample=[x for h in rng.choices(houses,k=len(houses)) for x in by_house[h]]
        values.append(statistics.mean(sample))
    values.sort()
    return dict(houses=len(houses),resamples=repetitions,interval95=[values[int(.025*repetitions)],values[int(.975*repetitions)]],
        interpretation='Descriptive house-cluster bootstrap; few exposed houses, correlated seeds and routes, not a blind-test generalization claim.')


def summary(run):
    p=u.read(run/'PROTOCOL.json');entries=u.read(run/'DATA_MANIFEST.json')['episodes'];rows=records(run)
    r=old.summarize(p,entries,rows)
    r.update(scope='PRIMARY_289_REMAINDER_FIRST_ONLY',discovery_routes_excluded=80,
        already_exposed=True,automatic_replacement=False)
    return r


def review(run):
    u.verify_sources(run);p=u.read(run/'PROTOCOL.json');entries=u.read(run/'DATA_MANIFEST.json')['episodes'];rows=records(run)
    result=summary(run)
    if not result['evaluation_complete']:raise ValueError('INCOMPLETE_FOLLOWUP_DENOMINATOR')
    table=[];failures={};decomposition={}
    for arm in ['NATIVE',*p['heads']]:
        failures[arm]=dict(far_stop=0,entered_range_without_success=0,budget_exhausted=0,collisions=0)
        decomposition[arm]={c:dict(n=0,spl_difference_sum=0.,step_difference_sum=0.) for c in ['both_success','rescue','regression','both_fail']}
    for i,r in sorted(rows.items()):
        for arm,outcome in r['outcomes'].items():
            path=Path(r['path']).parent/arm/'TRACE.jsonl'
            if u.sha(path)!=r['trace_hashes'][arm]:raise ValueError('TRACE_CHANGED')
            actions=[e for e in map(json.loads,path.read_text().splitlines()) if e['event']=='action'];last=actions[-1]
            f=failures[arm];f['far_stop']+=last['executed_action']==0 and last['distance']>=3
            f['entered_range_without_success']+=any(e['distance']<3 for e in actions) and not outcome['success']
            f['budget_exhausted']+=len(actions)>=500 and last['executed_action']!=0
            f['collisions']+=sum(e['collision'] for e in actions)
            native=r['outcomes']['NATIVE'];c='both_success' if outcome['success'] and native['success'] else 'rescue' if outcome['success'] else 'regression' if native['success'] else 'both_fail'
            b=decomposition[arm][c];b['n']+=1;b['spl_difference_sum']+=outcome['spl']-native['spl'];b['step_difference_sum']+=outcome['steps']-native['steps']
            table.append(dict(id=i,house=r['house'],arm=arm,success=outcome['success'],spl=outcome['spl'],steps=outcome['steps'],
                budget_penalty=outcome['steps']/500 if outcome['success'] else 1.,
                trace=str(path),trace_sha256=r['trace_hashes'][arm]))
    with (run/'ROLLOUTS.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    result['failure_types']=failures;result['efficiency_decomposition']=decomposition
    result['budget_penalty']={a:statistics.mean(x['budget_penalty'] for x in table if x['arm']==a) for a in failures}
    result['descriptive_uncertainty']={
        'DELTA_minus_CURRENT':bootstrap(rows,'CURRENT_FIRST_s{seed}','DELTA_FIRST_s{seed}',p['seeds']),
        'DELTA_minus_NATIVE':bootstrap(rows,'NATIVE','DELTA_FIRST_s{seed}',p['seeds'])}
    by_mode={}
    for mode in ['CURRENT','DELTA']:
        names=[f'{mode}_FIRST_s{s}' for s in p['seeds']]
        sr=statistics.mean(result['arms'][a]['sr'] for a in names)
        by_mode[mode]=dict(mean_sr=sr,delta_native_sr=sr-result['arms']['NATIVE']['sr'],
            mean_spl=statistics.mean(result['arms'][a]['spl'] for a in names),
            mean_steps=statistics.mean(result['arms'][a]['mean_steps'] for a in names),
            positive_seeds_vs_native=sum(result['arms'][a]['sr']>result['arms']['NATIVE']['sr'] for a in names))
    result['by_mode']=by_mode
    delta=by_mode['DELTA']['mean_sr']-by_mode['CURRENT']['mean_sr']
    positive=sum(result['arms'][f'DELTA_FIRST_s{s}']['sr']>result['arms'][f'CURRENT_FIRST_s{s}']['sr'] for s in p['seeds'])
    result['decision']=dict(delta_unique_development_signal=delta>0 and positive>=2,
        delta_minus_current=delta,positive_seeds_delta_minus_current=positive,
        retained_candidates=[m for m,v in by_mode.items() if v['delta_native_sr']>0 and v['positive_seeds_vs_native']>=2],
        uncertainty_and_spl_cost_tradeoffs_must_be_read=True,automatic_adoption=False)
    discovery=Path(p['discovery_run'])
    if u.sha(discovery/'RESULT.json')!=p['discovery_result_sha256']:raise ValueError('DISCOVERY_RESULT_CHANGED')
    prior=records(discovery);prior_entries=u.read(discovery/'DATA_MANIFEST.json')['episodes'];prior_p=u.read(discovery/'PROTOCOL.json')
    if len(prior)!=80 or set(prior)&set(rows):raise ValueError('DISCOVERY_FOLLOWUP_OVERLAP')
    for a,head in p['heads'].items():
        if head!=prior_p['heads'][a]:raise ValueError('CANNOT_POOL_CHANGED_HEAD_OR_SCOPE')
    prior={i:dict(r,outcomes={a:r['outcomes'][a] for a in ['NATIVE',*p['heads']]}) for i,r in prior.items()}
    pooled=old.summarize(p,prior_entries+entries,prior | rows)
    if pooled['complete_groups']!=369:raise ValueError('POOLED_DENOMINATOR_CHANGED')
    pooled.update(scope='SECONDARY_DESCRIPTIVE_DISCOVERY80_PLUS_FOLLOWUP289',
        selection_bias='FIRST selected using discovery80; this pooled score is not the primary confirmation result.',
        blocks=dict(discovery80=str(discovery),followup289=str(run)),head_registry_sizes_differ_by_block=True)
    u.write(run/'POOLED_369_DESCRIPTIVE.json',pooled)
    u.write(run/'RESULT.json',result)
    lines=['COMPLETE','',f"主分母 {len(rows)} 条剩余 unseen，{len(table)} 次执行；发现集80条另列。0次训练更新。",'']
    for a,v in result['arms'].items():lines.append(f"{a}: SR={v['sr']:.4f}, SPL={v['spl']:.4f}, steps={v['mean_steps']:.2f}")
    lines+=['','DELTA−CURRENT 平均 SR 差：'+str(delta),'各种子和房屋共享数据；区间为描述性，不是独立泛化证明。',
        '这289条没有参与本次FIRST范围选择，但此前基线/其他方法结果已有暴露。',
        '369条合并仅作次要描述，不能盖过289条主结果。旧CONCAT候选继续保留，不自动替换。']
    (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return result
