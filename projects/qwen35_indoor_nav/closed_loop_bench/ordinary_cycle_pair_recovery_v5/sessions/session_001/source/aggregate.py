"""Cumulative valid-pair denominator, unresolved attempts, effect and costs separately."""
import collections
import math
from pathlib import Path
import random
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c


def percentile(values,q):
    if not values: return None
    ordered=sorted(values)
    x=(len(ordered)-1)*q
    a=int(x);b=min(a+1,len(ordered)-1)
    return ordered[a]+(ordered[b]-ordered[a])*(x-a)


def metrics(rows):
    return dict(n=len(rows),**{key:statistics.mean(r[field] for r in rows) if rows else None
        for key,field in [('sr','success'),('spl','spl'),('ndtw','ndtw'),('osr','oracle_success')]})


def stop_class(e):
    if e['success']: return 'success'
    if min(e['distances'])<3:
        return 'left_range_then_stopped_far' if e['stopped'] else 'entered_range_budget_no_stop'
    return 'far_stop_never_entered' if e['stopped'] else 'budget_never_entered'


def main():
    found=c.committed_pairs()
    order=c.read(HERE/'PAIR_ORDER.json')
    rows=[found[r] for r in sorted(found)]
    issues=[]
    resources=[]
    gpu_seconds=0.
    identity_blocks={}
    for session in sorted((HERE/'sessions').glob('session_*')):
        if (session/'FAILURE.json').exists(): issues.append(dict(session=session.name,**c.read(session/'FAILURE.json')))
        if (session/'LAUNCH_RESULT.json').exists(): gpu_seconds+=c.read(session/'LAUNCH_RESULT.json')['gpu_seconds']
        elif (session/'RESOURCE.jsonl').exists():
            current=c.records(session/'RESOURCE.jsonl')
            if current: gpu_seconds+=current[-1]['elapsed_seconds']
        if (session/'RUNTIME_IDENTITY.json').exists():
            identity=c.read(session/'RUNTIME_IDENTITY.json')
            identity_blocks[session.name]=dict(gpu_uuid=identity['gpu_uuid'],device=identity['device'],software=identity['software'],
                loaded_state_sha256=identity['loaded_state_sha256'],protocol_sha256=identity['protocol_sha256'],
                source_hashes=identity['source_hashes'],completed_pairs=sum(r['session']==session.name for r in rows))
    arm_stats={}
    for arm in ('A','B'):
        episode_rows=[r['episodes'][arm] for r in rows]
        decisions=[];per_episode=[];timings={'competition':[],'no_foreign_observed':[]}
        for r in rows:
            folder=HERE/r['artifact'];folder=folder.parent/arm
            log=c.records(folder/'POLICY_STEPS.jsonl')
            for name,digest in r['logs'][arm].items(): assert c.sha(folder/name)==digest,'PAIR_LOG_CHANGED'
            decisions.extend(log)
            resource_path=HERE/'sessions'/r['session']/'RESOURCE.jsonl'
            snapshot_rows=c.records(resource_path) if resource_path.exists() else []
            j=0
            for d in log:
                while j+1<len(snapshot_rows) and snapshot_rows[j+1]['unix']<=d['unix']: j+=1
                competition=bool(snapshot_rows and snapshot_rows[j]['resource_competition'])
                timings['competition' if competition else 'no_foreign_observed'].append(d['inference_seconds'])
            per_episode.append(dict(episode_id=r['episode_id'],rank=r['rank'],decisions=len(log),
                repeats=sum(d['repeated_input'] for d in log),repeat_ratio=sum(d['repeated_input'] for d in log)/len(log),
                overrides=sum(d['override'] for d in log),first_override=next((d['step'] for d in log if d['override']),None),
                stop_class=stop_class(r['episodes'][arm]),final_stop_margin=log[-1]['stop_margin']))
        arm_stats[arm]=dict(**metrics(episode_rows),by_house={h:metrics([r['episodes'][arm] for r in rows if r['house']==h]) for h in sorted({r['house'] for r in order})},
            total_decisions=len(decisions),collisions=sum(r['collisions'] for r in episode_rows),
            repeated_decisions=sum(d['repeated_input'] for d in decisions),overrides=sum(d['override'] for d in decisions),
            failure_classes=dict(collections.Counter(stop_class(r) for r in episode_rows)),per_episode=per_episode,
            latency={field:dict(p50=percentile([d[field] for d in decisions],.5),p95=percentile([d[field] for d in decisions],.95))
                for field in ('inference_seconds','preprocess_seconds','controller_seconds')},
            inference_latency_by_observed_competition={key:dict(n=len(values),p50=percentile(values,.5),p95=percentile(values,.95)) for key,values in timings.items()})
    differences={key:arm_stats['B'][key]-arm_stats['A'][key] if rows else None for key in ('sr','spl','ndtw','osr')}
    wins=[r['episode_id'] for r in rows if r['episodes']['B']['success']>r['episodes']['A']['success']]
    losses=[r['episode_id'] for r in rows if r['episodes']['B']['success']<r['episodes']['A']['success']]
    uncertainty=None
    if rows:
        rng=random.Random(1209)
        deltas=[r['episodes']['B']['success']-r['episodes']['A']['success'] for r in rows]
        bootstrap=[statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(2000)]
        houses=sorted({r['house'] for r in rows})
        house_boot=[]
        for _ in range(2000):
            draw=[r['episodes']['B']['success']-r['episodes']['A']['success'] for h in rng.choices(houses,k=len(houses)) for r in rows if r['house']==h]
            house_boot.append(statistics.mean(draw))
        uncertainty=dict(episode_bootstrap95=[percentile(bootstrap,.025),percentile(bootstrap,.975)],
            house_bootstrap95=[percentile(house_boot,.025),percentile(house_boot,.975)],
            caveat='Descriptive development intervals; correlated episodes and only five houses, no blind/generalization inference')
    result=dict(protocol='V5 behavioral pair protocol, not V3/V4 bitwise acceptance',planned_pairs=100,
        complete_valid_pairs=len(rows),pending_pairs=100-len(rows),complete_episode_executions=2*len(rows),
        model_loaded=bool(identity_blocks),experiment_complete=len(rows)==100,
        behavioral_comparison_valid=bool(rows),full100_effect_evaluable=len(rows)==100,
        development_positive_signal=bool(rows and differences['sr']>0),adopted=False,
        effect_scope='FULL_INTERNAL_DEV100' if len(rows)==100 else 'PARTIAL_DEVELOPMENT_ONLY',
        benefit='UNKNOWN' if not rows else 'positive_delta_sr' if differences['sr']>0 else 'no_positive_delta_sr',
        delta=differences,wins=wins,losses=losses,retained_successes=sum(r['episodes']['A']['success'] and r['episodes']['B']['success'] for r in rows),
        by_house_delta={h:{key:arm_stats['B']['by_house'][h][key]-arm_stats['A']['by_house'][h][key] if arm_stats['A']['by_house'][h]['n'] else None for key in differences} for h in arm_stats['A']['by_house']},
        arms=arm_stats,audit=dict(input_prefix_matched=all(r['audit']['input_prefix_matched'] for r in rows) if rows else None,
            action_prefix_matched=all(r['audit']['action_prefix_matched'] for r in rows) if rows else None,
            logits_bitwise_equal=all(r['audit']['logits_bitwise_equal'] for r in rows) if rows else None,
            max_logit_delta=max((r['audit']['max_logit_delta'] for r in rows),default=None),
            argmax_flip_count=sum(r['audit']['argmax_flip_count'] for r in rows)),
        paired_uncertainty=uncertainty,identity_blocks=identity_blocks,gpu_hours=gpu_seconds/3600,
        issues=issues,pair_manifest=[dict(item,complete=item['rank'] in found,artifact=found.get(item['rank'],{}).get('artifact'),
            status='VALID_COMPLETE' if item['rank'] in found else 'UNRESOLVED_ATTEMPT' if any(x.get('pair_rank')==item['rank'] for x in issues) else 'PENDING') for item in order],
        timing_note='Shared device; competition inferred from5s snapshots. Preprocess includes CPU audit hashing; no exclusive deployment speed claim',
        scientific_generalization=False,old_results_reused=False)
    c.write(HERE/'RESULT.json',result)
    if len(rows) in (5,20,100) and not (HERE/f'MILESTONE_{len(rows):03d}.json').exists():
        c.write(HERE/f'MILESTONE_{len(rows):03d}.json',result,True)
    print(dict(complete_pairs=len(rows),delta_sr=differences['sr'],wins=len(wins),losses=len(losses),gpu_hours=result['gpu_hours']),flush=True)
    return result


if __name__=='__main__': main()
