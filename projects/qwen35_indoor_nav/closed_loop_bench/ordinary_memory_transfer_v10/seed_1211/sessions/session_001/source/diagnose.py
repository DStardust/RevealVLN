"""Read-only full-denominator STOP, repetition and shared-resource accounting."""
import bisect
from collections import Counter
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import review
c=review.c


def quantiles(values):
    ordered=sorted(values)
    return dict(count=len(values),p50=statistics.median(ordered) if ordered else None,
                p95=ordered[math.ceil(.95*len(ordered))-1] if ordered else None)


def summarize(seed):
    found=review.committed(seed)
    assert len(found)==100,'FULL_SEED_DENOMINATOR_REQUIRED'
    sessions={}
    resources={}
    for row in found.values():
        session=Path(row['path']).parents[2]
        key=str(session)
        if key not in sessions:
            samples=c.records(session/'RESOURCE.jsonl')
            resources[key]=([s['unix'] for s in samples],samples)
            sessions[key]=dict(launch=c.read(session/'LAUNCH_RESULT.json'),
                protocol_sha256=c.sha(session/'source/PROTOCOL.json'),method_sha256=c.sha(session/'METHOD_IDENTITY.json'),
                peak_own_gpu_mib=max(s['own_memory_mib'] for s in samples),
                peak_rss_bytes=max(s['rss_bytes'] for s in samples),peak_output_bytes=max(s['output_bytes'] for s in samples),
                foreign_task_samples=sum(bool(s['foreign']) for s in samples),resource_samples=len(samples))
        assert row['protocol_sha256']==sessions[key]['protocol_sha256']
        assert row['method_identity_sha256']==sessions[key]['method_sha256']
    arms={}
    for arm in review.ARMS:
        uses_memory=arm in ('B','C','D')
        episodes=[];categories=Counter();latency={'shared':[],'no_foreign_task_observed':[],'unclassified':[]}
        for rank,row in sorted(found.items()):
            folder=Path(row['path']).parent/arm
            steps=c.records(folder/'POLICY_STEPS.jsonl');e=row['episodes'][arm]
            assert len(steps)+1==len(e['distances'])
            near=any(distance<3 for distance in e['distances'])
            category=('success' if e['success'] else
                'far_stop_after_entering_range' if e['stopped'] and near else
                'far_stop_never_entered_range' if e['stopped'] else
                'budget_exhausted_after_entering_range' if near else 'budget_exhausted_never_entered_range')
            categories[category]+=1
            seen=set();full_repeats=0;near_native_margins=[];near_method_margins=[]
            times,samples=resources[str(Path(row['path']).parents[2])]
            for i,d in enumerate(steps):
                memory=d['memory_fingerprint'] if uses_memory else None
                key=(d['raw']['input_key'],memory['sha256'] if memory is not None else None)
                full_repeats+=int(key in seen);seen.add(key)
                index=bisect.bisect_right(times,d['unix'])-1
                load='unclassified' if index<0 else 'shared' if samples[index]['foreign'] else 'no_foreign_task_observed'
                latency[load].append(d['inference_seconds'])
                if e['distances'][i]<3:
                    near_native_margins.append(d['logits'][3]-max(d['logits'][:3]))
                    near_method_margins.append(d['method_logits'][3]-max(d['method_logits'][:3]))
            overrides=[d['step'] for d in steps if d['override']]
            episodes.append(dict(rank=rank,episode_id=e['episode_id'],house=e['house'],category=category,
                decisions=len(steps),short_window_repeats=sum(d['repeated_input'] for d in steps),
                short_window_repeat_ratio=sum(d['repeated_input'] for d in steps)/len(steps),
                policy_input_repeats=full_repeats,first_override=overrides[0] if overrides else None,
                overrides=len(overrides),collisions=e['collisions'],stopped=e['stopped'],
                native_caused_stop=e['stopped'] and steps[-1]['native_action']=='STOP',
                memory_caused_stop=e['stopped'] and steps[-1]['native_action']!='STOP',
                near_range_native_stop_margin=quantiles(near_native_margins),
                near_range_method_stop_margin=quantiles(near_method_margins)))
        assert sum(categories.values())==100
        arms[arm]=dict(episodes=episodes,policy_reads_memory=uses_memory,stop_categories=dict(categories),
            native_caused_stops=sum(e['native_caused_stop'] for e in episodes),
            memory_caused_stops=sum(e['memory_caused_stop'] for e in episodes),
            policy_input_repeats=sum(e['policy_input_repeats'] for e in episodes),
            mean_episode_short_window_repeat_ratio=statistics.mean(e['short_window_repeat_ratio'] for e in episodes),
            first_override_steps=quantiles([e['first_override'] for e in episodes if e['first_override'] is not None]),
            inference_latency_by_observed_resource_competition={k:quantiles(v) for k,v in latency.items()})
    return dict(seed=seed,complete_groups=100,sessions=sessions,arms=arms)


if __name__=='__main__':
    seeds=c.read(HERE/'PROTOCOL.json')['method_seeds']
    result=dict(status='ALL_FIXED_SEEDS_READ_ONLY_DIAGNOSIS',per_seed={str(seed):summarize(seed) for seed in seeds},
        source_sha256=c.sha(Path(__file__)),gpu_used=False,optimizer_updates=0,
        limits='STOP margins and co-occurrences are descriptive, not causal ablations. Changing floating memory values can remove exact repeats without reducing wasted actions. Native A and short-window-only E exclude the unused memory from the policy-input repeat key. Resource categories use the latest monitor sample; collection intervals are irregular and peaks are sampled. Absence of an observed foreign process is not an exclusive GPU reservation.')
    c.write(HERE/'DIAGNOSIS.json',result,True)
    print('Completed all fixed seeds: STOP types, repeats, interventions and resource accounting.')
