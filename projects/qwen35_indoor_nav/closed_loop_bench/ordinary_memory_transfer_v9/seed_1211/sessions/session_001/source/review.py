"""Whole-quartet denominators and paired SR effects; partial progress is explicit."""
import json
import math
from pathlib import Path
import statistics
import sys
import random

HERE=Path(__file__).resolve().parent
V5=HERE.parent/'ordinary_cycle_pair_recovery_v5'
sys.path.insert(0,str(V5))
import common as c
ARMS=('A','B','C','D')


def committed(seed=1209):
    found={}
    for session in sorted((HERE/f'seed_{seed}'/'sessions').glob('session_*')):
        covered=set()
        for path in session.glob('STATE_SEAL_*.json'):
            seal=c.read(path)
            if seal['unchanged']:covered.update(seal['quartet_ranks'])
        for path in session.glob('pairs/pair_*/QUARTET.json'):
            row=c.read(path)
            assert row['method_seed']==seed, 'SEED_BINDING_MISMATCH'
            if row['rank'] not in covered:continue
            assert row['valid_behavioral_quartet']
            assert row['rank'] not in found,'DUPLICATE_COMPLETE_QUARTET'
            found[row['rank']]=dict(row,path=str(path))
    return found


def contrast(rows,left,right):
    wins=[];losses=[];retained=[];houses={}
    for row in rows:
        a,b=row['episodes'][left],row['episodes'][right]
        assert a['episode_id']==b['episode_id'] and a['house']==b['house']
        if b['success']>a['success']:wins.append(a['episode_id'])
        elif b['success']<a['success']:losses.append(a['episode_id'])
        elif a['success']:retained.append(a['episode_id'])
        houses.setdefault(a['house'],[]).append(b['success']-a['success'])
    n=len(rows);discordant=len(wins)+len(losses)
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(len(wins),len(losses))+1))/2**discordant) if discordant else 1.
    return dict(delta_sr=(len(wins)-len(losses))/n,wins=wins,losses=losses,retained_successes=retained,
        by_house={h:dict(episodes=len(v),delta_sr=sum(v)/len(v)) for h,v in houses.items()},
        paired_discordance_exact_p=p,p_scope='Descriptive paired episode test; house correlation and only five houses limit generalization',
        delta_spl=sum(r['episodes'][right]['spl']-r['episodes'][left]['spl'] for r in rows)/n,
        delta_ndtw=sum(r['episodes'][right]['ndtw']-r['episodes'][left]['ndtw'] for r in rows)/n)


def summarize(seed=1209,verify=False):
    found=committed(seed);rows=[found[k] for k in sorted(found)]
    if not rows:return dict(status='NO_COMPLETE_QUARTET',complete_quartets=0,expected_quartets=100)
    arms={}
    for arm in ARMS:
        episodes=[row['episodes'][arm] for row in rows]
        steps=[]
        for row in rows:
            folder=Path(row['path']).parent/arm
            if verify:
                assert c.audit_episode(folder,row['index'])==row['episodes'][arm]
                for name,digest in row['logs'][arm].items():assert c.sha(folder/name)==digest
            steps.extend(c.records(folder/'POLICY_STEPS.jsonl'))
        latency=sorted(step['inference_seconds'] for step in steps)
        memory=sorted(step['controller_seconds'] for step in steps)
        arms[arm]=dict(episodes=len(episodes),sr=sum(e['success'] for e in episodes)/len(episodes),
            spl=sum(e['spl'] for e in episodes)/len(episodes),ndtw=sum(e['ndtw'] for e in episodes)/len(episodes),
            osr=sum(e['oracle_success'] for e in episodes)/len(episodes),decisions=len(steps),
            collisions=sum(e['collisions'] for e in episodes),repeated_decisions=sum(d['repeated_input'] for d in steps),
            overrides=sum(d['override'] for d in steps),
            entered_range_without_success=sum(e['oracle_success'] and not e['success'] for e in episodes),
            far_stop=sum(e['stopped'] and e['distances'][-1]>=3 for e in episodes),
            inference_p50=statistics.median(latency),inference_p95=latency[math.ceil(.95*len(latency))-1],
            memory_p50=statistics.median(memory),memory_p95=memory[math.ceil(.95*len(memory))-1])
    return dict(status='VALID_COMPLETE' if len(rows)==100 else 'VALID_PARTIAL_DEVELOPMENT',
        complete_quartets=len(rows),expected_quartets=100,episode_executions=len(rows)*len(ARMS),
        missing_ranks=sorted(set(range(100))-set(found)),arms=arms,
        comparisons={a+b:contrast(rows,a,b) for a,b in (('A','B'),('A','C'),('A','D'),('B','C'),('D','C'),('D','B'))},
        native_stop_preserved=True,adopted=False,seed=seed,seed_selected_by_score=False,
        data_exposure='Previously exposed INTERNAL_DEV100; not full val_unseen or blind generalization',
        base_prefix_logits_bitwise_equal=all(a['logits_bitwise_equal'] for row in rows for a in row['audits'].values()),
        base_prefix_max_logit_delta=max(a['max_logit_delta'] for row in rows for a in row['audits'].values()),
        metric_and_log_recomputed=verify)


def clustered_interval(values, draws=5000):
    """Resample whole episodes or houses, preserving all seed repeats together."""
    rng=random.Random(1209)
    episode_effects=[statistics.mean(v['deltas']) for v in values.values()]
    houses={}
    for row in values.values():
        houses.setdefault(row['house'],[]).append(statistics.mean(row['deltas']))
    clusters=list(houses.values())
    episode_samples=[];house_samples=[]
    for _ in range(draws):
        episode_samples.append(statistics.mean(rng.choices(episode_effects,k=len(episode_effects))))
        sampled=[effect for cluster in rng.choices(clusters,k=len(clusters)) for effect in cluster]
        house_samples.append(statistics.mean(sampled))
    return dict(episode_resample_95pct=[sorted(episode_samples)[int(draws*q)] for q in (.025,.975)],
        house_resample_95pct=[sorted(house_samples)[int(draws*q)] for q in (.025,.975)],
        independent_episode_units=len(values),houses=len(houses),seed_repeats_not_independent=True)


def aggregate(verify=False):
    config=c.read(HERE/'PROTOCOL.json')
    seeds=config['method_seeds']
    reports={str(seed):summarize(seed,verify) for seed in seeds}
    complete=all(r['complete_quartets']==100 for r in reports.values())
    output=dict(status='VALID_COMPLETE' if complete else 'PARTIAL_FIXED_SEED_REPLICATION',
        seeds=seeds,per_seed=reports,complete_quartets=sum(r['complete_quartets'] for r in reports.values()),
        expected_quartets=100*len(seeds),episode_executions=4*sum(r['complete_quartets'] for r in reports.values()),
        metric_and_log_recomputed=verify,adopted=False)
    if not complete:
        output['benefit']='UNKNOWN_UNTIL_ALL_FIXED_SEEDS_COMPLETE'
        return output
    results={seed:committed(seed) for seed in seeds}
    comparisons={}
    for left,right in [('A','B'),('A','C'),('A','D'),('B','C'),('D','C'),('D','B')]:
        values={}
        for seed,rows in results.items():
            for rank,row in rows.items():
                a,b=row['episodes'][left],row['episodes'][right]
                entry=values.setdefault(rank,dict(episode_id=a['episode_id'],house=a['house'],deltas=[]))
                assert (entry['episode_id'],entry['house'])==(a['episode_id'],a['house'])
                entry['deltas'].append(b['success']-a['success'])
        interval=clustered_interval(values)
        deltas=[reports[str(seed)]['comparisons'][left+right]['delta_sr'] for seed in seeds]
        comparisons[left+right]=dict(mean_delta_sr=statistics.mean(deltas),per_seed_delta_sr=deltas,
            **interval,development_support=all(x>=0 for x in deltas) and
            interval['episode_resample_95pct'][0]>0 and interval['house_resample_95pct'][0]>0)
    output.update(comparisons=comparisons,unique_navigation_episodes=100,
        mean_sr={arm:statistics.mean(reports[str(seed)]['arms'][arm]['sr'] for seed in seeds) for arm in ARMS},
        interpretation='All fixed seeds, shared exposed100 episodes. Three repetitions are not300 independent episodes. Five-house intervals remain descriptive; no scientific generalization or adoption follows automatically.',
        method_increment_supported=comparisons['BC']['development_support'] and comparisons['DC']['development_support'])
    return output


if __name__=='__main__':
    result=aggregate(verify=True)
    c.write(HERE/'RESULT.json',result,True)
    print(json.dumps({k:v for k,v in result.items() if k!='per_seed'},ensure_ascii=False))
