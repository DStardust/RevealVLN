"""Whole-quartet denominators and paired SR effects; partial progress is explicit."""
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
V5=HERE.parent/'ordinary_cycle_pair_recovery_v5'
sys.path.insert(0,str(V5))
import common as c
ARMS=('A','B','C','D')


def committed():
    found={}
    for session in sorted((HERE/'sessions').glob('session_*')):
        covered=set()
        for path in session.glob('STATE_SEAL_*.json'):
            seal=c.read(path)
            if seal['unchanged']:covered.update(seal['quartet_ranks'])
        for path in session.glob('pairs/pair_*/QUARTET.json'):
            row=c.read(path)
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


def summarize(verify=False):
    found=committed();rows=[found[k] for k in sorted(found)]
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
        native_stop_preserved=True,adopted=False,seed=1209,seed_selected_by_score=False,
        data_exposure='Previously exposed INTERNAL_DEV100; not full val_unseen or blind generalization',
        base_prefix_logits_bitwise_equal=all(a['logits_bitwise_equal'] for row in rows for a in row['audits'].values()),
        base_prefix_max_logit_delta=max(a['max_logit_delta'] for row in rows for a in row['audits'].values()),
        metric_and_log_recomputed=verify)


if __name__=='__main__':
    result=summarize(verify=True)
    c.write(HERE/'RESULT.json',result,True)
    print(json.dumps(result,ensure_ascii=False))
