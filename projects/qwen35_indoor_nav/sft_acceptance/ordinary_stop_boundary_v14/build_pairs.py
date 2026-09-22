"""Derived supervision from actual FIT observations; no synthetic policy inputs."""
import collections

def compile_pairs(labels,inputs):
    assert len(labels)==len(inputs)
    episodes=collections.defaultdict(list)
    for i,(label,policy) in enumerate(zip(labels,inputs)):
        assert label['record_id']==policy['record_id'],'RECORD_ALIGNMENT'
        assert set(policy)=={'record_id','instruction','rgb_sha256','executed_actions'},'POLICY_FIREWALL'
        for occurrence in label['occurrences']:
            assert occurrence['house']==label['house'],'HOUSE_ALIGNMENT'
            assert int(occurrence['distance']<3)==label['target'],'LABEL_DISTANCE_ALIGNMENT'
            episodes[occurrence['episode']].append(dict(index=i,**occurrence))
    pairs=[]
    for episode,rows in sorted(episodes.items()):
        assert len({r['house'] for r in rows})==1,'EPISODE_HOUSE_CONFLICT'
        assert len({inputs[r['index']]['instruction'] for r in rows})==1,'EPISODE_INSTRUCTION_CONFLICT'
        positive=[r for r in rows if r['target']==1]
        negative=[r for r in rows if 3<=r['distance']<=6]
        seen=set()
        for n in sorted(negative,key=lambda r:(r['step'],r['index'])):
            if not positive:continue
            p=min(positive,key=lambda r:(abs(r['step']-n['step']),r['step'],r['index']))
            key=(n['index'],p['index'])
            if key in seen:continue
            seen.add(key)
            assert n['index']!=p['index'],'CONTRADICTORY_IDENTICAL_INPUT'
            pairs.append(dict(episode=episode,house=n['house'],negative_index=n['index'],positive_index=p['index'],
                negative_id=labels[n['index']]['record_id'],positive_id=labels[p['index']]['record_id'],
                negative_step=n['step'],positive_step=p['step'],negative_distance=n['distance'],positive_distance=p['distance'],
                temporal_gap=abs(n['step']-p['step'])))
    return pairs

def pair_weights(pairs,houses):
    import torch
    he=collections.defaultdict(set);ep=collections.Counter()
    for p in pairs:
        if p['house'] in houses:he[p['house']].add(p['episode']);ep[p['episode']]+=1
    assert he,'NO_VALID_FIT_PAIRS'
    w=torch.tensor([1/len(he)/len(he[p['house']])/ep[p['episode']] if p['house'] in houses else 0. for p in pairs],dtype=torch.float64)
    assert abs(float(w.sum())-1)<1e-12,'PAIR_NORMALIZATION'
    return w
