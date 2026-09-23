"""Freeze up to64 FIT-only recovery prefixes before looking at V16 evaluation scores."""
import collections,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
def trigger(rows,private):
    streak=0
    for i,(r,s) in enumerate(zip(rows,private)):
        if r['executed_action']=='STOP' and s['distance_to_goal']>=3:
            if 1<=i<=480:return dict(cutoff=i,kind='premature_stop_proposal',event_step=i+1,event_action_executed=False)
        streak=streak+1 if s['action']=='move_forward' and s['collided'] else 0
        if streak>=3 and 1<=i+1<=480:return dict(cutoff=i+1,kind='three_forward_collisions',event_step=i+1,event_action_executed=True)
    return None
def main(run):
    cfg=u.read(run/'PROTOCOL.json');source=Path(cfg['upstream_run']);episodes=u.read(Path(cfg['manifests'])/'FIT_EPISODES.json');registry=u.read(Path(cfg['manifests'])/'FIT_ORDER.json');geo=u.read(source/'FIT_GEOMETRY.json');byhouse=collections.defaultdict(list);funnel=[]
    seals=sorted(source.glob('collect/sessions/*/pairs/*/COLLECT.json'),key=lambda p:u.read(p)['rank'])
    assert len(seals)==640
    for path in seals:
        seal=u.read(path)
        if seal['mode']!='POLICY':continue
        folder=path.parent/'C';pp=u.c.records(folder/'POLICY_STEPS.jsonl');ss=u.c.records(folder/'STEPS_PRIVILEGED.jsonl');t=trigger(pp,ss);funnel.append(dict(rank=seal['rank'],house=seal['house'],trigger=t))
        if t:
            source_row=registry[seal['rank']];byhouse[seal['house']].append(dict(t,house=seal['house'],source_rank=seal['rank'],source_index=source_row['index'],source_folder=str(folder),source_seal=str(path),source_seal_sha256=u.sha(path),policy_sha256=u.sha(folder/'POLICY_STEPS.jsonl'),private_sha256=u.sha(folder/'STEPS_PRIVILEGED.jsonl')))
    chosen=[]
    while len(chosen)<64 and any(byhouse.values()):
        for h in sorted(byhouse):
            if byhouse[h] and len(chosen)<64:chosen.append(byhouse[h].pop(0))
    assert chosen,'NO_REGISTERED_RECOVERY_PREFIX'
    folder=run/'recovery';folder.mkdir(exist_ok=True);out=[];geos=[];order=[]
    for i,x in enumerate(chosen):
        e=episodes[x['source_index']];out.append(e);g=dict(geo[x['source_index']],index=i);geos.append(g);order.append(dict(x,rank=i,index=i,episode_id=e['episode_id'],mode='RECOVERY',inference_order=['C']))
    for name,value in [('EPISODES.json',out),('ORDER.json',order),('GEOMETRY.json',geos),('FUNNEL.json',funnel)]:u.write(folder/name,value,True)
    u.write(folder/'MANIFEST.json',dict(eligible=sum(x['trigger'] is not None for x in funnel),selected=len(order),policy_denominator=320,selection_before_v16_test=True,source='V15 FIT policy traces only',new_training_after_collection=False),True)
if __name__=='__main__':main(Path(sys.argv[1]))
