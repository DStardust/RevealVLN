"""Certified frozen causal features; labels and hard-negative pairing stay offline."""
import collections,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
torch.set_num_threads(4)

def prepare(run,cfg):
    # Older entrypoints import a bare common module. Isolate that namespace instead
    # of letting the new runtime accidentally supply their dependency paths.
    saved_common=sys.modules['common'];saved_path=list(sys.path)
    try:
        sys.modules['common']=u.load('contrast20_v16_common',u.HERE.parent/'ordinary_action_repair_v16/common.py')
        old=u.load('contrast20_certified_builder',u.HERE.parent/'ordinary_action_repair_v16/train.py')
    finally:
        sys.modules['common']=saved_common;sys.path[:]=saved_path
    attempt=run/'data_attempts'/str(time.time_ns());attempt.mkdir(parents=True)
    h,z,y,labels,_,_,original=old.build(attempt,cfg)
    h=h.float();z=z.float();mapping={r['record_id']:i for i,r in enumerate(labels)}
    reference=torch.load(cfg['reference_checkpoint'],map_location='cpu',weights_only=True)['trainable']
    assert u.sha(Path(cfg['reference_checkpoint']))==cfg['reference_checkpoint_sha256']
    u.validate_head(original,reference)
    logits=h@reference['action_head.weight'].T+reference['action_head.bias']
    margin=logits[:,3]-logits[:,:3].max(1).values
    normalized=torch.nn.functional.normalize(h,dim=1)
    trajectories=[];pairs=[];bindings=[];counts=collections.Counter()
    source=Path(cfg['upstream_run'])
    for p in sorted(source.glob('collect/sessions/*/pairs/*/COLLECT.json'),key=lambda p:u.read(p)['rank']):
        seal=u.read(p);folder=p.parent/'C'
        records=u.c.records(folder/'POLICY_STEPS.jsonl');sup=u.c.records(folder/'SUPERVISION_ONLY.jsonl')
        assert len(records)==len(sup)==seal['steps']
        seen=set();negative=[];positive=None;distances={};indices=[];steps={}
        for step,(r,s) in enumerate(zip(records,sup),1):
            key=r['raw']['input_key'];idx=mapping[key]
            assert r['executed_action']==s['executed_action'] and s['target']==int(s['distance_before']<3)
            assert labels[idx]['house']==seal['house']
            indices.append(idx);steps.setdefault(idx,step);distances[idx]=s['distance_before']
            if s['distance_before']>=3 and idx not in seen:negative.append(idx);seen.add(idx)
            if r['executed_action']=='STOP' and s['distance_before']<3:
                assert step==seal['steps'] and seal['termination']=='STOP'
                positive=idx
        entry=dict(rank=seal['rank'],house=seal['house'],mode=seal['mode'],indices=indices,
                   negative=negative,positive=positive,actual_steps=seal['steps'])
        trajectories.append(entry)
        if positive is not None and negative:
            cos=normalized[negative]@normalized[positive]
            choices={'feature_nearest':negative[int(cos.argmax())],
                     'reference_hardest':negative[int(margin[negative].argmax())],
                     'boundary_nearest':min(negative,key=lambda i:(distances[i],i))}
            for kind,neg in choices.items():
                pairs.append(dict(rank=seal['rank'],house=seal['house'],kind=kind,positive=positive,negative=neg,
                    positive_step=steps[positive],negative_step=steps[neg],negative_distance_m=distances[neg],
                    cosine=float(normalized[neg]@normalized[positive]),
                    reference_positive_margin=float(margin[positive]),reference_negative_margin=float(margin[neg]),
                    raw_input_conflict=labels[positive]['record_id']==labels[neg]['record_id']))
        counts['physical_trajectories']+=1;counts['decisions']+=seal['steps']
        counts['actual_successful_terminal_stops']+=positive is not None
        counts['unique_negative_opportunities_within_trajectory']+=len(negative)
        bindings.append(dict(path=str(p),sha256=u.sha(p)))
    assert len(trajectories)==640 and not any(x['raw_input_conflict'] for x in pairs)
    split=u.read(Path(cfg['manifests'])/'SPLIT.json')
    assert not set(split['all_training_houses'])&set(split['dev_houses']+split['unseen_houses'])
    def summary(houses):
        result={}
        for kind in ('feature_nearest','reference_hardest','boundary_nearest'):
            pp=[x for x in pairs if x['house'] in houses and x['kind']==kind]
            result[kind]=dict(n=len(pp),reference_terminal_rank_fraction=sum(x['reference_positive_margin']>x['reference_negative_margin'] for x in pp)/len(pp),
                mean_cosine=sum(x['cosine'] for x in pp)/len(pp),near_boundary_count=sum(x['negative_distance_m']<4 for x in pp))
        return result
    audit=dict(counts=dict(counts),unique_features=len(h),feature_width=h.shape[1],split=split,
        fit=summary(split['fit_houses']),diagnostic=summary(split['diagnostic_houses']),
        source_seals=bindings,input_contract='Original instruction + latest two RGB224 + latest eight executed actions. No distance, progress, route id, future feature in policy.',
        diagnostic_exposure='Eight diagnostic houses were previously used in V15-V19 development. Not an independent generalization test.',
        teacher_and_policy_shared=True,recovery64_used=False,collision_does_not_change_original_public_stop_label=True)
    u.write(run/'DATA_AUDIT.json',audit,True);u.write(run/'TRAJECTORY_INDEX.json',trajectories,True)
    u.write(run/'MATCHED_TERMINAL_PAIRS.json',pairs,True)
    torch.save(dict(features=h,base_logits=z,reference_logits=logits,reference_margin=margin,record_ids=[x['record_id'] for x in labels]),run/'FEATURES.pt')
    u.write(run/'DATA_READY.json',dict(status='COMPLETE',cache_sha256=u.sha(run/'FEATURES.pt'),trajectories_sha256=u.sha(run/'TRAJECTORY_INDEX.json'),pairs_sha256=u.sha(run/'MATCHED_TERMINAL_PAIRS.json'),audit_sha256=u.sha(run/'DATA_AUDIT.json')),True)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('source_seals','split')},indent=2),flush=True)
    return audit

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    prepare(run,u.read(run/'PROTOCOL.json'))
