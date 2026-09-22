"""One fixed same-route boundary objective, with FIT-only diagnostic and final fit."""
import collections,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
from build_pairs import compile_pairs,pair_weights
from objective import optimize,ranking_loss
torch.set_num_threads(4)

def fold(original,theta,scale):
    new={k:v.clone() for k,v in original.items()}
    new['action_head.weight'][3]=(new['action_head.weight'][3].double()+theta[:-1]/scale).float()
    new['action_head.bias'][3]=(new['action_head.bias'][3].double()+theta[-1]).float()
    u.validate_head(original,new)
    return new

def main(run):
    u.verify();cfg=u.read(u.HERE/'PROTOCOL.json');began=time.time()
    reference=torch.load(cfg['reference_checkpoint'],map_location='cpu',weights_only=True)
    assert u.sha(Path(cfg['reference_checkpoint']))==cfg['reference_checkpoint_sha256'],'REFERENCE_CHANGED'
    old=u.load('stop14_readonly_fit',u.LINE/'sft_acceptance/ordinary_stop_row_v12/fit.py')
    cache=torch.load(u.OLD/'FEATURES.pt',map_location='cpu',weights_only=True)
    labels=u.c.records(u.OLD/'SUPERVISION_ONLY.jsonl');inputs=u.c.records(u.OLD/'POLICY_INPUTS.jsonl')
    assert cache['record_ids']==[x['record_id'] for x in labels]==[x['record_id'] for x in inputs]
    original=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)['trainable'];u.validate_head(original,reference['trainable'])
    assert cache['source_checkpoint_sha256']==cfg['checkpoint_sha256']
    h=cache['features'].double();z=cache['logits'].double();y=torch.tensor([x['target'] for x in labels],dtype=torch.float64)
    reconstructed=h.float()@original['action_head.weight'].T+original['action_head.bias']
    assert torch.isfinite(h).all() and torch.isfinite(z).all()
    assert float((reconstructed.double()-z).abs().max())<1e-4,'CACHE_HEAD_IDENTITY_MISMATCH'
    split=u.read(u.OLD/'SPLIT.json');eval_houses={x['house'] for x in u.read(u.V5/'PAIR_ORDER.json')}
    assert not eval_houses & set(split['all_houses']),'FIT_EVALUATION_HOUSE_LEAK'
    pairs=compile_pairs(labels,inputs);assert len(pairs)==888,'AUDITED_PAIR_COUNT_CHANGED'
    pos=torch.tensor([p['positive_index'] for p in pairs]);neg=torch.tensor([p['negative_index'] for p in pairs])
    counts=dict(inputs=len(labels),positive=sum(r['target'] for r in labels),near_negative=sum(3<=min(o['distance'] for o in r['occurrences'])<=6 for r in labels),
        source_occurrences=sum(len(r['occurrences']) for r in labels),physical_episodes=64,houses=16,boundary_pairs=len(pairs),paired_episodes=len({p['episode'] for p in pairs}),paired_houses=len({p['house'] for p in pairs}),new_physical_trajectories=0,new_qwen_forwards=0)
    u.write(run/'DATA_MANIFEST.json',dict(counts=counts,pairs=pairs,split=split,policy_fields=['instruction','rgb_sha256','executed_actions'],
        private_fields_train_only=['distance','target','episode','house','positive_index','negative_index'],
        evaluation_ids_used_by_objective=False,source_features_sha256=u.sha(u.OLD/'FEATURES.pt'),source_labels_sha256=u.sha(u.OLD/'SUPERVISION_ONLY.jsonl')),True)
    records=[]
    for mode,fit,check in [('house_probe',split['fit_houses'],split['heldout_houses']),('final',split['all_houses'],split['all_houses'])]:
        w=old.weights(labels,fit);pw=pair_weights(pairs,fit)
        theta,scale,info=optimize(h,z,y,w,pos,neg,pw,cfg,lambda value:u.append(run/'OPTIMIZER.jsonl',dict(mode=mode,**value)))
        new=fold(original,theta,scale)
        if mode=='house_probe':
            # V13 reference diagnostic must also use only the same twelve houses.
            rt,rs,ri=old.optimize(h,z,y,w);ref=fold(original,rt,rs)
        else:ref=reference['trainable']
        tw=old.weights(labels,check);tpw=pair_weights(pairs,check)
        rm=(h.float()@ref['action_head.weight'][3]+ref['action_head.bias'][3]).double()-z[:,:3].max(1).values
        nm=(h.float()@new['action_head.weight'][3]+new['action_head.bias'][3]).double()-z[:,:3].max(1).values
        stats=lambda m:dict(**old.metrics(m,y,tw),pair_order_accuracy=float((tpw*(m[pos]>m[neg])).sum()),ranking_loss=float(ranking_loss(m,pos,neg,tpw,cfg['ranking_margin'])))
        record=dict(mode=mode,fit_houses=fit,diagnostic_houses=check,before=stats(rm),after=stats(nm),optimization=info,
            fit_pair_count=int((pw>0).sum()),diagnostic_pair_count=int((tpw>0).sum()),parameters_before=u.state_stamp(ref)['sha256'],parameters_after=u.state_stamp(new)['sha256'],
            backbone_updates=0,learned_parameters=2049,init='zero residual over best4k; same as V13')
        records.append(record);u.append(run/'TRAIN_LOG.jsonl',record)
        if mode=='final':candidate=new
    result=dict(status='TRAINED_NOT_NAVIGATION_VALIDATED',counts=counts,fits=records,trained_on_cpu=True,wall_seconds=time.time()-began,
        reference_checkpoint_sha256=cfg['reference_checkpoint_sha256'],protocol_sha256=u.sha(u.HERE/'PROTOCOL.json'),
        fixed_recipe=True,navigation_benefit=None,scientific_novelty=False,learned_parameters=2049,base_updates=0)
    temp=run/'CANDIDATE.tmp.pt';torch.save(dict(trainable=candidate,source_checkpoint_sha256=cfg['checkpoint_sha256'],protocol_sha256=result['protocol_sha256'],training=result),temp);temp.rename(run/'CANDIDATE.pt')
    result['checkpoint_sha256']=u.sha(run/'CANDIDATE.pt');u.write(run/'TRAIN_RESULT.json',result,True);print(json.dumps(result),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
