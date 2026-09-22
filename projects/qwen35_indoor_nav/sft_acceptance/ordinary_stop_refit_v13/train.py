"""Reuse certified FIT features; train a real anchored STOP row, no cache replays."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
torch.set_num_threads(4)

def main(run):
    u.verify();cfg=u.read(u.HERE/'PROTOCOL.json')
    if (run/'TRAIN_RESULT.json').exists():
        result=u.read(run/'TRAIN_RESULT.json')
        assert u.sha(run/'CANDIDATE.pt')==result['checkpoint_sha256'];return
    old=u.load('stop13_readonly_fit',u.LINE/'sft_acceptance/ordinary_stop_row_v12/fit.py')
    cache=torch.load(u.OLD/'FEATURES.pt',map_location='cpu',weights_only=True)
    labels=u.c.records(u.OLD/'SUPERVISION_ONLY.jsonl');inputs=u.c.records(u.OLD/'POLICY_INPUTS.jsonl')
    assert cache['record_ids']==[x['record_id'] for x in labels]==[x['record_id'] for x in inputs]
    original=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    assert cache['source_checkpoint_sha256']==cfg['checkpoint_sha256']
    h=cache['features'].double();z=cache['logits'].double();y=torch.tensor([x['target'] for x in labels],dtype=torch.float64)
    assert torch.isfinite(h).all() and torch.isfinite(z).all()
    reconstructed=h.float()@original['trainable']['action_head.weight'].T+original['trainable']['action_head.bias']
    reconstruction_delta=float((reconstructed.double()-z).abs().max())
    assert reconstruction_delta<1e-4,'CACHE_HEAD_IDENTITY_MISMATCH'
    split=u.read(u.OLD/'SPLIT.json');eval_houses={x['house'] for x in u.read(u.V5/'PAIR_ORDER.json')}
    assert not eval_houses & set(split['all_houses']),'FIT_EVALUATION_HOUSE_LEAK'
    for r in inputs:assert set(r)=={'record_id','instruction','rgb_sha256','executed_actions'},'PRIVILEGED_INPUT'
    start=time.time();records=[];candidate=None
    for name,fit,check in [('house_probe',split['fit_houses'],split['heldout_houses']),('final',split['all_houses'],split['all_houses'])]:
        w=old.weights(labels,fit);theta,scale,info=old.optimize(h,z,y,w)
        new={k:v.clone() for k,v in original['trainable'].items()}
        new['action_head.weight'][3]=(new['action_head.weight'][3].double()+theta[:-1]/scale).float()
        new['action_head.bias'][3]=(new['action_head.bias'][3].double()+theta[-1]).float()
        u.validate_head(original['trainable'],new)
        assert not torch.equal(original['trainable']['action_head.weight'][3],new['action_head.weight'][3]),'NO_PARAMETER_UPDATE'
        tw=old.weights(labels,check);margin=z[:,3]-z[:,:3].max(1).values
        newmargin=(h.float()@new['action_head.weight'][3]+new['action_head.bias'][3]).double()-z[:,:3].max(1).values
        record=dict(mode=name,fit_houses=fit,diagnostic_houses=check,before=old.metrics(margin,y,tw),after=old.metrics(newmargin,y,tw),optimization=info,
            parameters_before=u.state_stamp(original['trainable'])['sha256'],parameters_after=u.state_stamp(new)['sha256'],backbone_updates=0,learned_parameters=2049)
        records.append(record);u.append(run/'TRAIN_LOG.jsonl',record)
        if name=='final':candidate=new
    result=dict(status='TRAINED_NOT_NAVIGATION_VALIDATED',counts=dict(inputs=len(labels),positive=sum(x['target'] for x in labels),houses=16,physical_episodes=64,source_occurrences=7225),
        cache_head_reconstruction_max_abs=reconstruction_delta,old_v12_numeric_status=u.read(u.OLD/'FEATURE_PARITY.json'),fits=records,
        cache_replay_forwards=0,trained_on_cpu=True,wall_seconds=time.time()-start,protocol_sha256=u.sha(u.HERE/'PROTOCOL.json'),
        scientific_novelty=False,navigation_benefit=None,recipe_selection='fixed lambda=.001; diagnostic split does not tune lambda or choose checkpoints')
    tmp=run/'CANDIDATE.tmp.pt';torch.save(dict(trainable=candidate,source_checkpoint_sha256=cfg['checkpoint_sha256'],protocol_sha256=result['protocol_sha256'],training=result),tmp);tmp.rename(run/'CANDIDATE.pt')
    result['checkpoint_sha256']=u.sha(run/'CANDIDATE.pt');u.write(run/'TRAIN_RESULT.json',result,True)
    print(json.dumps({k:v for k,v in result.items() if k!='fits'}),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
