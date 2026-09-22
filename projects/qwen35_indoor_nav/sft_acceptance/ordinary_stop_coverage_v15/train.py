"""Expand real-state coverage; keep the successful V13 objective and head architecture."""
import collections,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
torch.set_num_threads(8)

def main(run):
    u.verify();p=u.read(run/'PROTOCOL.json');split=u.read(Path(p['manifests'])/'SPLIT.json');old=u.load('coverage15_fit',u.LINE/'sft_acceptance/ordinary_stop_row_v12/fit.py');start=time.time()
    original=torch.load(p['checkpoint'],map_location='cpu',weights_only=True)['trainable'];ref=torch.load(p['reference_checkpoint'],map_location='cpu',weights_only=True)['trainable']
    assert u.sha(Path(p['reference_checkpoint']))==p['reference_checkpoint_sha256'];u.validate_head(original,ref)
    found={}
    for sealpath in sorted((run/'collect/sessions').glob('*/pairs/*/COLLECT.json')):
        seal=u.read(sealpath);assert seal['rank'] not in found and seal['base_unchanged'];found[seal['rank']]=(sealpath,seal)
    assert set(found)==set(range(640)),'INCOMPLETE_COLLECTION_DENOMINATOR'
    groups={};conflicts=set();physical=[];repeated=0;max_feature_delta=0.;max_logit_delta=0.
    for rank,(sealpath,seal) in sorted(found.items()):
        folder=sealpath.parent/'C'
        for file,h in seal['files'].items():assert u.sha(folder/file)==h,'COLLECTION_FILE_CHANGED'
        for file,h in seal['frames'].items():assert Path(file).is_relative_to(run/'collect') and u.sha(Path(file))==h,'RAW_FRAME_CHANGED'
        data=torch.load(folder/'FEATURES.pt',map_location='cpu',weights_only=True)
        labels=u.c.records(folder/'SUPERVISION_ONLY.jsonl') if seal['steps'] else []
        policy=u.c.records(folder/'POLICY_STEPS.jsonl') if seal['steps'] else []
        assert data['source_checkpoint_sha256']==p['checkpoint_sha256'] and len(labels)==len(policy)==len(data['features'])==seal['steps']
        assert torch.isfinite(data['features']).all() and torch.isfinite(data['logits']).all()
        physical.append(dict(rank=rank,house=seal['house'],mode=seal['mode'],steps=seal['steps'],positive=seal['positive'],termination=seal['termination']))
        for i,(lab,pol) in enumerate(zip(labels,policy)):
            key=data['input_keys'][i];assert key==pol['raw']['input_key'] and pol['step']==lab['step']==i+1
            assert lab['house'] in split['all_training_houses'],'NONTRAINING_SAMPLE'
            occ=dict(episode=rank,step=i,house=lab['house'],distance=lab['distance_before'],target=lab['target'])
            if key not in groups:groups[key]=dict(feature=data['features'][i].clone(),logits=data['logits'][i].clone(),target=lab['target'],raw=pol['raw'],house=lab['house'],occurrences=[occ])
            else:
                g=groups[key];assert g['raw']==pol['raw'] and g['house']==lab['house'],'INPUT_KEY_COLLISION'
                repeated+=1;g['occurrences'].append(occ)
                max_feature_delta=max(max_feature_delta,float((g['feature']-data['features'][i]).abs().max()));max_logit_delta=max(max_logit_delta,float((g['logits']-data['logits'][i]).abs().max()))
                if g['target']!=lab['target']:conflicts.add(key)
    selected=[(k,v) for k,v in sorted(groups.items()) if k not in conflicts]
    assert selected,'NO_TRAINABLE_INPUTS'
    h=torch.stack([v['feature'] for k,v in selected]).double();z=torch.stack([v['logits'] for k,v in selected]).double();y=torch.tensor([v['target'] for k,v in selected],dtype=torch.float64)
    labels=[dict(record_id=k,target=v['target'],house=v['house'],occurrences=v['occurrences']) for k,v in selected]
    reconstruction=h.float()@original['action_head.weight'].T+original['action_head.bias'];assert float((reconstruction.double()-z).abs().max())<1e-4,'FEATURE_BASE_IDENTITY'
    assert {v['house'] for k,v in selected}==set(split['all_training_houses']),'HOUSE_COVERAGE_GAP'
    for house in split['all_training_houses']:assert any(v['house']==house and v['target']==1 for k,v in selected),'NO_POSITIVE_IN_HOUSE:'+house
    audit=dict(physical_trajectories=len(physical),instructions=320,houses=40,observed_decisions=sum(v['steps'] for v in physical),unique_inputs=len(selected),positive_inputs=int(y.sum()),duplicate_observations=repeated,conflicting_input_ids=sorted(conflicts),max_duplicate_feature_delta=max_feature_delta,max_duplicate_logit_delta=max_logit_delta,physical=physical,old_5487_replayed=False,new_training_labels_from_dev_or_unseen=False)
    u.write(run/'DATA_AUDIT.json',audit,True)
    u.write(run/'TRAINING_LABEL_INDEX.json',labels,True)
    fits=[]
    for mode,houses,check in [('house_probe',split['fit_houses'],split['diagnostic_houses']),('final',split['all_training_houses'],split['all_training_houses'])]:
        w=old.weights(labels,houses);theta,scale,info=old.optimize(h,z,y,w)
        new={k:v.clone() for k,v in original.items()};new['action_head.weight'][3]=(new['action_head.weight'][3].double()+theta[:-1]/scale).float();new['action_head.bias'][3]=(new['action_head.bias'][3].double()+theta[-1]).float();u.validate_head(original,new)
        tw=old.weights(labels,check);before=(h.float()@ref['action_head.weight'][3]+ref['action_head.bias'][3]).double()-z[:,:3].max(1).values;after=(h.float()@new['action_head.weight'][3]+new['action_head.bias'][3]).double()-z[:,:3].max(1).values
        result=dict(mode=mode,fit_houses=houses,diagnostic_houses=check,reference=old.metrics(before,y,tw),candidate=old.metrics(after,y,tw),optimization=info,reference_prior_house_exposure='V13 used its earlier 16 FIT houses; this probe is descriptive, not a matched-data method comparison',parameter_sha256=u.state_stamp(new)['sha256'])
        fits.append(result);u.append(run/'TRAIN_LOG.jsonl',result)
        if mode=='final':candidate=new
    result=dict(status='TRAINED_PENDING_CLOSED_LOOP',fits=fits,learned_parameters=2049,backbone_updates=0,wall_seconds=time.time()-start,reference_checkpoint_sha256=p['reference_checkpoint_sha256'],protocol_sha256=u.sha(run/'PROTOCOL.json'),counts={k:v for k,v in audit.items() if k not in ('physical','conflicting_input_ids')},navigation_benefit=None)
    tmp=run/'CANDIDATE.tmp.pt';torch.save(dict(trainable=candidate,training=result,source_checkpoint_sha256=p['checkpoint_sha256']),tmp);tmp.rename(run/'CANDIDATE.pt');result['checkpoint_sha256']=u.sha(run/'CANDIDATE.pt');u.write(run/'TRAIN_RESULT.json',result,True)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
