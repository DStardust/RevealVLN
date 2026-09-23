"""Use exactly V15's admitted causal inputs; add real teacher motion supervision."""
import collections,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
torch.set_num_threads(4)
obj=u.load('motion16_objective',u.HERE/'objective.py')
def build(run,cfg):
    source=Path(cfg['upstream_run']);labels=u.read(source/'TRAINING_LABEL_INDEX.json');v15_result=u.read(source/'TRAIN_RESULT.json');assert u.sha(source/'CANDIDATE.pt')==v15_result['checkpoint_sha256']
    keys=[r['record_id'] for r in labels];wanted=set(keys);found={};teacher=collections.defaultdict(list);seals={};moves=list(u.c.ACTIONS[:3])
    for p in source.glob('collect/sessions/*/pairs/*/COLLECT.json'):
        seal=u.read(p);assert seal['rank'] not in seals;seals[seal['rank']]=(p,seal)
    assert set(seals)==set(range(640)),'FULL_SHARED_COLLECTION_REQUIRED'
    for rank,(p,s) in sorted(seals.items()):
        folder=p.parent/'C';assert s['base_unchanged']
        for f,sha in s['files'].items():assert u.sha(folder/f)==sha,'SOURCE_DATA_CHANGED'
        data=torch.load(folder/'FEATURES.pt',map_location='cpu',weights_only=True);ll=u.c.records(folder/'SUPERVISION_ONLY.jsonl') if s['steps'] else []
        assert len(ll)==len(data['features'])==len(data['input_keys'])==s['steps']
        for i,key in enumerate(data['input_keys']):
            if key not in wanted:continue
            if key not in found:found[key]=(data['features'][i].clone(),data['logits'][i].clone(),ll[i]['target'])
            if s['mode']=='TEACHER' and ll[i]['distance_before']>=3 and ll[i]['executed_action'] in moves:
                teacher[key].append(dict(episode=rank,step=i,target=moves.index(ll[i]['executed_action'])))
    assert set(found)==wanted and [found[k][2] for k in keys]==[r['target'] for r in labels],'SHARED_INDEX_CHANGED'
    h=torch.stack([found[k][0] for k in keys]).double();z=torch.stack([found[k][1] for k in keys]).double();y=torch.tensor([r['target'] for r in labels],dtype=torch.float64)
    my=torch.full((len(keys),),-1,dtype=torch.long);mlabels=[];conflicts=[]
    for i,(key,label) in enumerate(zip(keys,labels)):
        ts={x['target'] for x in teacher[key]}
        if len(ts)>1:conflicts.append(key)
        if len(ts)==1:my[i]=next(iter(ts))
        mlabels.append(dict(house=label['house'],occurrences=teacher[key] if len(ts)==1 else []))
    original=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)['trainable'];assert (h.float()@original['action_head.weight'].T+original['action_head.bias']-z.float()).abs().max()<1e-4
    binding=dict(upstream_checkpoint_sha256=v15_result['checkpoint_sha256'],upstream_label_index_sha256=u.sha(source/'TRAINING_LABEL_INDEX.json'),shared_inputs=len(keys),eligible_motion_inputs=int((my>=0).sum()),motion_label_conflicts=conflicts,feature_sha256=u.tensor_hash(h),base_logits_sha256=u.tensor_hash(z),source_rank_count=640)
    u.write(run/'DATA_BINDING.json',binding,True)
    return h,z,y,labels,my,mlabels,original
def main(run):
    u.verify();cfg=u.read(run/'PROTOCOL.json');h,z,y,labels,my,mlabels,original=build(run,cfg);fit=u.load('motion16_fit_weights',u.LINE/'sft_acceptance/ordinary_stop_row_v12/fit.py');split=u.read(Path(cfg['manifests'])/'SPLIT.json');records=[]
    reference=torch.load(Path(cfg['upstream_run'])/'CANDIDATE.pt',map_location='cpu',weights_only=True)['trainable'];began=time.time()
    for mode,houses,held in [('house_probe',split['fit_houses'],split['diagnostic_houses']),('final',split['all_training_houses'],split['all_training_houses'])]:
        sw=fit.weights(labels,houses);mh=[h for h in houses if any(x['house']==h and x['occurrences'] for x in mlabels)];mw=fit.weights(mlabels,mh);assert mw.sum()>0
        def progress(value):
            u.append(run/'TRAIN_LOG.jsonl',dict(mode=mode,**value));u.write(run/'TRAIN_PROGRESS.json',dict(status='TRAINING',mode=mode,**value,unix=time.time()))
        theta,scale,info=obj.optimize(h,z,y,sw,my,mw,progress);state={k:v.clone() for k,v in original.items()};state['action_head.weight']=(state['action_head.weight'].double()+theta[:,:-1]/scale).float();state['action_head.bias']=(state['action_head.bias'].double()+theta[:,-1]).float();u.validate_head(original,state)
        fit_motion_houses=mh;tw=fit.weights(labels,held);mh=[house for house in held if any(x['house']==house and x['occurrences'] for x in mlabels)];mtw=fit.weights(mlabels,mh)
        diagnostics={}
        for name,st in [('V15',reference),('V16',state)]:
            pred=h.float()@st['action_head.weight'].T+st['action_head.bias'];margin=pred[:,3].double()-pred[:,:3].max(1).values.double()
            diagnostics[name]=dict(stop=fit.metrics(margin,y,tw),motion_teacher_agreement=float((mtw*(pred[:,:3].argmax(1)==my)).sum()))
        rec=dict(mode=mode,optimizer=info,diagnostics=diagnostics,fit_motion_houses=fit_motion_houses,diagnostic_motion_houses=mh,learned_parameters=8196,backbone_updates=0,parameter_fingerprint=u.state_stamp(state));records.append(rec);u.write(run/(mode.upper()+'_RESULT.json'),rec,True)
        if mode=='final':candidate=state
    tmp=run/'CANDIDATE.tmp.pt';torch.save(dict(trainable=candidate,source_checkpoint_sha256=cfg['checkpoint_sha256'],source='motion16 frozen-feature four-row fit'),tmp);tmp.rename(run/'CANDIDATE.pt')
    u.write(run/'TRAIN_RESULT.json',dict(status='TRAINED_PENDING_CLOSED_LOOP',checkpoint_sha256=u.sha(run/'CANDIDATE.pt'),fits=records,wall_seconds=time.time()-began,backbone_updates=0,navigation_benefit=None),True);u.write(run/'TRAIN_PROGRESS.json',dict(status='COMPLETE',unix=time.time()),False)
if __name__=='__main__':main(Path(sys.argv[1]))
