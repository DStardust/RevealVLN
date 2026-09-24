"""Real frozen-feature STOP updates with one registered trajectory-risk objective."""
import collections,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
torch.set_num_threads(4)
obj=u.load('calibrated19_objective',u.HERE/'objective.py')
def groups_for(trajectories,houses):
    rows=[r for r in trajectories if r['house'] in houses];assert {r['house'] for r in rows}==set(houses)
    counts=collections.Counter(r['house'] for r in rows);width=max(1,max(len(r['negative']) for r in rows))
    indices=torch.zeros((len(rows),width),dtype=torch.long);mask=torch.zeros_like(indices,dtype=torch.bool)
    positives=[];pmask=[];weights=[]
    for i,r in enumerate(rows):
        nn=len(r['negative']);indices[i,:nn]=torch.tensor(r['negative'],dtype=torch.long);mask[i,:nn]=True
        positives.append(r['positive'] if r['positive'] is not None else 0);pmask.append(r['positive'] is not None);weights.append(1/len(houses)/counts[r['house']])
    return dict(negative_indices=indices,negative_mask=mask,positive_indices=torch.tensor(positives),positive_mask=torch.tensor(pmask,dtype=torch.float64),trajectory_weights=torch.tensor(weights,dtype=torch.float64))
def trajectory_metrics(m,groups):
    neg=m[groups['negative_indices']].masked_fill(~groups['negative_mask'],-1e9).max(1).values
    pos=m[groups['positive_indices']];w=groups['trajectory_weights'];pm=groups['positive_mask']
    return dict(any_false_stop=float((w*(neg>0)).sum()),verified_terminal_stop_recall=float((w*pm*(pos>0)).sum()/(w*pm).sum()),has_positive_weight=float((w*pm).sum()),scope='fixed observed training trajectories; not counterfactual navigation success')
def main(run):
    u.verify();cfg=u.read(run/'PROTOCOL.json');source=Path(cfg['upstream_run']);start=time.time()
    # Read-only reuse of the certified canonical input builder; no model or native-input replay.
    old=u.load('calibrated19_shared_builder',u.HERE.parent/'ordinary_action_repair_v16/train.py')
    h,z,y,labels,unused_motion,unused_labels,original=old.build(run,cfg)
    assert u.sha(Path(cfg['reference_checkpoint']))==cfg['reference_checkpoint_sha256']
    reference=torch.load(cfg['reference_checkpoint'],map_location='cpu',weights_only=True)['trainable'];u.validate_head(original,reference)
    mapping={r['record_id']:i for i,r in enumerate(labels)};trajectories=[];quality=collections.Counter();bindings=[]
    for p in sorted(source.glob('collect/sessions/*/pairs/*/COLLECT.json'),key=lambda p:u.read(p)['rank']):
        seal=u.read(p);folder=p.parent/'C';records=u.c.records(folder/'POLICY_STEPS.jsonl');supervision=u.c.records(folder/'SUPERVISION_ONLY.jsonl');private=u.c.records(folder/'STEPS_PRIVILEGED.jsonl')
        assert len(records)==len(supervision)==len(private)==seal['steps']
        negative=[];seen=set();positive=None;first=None
        for t,(r,s,priv) in enumerate(zip(records,supervision,private)):
            key=r['raw']['input_key'];assert key in mapping and s['target']==int(s['distance_before']<3)
            assert r['executed_action']==s['executed_action']==priv['action']
            if seal['mode']=='TEACHER' and s['distance_before']>=3 and s['executed_action'] in u.c.ACTIONS[:3]:
                quality['motion_label_occurrences']+=1;quality['colliding_motion_labels']+=int(priv['collided'])
            if not s['target'] and key not in seen:
                negative.append(mapping[key]);seen.add(key)
            if r['executed_action']=='STOP' and s['target']:
                assert t==len(records)-1 and seal['termination']=='STOP','TERMINAL_LABEL'
                positive=mapping[key];first=t
            elif s['target']:quality['in_range_nonterminal_inputs_not_used_as_labels']+=1
        trajectories.append(dict(rank=seal['rank'],house=seal['house'],mode=seal['mode'],negative=negative,positive=positive,first_positive_step=first,actual_steps=seal['steps']))
        bindings.append(dict(path=str(p),sha256=u.sha(p)))
    assert len(trajectories)==640
    u.write(run/'TRAJECTORY_INDEX.json',trajectories,True)
    u.write(run/'DATA_AUDIT.json',dict(physical_trajectories=640,unique_features=len(h),negative_opportunities=sum(len(t['negative']) for t in trajectories),verified_terminal_stop_opportunities=sum(t['positive'] is not None for t in trajectories),quality=dict(quality),source_seals=bindings,recovery64_used=False,observed_history_only=True),True)
    ref_logits=h.float()@reference['action_head.weight'].T+reference['action_head.bias'];offset=(ref_logits[:,3]-ref_logits[:,:3].max(1).values).double()
    fit=u.load('calibrated19_old_weights',u.LINE/'sft_acceptance/ordinary_stop_row_v12/fit.py');split=u.read(Path(cfg['manifests'])/'SPLIT.json');fits=[]
    for mode,houses,check in [('calibrated_final',split['fit_houses'],split['diagnostic_houses'])]:
        sw=fit.weights(labels,houses);scale=(sw[:,None]*h.square()).sum().sqrt();groups=groups_for(trajectories,houses);test=groups_for(trajectories,check)
        def progress(row):
            value=dict(mode=mode,unix=time.time(),**row);u.append(run/'TRAIN_LOG.jsonl',value);u.write(run/'TRAIN_PROGRESS.json',dict(status='TRAINING',**value))
        theta,info=obj.optimize(h,offset,scale,groups,progress)
        state={k:v.clone() for k,v in reference.items()};state['action_head.weight'][3]=(state['action_head.weight'][3].double()+theta[:-1]/scale).float();state['action_head.bias'][3]=(state['action_head.bias'][3].double()+theta[-1]).float();u.validate_head(reference,state)
        actual=h.float()@state['action_head.weight'].T+state['action_head.bias'];margin=(actual[:,3]-actual[:,:3].max(1).values).double()
        predicted=obj.margins(theta,h/scale,offset);assert float((predicted-margin).abs().max())<1e-4,'FOLDED_HEAD_MISMATCH'
        before_calibration=trajectory_metrics(margin,test)
        from calibration import fit_bias
        adjustment=fit_bias(margin,offset,test)
        state['action_head.bias'][3]-=adjustment['bias_reduction']
        calibrated=h.float()@state['action_head.weight'].T+state['action_head.bias']
        calibrated_margin=(calibrated[:,3]-calibrated[:,:3].max(1).values).double()
        after_calibration=trajectory_metrics(calibrated_margin,test)
        assert after_calibration['any_false_stop']<=trajectory_metrics(offset,test)['any_false_stop']+1e-12,'CALIBRATION_NUMERIC_BUDGET'
        u.validate_head(reference,state)
        u.write(run/'CALIBRATION.json',dict(adjustment=adjustment,before=before_calibration,after=after_calibration,fit_houses=houses,calibration_houses=check,reference=trajectory_metrics(offset,test),no_refit_on_calibration_houses=True),True)
        margin=calibrated_margin
        record=dict(mode=mode,optimization=info,reference=trajectory_metrics(offset,test),candidate=trajectory_metrics(margin,test),motion_parameters_exact_unchanged=True,base_updates=0,scoring_houses=check,prior_V13_house_exposure=True,parameter_fingerprint=u.state_stamp(state))
        fits.append(record);u.write(run/(mode.upper()+'_RESULT.json'),record,True)
        candidate=state
    tmp=run/'CANDIDATE.tmp.pt';torch.save(dict(trainable=candidate,source_checkpoint_sha256=cfg['checkpoint_sha256'],anchor_sha256=cfg['reference_checkpoint_sha256']),tmp);tmp.rename(run/'CANDIDATE.pt')
    u.write(run/'TRAIN_RESULT.json',dict(status='TRAINED_PENDING_CLOSED_LOOP',fits=fits,checkpoint_sha256=u.sha(run/'CANDIDATE.pt'),learned_parameters=2049,calibration_parameters=1,base_updates=0,wall_seconds=time.time()-start,navigation_benefit=None),True);u.write(run/'TRAIN_PROGRESS.json',dict(status='COMPLETE',unix=time.time()))
if __name__=='__main__':main(Path(sys.argv[1]))
