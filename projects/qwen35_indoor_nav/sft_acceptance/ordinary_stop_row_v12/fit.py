"""One preregistered anchored linear STOP fit; no search or motion updates."""
import collections,importlib.util,json,math,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
import torch
torch.set_num_threads(4)
def weights(labels,houses):
    ep=collections.defaultdict(set);he=collections.defaultdict(set)
    for i,row in enumerate(labels):
        if row['house'] not in houses:continue
        for e in {x['episode'] for x in row['occurrences']}:ep[e].add(i);he[row['house']].add(e)
    assert set(he)==set(houses)
    w=torch.zeros(len(labels),dtype=torch.float64)
    for house,episodes in he.items():
        for e in episodes:
            for i in ep[e]:w[i]+=1/len(houses)/len(episodes)/len(ep[e])
    assert abs(w.sum().item()-1)<1e-12
    return w
def metrics(margin,y,w):
    pred=margin>0;yes=y>0;tp=float((w*pred*yes).sum());fp=float((w*pred*~yes).sum());fn=float((w*~pred*yes).sum())
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return dict(bce=float((w*torch.nn.functional.binary_cross_entropy_with_logits(margin,y,reduction='none')).sum()),precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,tp=tp,fp=fp,fn=fn)
def optimize(h,z,y,w):
    keep=w>0;scale=(w[:,None]*h.square()).sum().sqrt();assert scale>0
    x=h[keep]/scale;offset=z[keep,3]-z[keep,:3].max(1).values;target=y[keep];ww=w[keep]
    theta=torch.zeros(h.shape[1]+1,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([theta],lr=1,max_iter=200,max_eval=250,history_size=50,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe')
    calls=0;began=time.monotonic()
    def closure():
        nonlocal calls
        calls+=1;assert calls<=250 and time.monotonic()-began<600
        opt.zero_grad();margin=offset+x@theta[:-1]+theta[-1]
        loss=(ww*torch.nn.functional.binary_cross_entropy_with_logits(margin,target,reduction='none')).sum()+.0005*theta.square().sum()
        assert torch.isfinite(loss);loss.backward();return loss
    opt.step(closure)
    assert torch.isfinite(theta).all()
    return theta.detach(),scale,dict(closure_calls=calls,optimizer_iterations=int(opt.state[theta]['n_iter']),wall_seconds=time.monotonic()-began,scale=float(scale),residual_norm=float(theta.norm()))
def main(mode):
    assert mode in ('probe','final');c.verify()
    assert c.read(HERE/'FEATURE_PARITY.json')['status']=='PASS'
    if mode=='final':assert c.read(HERE/'PROBE_RESULT.json')['pass_gate'] is True
    assert not (HERE/(mode.upper()+'_RESULT.json')).exists()
    cache=torch.load(HERE/'FEATURES.pt',map_location='cpu',weights_only=True)
    labels=c.rows(HERE/'SUPERVISION_ONLY.jsonl');assert cache['record_ids']==[x['record_id'] for x in labels]
    h=cache['features'].double();z=cache['logits'].double();y=torch.tensor([x['target'] for x in labels],dtype=torch.float64)
    split=c.read(HERE/'SPLIT.json');fit_houses=split['fit_houses'] if mode=='probe' else split['all_houses']
    w=weights(labels,fit_houses);theta,scale,info=optimize(h,z,y,w)
    original=torch.load(c.BEST,map_location='cpu',weights_only=True);trainable={k:v.clone() for k,v in original['trainable'].items()}
    assert trainable['action_head.weight'].shape==(4,2048) and trainable['action_head.bias'].shape==(4,)
    trainable['action_head.weight'][3]=(trainable['action_head.weight'][3].double()+theta[:-1]/scale).float()
    trainable['action_head.bias'][3]=(trainable['action_head.bias'][3].double()+theta[-1]).float()
    for name,value in trainable.items():
        if name.startswith('action_head.'):assert torch.equal(value[:3],original['trainable'][name][:3])
        else:assert torch.equal(value,original['trainable'][name])
    # Re-evaluate the actually serialized FP32 head, not an unrounded optimizer vector.
    newstop=(h.float()@trainable['action_head.weight'][3]+trainable['action_head.bias'][3]).double()
    folded=h@trainable['action_head.weight'][3].double()+trainable['action_head.bias'][3].double()
    assert (newstop-folded).abs().max()<1e-4
    testhouses=split['heldout_houses'] if mode=='probe' else split['all_houses'];tw=weights(labels,testhouses)
    oldmargin=z[:,3]-z[:,:3].max(1).values;newmargin=newstop-z[:,:3].max(1).values
    before=metrics(oldmargin,y,tw);after=metrics(newmargin,y,tw)
    gate=after['bce']<before['bce'] and after['precision']>=before['precision'] and after['recall']>=before['recall'] and after['f1']>before['f1']
    ckpt=HERE/(mode+'_stop_row.pt');assert not ckpt.exists()
    torch.save(dict(trainable=trainable,binding=dict(protocol_sha256=c.sha(HERE/'PLAN_ZH.md'),sample_index_sha256=original['binding']['sample_index_sha256']),cursor=dict(updates=4000),stop_row_v12=dict(mode=mode,not_optimizer_resume_checkpoint=True,backbone_optimizer_steps=4000,fit=info,feature_sha256=c.sha(HERE/'FEATURES.pt'))),ckpt)
    c.write(HERE/(mode.upper()+'_RESULT.json'),dict(status='COMPLETE',unix=time.time(),mode=mode,fit_houses=fit_houses,scoring_houses=testhouses,before=before,after=after,pass_gate=gate if mode=='probe' else None,fit=info,checkpoint=str(ckpt),checkpoint_sha256=c.sha(ckpt),motion_parameters_exact_unchanged=True,other_parameters_exact_unchanged=True,learned_parameters=2049,positive_navigation_result=False,scope='FIT only; feature-probe metrics not navigation'))
    print(json.dumps(c.read(HERE/(mode.upper()+'_RESULT.json'))))
if __name__=='__main__':main(sys.argv[1])
