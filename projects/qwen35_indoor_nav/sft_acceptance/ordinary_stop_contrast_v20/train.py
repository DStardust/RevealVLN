"""One fixed FIT-only convex head fit. Diagnostic scores never select a checkpoint."""
import collections,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
import objective as o
torch.set_num_threads(4)

def groups_for(rows,pairs,houses):
    rows=[r for r in rows if r['house'] in houses];counts=collections.Counter(r['house'] for r in rows)
    assert set(counts)==set(houses)
    width=max(1,max(len(r['negative']) for r in rows));nn=torch.zeros(len(rows),width,dtype=torch.long);mask=torch.zeros_like(nn,dtype=torch.bool)
    pos=[];pm=[];w=[]
    for i,r in enumerate(rows):
        n=len(r['negative']);nn[i,:n]=torch.tensor(r['negative'],dtype=torch.long);mask[i,:n]=True
        pos.append(r['positive'] or 0);pm.append(r['positive'] is not None);w.append(1/len(houses)/counts[r['house']])
    pp={}
    for r in pairs:
        if r['house'] in houses:pp[(r['rank'],r['positive'],r['negative'])]=r
    pairs=list(pp.values());pc=collections.Counter(r['rank'] for r in pairs);hc=collections.defaultdict(set)
    for r in pairs:hc[r['house']].add(r['rank'])
    pw=[1/len(hc)/len(hc[r['house']])/pc[r['rank']] for r in pairs]
    return dict(negative_indices=nn,negative_mask=mask,positive_indices=torch.tensor(pos),positive_mask=torch.tensor(pm,dtype=torch.float64),
        trajectory_weights=torch.tensor(w,dtype=torch.float64),pair_positive=torch.tensor([r['positive'] for r in pairs]),
        pair_negative=torch.tensor([r['negative'] for r in pairs]),pair_weights=torch.tensor(pw,dtype=torch.float64))

def metrics(m,ref,g):
    v=m[g['negative_indices']].masked_fill(~g['negative_mask'],-1e9)
    rv=ref[g['negative_indices']].masked_fill(~g['negative_mask'],-1e9)
    w=g['trajectory_weights'];pm=g['positive_mask'];pw=w*pm
    return dict(any_false_stop=float((w*(v.max(1).values>0)).sum()),
        verified_terminal_stop_recall=float((pw*(m[g['positive_indices']]>0)).sum()/pw.sum()),
        paired_terminal_rank=float((g['pair_weights']*(m[g['pair_positive']]>m[g['pair_negative']])).sum()),
        reference_continue_to_stop_trajectories=int(((v>0)&(rv<=0)&g['negative_mask']).any(1).sum()),
        reference_terminal_preserved=int(((m[g['positive_indices']]>0)&(ref[g['positive_indices']]>0)&pm.bool()).sum()),
        terminal_count=int(pm.sum()),negative_occurrences=int(g['negative_mask'].sum()),matched_pairs=len(g['pair_weights']),scope='Fixed observed histories; not a closed-loop success estimate')

def main(run):
    u.verify();cfg=u.read(run/'PROTOCOL.json');ready=u.read(run/'DATA_READY.json')
    for key,name in [('cache_sha256','FEATURES.pt'),('trajectories_sha256','TRAJECTORY_INDEX.json'),('pairs_sha256','MATCHED_TERMINAL_PAIRS.json'),('audit_sha256','DATA_AUDIT.json')]:assert u.sha(run/name)==ready[key],'DATA_CHANGED'
    cache=torch.load(run/'FEATURES.pt',map_location='cpu',weights_only=True);h=cache['features'];ref=cache['reference_margin'].double()
    audit=u.read(run/'DATA_AUDIT.json');rows=u.read(run/'TRAJECTORY_INDEX.json');pairs=u.read(run/'MATCHED_TERMINAL_PAIRS.json')
    fit=groups_for(rows,pairs,audit['split']['fit_houses']);check=groups_for(rows,pairs,audit['split']['diagnostic_houses'])
    indices=sorted({i for r in rows if r['house'] in audit['split']['fit_houses'] for i in r['indices']})
    scale=h[indices].double().square().sum(1).mean().sqrt()
    def progress(r):
        r=dict(mode='matched_terminal_fit32',unix=time.time(),**r);u.append(run/'TRAIN_LOG.jsonl',r);u.write(run/'TRAIN_PROGRESS.json',dict(status='TRAINING',**r))
    theta,info=o.optimize(h,ref,scale,fit,progress)
    reference=torch.load(cfg['reference_checkpoint'],map_location='cpu',weights_only=True)['trainable']
    state={k:v.clone() for k,v in reference.items()}
    state['action_head.weight'][3]=(state['action_head.weight'][3].double()+theta[:-1]/scale).float()
    state['action_head.bias'][3]=(state['action_head.bias'][3].double()+theta[-1]).float();u.validate_head(reference,state)
    final=h@state['action_head.weight'].T+state['action_head.bias'];m=(final[:,3]-final[:,:3].max(1).values).double()
    assert float((o.margins(theta,h.double()/scale,ref)-m).abs().max())<1e-4,'SERIALIZED_HEAD_PARITY'
    diagnostics={tag:dict(reference=metrics(ref,ref,g),candidate=metrics(m,ref,g)) for tag,g in [('FIT32',fit),('DIAGNOSTIC8',check)]}
    tmp=run/'CANDIDATE.tmp.pt';torch.save(dict(trainable=state,source_checkpoint_sha256=cfg['checkpoint_sha256'],anchor_sha256=cfg['reference_checkpoint_sha256']),tmp);tmp.rename(run/'CANDIDATE.pt')
    record=dict(mode='matched_terminal_fit32',optimization=info,reference=diagnostics['DIAGNOSTIC8']['reference'],candidate=diagnostics['DIAGNOSTIC8']['candidate'],parameter_fingerprint=u.state_stamp(state))
    result=dict(status='TRAINED_PENDING_CLOSED_LOOP',fits=[record],diagnostics=diagnostics,checkpoint_sha256=u.sha(run/'CANDIDATE.pt'),base_updates=0,learned_parameters=2049,
                navigation_benefit=None,model_choice='One fixed objective/seed/final optimizer result. No bias calibration, refit, diagnostic/DEV/unseen checkpoint selection.')
    u.write(run/'TRAIN_RESULT.json',result,True);u.write(run/'TRAIN_PROGRESS.json',dict(status='COMPLETE',unix=time.time()))
    print(__import__('json').dumps(diagnostics,indent=2),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
