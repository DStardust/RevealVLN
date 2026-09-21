"""Frozen causal-state cache plus bounded STOP-readout training; complete optimizer resume."""
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def cache_rows(net,rows,features):
    import torch
    lengths=[len(r['features']) for r in rows];maximum=max(lengths)
    indices=torch.tensor([r['features']+[0]*(maximum-n) for r,n in zip(rows,lengths)],device='cuda')
    alive=torch.tensor([[True]*n+[False]*(maximum-n) for n in lengths],device='cuda')
    with torch.inference_mode():pred=net(features['features'][indices],features['logits'][indices],alive)
    output=[]
    for i,(r,n) in enumerate(zip(rows,lengths)):
        result=dict(logits=pred['logits'][i,:n].cpu(),state=pred['state'][i,:n].cpu(),native=features['logits'][indices[i,:n]].cpu(),
            targets=torch.tensor(r['targets'],dtype=torch.long))
        if 'action_masks' in r:
            result.update(action_mask=torch.tensor(r['action_masks'],dtype=torch.bool),truth=torch.tensor(r['state_targets']),cutoff=r['cutoff'],
                preservation=torch.arange(n)<min(156,r['cutoff']+1),history=r['history'],task=r['task'])
        output.append(result)
    return output

def build_cache(run,seed,net):
    import torch
    path=run/'cache'/f'STATE_{seed}.pt';meta=path.with_suffix('.json')
    if meta.exists():
        if sha(path)!=read(meta)['sha256']:raise ValueError('STATE_CACHE_CHANGED')
        return torch.load(path,map_location='cpu',weights_only=True)
    data=read(run/'DATA.json');ordinary=read(run/'ORDINARY_DATA.json')
    checkpoint=read(run/'train'/f'MONOTONIC_{seed}'/'RESULT.json');before=c.model_identity(net)['sha256']
    if before!=checkpoint['final']:raise ValueError('FROZEN_CACHE_MODEL')
    features={k:v.float().cuda() for k,v in torch.load(run/'features/FEATURES.pt',map_location='cpu',weights_only=True).items()}
    families={};started=time.monotonic()
    for i,f in enumerate(data['families']):
        if f['split'] not in ('FIT','DEV'):raise ValueError('UNREGISTERED_DATA_SPLIT')
        families[f['family_id']]=dict(split=f['split'],rows=cache_rows(net,f['sequences'],features))
        write(run/f'TRAIN_PROGRESS_{seed}.json',dict(stage='frozen_state_cache',family=i+1,total_families=len(data['families']),step=0,seed=seed))
    del features
    ordinary_features={k:v.float().cuda() for k,v in torch.load(run/'features/ORDINARY_FEATURES.pt',map_location='cpu',weights_only=True).items()}
    natural=[]
    for i in range(0,len(ordinary['records']),16):
        rows=ordinary['records'][i:i+16];cached=cache_rows(net,rows,ordinary_features)
        natural.extend(dict(value,partition=row['partition']) for value,row in zip(cached,rows))
    del ordinary_features
    after=c.model_identity(net)['sha256']
    if before!=after:raise ValueError('FROZEN_MODEL_UPDATED')
    value=dict(families=families,ordinary=natural)
    path.parent.mkdir(exist_ok=True);atomic_torch(path,value)
    immutable(meta,dict(sha256=sha(path),model_state_sha256=before,model_unchanged=True,seconds=time.monotonic()-started,
        causal_features_only=True,query_in_readout=False,base_loaded=False,base_updates=0))
    return value

def stack(rows,device,masked=False):
    import torch
    # No ground truth or future query is packed into readout input.
    tensors={key:torch.cat([r[key][r['action_mask']] if masked else r[key] for r in rows]).to(device) for key in ('logits','state','targets')}
    return tensors

def objective(head,family,ordinary):
    import torch
    from torch.nn import functional as F
    device=next(head.parameters()).device;rows=family['rows'];a=stack(rows,device,True)
    logits=head(a['logits'],a['state']);action=F.cross_entropy(logits,a['targets'])
    cutrows=[{k:r[k][r['cutoff']:r['cutoff']+1] for k in ('logits','state','targets')} for r in rows if r['action_mask'][r['cutoff']]]
    cut=stack(cutrows,device);cutloss=F.cross_entropy(head(cut['logits'],cut['state']),cut['targets'])
    old=torch.cat([r['logits'][r['preservation']] for r in rows]).to(device)
    z=torch.cat([r['state'][r['preservation']] for r in rows]).to(device)
    native=torch.cat([r['native'][r['preservation']] for r in rows]).to(device)
    kl=F.kl_div(head(old,z).log_softmax(-1),native.softmax(-1),reduction='batchmean')
    natural=[]
    for r in ordinary:
        b=stack([r],device);natural.append(F.cross_entropy(head(b['logits'],b['state']),b['targets']))
    ordinary_ce=torch.stack(natural).mean()
    total=action+cutloss+kl+ordinary_ce
    return total,dict(action_ce=float(action.detach()),cutoff_ce=float(cutloss.detach()),preservation_kl=float(kl.detach()),ordinary_ce=float(ordinary_ce.detach()))

def diagnose(head,cache):
    import torch
    values=[];device=next(head.parameters()).device
    with torch.inference_mode():
        for split in ('FIT','DEV'):
            for task in ('task_A','task_T'):
                rows=[r for f in cache['families'].values() if f['split']==split for r in f['rows'] if r['task']==task and bool(r['action_mask'].any())]
                a=stack(rows,device,True);pred=head(a['logits'],a['state']).argmax(-1);old=a['logits'].argmax(-1);target=a['targets']
                truth=torch.cat([r['truth'][r['action_mask']] for r in rows]).to(device)
                correct_state=((a['state']>=.5)==truth.bool()).all(-1)
                def metric(actions):
                    return dict(n=len(actions),correct=int((actions==target).sum()),stop_n=int((target==3).sum()),continue_n=int((target!=3).sum()),
                        missed_stop=int(((target==3)&(actions!=3)).sum()),false_stop=int(((target!=3)&(actions==3)).sum()),
                        correct_state_stop_n=int(((target==3)&correct_state).sum()),correct_state_missed_stop=int(((target==3)&correct_state&(actions!=3)).sum()))
                values.append(dict(split=split,task=task,old=metric(old),repair=metric(pred)))
        for partition in ('fit','check'):
            a=stack([r for r in cache['ordinary'] if r['partition']==partition],device);target=a['targets'];old=a['logits'].argmax(-1);pred=head(a['logits'],a['state']).argmax(-1)
            def ordinary_metric(actions):return dict(n=len(actions),correct=int((actions==target).sum()),continue_n=int((target!=3).sum()),stop_n=int((target==3).sum()),missed_stop=int(((target==3)&(actions!=3)).sum()),false_stop=int(((target!=3)&(actions==3)).sum()))
            values.append(dict(split='ORDINARY_'+partition.upper(),task='ordinary',old=ordinary_metric(old),repair=ordinary_metric(pred)))
    return values

def main(run):
    import torch
    import numpy as np
    cfg=config(run);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for tag in os.environ['B2_TRAIN_TAGS'].split(','):
        seed=int(tag.split('_')[1]);target=run/'train'/tag
        if (target/'RESULT.json').exists():continue
        target.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        net=make_head(tag).cuda().eval();source=run/'train'/f'MONOTONIC_{seed}'/'FINAL.pt'
        net.frozen.load_state_dict(torch.load(source,map_location='cpu',weights_only=True),strict=True)
        initial_readout=c.model_identity(net.readout)['sha256']
        before=c.model_identity(net.frozen)['sha256'];cache=build_cache(run,seed,net.frozen)
        opt=torch.optim.AdamW(net.readout.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        schedule=read(run/'SCHEDULES.json')[str(seed)];cursor=0
        checkpoints=sorted(p for p in target.glob('STEP_*.pt') if p.with_suffix('.json').exists())
        binding=dict(protocol_sha256=sha(run/'PROTOCOL.json'),source_lock_sha256=sha(run/'SOURCE_LOCK.json'),cache_sha256=sha(run/'cache'/f'STATE_{seed}.pt'),source_head_sha256=sha(source),seed=seed)
        if checkpoints:
            last=checkpoints[-1];meta=read(last.with_suffix('.json'))
            if meta['sha256']!=sha(last) or meta['binding']!=binding:raise ValueError('RESUME_BINDING')
            state=torch.load(last,map_location='cuda',weights_only=False);net.readout.load_state_dict(state['readout']);opt.load_state_dict(state['optimizer']);cursor=state['step']
            torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']]);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        attempt=target/f'ATTEMPT_{len(list(target.glob("ATTEMPT_*.jsonl")))+1:03d}.jsonl';started=time.monotonic()
        for i in range(cursor,cfg['repair_updates']):
            item=schedule[i];family=cache['families'][item['family']];natural=[cache['ordinary'][j] for j in item['ordinary']]
            if family['split']!='FIT' or any(r['partition']!='fit' for r in natural):raise ValueError('NONFIT_TRAINING')
            opt.zero_grad(set_to_none=True);loss,stats=objective(net.readout,family,natural)
            if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
            loss.backward();gradient=float(torch.sqrt(sum(p.grad.square().sum() for p in net.readout.parameters())))
            if not math_isfinite(gradient) or any(p.grad is not None for p in net.frozen.parameters()):raise ValueError('NONFINITE_OR_FROZEN_GRADIENT')
            opt.step();append(attempt,dict(step=i+1,loss=float(loss.detach()),gradient_norm=gradient,family=item['family'],**stats))
            if (i+1)%20==0:write(run/f'TRAIN_PROGRESS_{seed}.json',dict(stage='readout_train',step=i+1,target=cfg['repair_updates'],seed=seed,loss=float(loss.detach()),gradient_norm=gradient))
            if (i+1)%200==0 or i+1==cfg['repair_updates']:
                p=target/f'STEP_{i+1:04d}.pt';atomic_torch(p,dict(readout=net.readout.state_dict(),optimizer=opt.state_dict(),step=i+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),binding=binding))
                immutable(p.with_suffix('.json'),dict(sha256=sha(p),binding=binding,step=i+1))
        if c.model_identity(net.frozen)['sha256']!=before:raise ValueError('ORIGINAL_MODEL_CHANGED')
        immutable(target/'LOCAL_DIAGNOSIS.json',dict(rows=diagnose(net.readout,cache),checkpoint_selection='fixed final, no DEV-selected step'))
        atomic_torch(target/'FINAL.pt',net.state_dict());final=c.model_identity(net)['sha256']
        immutable(target/'RESULT.json',dict(status='STOP_READOUT_TRAINING_COMPLETE',updates=cfg['repair_updates'],base_updates=0,original_model_updates=0,
            source_model_state=before,frozen_model_unchanged=True,initial_readout=initial_readout,final_readout=c.model_identity(net.readout)['sha256'],readout_changed=initial_readout!=c.model_identity(net.readout)['sha256'],final=final,checkpoint_sha256=sha(target/'FINAL.pt'),trainable_parameters=sum(p.numel() for p in net.readout.parameters()),seconds=time.monotonic()-started))

def math_isfinite(value):
    import math
    return math.isfinite(value)

if __name__=='__main__':main(Path(sys.argv[1]))
