"""Matched BC/B2/history-effect optimization on admitted causal feature groups.

This is a real training entry, not automatically run on ordinary DEV caches.
Each input batch contains complete causal histories for both histories and all
tasks; future outcomes occur only in the loss. No base model is optimized.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
import torch.nn.functional as F
from memory_v2 import ExecutionMemory,interaction_loss,masked_state_loss


def validate(data):
    if data.get('action_boundary')!='AFTER_NATIVE_ASSISTANT_HEADER_V2':raise ValueError('OLD_ACTION_POSITION_CACHE')
    if data['split']!='FIT' or data['scope']!='STREAMVLN_EXECUTED_HISTORY_FEATURES':
        raise ValueError('UNADMITTED_TRAINING_SPLIT_OR_FEATURE_PATH')
    if data.get('future_in_policy_features') is not False:
        raise ValueError('FUTURE_INPUT_AUDIT_REQUIRED')
    if not data.get('groups'):raise ValueError('EMPTY_FIT_POOL')
    for group in data['groups']:
        if not group['actual_execution_certified'] or not group['policy_witness_certified']:
            raise ValueError('UNVERIFIED_HISTORY_OR_WITNESS')
        if not group['same_current_input_certified'] or not group['fork_is_runtime_query']:
            raise ValueError('UNMATCHED_INPUT_OR_UNEXECUTABLE_FORK')
        for path,digest in group['certificates'].items():
            if u.sha(path)!=digest:raise ValueError('DATA_CERTIFICATE_CHANGED')
        if not group['certificates']:raise ValueError('MISSING_EXECUTION_CERTIFICATES')
        x=group['features'];lengths=group['lengths']
        if x.ndim!=4 or x.shape[1]!=2 or tuple(lengths.shape)!=tuple(x.shape[:2]):
            raise ValueError('TASK_HISTORY_SEQUENCE_SHAPE')
        if not bool(torch.isfinite(x).all()) or not bool(((lengths>0)&(lengths<=x.shape[2])).all()):
            raise ValueError('INVALID_CAUSAL_SEQUENCE')
    return data


def objective(model,group,arm):
    x=group['features'];tasks,histories,steps,width=x.shape
    out=model(x.reshape(tasks*histories,steps,width),group['lengths'].flatten(),**({'actor_features':group['actor_features'].reshape(tasks*histories,steps,width)} if 'actor_features' in group else {}))
    logits=group['base_action_logits']+out['delta'].reshape(tasks,histories,steps,4)
    mask=torch.arange(steps,device=x.device)[None,None,:]<group['lengths'][:,:,None]
    supervised=mask & group['action_known']
    if not bool(supervised.any()):raise ValueError('NO_SHARED_ACTION_SUPERVISION')
    bc=F.cross_entropy(logits[supervised],group['action_targets'][supervised])
    preservation=mask & group['preservation_mask']
    kl=F.kl_div(F.log_softmax(logits[preservation],-1),F.softmax(group['base_action_logits'][preservation],-1),reduction='batchmean') if bool(preservation.any()) else bc*0
    auxiliary=bc*0
    if arm=='B2':
        states=model.state_reader(out['memory'].flatten(-2)).reshape(tasks,histories,steps,4)
        auxiliary=masked_state_loss(states,group['state_targets'],group['state_known'] & mask[:,:,:,None])
    elif arm=='OURS':
        # Actual choice point after the full registered causal prefix.
        at=logits.gather(2,(group['lengths']-1)[:,:,None,None].expand(tasks,histories,1,4)).squeeze(2)
        auxiliary=interaction_loss(at[None],group['action_pairs'][None],group['returns'][None],group['return_known'][None])
    return bc+.2*kl+auxiliary,dict(action_ce=bc,ordinary_kl=kl,auxiliary=auxiliary,
        supervised_actions=int(supervised.sum()),known_outcomes=int(group['return_known'].sum()))


def main(data_path,run,arm,seed,steps,device,initialize_only=False):
    data=validate(torch.load(data_path,map_location='cpu',weights_only=True))
    run.mkdir(parents=True,exist_ok=True);folder=run/arm;folder.mkdir(exist_ok=True)
    if (folder/'FINAL.pt').exists():raise ValueError('FINAL_ALREADY_EXISTS')
    torch.set_num_threads(4);torch.manual_seed(seed);random.seed(seed)
    width=data['groups'][0]['features'].shape[-1]
    model=ExecutionMemory(width).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    source={str(p):u.sha(p) for p in (u.HERE/'memory_v2.py',Path(__file__))}
    binding=dict(data_sha256=u.sha(data_path),source=source,seed=seed,steps=steps,learning_rate=1e-4,
        preservation_kl_weight=.2,auxiliary_weight=1.,base_updates=0,selection='fixed final step; no DEV/TEST selection')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_path=run/f'INITIAL_{seed}.pt'
    if initial_path.exists():
        stored=torch.load(initial_path,map_location='cpu',weights_only=True)
        assert all(torch.equal(v,stored[k]) for k,v in initial.items()),'INITIALIZATION_CHANGED'
    else:torch.save(initial,initial_path)
    order=[];rng=random.Random(seed)
    while len(order)<steps:
        indices=list(range(len(data['groups'])));rng.shuffle(indices);order.extend(indices)
    order=order[:steps];schedule_path=run/f'SCHEDULE_{seed}.json'
    if schedule_path.exists():assert u.read(schedule_path)==order,'SCHEDULE_CHANGED'
    else:u.write(schedule_path,order)
    if initialize_only:return
    checkpoint=folder/'LATEST.pt';start=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['binding']==binding
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
        torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng'])
        if device.startswith('cuda'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
    u.write(folder/'PROTOCOL.json',dict(arm=arm,**binding));began=time.time()
    attempt=time.time_ns()
    u.write(folder/f'ATTEMPT_{attempt}.json',dict(resumed_step=start,planned=steps,binding=binding))
    with (folder/'STEPS.jsonl').open('a',buffering=1) as log:
        for step in range(start,steps):
            group={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in data['groups'][order[step]].items()}
            loss,items=objective(model,group,arm);optimizer.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('NONFINITE_LOSS_OR_GRADIENT')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            row=dict(attempt=attempt,step=step+1,arm=arm,group=order[step],loss=float(loss),gradient_norm=float(norm),
                **{k:float(v) if isinstance(v,torch.Tensor) else v for k,v in items.items()},wall_seconds=time.time()-began)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            if (step+1)%10==0:u.write(folder/'PROGRESS.json',dict(status='TRAINING',planned=steps,**row))
            if (step+1)%200==0 or step+1==steps:
                temp=folder/'LATEST.tmp';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                    python_rng=random.getstate(),binding=binding),temp);temp.replace(checkpoint)
    torch.save(dict(model=model.state_dict(),binding=binding,arm=arm,step=steps),folder/'FINAL.pt')
    changed=any(not torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())
    assert changed,'NO_PARAMETER_UPDATE'
    u.write(folder/'RESULT.json',dict(status='TRAINING_COMPLETE',updates=steps,base_updates=0,parameters_changed=True,
        final_sha256=u.sha(folder/'FINAL.pt'),closed_loop_benefit='NOT_YET_MEASURED'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--arm',choices=['BC','B2','OURS'],required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--initialize-only',action='store_true');p.add_argument('--steps',type=int,default=1200);p.add_argument('--device',default='cuda');a=p.parse_args()
    main(a.data,a.run,a.arm,a.seed,a.steps,a.device,a.initialize_only)
