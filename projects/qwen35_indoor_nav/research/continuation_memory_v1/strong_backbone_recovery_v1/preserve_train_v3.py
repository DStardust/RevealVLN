"""Matched repair training; ordinary preservation is never agreement-masked."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
from memory_v2 import ExecutionMemory
from train_memory_v2 import validate
from preserve_objective_v3 import objective


def main(root,trial,arm,initialize):
    u.verify_sources(root);torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
    p=u.read(root/'PROTOCOL.json');weight=p['preservation_weights'][trial]
    special=validate(torch.load(Path(p['memory_run'])/'data/FIT.pt',map_location='cpu',weights_only=True))
    admission=u.read(root/'data/ADMISSION.json');assert u.sha(root/'data/POOLS.pt')==admission['pools_sha256']
    pools=torch.load(root/'data/POOLS.pt',map_location='cpu',weights_only=True)
    training=root/'trials'/str(trial)/'training';training.mkdir(parents=True,exist_ok=True)
    folder=training/arm;folder.mkdir(exist_ok=True)
    if (folder/'FINAL.pt').exists():raise ValueError('FINAL_ALREADY_EXISTS')
    model=ExecutionMemory(special['groups'][0]['features'].shape[-1])
    initial={k:v.clone() for k,v in model.state_dict().items()};init_path=training/'INITIAL_42.pt'
    if init_path.exists():
        stored=torch.load(init_path,map_location='cpu',weights_only=True)
        assert all(torch.equal(v,stored[k]) for k,v in initial.items())
    else:torch.save(initial,init_path)
    rng=random.Random(42);schedule=[]
    indices=[list(range(len(special['groups']))),list(range(len(pools['ordinary_fit']))),list(range(len(pools['dense'])))]
    for step in range(1200):
        row=[]
        for order in indices:
            if step%len(order)==0:rng.shuffle(order)
            row.append(order[step%len(order)])
        schedule.append(row)
    sp=training/'SCHEDULE.json'
    if sp.exists():assert u.read(sp)==schedule
    else:u.write(sp,schedule)
    if initialize:return
    model=model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    binding=dict(seed=42,steps=1200,preservation_weight=weight,pools_sha256=admission['pools_sha256'],
        sparse_sha256=u.sha(Path(p['memory_run'])/'data/FIT.pt'),source_lock_sha256=u.sha(root/'SOURCE_LOCK.json'),base_updates=0,
        loss='sparse_task + weight*(ordinary_KL+native_margin) + .5*ordinary_CE + .5*class_balanced_legal_suffix_CE')
    checkpoint=folder/'LATEST.pt';start=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['binding']==binding
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['step']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);random.setstate(saved['python_rng'])
    def cuda(row):return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in row.items()}
    groups=[cuda(g) for g in special['groups']];ordinary=[cuda(g) for g in pools['ordinary_fit']];dense=[cuda(g) for g in pools['dense']]
    class_weights=pools['dense_class_weights'].cuda();began=time.time();attempt=time.time_ns()
    u.write(folder/'PROTOCOL.json',dict(arm=arm,**binding))
    with (folder/'STEPS.jsonl').open('a',buffering=1) as log:
        for step in range(start,1200):
            g,o,d=schedule[step]
            loss,items=objective(model,groups[g],ordinary[o],dense[d],arm,weight,class_weights)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or not all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()):
                raise ValueError('NONFINITE_LOSS_OR_GRADIENT')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            row=dict(step=step+1,planned=1200,arm=arm,trial=trial,attempt=attempt,loss=float(loss),gradient_norm=float(norm),
                schedule=[g,o,d],wall_seconds=time.time()-began,
                **{k:float(v) if isinstance(v,torch.Tensor) else v for k,v in items.items()})
            assert row['ordinary_queries']>0 and row['dense_queries']>0
            log.write(json.dumps(row,allow_nan=False)+'\n')
            if (step+1)%10==0:u.write(folder/'PROGRESS.json',dict(status='TRAINING',**row))
            if (step+1)%200==0:
                temp=folder/'LATEST.tmp';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    step=step+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate(),binding=binding),temp);temp.replace(checkpoint)
    assert any(not torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items()),'NO_PARAMETER_UPDATE'
    temp=folder/'FINAL.tmp';torch.save(dict(model=model.state_dict(),binding=binding,arm=arm,step=1200),temp);temp.replace(folder/'FINAL.pt')
    u.write(folder/'RESULT.json',dict(status='TRAINING_COMPLETE',updates=1200,base_updates=0,final_sha256=u.sha(folder/'FINAL.pt')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--trial',type=int,required=True)
    p.add_argument('--arm',choices=['BC','B2','OURS'],required=True);p.add_argument('--initialize',action='store_true');a=p.parse_args()
    main(a.root,a.trial,a.arm,a.initialize)

