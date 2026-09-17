"""Bounded, rank-isolated Qwen SFT learnability diagnostic; no simulator."""
import contextlib
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
OLD = OUT.parent/'v1'
RANK = int(os.environ['RANK'])
DEVICE = 'cuda:0'

def save(name, value):
    with (OUT/f'rank{RANK}_{name}.json').open('x') as f: json.dump(value,f,indent=2,allow_nan=False)

def event(kind, **kw):
    value = dict(event=kind, rank=RANK, unix=time.time(), **kw)
    with (OUT/f'rank{RANK}_events.jsonl').open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')
    print(json.dumps(value,allow_nan=False),flush=True)

spec = importlib.util.spec_from_file_location('sealed_q35n_policy', OLD/'recovery_r1/policy.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

class Chunk(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, record, start, end, memory):
        instruction, images, actions = record
        loss = []
        stats = []
        for t in range(start,end):
            memory, logits, details = old.policy_forward(self.policy,old.payload(instruction,images,actions,t),memory)
            target = old.ACTIONS.index(actions[t])
            ce = F.cross_entropy(logits,torch.tensor([target],device=DEVICE))
            loss.append(ce)
            stats.append(dict(t=t,target=target,prediction=int(logits.detach().argmax(-1)),ce=float(ce.detach()),
                              memory_rms=float(memory.detach().float().square().mean().sqrt()), **details))
        assert loss
        return torch.stack(loss).sum(), memory.detach(), stats

def params(policy): return [p for p in policy.parameters() if p.requires_grad]

def gradient(policy):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().float().flatten().cpu() for p in params(policy)])

def values(policy):
    return torch.cat([p.detach().float().flatten().cpu() for p in params(policy)])

def optimizer(policy):
    return torch.optim.AdamW(params(policy),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)

def normalize_clip(policy, denominator):
    for p in params(policy):
        if p.grad is not None: p.grad.div_(denominator)
    return float(torch.nn.utils.clip_grad_norm_(params(policy),1.,error_if_nonfinite=True))

def compare(a,b):
    delta=(a-b).norm()
    return dict(relative_L2=float(delta/a.norm().clamp_min(1e-12)),max_abs=float((a-b).abs().max()),
                cosine=float(F.cosine_similarity(a.double(),b.double(),dim=0)))

def rank_identity(policy):
    state=old.trainable_state(policy)
    h=hashlib.sha256()
    for n,p in state.items(): h.update(n.encode());h.update(p.contiguous().view(torch.uint8).numpy().tobytes())
    hashes=[None,None]
    dist.all_gather_object(hashes,h.hexdigest())
    assert hashes[0]==hashes[1], 'RANK_PARAMETER_DIVERGENCE'
    return hashes[0]

def toy():
    torch.manual_seed(19)
    net=nn.Linear(3,2).to(DEVICE)
    x=torch.arange(12,dtype=torch.float32,device=DEVICE).reshape(4,3)/10
    y=torch.tensor([0,1,0,1],device=DEVICE)
    F.cross_entropy(net(x),y).backward()
    ref=torch.cat([p.grad.flatten() for p in net.parameters()]).clone()
    net.zero_grad(set_to_none=True)
    ddp=DDP(net,device_ids=[0],broadcast_buffers=False)
    F.cross_entropy(ddp(x[RANK*2:RANK*2+2]),y[RANK*2:RANK*2+2]).backward()
    actual=torch.cat([p.grad.flatten() for p in net.parameters()])
    err=float((ref-actual).abs().max())
    save('NCCL_TOY',dict(max_abs_gradient_error=err,pass_gate=err<=1e-6))
    assert err<=1e-6
    del ddp,net

def d0(policy, records):
    old.load_trainable(policy,torch.load(OLD/'recovery_r1/checkpoints/terminal.pt',map_location='cpu',weights_only=True)['trainable'])
    policy.eval()
    record=records[RANK]
    other=records[1-RANK]
    summary={}
    for mode in ['normal','zero_memory','mismatched_instruction','mismatched_images']:
        memory=old.zero_memory();steps=[]
        with torch.no_grad():
            for t in range(4):
                ins=old.payload(*record,t)
                if mode=='zero_memory': memory=old.zero_memory()
                if mode=='mismatched_instruction': ins['instruction']=other[0]
                if mode=='mismatched_images': ins['images']=other[1][max(0,t-1):t+1]
                memory,logits,_=old.policy_forward(policy,ins,memory)
                steps.append(dict(t=t,logits=logits.cpu().tolist(),memory_rms=float(memory.float().square().mean().sqrt())))
        summary[mode]=steps
    policy.zero_grad(set_to_none=True)
    memory=old.zero_memory()
    wrapper=Chunk(policy)
    loss,_,_=wrapper(record,0,4,memory)
    (loss/4).backward()
    summary['module_gradients']={n:dict(dtype=str(p.dtype),grad_norm=float(p.grad.float().norm()) if p.grad is not None else None,
                                    parameter_norm=float(p.detach().float().norm())) for n,p in policy.named_parameters() if p.requires_grad}
    summary['embedding_weight_rms']=float(policy.base.get_input_embeddings().weight.detach().float().square().mean().sqrt())
    summary['optimizer_updates']=0
    save('D0',summary)
    policy.zero_grad(set_to_none=True)

def equivalence(policy, initial, records):
    old.load_trainable(policy,initial)
    policy.train()
    chunk=Chunk(policy)
    opt=optimizer(policy)
    initial_vector=values(policy)
    for rec in records[:2]:
        loss,_,_=chunk(rec,0,4,old.zero_memory());loss.backward()
    for p in params(policy):
        if p.grad is not None:p.grad.div_(8)
    reference_gradient=gradient(policy)
    torch.nn.utils.clip_grad_norm_(params(policy),1.,error_if_nonfinite=True)
    opt.step()
    reference_delta=values(policy)-initial_vector
    del opt
    old.load_trainable(policy,initial);policy.zero_grad(set_to_none=True)
    ddp=DDP(chunk,device_ids=[0],broadcast_buffers=False,find_unused_parameters=True)
    opt=optimizer(policy)
    loss,_,_=ddp(records[RANK],0,4,old.zero_memory());loss.backward()
    for p in params(policy):
        if p.grad is not None:p.grad.div_(4)
    actual_gradient=gradient(policy)
    gc=compare(reference_gradient,actual_gradient)
    torch.nn.utils.clip_grad_norm_(params(policy),1.,error_if_nonfinite=True)
    opt.step()
    uc=compare(reference_delta,values(policy)-initial_vector)
    identity=rank_identity(policy)
    passed=gc['relative_L2']<=.10 and gc['cosine']>=.99 and uc['relative_L2']<=.15
    save('REAL_EQUIVALENCE',dict(gradient=gc,update=uc,pass_gate=passed,rank_parameter_sha256=identity,
                                single_rank_reference_diagnostic_updates=1,DDP_shared_diagnostic_updates=1))
    assert passed, 'REAL_DDP_EQUIVALENCE_FAILED'
    old.load_trainable(policy,initial);policy.zero_grad(set_to_none=True)
    del opt
    return ddp

@torch.no_grad()
def evaluate(policy,records,rows,stage):
    policy.eval();cm=torch.zeros((4,4),dtype=torch.float64,device=DEVICE);total=torch.zeros(2,dtype=torch.float64,device=DEVICE)
    for i in range(RANK,len(records),2):
        ins,images,actions=records[i];memory=old.zero_memory()
        for t,a in enumerate(actions):
            memory,logits,_=old.policy_forward(policy,old.payload(ins,images,actions,t),memory)
            target=old.ACTIONS.index(a);prediction=int(logits.argmax(-1))
            ce=float(F.cross_entropy(logits,torch.tensor([target],device=DEVICE)))
            cm[target,prediction]+=1;total[0]+=ce;total[1]+=1
            with (OUT/f'rank{RANK}_eval_{stage}.jsonl').open('a') as f:
                f.write(json.dumps(dict(row=i,job_id=rows[i]['job_id'],t=t,target=target,prediction=prediction,ce=ce))+'\n')
    dist.all_reduce(cm);dist.all_reduce(total)
    recall=cm.diag()/cm.sum(1).clamp_min(1)
    result=dict(stage=stage,decisions=int(total[1]),CE=float(total[0]/total[1]),accuracy=float(cm.diag().sum()/total[1]),
                macro_recall=float(recall.mean()),recall=recall.tolist(),STOP_precision=float(cm[3,3]/cm[:,3].sum().clamp_min(1)),
                STOP_recall=float(recall[3]),confusion=cm.long().tolist())
    result['learnability_pass']=result['accuracy']>=.95 and result['macro_recall']>=.90 and result['STOP_precision']>=.90 and result['STOP_recall']>=.90
    save('EVAL_'+stage,result);event('evaluation_complete',**result)
    return result

def train(ddp,policy,records,schedule):
    policy.train();opt=optimizer(policy);memory=old.zero_memory();total_tokens=0
    for update in range(200):
        opt.zero_grad(set_to_none=True);local_count=0;local_loss=0.;local_tokens=0;step_rows=[]
        for j in range(4):
            item=schedule[update*4+j]
            if item['start']==0:memory=old.zero_memory()
            with ddp.no_sync() if j<3 else contextlib.nullcontext():
                loss,memory,stats=ddp(records[item['row']],item['start'],item['end'],memory)
                loss.backward()
            local_count+=len(stats);local_loss+=float(loss.detach());local_tokens+=sum(s['tokens'] for s in stats)
            step_rows.extend(dict(**s,update=update+1,row=item['row'],epoch=item['epoch']) for s in stats)
        totals=torch.tensor([local_count,local_loss,local_tokens],dtype=torch.float64,device=DEVICE)
        dist.all_reduce(totals)
        # DDP has divided the gradient sum by world_size=2.
        norm=normalize_clip(policy,float(totals[0])/2)
        assert norm>0
        opt.step()
        assert all(torch.isfinite(p).all() for p in params(policy))
        total_tokens+=int(totals[2])
        assert policy.forward_tokens<=5000000, 'PER_RANK_FORWARD_TOKEN_CAP'
        with (OUT/f'rank{RANK}_train_steps.jsonl').open('a') as f:
            for row in step_rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        event('update',update=update+1,CE=float(totals[1]/totals[0]),decisions=int(totals[0]),grad_norm=norm,
              global_train_tokens=total_tokens,local_forward_tokens=policy.forward_tokens,peak_allocated=torch.cuda.max_memory_allocated())
    identity=rank_identity(policy)
    state=dict(trainable=old.trainable_state(policy),optimizer=opt.state_dict(),updates=200,schedule_cursor=800,
               memory=memory.cpu(),rank=RANK,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
               python_rng=random.getstate(),numpy_rng=np.random.get_state(),rank_parameter_sha256=identity,
               global_train_tokens=total_tokens,local_forward_tokens=policy.forward_tokens)
    path=OUT/f'rank{RANK}_terminal.pt';assert not path.exists();torch.save(state,path)
    return state

def main():
    lock=json.loads((OUT/'LOCK.json').read_text())
    for name,h in lock['code'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==h
    assert hashlib.sha256((OUT/'SUBSET.json').read_bytes()).hexdigest()==lock['subset_sha256']
    torch.set_num_threads(4);torch.cuda.set_device(0)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=300))
    toy();event('NCCL_toy_pass')
    subset=json.loads((OUT/'SUBSET.json').read_text());rows=subset['rows']
    records=[old.load_record(r) for r in rows]
    for r in records:
        assert old.payload(*r,0)['executed']==[]
        assert all(old.payload(*r,t)['executed']==r[2][max(0,t-8):t] for t in range(len(r[2])))
    event('model_load_start');policy=old.build();event('model_loaded')
    d0(policy,records);event('D0_complete')
    initial=torch.load(OLD/'checkpoints/initial.pt',map_location='cpu',weights_only=True)['trainable']
    ddp=equivalence(policy,initial,records);event('real_equivalence_pass')
    before=evaluate(policy,records,rows,'before')
    state=train(ddp,policy,records,subset['schedules'][RANK])
    after=evaluate(policy,records,rows,'after')
    with torch.no_grad():
        rec=records[RANK];_,ref,_=old.policy_forward(policy,old.payload(*rec,0),old.zero_memory())
        policy.action_head.bias.add_(1.)
        # This is our own newly written checkpoint, including explicit RNG objects.
        loaded=torch.load(OUT/f'rank{RANK}_terminal.pt',map_location='cpu',weights_only=False)
        old.load_trainable(policy,loaded['trainable'])
        _,actual,_=old.policy_forward(policy,old.payload(*rec,0),old.zero_memory())
        error=float((ref-actual).abs().max());assert error<=1e-5
    save('RESULT',dict(DDP_interface_pass=True,learnability_pass=after['learnability_pass'],before=before,after=after,
                       terminal_updates=200,reload_max_abs=error,scientific_pass=False,local_forward_tokens=policy.forward_tokens,
                       claim='ten training routes only; no generalization or navigation claim'))
    event('worker_complete');dist.destroy_process_group()

if __name__=='__main__':
    try:main()
    except BaseException as ex:
        save('FAILURE',dict(error=repr(ex),traceback=traceback.format_exc()))
        raise
