"""Matched small-set weighted SFT; sealed policy and evaluation reused read-only."""
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
OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
RANK=int(os.environ['RANK'])
DEVICE='cuda:0'
spec=importlib.util.spec_from_file_location('sealed_d1_helpers',OUT.parent/'d1_v1/worker.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
h.OUT=OUT
old=h.old
optimizer=h.optimizer
normalize_clip=h.normalize_clip
params=h.params
rank_identity=h.rank_identity
event=h.event
class Chunk(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.weights = json.loads((OUT/'WEIGHTS.json').read_text())['values']

    def forward(self, record, start, end, memory):
        instruction, images, actions = record
        loss = []
        stats = []
        for t in range(start,end):
            memory, logits, details = old.policy_forward(self.policy,old.payload(instruction,images,actions,t),memory)
            target = old.ACTIONS.index(actions[t])
            ce = F.cross_entropy(logits,torch.tensor([target],device=DEVICE))
            loss.append(ce*self.weights[target])
            stats.append(dict(t=t,target=target,prediction=int(logits.detach().argmax(-1)),ce=float(ce.detach()),
                              weight=self.weights[target],weighted_ce=float(ce.detach())*self.weights[target],memory_rms=float(memory.detach().float().square().mean().sqrt()), **details))
        assert loss
        return torch.stack(loss).sum(), memory.detach(), stats

def audited_allreduce(state,bucket):
    buf=bucket.buffer();index=bucket.index()
    if state['capture']:state['raw'][index]=buf.detach().cpu().clone()
    future=dist.all_reduce(buf,async_op=True).get_future()
    def finish(fut):
        value=fut.value()[0].div_(dist.get_world_size())
        if state['capture']:state['reduced'][index]=value.detach().cpu().clone()
        return value
    return future.then(finish)

def real_bucket_gate(policy,records,initial):
    old.load_trainable(policy,initial);policy.zero_grad(set_to_none=True);policy.train()
    ddp=DDP(Chunk(policy),device_ids=[0],broadcast_buffers=False,find_unused_parameters=True)
    state=dict(capture=True,raw={},reduced={})
    ddp.register_comm_hook(state,audited_allreduce)
    loss,_,stats=ddp(records[RANK],0,4,old.zero_memory());loss.backward()
    gathered=[None,None];dist.all_gather_object(gathered,dict(raw=state['raw'],reduced=state['reduced']))
    checks=[]
    assert set(gathered[0]['raw'])==set(gathered[1]['raw'])
    for key in sorted(state['raw']):
        a=gathered[0]['raw'][key];b=gathered[1]['raw'][key]
        expected=(a.float()+b.float()).to(a.dtype).div_(2)
        actual=state['reduced'][key]
        error=float((expected.float()-actual.float()).abs().max())
        tolerance=2*torch.finfo(a.dtype).eps*max(1.,float(expected.float().abs().max()))
        assert torch.isfinite(actual).all() and error<=tolerance
        assert torch.equal(gathered[0]['reduced'][key],gathered[1]['reduced'][key])
        checks.append(dict(bucket=key,dtype=str(a.dtype),numel=a.numel(),max_abs_error=error,tolerance=tolerance))
    # Verify actual parameter gradients are identical, not merely the hook buffers.
    gradient=h.gradient(policy);digest=hashlib.sha256(gradient.numpy().tobytes()).hexdigest()
    hashes=[None,None];dist.all_gather_object(hashes,digest);assert hashes[0]==hashes[1]
    total_weight=torch.tensor(sum(s['weight'] for s in stats),device=DEVICE,dtype=torch.float64)
    dist.all_reduce(total_weight)
    opt=h.optimizer(policy);h.normalize_clip(policy,float(total_weight)/2);opt.step();identity=h.rank_identity(policy)
    h.save('BUCKET_CORRECTNESS',dict(pass_gate=True,buckets=checks,actual_gradient_sha256=hashes,
        updated_parameter_sha256=identity,shared_diagnostic_updates=1,old_single_rank_update_equivalence_pass=False))
    old.load_trainable(policy,initial);policy.zero_grad(set_to_none=True)
    state['capture']=False;state['raw'].clear();state['reduced'].clear()
    del opt
    return ddp

def train(ddp,policy,records,schedule):
    policy.train();opt=optimizer(policy);memory=old.zero_memory();total_tokens=0
    for update in range(200):
        opt.zero_grad(set_to_none=True);local_count=0;local_loss=0.;local_tokens=0;local_weight=0.;local_raw_loss=0.;step_rows=[]
        for j in range(4):
            item=schedule[update*4+j]
            if item['start']==0:memory=old.zero_memory()
            with ddp.no_sync() if j<3 else contextlib.nullcontext():
                loss,memory,stats=ddp(records[item['row']],item['start'],item['end'],memory)
                loss.backward()
            local_count+=len(stats);local_loss+=float(loss.detach());local_tokens+=sum(s['tokens'] for s in stats)
            local_weight+=sum(s['weight'] for s in stats);local_raw_loss+=sum(s['ce'] for s in stats)
            step_rows.extend(dict(**s,update=update+1,row=item['row'],epoch=item['epoch']) for s in stats)
        totals=torch.tensor([local_count,local_loss,local_tokens,local_weight,local_raw_loss],dtype=torch.float64,device=DEVICE)
        dist.all_reduce(totals)
        # DDP has divided the gradient sum by world_size=2.
        norm=normalize_clip(policy,float(totals[3])/2)
        assert norm>0
        opt.step()
        assert all(torch.isfinite(p).all() for p in params(policy))
        total_tokens+=int(totals[2])
        assert policy.forward_tokens<=5000000, 'PER_RANK_FORWARD_TOKEN_CAP'
        with (OUT/f'rank{RANK}_train_steps.jsonl').open('a') as f:
            for row in step_rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        event('update',update=update+1,CE=float(totals[4]/totals[0]),weighted_CE=float(totals[1]/totals[3]),global_weight_sum=float(totals[3]),decisions=int(totals[0]),grad_norm=norm,
              global_train_tokens=total_tokens,local_forward_tokens=policy.forward_tokens,peak_allocated=torch.cuda.max_memory_allocated())
    identity=rank_identity(policy)
    state=dict(trainable=old.trainable_state(policy),optimizer=opt.state_dict(),updates=200,schedule_cursor=800,
               weights=json.loads((OUT/'WEIGHTS.json').read_text()),memory=memory.cpu(),rank=RANK,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
               python_rng=random.getstate(),numpy_rng=np.random.get_state(),rank_parameter_sha256=identity,
               global_train_tokens=total_tokens,local_forward_tokens=policy.forward_tokens)
    path=OUT/f'rank{RANK}_terminal.pt';assert not path.exists();torch.save(state,path)
    return state

def main():
    lock=json.loads((OUT/'LOCK.json').read_text())
    for name,digest in lock['code'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((OUT/'WEIGHTS.json').read_bytes()).hexdigest()==lock['weights_sha256']
    assert hashlib.sha256((OUT/'SUBSET.json').read_bytes()).hexdigest()==lock['subset_sha256']
    torch.set_num_threads(4);torch.cuda.set_device(0)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=300))
    h.toy();event('NCCL_pass')
    subset=json.loads((OUT/'SUBSET.json').read_text());records=[old.load_record(r) for r in subset['rows']]
    event('model_load_start');policy=old.build();event('model_loaded')
    initial=torch.load(h.OLD/'checkpoints/initial.pt',map_location='cpu',weights_only=True)['trainable']
    ddp=real_bucket_gate(policy,records,initial);event('real_bucket_correctness_pass')
    before=h.evaluate(policy,records,subset['rows'],'before')
    start=time.time();state=train(ddp,policy,records,subset['schedules'][RANK]);train_wall=time.time()-start
    after=h.evaluate(policy,records,subset['rows'],'after')
    with torch.no_grad():
        _,ref,_=old.policy_forward(policy,old.payload(*records[RANK],0),old.zero_memory())
        policy.action_head.bias.add_(1.)
        loaded=torch.load(OUT/f'rank{RANK}_terminal.pt',map_location='cpu',weights_only=False)
        old.load_trainable(policy,loaded['trainable'])
        _,actual,_=old.policy_forward(policy,old.payload(*records[RANK],0),old.zero_memory())
        error=float((ref-actual).abs().max());assert error<=1e-5
    h.save('RESULT',dict(DDP_bucket_correctness_pass=True,learnability_pass=after['learnability_pass'],
        before=before,after=after,terminal_updates=200,train_wall_seconds=train_wall,
        reload_max_abs=error,scientific_pass=False,local_forward_tokens=policy.forward_tokens,
        weights=json.loads((OUT/'WEIGHTS.json').read_text()),cache_enabled=False))
    event('worker_complete');dist.destroy_process_group()

if __name__=='__main__':
    try:main()
    except BaseException as ex:
        h.save('FAILURE',dict(error=repr(ex),traceback=traceback.format_exc()));raise
