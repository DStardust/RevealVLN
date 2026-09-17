"""Profile exact frozen cache, audit real gradient buckets, run bounded D1."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
RANK=int(os.environ['RANK'])

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

h=module('q35n_d1_helpers',OUT.parent/'d1_v1/worker.py')
h.OUT=OUT  # All helper artifact writes must target this new run.
c=module('q35n_frozen_cache',OUT.parent/'efficiency_v1/frozen_cache.py')
old=h.old

class CacheSwitch:
    def __init__(self,policy):
        self.policy=policy;self.processor=policy.processor;self.encoder=policy.mm.get_image_features
        self.cached_processor=c.CachedProcessor(self.processor)
        self.cached_encoder=c.CachedFrozenVision(policy.mm.visual,self.encoder)
    def use(self,enabled):
        self.policy.processor=self.cached_processor if enabled else self.processor
        self.policy.mm.get_image_features=self.cached_encoder if enabled else self.encoder
    def clear(self):
        self.cached_processor.cache.clear();self.cached_encoder.cache.clear()
    def stats(self):return dict(processor=self.cached_processor.cache.stats(),vision=self.cached_encoder.cache.stats())

def one_chunk(policy,record):
    policy.zero_grad(set_to_none=True)
    memory=old.zero_memory();losses=[];outputs=[]
    torch.cuda.synchronize();start=time.perf_counter()
    for t in range(4):
        memory,logits,_=old.policy_forward(policy,old.payload(*record,t),memory)
        outputs.append(logits.detach().cpu())
        target=old.ACTIONS.index(record[2][t])
        losses.append(torch.nn.functional.cross_entropy(logits,torch.tensor([target],device='cuda:0')))
    (torch.stack(losses).sum()/4).backward()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    gradient=h.gradient(policy)
    return dict(seconds=elapsed,logits=torch.cat(outputs),gradient=gradient)

def cache_profile(policy,records,switch):
    policy.eval();record=records[RANK]
    switch.use(False)
    one_chunk(policy,record)  # Warm CUDA kernels, charged but not steady-state timing.
    raw=[one_chunk(policy,record) for _ in range(3)]
    switch.use(True)
    cold=one_chunk(policy,record)
    # Processor and vision hits must preserve all tensor values and containers.
    pp=old.payload(*record,0)
    text=policy.processor.apply_chat_template([{'role':'user','content':[
        *[{'type':'image'} for _ in pp['images']],{'type':'text','text':pp['instruction']}]}],tokenize=False,add_generation_prompt=True)
    a=switch.processor(text=[text],images=pp['images'],return_tensors='pt')
    b=switch.cached_processor(text=[text],images=pp['images'],return_tensors='pt')
    assert set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
    b=b.to('cuda:0')
    with torch.no_grad():
        first=switch.cached_encoder(b['pixel_values'],b['image_grid_thw'],return_dict=True)
        second=switch.cached_encoder(b['pixel_values'],b['image_grid_thw'],return_dict=True)
    assert c.fingerprint(first)==c.fingerprint(second),'Cached feature content changed'
    hot=[one_chunk(policy,record) for _ in range(3)]
    noise_logits=max(float((x['logits']-raw[0]['logits']).abs().max()) for x in raw[1:])
    noise_grad=max(h.compare(raw[0]['gradient'],x['gradient'])['relative_L2'] for x in raw[1:])
    cache_logits=max(float((x['logits']-raw[0]['logits']).abs().max()) for x in [cold,*hot])
    cache_grad=max(h.compare(raw[0]['gradient'],x['gradient'])['relative_L2'] for x in [cold,*hot])
    raw_median=statistics.median(x['seconds'] for x in raw)
    hot_median=statistics.median(x['seconds'] for x in hot)
    correct=noise_grad<=.10 and cache_logits<=max(1e-5,3*noise_logits) and cache_grad<=max(.10,3*noise_grad)
    result=dict(raw_seconds=[x['seconds'] for x in raw],cold_seconds=cold['seconds'],hot_seconds=[x['seconds'] for x in hot],
        raw_median=raw_median,hot_median=hot_median,hot_speedup=raw_median/hot_median,
        same_card_logit_noise=noise_logits,same_card_gradient_noise=noise_grad,cache_logit_difference=cache_logits,
        cache_gradient_difference=cache_grad,correctness_pass=correct,local_cache_candidate=correct and hot_median<=raw_median*.95,
        tensor_cache_hit_exact=True,stats=switch.stats(),optimizer_updates=0)
    h.save('CACHE_PROFILE',result);h.event('cache_profile_complete',**result)
    votes=[None,None];dist.all_gather_object(votes,result['local_cache_candidate'])
    selected=all(votes);switch.use(selected)
    h.save('CACHE_SELECTION',dict(enabled=selected,rank_votes=votes,selection='engineering throughput only; no dev'))
    policy.zero_grad(set_to_none=True)
    return selected

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
    ddp=DDP(h.Chunk(policy),device_ids=[0],broadcast_buffers=False,find_unused_parameters=True)
    state=dict(capture=True,raw={},reduced={})
    ddp.register_comm_hook(state,audited_allreduce)
    loss,_,_=ddp(records[RANK],0,4,old.zero_memory());loss.backward()
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
    opt=h.optimizer(policy);h.normalize_clip(policy,4);opt.step();identity=h.rank_identity(policy)
    h.save('BUCKET_CORRECTNESS',dict(pass_gate=True,buckets=checks,actual_gradient_sha256=hashes,
        updated_parameter_sha256=identity,shared_diagnostic_updates=1,old_single_rank_update_equivalence_pass=False))
    old.load_trainable(policy,initial);policy.zero_grad(set_to_none=True)
    state['capture']=False;state['raw'].clear();state['reduced'].clear()
    del opt
    return ddp

def main():
    lock=json.loads((OUT/'LOCK.json').read_text())
    for name,digest in lock['code'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    torch.set_num_threads(4);torch.cuda.set_device(0)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=300))
    h.toy();h.event('NCCL_pass')
    subset=json.loads((OUT/'SUBSET.json').read_text());records=[old.load_record(r) for r in subset['rows']]
    h.event('model_load_start');policy=old.build();h.event('model_loaded')
    initial=torch.load(h.OLD/'checkpoints/initial.pt',map_location='cpu',weights_only=True)['trainable']
    old.load_trainable(policy,initial)
    switch=CacheSwitch(policy);selected=cache_profile(policy,records,switch)
    ddp=real_bucket_gate(policy,records,initial);h.event('real_bucket_correctness_pass')
    before=h.evaluate(policy,records,subset['rows'],'before')
    start=time.time();state=h.train(ddp,policy,records,subset['schedules'][RANK]);train_wall=time.time()-start
    after=h.evaluate(policy,records,subset['rows'],'after')
    with torch.no_grad():
        _,ref,_=old.policy_forward(policy,old.payload(*records[RANK],0),old.zero_memory())
        policy.action_head.bias.add_(1.)
        loaded=torch.load(OUT/f'rank{RANK}_terminal.pt',map_location='cpu',weights_only=False)
        old.load_trainable(policy,loaded['trainable'])
        _,actual,_=old.policy_forward(policy,old.payload(*records[RANK],0),old.zero_memory())
        error=float((ref-actual).abs().max());assert error<=1e-5
    h.save('RESULT',dict(DDP_bucket_correctness_pass=True,cache_enabled=selected,learnability_pass=after['learnability_pass'],
        before=before,after=after,terminal_updates=200,train_wall_seconds=train_wall,cache_stats=switch.stats(),
        reload_max_abs=error,scientific_pass=False,local_forward_tokens=policy.forward_tokens))
    h.event('worker_complete');dist.destroy_process_group()

if __name__=='__main__':
    try:main()
    except BaseException as ex:
        h.save('FAILURE',dict(error=repr(ex),traceback=traceback.format_exc()));raise
