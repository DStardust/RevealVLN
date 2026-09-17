"""Fixed paired ordinary BC: three ranks, exact global batch96, accumulated microbatches."""
import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import time
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('paired_runtime',HERE/'runtime.py')
r=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(r)

def lr_at(step):
    if step<120:return 5e-5*(step+1)/120
    return 5e-5*.5*(1+math.cos(math.pi*(step-120)/(4000-120)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['control_recent2','treatment_prefix8'],required=True)
    args=parser.parse_args();protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    assert protocol['runtime_allowed'] and protocol['id']=='ORDINARY_HISTORY8_PAIRED_TRAIN_R1'
    for path,digest in protocol['code_sha256'].items():assert r.sha(Path(path))==digest,'SOURCE_CHANGED:'+path
    assert r.sha(HERE.parent/'ordinary_history8_paired_train_v1/SELECTED_SAMPLES.jsonl')==protocol['sample_index_sha256']
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    assert os.environ['HF_HUB_OFFLINE']==os.environ['TRANSFORMERS_OFFLINE']=='1'
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==3 and rank in range(3)
    out=HERE/args.arm/'run_001';assert out.is_dir()
    r.configure()
    import torch
    import datetime
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.cuda.set_per_process_memory_fraction(26*1024**3/torch.cuda.get_device_properties(rank).total_memory,rank)
    torch.distributed.init_process_group('nccl',init_method=Path(os.environ['Q35N_STORE']).as_uri(),rank=rank,world_size=world,
                                        timeout=datetime.timedelta(seconds=180))
    data=r.ordinary();rows,report=data.load_rows()
    assert report['training_index_sha256']==protocol['snapshot_sha256']
    selected=[json.loads(line) for line in (HERE.parent/'ordinary_history8_paired_train_v1/SELECTED_SAMPLES.jsonl').read_text().splitlines()]
    assert len(selected)==384000
    samples=[dict(record_idx=v['entry'][0],t=v['entry'][1],target=v['entry'][2],weight=v['entry'][3],est=v['entry'][4]) for v in selected]
    model,policy,source=r.initialize()
    torch.set_rng_state(source['torch_rng']);torch.cuda.set_rng_state(source['cuda_rng'],device)
    random.setstate(source['python_rng']);del source
    policy.train()
    trainable=[p for p in policy.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    assert not opt.state,'OPTIMIZER_MUST_RESET'
    start_fingerprint=r.fingerprint(policy)
    starts=[None]*world;torch.distributed.all_gather_object(starts,start_fingerprint)
    assert len(set(starts))==1,'INITIAL_RANK_MISMATCH'
    r.save(out/f'INITIAL_RANK{rank}.json',dict(unix=time.time(),rank=rank,source_sha256=r.BRIDGE_SHA,
        trainable_fingerprint=start_fingerprint,optimizer_reset=True,optimizer_steps=0,fp32_trainable_count=len(trainable)),True)
    micro=protocol['microbatch'];assert micro==4 and 32%micro==0
    store=r.stores(rows)[args.arm]
    dataset=model.DecisionDataset(samples,store,policy.processor)
    collate=model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
    packing=r.load('paired_packing',HERE/'batching_fixed4.py')
    step_batches=[packing.partition(samples[step*96+rank*32:step*96+(rank+1)*32],step*96+rank*32) for step in range(4000)]
    batches=(batch for groups in step_batches for batch in groups)
    prefetch=r.load('paired_cpu_prefetch',HERE/'prefetch.py')
    def batch_stream():
        torch.cuda.set_device(rank)  # Pin-memory allocator stays on this rank, not thread-default GPU0.
        for indices in batches:
            batch=collate([dataset[i] for i in indices])
            assert all(v.device.type=='cpu' for v in batch.values())
            yield {k:v.pin_memory() for k,v in batch.items()}
    # No data process is forked after the model is loaded. One bounded CPU producer.
    loader=prefetch.OrderedPrefetch(batch_stream,capacity=2,timeout=90,name=f'cpu-prefetch-rank{rank}')
    # Bound the actual longest padded batch before any policy update.
    longest=sorted(range(len(samples)),key=lambda i:packing.history8_est(samples[i]),reverse=True)[:4]
    probe=collate([dataset[i] for i in longest]);py=probe.pop('targets').to(device);pw=probe.pop('weights').to(device)
    assert probe['input_ids'].numel()<=6144,'ACTUAL_TOKEN_BOUND'
    before_probe=r.fingerprint(policy)
    pz=policy.forward_batch(**{k:v.to(device) for k,v in probe.items()})
    ploss=(torch.nn.functional.cross_entropy(pz,py,reduction='none')*pw).sum()/pw.sum();ploss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable),'WORST_BATCH_GRADIENT'
    policy.zero_grad(set_to_none=True)
    assert r.fingerprint(policy)==before_probe and not opt.state
    r.save(out/f'CAPACITY_RANK{rank}.json',dict(rank=rank,source_indices=[selected[i]['source_sample_id'] for i in longest],
        padded_tokens=probe['input_ids'].numel(),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        diagnostic_forward_decisions=4,diagnostic_backward_calls=1,optimizer_steps=0,parameters_unchanged=True),True)
    del probe,py,pw,pz,ploss;gc.collect();torch.cuda.empty_cache()
    torch.distributed.barrier()
    iterator=iter(loader)
    own=r.load('paired_child_identity',HERE.parent/'ordinary_prefix_history8_v1/owned_process_r1.py')
    r.save(out/f'WORKERS_RANK{rank}.json',dict(rank=rank,rank_pid=os.getpid(),workers=[],
        prefetch_thread=iterator.identity(),process_workers=0,queued_batches_max=2),True)
    loss_fn=r.load('paired_global_loss',HERE/'loss.py').local_micro_loss
    stop=[];signal.signal(signal.SIGTERM,lambda *_:stop.append('SIGTERM'));signal.signal(signal.SIGINT,lambda *_:stop.append('SIGINT'))
    deadline=float(os.environ['Q35N_DEADLINE'])
    began=time.monotonic();last_report=began;completed=0;latest=None;stop_code=0
    stats=torch.zeros(18,dtype=torch.float64,device=device)
    def save_checkpoint():
        path=out/('checkpoint_%09d.pt'%completed)
        assert not path.exists()
        value=dict(binding=dict(protocol_sha256=r.sha(HERE/'PROTOCOL.json'),sample_index_sha256=protocol['sample_index_sha256'],arm=args.arm),
            cursor=dict(epoch=1 if completed==4000 else 0,position=0 if completed==4000 else completed,updates=completed,decisions=completed*32),
            global_decisions=completed*96,charged_compute_decisions=completed*96,
            trainable=model.trainable_state(policy),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device),python_rng=random.getstate())
        temp=path.with_suffix('.pt.tmp')
        with temp.open('xb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
        temp.replace(path)
        r.save(Path(str(path)+'.json'),dict(sha256=r.sha(path),cursor=value['cursor'],global_decisions=completed*96,
            charged_compute_decisions=completed*96,arm=args.arm,unix=time.time()),True)
        return str(path)
    try:
        for step in range(4000):
            flag=torch.tensor(int(bool(stop) or time.time()>deadline),device=device)
            torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX)
            if int(flag):stop_code=1;break
            group=samples[step*96:(step+1)*96]
            global_weights=sum(float(torch.tensor(x['weight'],dtype=torch.float32)) for x in group)
            policy.zero_grad(set_to_none=True)
            offset=0
            for planned_micro in step_batches[step]:
                batch=next(iterator)
                actual_micro=len(planned_micro)
                expected=group[rank*32+offset:rank*32+offset+actual_micro]
                offset+=actual_micro
                assert batch['input_ids'].numel()<=6144,'ACTUAL_TOKEN_BOUND'
                assert batch['targets'].tolist()==[x['target'] for x in expected]
                assert torch.equal(batch['weights'],torch.tensor([x['weight'] for x in expected]))
                y=batch.pop('targets').to(device,non_blocking=True);w=batch.pop('weights').to(device,non_blocking=True)
                batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
                logits=policy.forward_batch(**batch)
                loss=loss_fn(logits,y,w,global_weights,world);loss.backward()
                with torch.no_grad():
                    ce=torch.nn.functional.cross_entropy(logits,y,reduction='none')
                    stats[0]+=(ce*w).sum().double();stats[1]+=w.sum().double()
                    bins=torch.bincount(y*4+logits.argmax(1),minlength=16).double()
                    stats[2:]+=bins
                del batch,logits,loss,ce,y,w
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable),'BAD_GRADIENT'
            model.sync_grads(policy,world)
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);assert float(norm)>0
            for group_opt in opt.param_groups:group_opt['lr']=lr_at(step)
            opt.step();opt.zero_grad(set_to_none=True);completed+=1
            assert torch.cuda.max_memory_reserved(device)<=26*1024**3,'RESERVED_MEMORY_BUDGET'
            if completed%20==0 or completed==1 or completed==4000:
                reduced=stats.clone();torch.distributed.all_reduce(reduced)
                now=time.monotonic();count=int(reduced[2:].sum())
                if rank==0:
                    matrix=reduced[2:].reshape(4,4).cpu().tolist();support=[sum(x) for x in matrix]
                    event=dict(status='TRAINING',unix=time.time(),arm=args.arm,updates=completed,total_updates=4000,
                        decisions=completed*96,planned_decisions=384000,microbatch_cap=micro,microbatch_sizes=[len(x) for x in step_batches[step]],accumulation=len(step_batches[step]),
                        metrics=dict(mean_ce=float(reduced[0]/reduced[1]),accuracy=sum(matrix[i][i] for i in range(4))/count,
                            action_recall=[matrix[i][i]/support[i] if support[i] else None for i in range(4)],
                            action_target_counts=support,confusion=matrix,lr=lr_at(step),grad_norm=float(norm)),
                        throughput=count/max(.001,now-last_report),session_throughput=completed*96/max(.001,now-began),
                        deadline_unix=deadline,latest_checkpoint=latest,navigation_gain_verified=False)
                    r.save(out/'PROGRESS.json',event)
                    with (out/'PROGRESS.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
                    print(json.dumps(dict(arm=args.arm,updates=completed,ce=event['metrics']['mean_ce'],throughput=event['throughput'])),flush=True)
                stats.zero_();last_report=now
            if completed in (200,1000,2000,3000,4000):
                if rank==0:latest=save_checkpoint()
                torch.distributed.barrier()
        if rank==0 and completed and completed not in (200,1000,2000,3000,4000):latest=save_checkpoint()
        # Stop and join this rank's bounded CPU producer thread.
        iterator.close()
        assert not iterator.is_alive()
        final_fingerprint=r.fingerprint(policy)
        hashes=[None]*world;torch.distributed.all_gather_object(hashes,final_fingerprint)
        assert len(set(hashes))==1,'FINAL_RANK_MISMATCH'
        r.save(out/f'FINAL_RANK{rank}.json',dict(rank=rank,updates=completed,decisions=completed*32,
            trainable_fingerprint=final_fingerprint,all_parameters_finite=all(torch.isfinite(p).all().item() for p in trainable),
            optimizer_steps=sorted({int(v['step']) for v in opt.state.values()}),
            data_workers_joined=True,prefetch_thread_joined=not iterator.is_alive(),peak_reserved_bytes=torch.cuda.max_memory_reserved(device)),True)
        if rank==0:
            r.save(out/'RESULT.json',dict(status='COMPLETE' if completed==4000 else 'STOPPED',unix=time.time(),
                arm=args.arm,updates=completed,global_decisions=completed*96,latest_checkpoint=latest,
                stop_code=stop_code,wall_seconds=time.monotonic()-began,navigation_gain_verified=False),True)
        torch.distributed.barrier()
    finally:
        iterator.close()
    gc.collect();torch.cuda.empty_cache()
    os._exit(0 if completed==4000 else 2)

if __name__=='__main__':main()
