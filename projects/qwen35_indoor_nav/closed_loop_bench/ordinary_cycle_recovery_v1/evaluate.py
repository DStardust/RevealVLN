"""Fixed ordinary policy, isolated lane windows, bounded batched inference only."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
import sys

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
OUT=HERE/'run_001'


def main():
    c.verify_lock();p=json.loads((HERE/'PROTOCOL.json').read_text())
    speed=c.LINE/'sft_acceptance/ordinary_speedup_10x_v1'
    sys.path.insert(0,str(speed/'official_einops_0_8_1/deps'));sys.path.insert(0,str(speed/'official_fla_0_5_2/deps'))
    import fla.ops.gated_delta_rule
    import torch
    from PIL import Image
    torch.cuda.set_device(0);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(25*1024**3/torch.cuda.get_device_properties(0).total_memory)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    bound=modeling.torch_chunk_gated_delta_rule
    cells=dict(zip(bound.__code__.co_freevars,(x.cell_contents for x in bound.__closure__)))
    assert getattr(cells.get('implementation'),'__module__','').startswith('fla.')
    model=c.load('frozen_policy',c.TRAIN/'model.py')
    checkpoint=Path(p['checkpoint']);assert c.sha(checkpoint)==p['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['binding']['protocol_sha256']==p['training_protocol_sha256']
    assert state['binding']['sample_index_sha256']==p['sample_index_sha256']
    assert state['cursor']['updates']==p['checkpoint_updates']
    assert all(bool(torch.isfinite(x).all()) for x in state['trainable'].values())
    policy=model.build_policy(p['seed']);model.load_trainable(policy,state['trainable']);policy.eval()
    def fingerprint():
        h=hashlib.sha256()
        for name,param in policy.named_parameters():
            if param.requires_grad:
                h.update(name.encode());h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    initial_sha=fingerprint();del state;gc.collect()
    c.write(OUT/'MODEL_LOADED.json',dict(checkpoint_updates=p['checkpoint_updates'],checkpoint_sha256=p['checkpoint_sha256'],
        trainable_sha256=initial_sha,parameter_updates=0,logical_device=0,physical_device=p['gpu'],
        device_name=torch.cuda.get_device_name(0),model_module_sha256=c.sha(c.TRAIN/'model.py'),model_eval=True),True)
    collate=model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
    def forward(items):
        batch=collate(items);batch.pop('targets');batch.pop('weights')
        batch={k:v.to('cuda:0') for k,v in batch.items()}
        output=policy.forward_batch(**batch);torch.cuda.synchronize()
        assert output.shape==(len(items),4) and torch.isfinite(output).all()
        return output
    # Before any new benchmark action: old exposed frames, synthetic interface
    # combinations only. No labels, logits or scores are used to modify weights.
    fixtures=json.loads((HERE/'PARITY_FIXTURES.json').read_text())
    class FixtureStore:
        def __init__(self,rows):self.rows=rows
        def get(self,idx,t):
            row=self.rows[idx]
            return dict(instruction=row['instruction'],executed=row['executed'],
                        images=[Image.open(x).convert('RGB') for x in row['images']])
    checks=[]
    with torch.inference_mode():
        for group in fixtures:
            dataset=model.DecisionDataset([dict(record_idx=i,t=0,target=0,weight=1.) for i in range(8)],FixtureStore(group),policy.processor)
            items=[dataset[i] for i in range(8)]
            tick=time.perf_counter();single=torch.cat([forward([x]) for x in items]);single_seconds=time.perf_counter()-tick
            tick=time.perf_counter();batch=forward(items);batch_seconds=time.perf_counter()-tick
            diff=(batch-single).float();relative=(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(single.float()).clamp_min(1e-12)).item()
            checks.append(dict(action_match=bool(torch.equal(batch.argmax(-1),single.argmax(-1))),
                max_abs=float(diff.abs().max()),relative_l2=relative,single_logits=single.float().cpu().tolist(),
                batch_logits=batch.float().cpu().tolist(),single_seconds=single_seconds,batch_seconds=batch_seconds))
    assert p['parity_pass_batch_size']==p['parity_fallback_batch_size']==1
    batch_size=1  # Matched comparison; diagnostic parity cannot change this.
    c.write(OUT/'BATCH_PARITY_GATE.json',dict(passed=c.parity_accept(checks),selected_batch_size=batch_size,checks=checks,comparison_batch_size_forced=1,
        fallback_predeclared=True,benchmark_actions_before_gate=0,parameter_updates=0),True)
    del dataset,items,single,batch,diff;gc.collect();torch.cuda.empty_cache()
    cycle=c.load('cycle_policy',HERE/'cycle_policy.py')
    guards=[cycle.CycleRecovery() for _ in range(p['lanes'])]
    windows=[c.Window() for _ in range(p['lanes'])]
    class Store:
        def get(self,idx,t):
            assert t==0 and 0<=idx<len(windows)
            return windows[idx].item()
    dataset=model.DecisionDataset([dict(record_idx=i,t=0,target=0,weight=1.) for i in range(p['lanes'])],Store(),policy.processor)
    schedules=c.schedules(p['episode_count'],p['lanes']);cursors=[0]*p['lanes'];episode_steps=[0]*p['lanes']
    simenv=os.environ.copy();simenv.pop('CUDA_VISIBLE_DEVICES',None)
    procs=[];streams=[];sockets=[];logs=[];completed=0;actions=0;batches=0;began=time.time()
    def call(lane,obj):
        stream=streams[lane];stream.write(json.dumps(obj)+'\n');stream.flush();line=stream.readline()
        if not line:raise RuntimeError('SIMULATOR_EOF_LANE_'+str(lane))
        return json.loads(line)
    def reset(lane):
        index=schedules[lane][cursors[lane]]
        assert windows[lane].receive(call(lane,dict(op='reset',index=index)))
        guards[lane].reset()
        episode_steps[lane]=0
    try:
        for lane in range(p['lanes']):
            folder=OUT/'lanes'/f'lane_{lane:02d}'
            parent,child=socket.socketpair();parent.settimeout(180)
            log=(folder/'simulator.log').open('x');logs.append(log)
            proc=subprocess.Popen([str(c.LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'),'-I','-B',str(HERE/'executor.py'),str(child.fileno()),str(lane)],
                env=simenv,cwd=c.ROOT,pass_fds=(child.fileno(),),stdout=log,stderr=subprocess.STDOUT)
            child.close();procs.append(proc);sockets.append(parent);streams.append(parent.makefile('rw'))
            c.append(OUT/'CHILD_PROCESSES.jsonl',dict(pid=proc.pid,lane=lane,role='simulator'))
        for lane in range(p['lanes']):reset(lane)
        with torch.inference_mode():
            while completed<p['episode_count']:
                active=[i for i in range(p['lanes']) if cursors[i]<len(schedules[i])]
                tick=time.perf_counter();encoded=[dataset[i] for i in active];outputs=[]
                for offset in range(0,len(encoded),batch_size):outputs.extend(forward(encoded[offset:offset+batch_size]).float().cpu().tolist())
                elapsed=time.perf_counter()-tick;batches+=1
                c.append(OUT/'INFERENCE_BATCHES.jsonl',dict(batch=batches,size=len(active),seconds=elapsed,selected_batch_size=batch_size))
                for lane,values in zip(active,outputs):
                    window=windows[lane];tick_cycle=time.perf_counter()
                    action,cycle_info=guards[lane].choose(window.instruction,window.images,window.executed,values)
                    cycle_info["cycle_seconds"]=time.perf_counter()-tick_cycle
                    index=schedules[lane][cursors[lane]];episode_steps[lane]+=1;actions+=1
                    c.append(OUT/'lanes'/f'lane_{lane:02d}'/'POLICY_STEPS.jsonl',dict(index=index,step=episode_steps[lane],
                        action=action,logits=values,inference_batch=batches,images=len(windows[lane].images),executed_history=len(windows[lane].executed),**cycle_info))
                    alive=windows[lane].receive(call(lane,dict(op='action',action=action)),executed=action)
                    if not alive:
                        completed+=1;cursors[lane]+=1
                        print(json.dumps(dict(completed=completed,total=p['episode_count'],lane=lane,index=index,steps=episode_steps[lane],total_actions=actions)),flush=True)
                        if cursors[lane]<len(schedules[lane]):reset(lane)
                        else:assert call(lane,dict(op='close'))==dict(closed=True)
                c.write(OUT/'PROGRESS.json',dict(status='EVALUATING',unix=time.time(),completed=completed,total=p['episode_count'],
                    total_actions=actions,checkpoint_updates=p['checkpoint_updates'],selected_batch_size=batch_size,
                    wall_seconds=time.time()-began,lanes=[dict(lane=i,completed=cursors[i],total=len(schedules[i]),
                    index=schedules[i][cursors[i]] if cursors[i]<len(schedules[i]) else None,step=episode_steps[i]) for i in range(p['lanes'])]))
        for proc in procs:proc.wait(timeout=30);assert proc.returncode==0
        final_sha=fingerprint();assert initial_sha==final_sha,'MODEL_CHANGED_DURING_EVALUATION'
        c.write(OUT/'INFERENCE_RESULT.json',dict(status='COMPLETE',completed=completed,total_actions=actions,
            trainable_unchanged=True,trainable_sha256=final_sha,selected_batch_size=batch_size,
            wall_seconds=time.time()-began,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            optimizer_updates=0,scientific_gain_verified=False),True)
        c.write(OUT/'PROGRESS.json',dict(status='COMPLETE_PENDING_AUDIT',unix=time.time(),completed=completed,total=p['episode_count'],
            total_actions=actions,checkpoint_updates=p['checkpoint_updates'],selected_batch_size=batch_size))
    except BaseException as exc:
        c.write(OUT/'INFERENCE_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed=completed,total_actions=actions),True)
        raise
    finally:
        for proc in procs:
            if proc.poll() is None:proc.terminate()
        for proc in procs:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        for stream in streams:stream.close()
        for sock in sockets:sock.close()
        for log in logs:log.close()


if __name__=='__main__':main()

