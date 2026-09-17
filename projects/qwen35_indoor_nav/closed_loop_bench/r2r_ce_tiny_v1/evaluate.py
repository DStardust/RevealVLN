"""Frozen ordinary-policy inference client. Simulator metadata never enters model input."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
OUT=HERE/'run_001'


def main():
    c.verify_lock();p=json.loads((HERE/'PROTOCOL.json').read_text())
    speed=c.LINE/'sft_acceptance/ordinary_speedup_10x_v1'
    sys.path.insert(0,str(speed/'official_einops_0_8_1/deps'))
    sys.path.insert(0,str(speed/'official_fla_0_5_2/deps'))
    import fla.ops.gated_delta_rule
    import torch
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
    assert state['cursor']['updates']==34800
    assert all(bool(torch.isfinite(x).all()) for x in state['trainable'].values())
    policy=model.build_policy(p['seed']);model.load_trainable(policy,state['trainable']);policy.eval()
    def fingerprint():
        h=hashlib.sha256()
        for name,param in policy.named_parameters():
            if param.requires_grad:
                h.update(name.encode());h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    initial_sha=fingerprint();del state;gc.collect()
    c.write(OUT/'MODEL_LOADED.json',dict(checkpoint_updates=34800,checkpoint_sha256=p['checkpoint_sha256'],
        trainable_sha256=initial_sha,parameter_updates=0,logical_device=0,physical_device=p['gpu'],
        device_name=torch.cuda.get_device_name(0),model_module_sha256=c.sha(c.TRAIN/'model.py'),
        model_eval=True,causal_encoder='unchanged DecisionDataset + make_collate',
        float_dtype='bfloat16 backbone; float32 action head'),True)
    collate=model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
    window=c.Window()
    class Store:
        def get(self,record_idx,t):
            assert record_idx==0 and t==0
            return window.item()
    dataset=model.DecisionDataset([dict(record_idx=0,t=0,target=0,weight=1.)],Store(),policy.processor)
    simenv=os.environ.copy();simenv.pop('CUDA_VISIBLE_DEVICES',None)
    parent,child=socket.socketpair();parent.settimeout(180)
    log=(OUT/'simulator.log').open('x')
    simproc=subprocess.Popen([str(c.LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'),'-I','-B',str(HERE/'executor.py'),str(child.fileno())],
        env=simenv,cwd=c.ROOT,pass_fds=(child.fileno(),),stdout=log,stderr=subprocess.STDOUT)
    child.close()
    c.append(OUT/'CHILD_PROCESSES.jsonl',dict(pid=simproc.pid,role='simulator'))
    stream=parent.makefile('rw');latencies=[];actions=0;completed=0;began=time.time()
    def call(obj):
        stream.write(json.dumps(obj)+'\n');stream.flush();line=stream.readline()
        if not line:raise RuntimeError('SIMULATOR_EOF')
        return json.loads(line)
    try:
        with torch.inference_mode():
            for index in range(p['episode_count']):
                response=call(dict(op='reset',index=index));assert window.receive(response)
                step=0
                while True:
                    tick=time.perf_counter();item=dataset[0];batch=collate([item])
                    batch.pop('targets');batch.pop('weights')
                    batch={k:v.to('cuda:0') for k,v in batch.items()}
                    logits=policy.forward_batch(**batch)
                    assert logits.shape==(1,4) and torch.isfinite(logits).all()
                    torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                    values=logits[0].float().cpu().tolist();action=c.ACTIONS[int(logits.argmax(-1).item())]
                    if actions==0:
                        repeat=policy.forward_batch(**batch);torch.cuda.synchronize()
                        assert torch.allclose(repeat,logits,atol=1e-5,rtol=1e-5),'REPEATED_INFERENCE_MISMATCH'
                        c.write(OUT/'FIRST_FORWARD_CHECK.json',dict(repeated_logits_match=True,
                            logits=values,images=len(window.images),executed_actions=len(window.executed),
                            output_action=action,token_count=int(batch['input_ids'].numel())),True)
                    latencies.append(elapsed);step+=1;actions+=1
                    c.append(OUT/'POLICY_STEPS.jsonl',dict(index=index,step=step,action=action,logits=values,inference_seconds=elapsed,
                        images=len(window.images),executed_history=len(window.executed)))
                    response=call(dict(op='action',action=action))
                    alive=window.receive(response,executed=action)
                    c.write(OUT/'PROGRESS.json',dict(status='EVALUATING',unix=time.time(),episode_index=index,
                        completed=completed,total=8,episode_step=step,total_actions=actions,checkpoint_updates=34800))
                    if not alive:break
                completed+=1
                print(json.dumps(dict(completed=completed,total=8,last_steps=step,total_actions=actions)),flush=True)
            assert call(dict(op='close'))=={'closed':True}
        simproc.wait(timeout=30);assert simproc.returncode==0
        final_sha=fingerprint();assert initial_sha==final_sha,'MODEL_CHANGED_DURING_EVALUATION'
        c.write(OUT/'INFERENCE_RESULT.json',dict(status='COMPLETE',completed=completed,total_actions=actions,
            trainable_unchanged=True,trainable_sha256=final_sha,latencies_seconds=latencies,
            wall_seconds=time.time()-began,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            optimizer_updates=0,scientific_gain_verified=False),True)
    except BaseException as exc:
        c.write(OUT/'INFERENCE_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),
                completed=completed,total_actions=actions),True)
        raise
    finally:
        stream.close();parent.close()
        if simproc.poll() is None:
            simproc.terminate()
            try:simproc.wait(timeout=10)
            except subprocess.TimeoutExpired:simproc.kill();simproc.wait(timeout=10)
        log.close()

if __name__=='__main__':main()
