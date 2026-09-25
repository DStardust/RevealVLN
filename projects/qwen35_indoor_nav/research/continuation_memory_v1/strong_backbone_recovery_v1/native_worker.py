"""Execute the public evaluator, with read-only evidence and a zero-residual check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as u


def main(run, output, ids, identity_pairs):
    u.verify_sources(run); output.mkdir(parents=True, exist_ok=False)
    u.setup_imports(output)
    import numpy as np
    import torch
    import transformers
    import habitat
    import habitat_sim
    import streamvln_eval as official
    from memory import ExecutionMemory
    started=time.time(); torch.set_num_threads(4)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tokenizer=transformers.AutoTokenizer.from_pretrained(str(u.MODEL),model_max_length=4096,padding_side='right',local_files_only=True)
    action_tokens=[tokenizer.encode(s,add_special_tokens=False) for s in ['STOP','↑','←','→']]
    assert all(len(x)==1 for x in action_tokens), ('MULTITOKEN_ACTION',action_tokens)
    token_ids=[x[0] for x in action_tokens]
    cfg=transformers.AutoConfig.from_pretrained(str(u.MODEL),local_files_only=True)
    cfg.mm_vision_tower=str(u.MODEL/'siglip-so400m-patch14-384');cfg.vision_tower=cfg.mm_vision_tower
    model,loading=official.StreamVLNForCausalLM.from_pretrained(str(u.MODEL),config=cfg,
        attn_implementation='flash_attention_2',torch_dtype=torch.bfloat16,low_cpu_mem_usage=False,
        local_files_only=True,output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),loading
    model.model.num_history=8;model.requires_grad_(False);model.to(0);model.eval();model.reset(1)
    adapter=ExecutionMemory(cfg.hidden_size).cuda().eval()
    def state_hash():
        digest=hashlib.sha256()
        for name,t in model.state_dict().items():
            digest.update(name.encode());digest.update(str(t.dtype).encode());digest.update(str(tuple(t.shape)).encode())
            raw=t.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw))
        return digest.hexdigest()
    frozen=state_hash()
    identity=dict(model_loaded=True,base_state_sha256=frozen,source_commit=u.read(run/'PROTOCOL.json')['source_commit'],
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,habitat_sim=habitat_sim.__version__),
        imports={k:str(Path(m.__file__).resolve()) for k,m in [('torch',torch),('habitat',habitat),('official',official)]},
        action_token_ids=token_ids,hidden_size=cfg.hidden_size,attention='flash_attention_2',dtype='bfloat16',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(0),
        loading_info=loading,loaded_seconds=time.time()-started,base_updates=0)
    u.write(output/'RUNTIME_IDENTITY.json',identity)
    active={}; real_env=official.Env; generate=model.generate
    def array_hash(x):
        x=np.ascontiguousarray(x);d=hashlib.sha256(str((x.shape,str(x.dtype))).encode());d.update(x.tobytes());return d.hexdigest()
    def emit(value):
        active['log'].write(json.dumps(value,allow_nan=False)+'\n');active['log'].flush()
    class LoggedEnv(real_env):
        def reset(self,*a,**kw):
            obs=super().reset(*a,**kw);active['steps']=0
            active['obs_hash']=array_hash(obs['rgb']);active['env']=self
            emit(dict(event='reset',rgb_sha256=active['obs_hash']))
            return obs
        def step(self,action,*a,**kw):
            before=active['obs_hash'];obs=super().step(action,*a,**kw);active['steps']+=1
            active['obs_hash']=array_hash(obs['rgb']);metrics=self.get_metrics()
            emit(dict(event='action',step=active['steps'],executed_action=int(action),before_rgb_sha256=before,
                after_rgb_sha256=active['obs_hash'],distance=float(metrics['distance_to_goal']),
                collision=bool(metrics.get('collisions',{}).get('is_collision',False))))
            return obs
    official.Env=LoggedEnv
    def observed_feature(module,args,result):
        if active.get('pending_observation'):
            feature=result.last_hidden_state[:,-1].detach().float()
            active['feature']=feature;active['memory']=adapter.update(feature,active['memory'])
            active['features'].append(feature[0].cpu())
            active['pending_observation']=False
    hook=model.model.register_forward_hook(observed_feature)
    class ZeroResidual(transformers.LogitsProcessor):
        def __call__(self,input_ids,scores):
            feature=active['feature'];bias=adapter.action_delta(feature,active['memory']).to(scores.dtype)
            assert torch.isfinite(scores[:,token_ids]).all(), 'NONFINITE_NATIVE_ACTION_LOGITS'
            revised=scores.clone();revised[:,token_ids]+=bias
            assert torch.equal(scores,revised),'ZERO_ADAPTER_CHANGED_SCORES'
            return revised
    def logged_generate(*a,**kw):
        active['pending_observation']=True
        evidence={}
        for name in ['inputs','images']:
            t=kw[name].detach().contiguous().cpu()
            evidence[name]=dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())
        if active['arm']=='ZERO_ADAPTER':kw['logits_processor']=transformers.LogitsProcessorList([ZeroResidual()])
        begin=time.time();value=generate(*a,**kw);torch.cuda.synchronize()
        assert not active['pending_observation'],'NO_REAL_OBSERVATION_FEATURE'
        emit(dict(event='generation',environment_step=active['steps'],input=evidence,
            generated_ids=value.sequences.detach().cpu().tolist(),seconds=time.time()-begin,
            feature_sha256=hashlib.sha256(active['feature'].cpu().numpy().tobytes()).hexdigest()))
        return value
    model.generate=logged_generate
    planned={x['id']:x for x in u.read(run/'DATA_MANIFEST.json')['episodes']};completed=[]
    for index in ids:
        episode=planned[index];arms=['NATIVE','ZERO_ADAPTER'] if identity_pairs else ['NATIVE'];group=output/'episodes'/str(index)
        group.mkdir(parents=True,exist_ok=False);results={};traces={}
        for arm in arms:
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            path=group/arm;path.mkdir();features=[]
            with (path/'TRACE.jsonl').open('x') as log:
                active.clear();active.update(log=log,arm=arm,memory=adapter.reset(),features=features,pending_observation=False)
                args=argparse.Namespace(save_video=False,num_frames=32,num_future_steps=4,num_history=8)
                evaluator=official.VLNEvaluator(episode['config'],split='val_unseen',env_num=1,
                    output_path=str(path),model=model,tokenizer=tokenizer,epoch=0,args=args)
                evaluator.eval_action(0)
            rows=[json.loads(x) for x in (path/'result.json').read_text().splitlines()]
            assert len(rows)==1 and str(rows[0]['episode_id'])==str(episode['episode_id'])
            assert rows[0]['steps']<=500
            results[arm]=rows[0];traces[arm]=u.sha(path/'TRACE.jsonl')
            torch.save(dict(features=torch.stack(features),episode_id=episode['episode_id'],arm=arm,
                scope='Causal features at upstream model-query times, not every environment step; no task-state labels'),path/'FEATURES.pt')
        if identity_pairs:
            a=[json.loads(x) for x in (group/'NATIVE/TRACE.jsonl').read_text().splitlines()]
            b=[json.loads(x) for x in (group/'ZERO_ADAPTER/TRACE.jsonl').read_text().splitlines()]
            for values in (a,b):
                for value in values:value.pop('seconds',None)
            assert a==b,'ZERO_ADAPTER_PREFIX_OR_ACTION_MISMATCH'
            assert results['NATIVE']==results['ZERO_ADAPTER'],'ZERO_ADAPTER_TERMINAL_MISMATCH'
        u.write(group/'COMPLETE.json',dict(id=index,house=episode['house'],outcomes=results,trace_hashes=traces,
            zero_adapter_identity_checked=identity_pairs,runtime_identity_sha256=u.sha(output/'RUNTIME_IDENTITY.json')))
        completed.append(index);u.write(output/'PROGRESS.json',dict(status='RUNNING',complete=completed,planned=ids,model_loaded=True))
    assert state_hash()==frozen,'BASE_STATE_CHANGED'
    hook.remove()
    u.write(output/'RESULT.json',dict(status='COMPLETE',complete=completed,planned=ids,base_unchanged=True,
        base_updates=0,wall_seconds=time.time()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ids',required=True);p.add_argument('--identity-pairs',action='store_true');a=p.parse_args()
    main(a.run,a.output,[int(x) for x in a.ids.split(',')],a.identity_pairs)
