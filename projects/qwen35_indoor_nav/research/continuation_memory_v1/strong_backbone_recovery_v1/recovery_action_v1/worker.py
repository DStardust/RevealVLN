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
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parent.parent)]
import common as u
from action_boundary_v2 import ActionBoundary, assistant_header
from replay import replay_token, forced_scores


def main(run, output, ids, identity_pairs):
    u.verify_sources(run); output.mkdir(parents=True, exist_ok=False)
    u.setup_imports(output)
    import numpy as np
    import torch
    import transformers
    import habitat
    import habitat_sim
    import streamvln_eval as official
    from recovery_model import RecoveryMemory
    from memory_v2 import ExecutionMemory
    from capture_runtime_v3 import DenseRuntime, tensor_hash
    from transfer_audit import audit_pair
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
    heads={'NATIVE':adapter,'ZERO_ADAPTER':adapter,'PRISTINE':adapter,'NONZERO_PROBE':adapter}
    header=assistant_header(tokenizer)
    for arm,record in u.read(run/'PROTOCOL.json')['heads'].items():
        assert u.sha(record['path'])==record['sha256'],'HEAD_FILE_CHANGED'
        saved=torch.load(record['path'],map_location='cpu',weights_only=False)
        assert saved['step']==u.read(run/'PROTOCOL.json')['steps'] and saved['arm']==arm
        head=RecoveryMemory(cfg.hidden_size,arm).cuda().eval();head.load_state_dict(saved['model']);head.requires_grad_(False)
        heads[arm]=head
    def head_hashes():
        return {arm:{name:tensor_hash(t) for name,t in head.state_dict().items()} for arm,head in heads.items()}
    initial_heads=head_hashes()
    def state_hash():
        digest=hashlib.sha256()
        for name,t in model.state_dict().items():
            digest.update(name.encode());digest.update(str(t.dtype).encode());digest.update(str(tuple(t.shape)).encode())
            raw=t.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw))
        return digest.hexdigest()
    frozen=state_hash()
    assert frozen==u.read(run/'PROTOCOL.json')['expected_base_state_sha256'],'LOADED_BASE_IDENTITY_CHANGED'
    identity=dict(model_loaded=True,base_state_sha256=frozen,source_commit=u.read(run/'PROTOCOL.json')['source_commit'],
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,habitat_sim=habitat_sim.__version__),
        imports={k:str(Path(m.__file__).resolve()) for k,m in [('torch',torch),('habitat',habitat),('official',official)]},
        action_token_ids=token_ids,hidden_size=cfg.hidden_size,attention='flash_attention_2',dtype='bfloat16',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(0),
        loading_info=loading,loaded_seconds=time.time()-started,base_updates=0)
    identity['heads']=u.read(run/'PROTOCOL.json')['heads']
    u.write(output/'RUNTIME_IDENTITY.json',identity)
    if identity_pairs and 'golden' in u.read(run/'PROTOCOL.json'):
        golden=u.read(run/'PROTOCOL.json')['golden']
        cached=torch.load(golden['feature_path'],map_location='cpu',weights_only=False)
        trace=u.read(golden['trace_path'])
        helper=official.VLNEvaluator(planned_config:=u.read(run/'DATA_MANIFEST.json')['episodes'][0]['config'],split='val_unseen',env_num=1,output_path=str(output),model=model,tokenizer=tokenizer,epoch=0,args=argparse.Namespace(save_video=False,num_frames=32,num_future_steps=4,num_history=8))
        dense=DenseRuntime(model,tokenizer,helper.image_processor,adapter,token_ids,cached['instruction'])
        parity=[]
        for step in golden['steps']:
            obs=trace['observations'][step]
            rgb=np.load(Path(golden['content'])/(obs['rgb_hash']+'.rgb.npy'),allow_pickle=False)
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==obs['rgb_hash']
            previous=None if step==0 else {'S':0,'F':1,'L':2,'R':3}[trace['actions'][step-1]]
            live=dense.feature(rgb,previous);old=cached['memory_features'][step]
            assert torch.isfinite(live).all() and live.shape==old.shape
            parity.append(dict(step=step,live_sha256=tensor_hash(live),cached_sha256=tensor_hash(old),max_abs_delta=float((live-old).abs().max()),rms_delta=float((live-old).square().mean().sqrt()),cached_rms=float(old.square().mean().sqrt())))
        u.write(output/'CACHE_LIVE_PARITY.json',dict(rows=parity,scope='Three frozen FIT current-observation encoder inputs, not a new navigation or training experiment',numerical_drift_recorded=True))
    class ProbeExecuted(Exception):pass
    active={}; real_env=official.Env; generate=model.generate
    def array_hash(x):
        x=np.ascontiguousarray(x);d=hashlib.sha256(str((x.shape,str(x.dtype))).encode());d.update(x.tobytes());return d.hexdigest()
    def emit(value):
        active['log'].write(json.dumps(value,allow_nan=False)+'\n');active['log'].flush()
    def observe(rgb,previous):
        if active['arm']=='PRISTINE':return
        begin=time.time();evidence=active['dense'].observe(rgb,previous);torch.cuda.synchronize()
        if active.get('capture'):active['dense_features'].append(active['dense'].last_feature.clone())
        emit(dict(event='memory_write',environment_step=active['steps'],seconds=time.time()-begin,**evidence))
    class LoggedEnv(real_env):
        def reset(self,*a,**kw):
            obs=super().reset(*a,**kw);active['steps']=0
            if active.get('capture'):assert array_hash(obs['rgb'])==active['replay']['rgb_sha256'][0],'REPLAY_START_MISMATCH'
            active['obs_hash']=array_hash(obs['rgb']);active['env']=self
            assert self.current_episode.instruction.instruction_text==active['instruction'],'INSTRUCTION_MISMATCH'
            active['dense']=DenseRuntime(model,tokenizer,active['helper'].image_processor,
                heads[active['arm']],token_ids,active['instruction'])
            emit(dict(event='reset',rgb_sha256=active['obs_hash'],instruction_sha256=hashlib.sha256(active['instruction'].encode()).hexdigest()))
            observe(obs['rgb'],None)
            return obs
        def step(self,action,*a,**kw):
            before=active['obs_hash']
            if active.get('replay') and (active.get('capture') or active['steps']<active['replay']['cutoff']):
                t=active['steps'];assert int(action)==active['replay']['actions'][t],'UNEXECUTED_TEACHER_ACTION'
                assert before==active['replay']['rgb_sha256'][t],'REPLAY_RGB_MISMATCH'
            obs=super().step(action,*a,**kw);active['steps']+=1
            active['obs_hash']=array_hash(obs['rgb']);metrics=self.get_metrics()
            emit(dict(event='action',step=active['steps'],executed_action=int(action),before_rgb_sha256=before,
                after_rgb_sha256=active['obs_hash'],distance=float(metrics['distance_to_goal']),
                collision=bool(metrics.get('collisions',{}).get('is_collision',False))))
            if active['arm']=='NONZERO_PROBE':
                assert int(action)==active['probe_action'],'NONZERO_NOT_CONSUMED_BY_ENV'
                assert active['decision']['native_token']!=active['decision']['method_token'],'NONZERO_DID_NOT_CHANGE_TOKEN'
                u.write(active['path']/'PROBE_EXECUTION.json',dict(status='NONZERO_ACTION_EXECUTED',executed_action=int(action),decision=active['decision'],environment_steps=active['steps'],diagnostic_only=True,not_navigation_sr=True))
                raise ProbeExecuted()
            # STOP supplies no new physical observation. Budget termination also
            # has no next actor call and needs no memory update.
            if int(action)!=0 and not self.episode_over:observe(obs['rgb'],int(action))
            return obs
    official.Env=LoggedEnv
    def observed_feature(module,args,result):
        active['last_feature']=result.last_hidden_state[:,-1].detach().float()
    hook=model.model.register_forward_hook(observed_feature)
    class Residual(transformers.LogitsProcessor):
        def __init__(self):self.boundary=ActionBoundary(header)
        def __call__(self,input_ids,scores):
            offset=self.boundary.offset(input_ids)
            if active.get('replay') and (active.get('capture') or active['steps']<active['replay']['cutoff']) and offset is not None and offset>0:
                return forced_scores(scores,replay_token(active['replay'],active['steps'],offset,token_ids,tokenizer.eos_token_id))
            if offset!=0:return scores
            active['feature']=active['last_feature']
            active['features'].append(active['feature'][0].cpu())
            active['pending_observation']=False
            assert scores.dtype==torch.float32,'ACTION_SCORE_DTYPE_CHANGED'
            assert torch.isfinite(scores[:,token_ids]).all(),'NONFINITE_NATIVE_ACTION_LOGITS'
            bias=active['dense'].delta(active['feature']).to(scores.dtype)
            revised=scores.clone()
            forced=active.get('replay') and (active.get('capture') or active['steps']<active['replay']['cutoff'])
            if forced:
                revised=forced_scores(scores,replay_token(active['replay'],active['steps'],0,token_ids,tokenizer.eos_token_id))
            if active['arm']=='NONZERO_PROBE':
                # Mechanical positive control only, never an efficacy arm.
                target=(int(scores[0,token_ids].argmax())+1)%4
                bias=torch.zeros_like(bias);bias[0,target]=scores.max()-scores[0,token_ids[target]]+1
                active['probe_action']=target
            if active['arm'] not in ('PRISTINE','NATIVE'):revised[:,token_ids]+=bias
            if active['arm'] in ('PRISTINE','NATIVE','ZERO_ADAPTER') and not forced:
                assert torch.equal(scores,revised),'ZERO_OR_NATIVE_CHANGED_SCORES'
            active['decision']=dict(native_logits=scores[0,token_ids].float().cpu().tolist(),
                method_logits=revised[0,token_ids].float().cpu().tolist(),score_dtype=str(scores.dtype),forced_prefix=bool(forced and not active.get('capture')),
                residual=bias[0].float().cpu().tolist(),native_token=int(scores.argmax(-1)[0]),
                method_token=int(revised.argmax(-1)[0]),action_boundary='AFTER_NATIVE_ASSISTANT_HEADER_V2',
                header_ids=header,action_generation_offset=len(header))
            return revised
    def logged_generate(*a,**kw):
        active['pending_observation']=True
        evidence={}
        for name in ['inputs','images','depths','poses','intrinsics']:
            t=kw[name];evidence[name]=dict(shape=list(t.shape),dtype=str(t.dtype),sha256=tensor_hash(t))
        evidence['time_ids']=kw['time_ids'];evidence['rgb_sha256']=active['obs_hash']
        kw['logits_processor']=transformers.LogitsProcessorList([Residual()])
        begin=time.time();value=generate(*a,**kw);torch.cuda.synchronize()
        assert not active['pending_observation'],'NO_REAL_ACTION_POSITION_FEATURE'
        assert value.sequences[0,:len(header)].tolist()==header,'HEADER_CHANGED'
        assert int(value.sequences[0,len(header)])==active['decision']['method_token'],'LOGGED_ACTION_TOKEN_NOT_GENERATED'
        emit(dict(event='generation',environment_step=active['steps'],input=evidence,
            generated_ids=value.sequences.detach().cpu().tolist(),seconds=time.time()-begin,
            feature_sha256=tensor_hash(active['feature']),**active['decision']))
        return value
    model.generate=logged_generate
    capture=u.read(run/'PROTOCOL.json').get('capture_only',False)
    planned={x['id']:x for x in u.read(run/'DATA_MANIFEST.json')['episodes']};completed=[]
    for index in ids:
        episode=planned[index];arms=(['PRISTINE','ZERO_ADAPTER','NONZERO_PROBE'] if index==ids[0] else ['PRISTINE','ZERO_ADAPTER']) if identity_pairs else ['NATIVE']+(list(u.read(run/'PROTOCOL.json')['heads'])[index%2:]+list(u.read(run/'PROTOCOL.json')['heads'])[:index%2]);group=output/'episodes'/str(index)
        if capture:arms=['NATIVE']
        group.mkdir(parents=True,exist_ok=False);results={};traces={}
        for arm in arms:
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            path=group/arm;path.mkdir();features=[]
            with (path/'TRACE.jsonl').open('x') as log:
                active.clear();active.update(log=log,arm=arm,features=features,pending_observation=False,instruction=episode['instruction'],path=path,capture=capture,dense_features=[],replay=u.read(episode['trajectory']) if capture else u.read(episode['prefix_path']) if 'prefix_path' in episode else None)
                args=argparse.Namespace(save_video=False,num_frames=32,num_future_steps=4,num_history=8)
                evaluator=official.VLNEvaluator(episode['config'],split='val_unseen',env_num=1,
                    output_path=str(path),model=model,tokenizer=tokenizer,epoch=0,args=args)
                active['helper']=evaluator
                try:
                    with torch.inference_mode():evaluator.eval_action(0)
                except ProbeExecuted:
                    active['env'].close()
            if arm=='NONZERO_PROBE':
                results[arm]=u.read(path/'PROBE_EXECUTION.json');traces[arm]=u.sha(path/'TRACE.jsonl');continue
            rows=[json.loads(x) for x in (path/'result.json').read_text().splitlines()]
            assert len(rows)==1 and str(rows[0]['episode_id'])==str(episode['episode_id'])
            assert rows[0]['steps']<=500
            results[arm]=rows[0];traces[arm]=u.sha(path/'TRACE.jsonl')
            torch.save(dict(features=torch.stack(features),episode_id=episode['episode_id'],arm=arm,
                scope='Causal features at upstream model-query times, not every environment step; no task-state labels'),path/'FEATURES.pt')
        if capture:
            trace=[json.loads(line) for line in (group/'NATIVE/TRACE.jsonl').read_text().splitlines()]
            generations=[x for x in trace if x['event']=='generation']
            actions={x['step']-1:x['executed_action'] for x in trace if x['event']=='action'}
            assert len(active['dense_features'])==len(actions),'OBSERVATION_ACTION_COUNT_MISMATCH'
            assert len(features)==len(generations)
            assert results['NATIVE']['success']==float(active['replay']['success']),'REPLAY_TERMINAL_MISMATCH'
            assert len(actions)==len(active['replay']['actions'])
            cache=group/'CACHE.pt'
            torch.save(dict(memory_features=torch.stack(active['dense_features']),actor_features=torch.stack(features),
                base_logits=torch.tensor([x['native_logits'] for x in generations]),
                query_steps=torch.tensor([x['environment_step'] for x in generations]),
                targets=torch.tensor([actions[x['environment_step']] for x in generations]),
                known=torch.tensor([episode['kind']=='PRESERVATION' or x['environment_step']>=active['replay']['cutoff'] for x in generations]),
                kind=episode['kind'],cutoff=active['replay']['cutoff'],
                episode_id=episode['episode_id'],house=episode['house'],partition=episode['partition'],
                action_boundary='AFTER_NATIVE_ASSISTANT_HEADER_V2',source_trace_sha256=traces['NATIVE'],
                source='ACTUAL_TEACHER_RECOVERY_OR_NATIVE_SUCCESS_REPLAY'),cache)
        reference='PRISTINE' if identity_pairs else 'NATIVE';audits={}
        for arm in [x for x in arms[1:] if x!='NONZERO_PROBE']:
            a=[json.loads(x) for x in (group/reference/'TRACE.jsonl').read_text().splitlines()]
            b=[json.loads(x) for x in (group/arm/'TRACE.jsonl').read_text().splitlines()]
            try:audits[arm]=audit_pair(a,b,results[reference],results[arm],zero=identity_pairs)
            except ValueError as error:
                u.write(group/'FIRST_DIVERGENCE.json',dict(error=str(error),reference=reference,arm=arm,
                    reference_trace=str(group/reference/'TRACE.jsonl'),method_trace=str(group/arm/'TRACE.jsonl')))
                raise
        u.write(group/'COMPLETE.json',dict(id=index,house=episode['house'],outcomes=results,trace_hashes=traces,
            cache=dict(path=str(cache),sha256=u.sha(cache)) if capture else None,audits=audits,zero_adapter_identity_checked=identity_pairs,runtime_identity_sha256=u.sha(output/'RUNTIME_IDENTITY.json')))
        completed.append(index);u.write(output/'PROGRESS.json',dict(status='RUNNING',complete=completed,planned=ids,model_loaded=True))
    assert state_hash()==frozen,'BASE_STATE_CHANGED'
    assert head_hashes()==initial_heads,'HEAD_STATE_CHANGED'
    u.write(output/'STATE_SEAL.json',dict(base_before=frozen,base_after=frozen,heads_unchanged=True,head_states=initial_heads))
    hook.remove()
    u.write(output/'RESULT.json',dict(status='COMPLETE',complete=completed,planned=ids,base_unchanged=True,
        base_updates=0,heads_unchanged=True,wall_seconds=time.time()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ids',required=True);p.add_argument('--identity-pairs',action='store_true');a=p.parse_args()
    main(a.run,a.output,[int(x) for x in a.ids.split(',')],a.identity_pairs)
