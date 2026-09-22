"""Frozen six-model complete groups, real autonomous forwards and atomic seals."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from select_action import select

def prefix_audit(left,right):
    result=dict(input_prefix_matched=True,action_prefix_matched=True,logits_bitwise_equal=True,max_logit_delta=0.,argmax_flip_count=0,decisions=0,first_method_action_difference=None)
    for a,b in zip(left,right):
        row=c.prefix_compare(a,b);result['decisions']+=1
        result['logits_bitwise_equal'] &= row['logits_bitwise_equal'];result['max_logit_delta']=max(result['max_logit_delta'],row['max_logit_delta']);result['argmax_flip_count']+=row['argmax_flip_count']
        if not row['input_prefix_matched'] or row['argmax_flip_count']:
            return dict(result,input_prefix_matched=row['input_prefix_matched'],action_prefix_matched=not row['argmax_flip_count'],first_divergence=dict(left=a,right=b))
        if a['executed_action']!=b['executed_action']:return dict(result,first_method_action_difference=a['decision'])
    if len(left)!=len(right):return dict(result,action_prefix_matched=False,first_divergence='UNMATCHED_TERMINATION')
    return result

def registry_value(raw_families, config):
    families=sorted([f for f in raw_families if f['split']=='DEV'],key=lambda f:(f['house'],f['family_id']))
    if len(families)!=16:raise ValueError('REGISTERED_DEV_VARIANTS')
    conditions=[]
    for f in families:
        for history in ('seen','seen_sham','missing','missing_sham'):
            for task in ('task_A','task_T'):
                conditions.append(dict(family_id=f['family_id'],house=f['house'],history_id=history,task_id=task,
                    endpoint='main' if task=='task_A' else 'control',parent_family_id=f['parent_family_id'],stratum=f['stratum'],available=True))
    models=[f'{a}_{s}' for s in config['seeds'] for a in config['arms']];slots=[]
    for i,condition in enumerate(conditions):
        offset=i%len(models)
        for model in models[offset:]+models[:offset]:
            slots.append(dict(rank=len(slots),condition=i,model=model,arm=model.split('_')[0],seed=int(model.split('_')[1]),planned=True))
    n=len(conditions)//2*len(config['seeds'])
    return dict(scope='EXPOSED_DEV_TEACHER_ALIGNMENT_V1',conditions=conditions,models=models,slots=slots,
                expected_prefix_comparisons=len(conditions)*len(config['seeds']),models_per_condition=len(models),
                main_denominator_per_arm=n,control_denominator_per_arm=n)


def registry(run):
    data=read(run/'DATA.json');config=read(run/'PROTOCOL.json')
    binding=read(run/'DATA_AUDIT.json')
    if sha(run/'DATA.json')!=binding['data_sha256'] or sha(run/'EVALUATION_REGISTRY.json')!=binding['registry_sha256']:
        raise ValueError('REGISTERED_DATA_IDENTITY_CHANGED')
    value=registry_value(data['raw_families'],config)
    if read(run/'EVALUATION_REGISTRY.json')!=value:raise ValueError('REGISTRY_CHANGED')
    return value

def admitted(run,reg):
    groups={}
    for session in sorted((run/'evaluate').glob('session_*')):
        for path in session.glob('GROUP_*.json'):
            row=read(path);condition=row['condition']
            sealpath=session/f'STATE_SEAL_{condition:03d}.json'
            if not sealpath.exists():continue
            seal=read(sealpath)
            if not seal['base_unchanged'] or not seal['heads_unchanged']:raise ValueError('GROUP_PARAMETER_STATE_CHANGED')
            expected=[r['rank'] for r in reg['slots'] if r['condition']==condition]
            if row['ranks']!=expected or not set(expected)<=set(seal['completed_ranks']):raise ValueError('UNSEALED_OR_INCOMPLETE_GROUP')
            if condition in groups:raise ValueError('DUPLICATE_COMPLETE_GROUP')
            for key,digest_ in row['files'].items():
                if sha(session/key)!=digest_:raise ValueError('GROUP_CONTENT_CHANGED')
            groups[condition]=session
    return groups

def main(run):
    import torch
    import encoder
    config=runtime_config(run);verify_lock(read(run/'SOURCE_LOCK.json'));reg=registry(run)
    out=run/'evaluate';out.mkdir(exist_ok=True);done=admitted(run,reg)
    assigned=set(read(Path(os.environ['B2_DEVICE']))['conditions'])
    remaining=[r for r in reg['slots'] if r['condition'] in assigned and r['condition'] not in done and reg['conditions'][r['condition']]['available']]
    if not remaining:return
    prefix='session_gpu'+os.environ['B2_GPU']+'_'
    session=out/f'{prefix}{len(list(out.glob(prefix+"*")))+1:03d}';session.mkdir()
    p=dict(config,run=str(run),ranks=[r['rank'] for r in remaining],source_hashes=read(run/'SOURCE_LOCK.json')['files'],v16_protocol_sha256=sha(run/'PROTOCOL.json'))
    write(session/'CONFIG.json',p,True);policy=None;heads={};initial=None;head_initial={};completed=[];proc=None;stream=None;forward=None;log=None
    try:
        model,policy,initial=encoder.load_policy(session,p)
        if initial['sha256']!=read(run/'features/FEATURE_RESULT.json')['base_state_sha256']:raise ValueError('RUNTIME_BASE_DIFFERS_FROM_CACHE')
        for parameter in policy.parameters():parameter.requires_grad_(False)
        for tag in reg['models']:
            result=read(run/'train'/tag/'RESULT.json');path=run/'train'/tag/'FINAL.pt'
            if result['updates']!=config['steps'] or sha(path)!=result['checkpoint_sha256']:raise ValueError('FINAL_CHECKPOINT_IDENTITY')
            net=make_head(tag);net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            if c.model_identity(net)['sha256']!=result['final']:raise ValueError('LOADED_HEAD_IDENTITY')
            heads[tag]=net.cuda().eval()
            for parameter in heads[tag].parameters():parameter.requires_grad_(False)
        head_initial={tag:c.model_identity(net)['sha256'] for tag,net in heads.items()}
        data=read(run/'DATA.json');golden=read(run/'WARMUP_DATA.json');families={f['family_id']:f for f in data['raw_families']}
        warmforward=encoder.Forward(model,policy,encoder.RawStore(golden),[dict(record_idx=i,t=0,target=0,weight=1.) for i in range(len(golden['features']))])
        with torch.inference_mode():warm=encoder.warmup(warmforward,golden,config)
        warmforward.close();write(session/'WARMUP.json',warm,True)
        window=c.Window()
        class Store:
            def get(self,index,t):return window.item()
        forward=encoder.Forward(model,policy,Store(),[dict(record_idx=0,t=0,target=0,weight=1.)])
        cache=torch.load(run/'features/FEATURES.pt',map_location='cpu',weights_only=True)
        lookup={(r['instruction'],tuple(x[7:] for x in r['rgb_refs']),tuple(r['executed'])):i for i,r in enumerate(golden['features'])}
        old_inputs={}
        for f in (run/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
            if not (f.parent/'STATE_SEAL.json').exists():continue
            for row in c.records(f):old_inputs[row['index']]=row['processed']
        sock,child=socket.socketpair();sock.settimeout(240);env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
        log=(session/'simulator.log').open('x')
        sim_python=Path(config.get('sim_python',DATA_LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'))
        proc=subprocess.Popen([str(sim_python),'-I','-B',str(HERE/'continuation_service.py'),str(child.fileno()),str(session)],
            pass_fds=(child.fileno(),),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        child.close();stream=sock.makefile('rw')
        def call(message):
            stream.write(json.dumps(message)+'\n');stream.flush();line=stream.readline()
            if not line:raise EOFError('SIMULATOR_EOF')
            return json.loads(line)
        write(session/'METHOD_IDENTITY.json',dict(base_state_sha256=initial['sha256'],head_states=head_initial,registry_sha256=sha(run/'EVALUATION_REGISTRY.json'),native_stop_guard=False,query_in_actor=False),True)
        current=None;prefix_features=[];prefix_raw=[];prefix_processed=[];contrast={};forwards=0;parity=[];session_began=time.monotonic()
        with torch.inference_mode():
            for slot in remaining:
                rank=slot['rank'];condition=reg['conditions'][slot['condition']];family=families[condition['family_id']]
                history=family['histories'][condition['history_id']];tag=slot['model'];net=heads[tag]
                folder=session/'rollouts'/f'{rank:04d}';folder.mkdir(parents=True)
                new=current!=slot['condition']
                if new:current=slot['condition'];prefix_features=[];prefix_raw=[];prefix_processed=[];contrast={}
                if not window.receive(call(dict(op='reset',rank=rank))):raise ValueError('INITIAL_ENVIRONMENT')
                for t,action in enumerate(history):
                    raw=c.raw_input(window)
                    if new:
                        feature,logits,processed,timing=forward(0);forwards+=1;prefix_features.append(feature);prefix_raw.append(raw);prefix_processed.append(processed)
                        append(folder/'PREFILL.jsonl',dict(step=t,raw=raw,processed=processed,feature=c.tensor_identity(feature),**timing))
                    else:
                        _,processed,seconds=forward.prepare(0)
                        if raw!=prefix_raw[t] or processed!=prefix_processed[t]:raise ValueError('FORCED_PREFIX_INPUT_MISMATCH')
                        append(folder/'PREFILL.jsonl',dict(step=t,raw=raw,processed=processed,shared_feature=True,preprocess_seconds=seconds))
                    executed=c.ACTIONS['FLR'.index(action)]
                    if not window.receive(call(dict(op='action',action=executed)),executed=executed):raise ValueError('PREFIX_EARLY_DONE')
                memory=prefill(net,prefix_features)
                write(folder/'PREFILL_AUDIT.json',dict(prefix_steps=len(history),matched=True,feature_origin_rank=remaining[[r['rank'] for r in remaining].index(rank)-len(contrast)]['rank'],memory_before_cutoff=state_identity(memory)),True)
                rows=[];alive=True
                references={other:True for other in contrast if other.split('_')[1]==tag.split('_')[1] and other.split('_')[0]!=tag.split('_')[0]}
                while alive:
                    raw=c.raw_input(window);feature,logits,processed,timing=forward(0);forwards+=1
                    tick=time.perf_counter();before=memory;method,memory,detail=net.step(feature,logits,memory)
                    native=logits[0].cpu().tolist();values=method[0].cpu().tolist();selection=select(native,values)
                    row=dict(decision=len(history)+len(rows),raw=raw,processed=processed,logits=native,method_logits=values,
                             controller_seconds=time.perf_counter()-tick,predicted_state=detail['state'][0].cpu().tolist(),event_probabilities=detail['event_logits'][0].sigmoid().cpu().tolist(),revision=float(detail['revision'][0]),**selection,**timing)
                    idx=lookup.get((raw['instruction'],tuple(raw['rgb_sha256']),tuple(raw['executed_history'])))
                    if idx is not None:
                        if old_inputs[idx]!=processed:raise ValueError('CACHE_LIVE_INPUT_MISMATCH')
                        cached=cache['features'][idx].unsqueeze(0).cuda();cl=cache['logits'][idx].unsqueeze(0).cuda()
                        oldmethod,cm,_=net.step(cached,cl,before)
                        par=dict(rank=rank,decision=row['decision'],index=idx,processed_equal=True,feature_delta=float((cached-feature).abs().max()),native_delta=float((cl-logits).abs().max()),method_delta=float((oldmethod-method).abs().max()),
                                 native_flip=int(cl.argmax()!=logits.argmax()),method_flip=int(oldmethod.argmax()!=method.argmax()),native_margin=selection['native_margin'],method_margin=selection['method_margin'])
                        append(session/'CACHE_LIVE_PARITY.jsonl',par);parity.append(par)
                    for other,matched in references.items():
                        if not matched:continue
                        audit=prefix_audit(contrast[other][:len(rows)+1],rows+[row])
                        if 'first_divergence' in audit:
                            write(session/'FIRST_DIVERGENCE.json',dict(rank=rank,left=other,right=tag,**audit),True)
                            raise ValueError('INPUT_OR_NATIVE_ACTION_PREFIX_MISMATCH')
                        references[other]=audit['first_method_action_difference'] is None
                    append(folder/'POLICY_STEPS.jsonl',row);rows.append(row)
                    alive=window.receive(call(dict(op='action',action=row['executed_action'])),executed=row['executed_action'])
                    if len(rows)%20==0:write(run/('EVALUATION_PROGRESS_'+os.environ['B2_GPU']+'.json'),dict(rank=rank,condition=current,model=tag,step=row['decision'],completed_slots=len(done)*len(reg['models'])+len(completed),forwards=forwards,unix=time.time()))
                contrast[tag]=rows;task=read(folder/'TASK_RESULT.json')
                if task['autonomous_decisions']!=len(rows):raise ValueError('AUTONOMOUS_DECISION_COUNT')
                write(folder/'ROLLOUT.json',dict(slot=slot,task_result=task,policy_sha256=sha(folder/'POLICY_STEPS.jsonl'),trace_sha256=sha(folder/'TRACE_PRIVILEGED.json')),True)
                completed.append(rank);print(rank,tag,task['safe_v16_label'],flush=True)
                if len(contrast)==len(reg['models']):
                    audits=[]
                    for seed in config['seeds']:
                        for arm in ('ORIGINAL',):
                            left=f'{arm}_{seed}';right=f'ALIGNED_{seed}';audit=prefix_audit(contrast[left],contrast[right])
                            if 'first_divergence' in audit:raise ValueError('FINAL_GROUP_PREFIX_AUDIT')
                            audits.append(dict(condition=current,left=left,right=right,**audit))
                    ranks=[r['rank'] for r in remaining if r['condition']==current]
                    files={str(p.relative_to(session)):sha(p) for r in ranks for p in (session/'rollouts'/f'{r:04d}').glob('*.json*')}
                    write(session/f'GROUP_{current:03d}.json',dict(condition=current,ranks=ranks,audits=audits,files=files),True)
                    base_end=c.model_identity(policy)['sha256'];head_end={tag:c.model_identity(net)['sha256'] for tag,net in heads.items()}
                    if base_end!=initial['sha256'] or head_end!=head_initial:raise ValueError('GROUP_MODEL_STATE_CHANGED')
                    write(session/f'STATE_SEAL_{current:03d}.json',dict(base_unchanged=True,heads_unchanged=True,base_state_sha256=base_end,head_states=head_end,completed_ranks=ranks),True)
                    if time.monotonic()-session_began>config['max_session_hours']*3600-600:break
        if call(dict(op='close'))!=dict(closed=True):raise ValueError('SERVICE_CLOSE')
        if proc.wait(timeout=30)!=0:raise ValueError('SERVICE_EXIT')
        write(session/'RESULT.json',dict(completed=len(completed),real_qwen_forwards=forwards,base_updates=0),True)
    finally:
        from kernel_metadata import record
        record(session/'TRITON_CACHE_METADATA.json')
        if policy is not None and initial is not None:
            final=c.model_identity(policy);last={tag:c.model_identity(net)['sha256'] for tag,net in heads.items()}
            write(session/'STATE_SEAL.json',dict(base_unchanged=final['sha256']==initial['sha256'],heads_unchanged=last==head_initial,
                base_state_sha256=final['sha256'],head_states=last,completed_ranks=completed),True)
        if forward:forward.close()
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        if stream:stream.close()
        if log:log.close()

if __name__=='__main__':main(Path(sys.argv[1]))
