"""Actual autonomous continuations, fixed-history prefill, no query/oracle actor input."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import objective as o
c=o.c
base=c.load('v15_continuation_base',c.HERE/'evaluate.py')
runtime=c.load('v15_stop_guard',c.LINE/'closed_loop_bench/ordinary_memory_transfer_v10/evaluate.py')


def prefix_audit(left,right):
    report=dict(input_prefix_matched=True,action_prefix_matched=True,logits_bitwise_equal=True,
        max_logit_delta=0.,argmax_flip_count=0,decisions=0,first_method_action_difference=None)
    for a,b in zip(left,right):
        comparison=c.prefix_compare(a,b)
        report['decisions']+=1
        report['logits_bitwise_equal'] &= comparison['logits_bitwise_equal']
        report['max_logit_delta']=max(report['max_logit_delta'],comparison['max_logit_delta'])
        report['argmax_flip_count']+=comparison['argmax_flip_count']
        if not comparison['input_prefix_matched'] or comparison['argmax_flip_count']:
            return dict(report,input_prefix_matched=comparison['input_prefix_matched'],
                action_prefix_matched=not bool(comparison['argmax_flip_count']),first_divergence=dict(left=a,right=b))
        if a['executed_action']!=b['executed_action']:
            return dict(report,first_method_action_difference=a['decision'])
    if len(left)!=len(right):
        return dict(report,action_prefix_matched=False,first_divergence='UNMATCHED_TERMINATION')
    return report


def main(run):
    protocol=c.read(HERE/'CONTINUATION_PROTOCOL.json');train=HERE/'train_run_001'
    for path,digest in protocol['source_hashes'].items():assert c.sha(c.LINE/path)==digest,path
    assert len(c.read(train/'RESULT.json')['runs'])==18
    proc=None;stream=None;sock=None;hook=None;log=None;policy=None;completed=[];heads={};initial=None;head_initial={}
    def progress(**extra):c.write(run/'PROGRESS.json',dict(unix=time.time(),completed=len(completed),**extra))
    def call(value):
        stream.write(json.dumps(value)+'\n');stream.flush()
        line=stream.readline()
        if not line:raise EOFError('SIMULATOR_EOF')
        return json.loads(line)
    try:
        progress(phase='LOAD_MODEL')
        model,policy,initial=base.load_policy(run,c.read(run/'CONFIG.json'))
        assert initial['sha256']==c.read(HERE/'features_run_001/FEATURE_RESULT.json')['base_state_sha256']
        for tag in protocol['models']:
            recorded=c.read(train/(tag+'_RESULT.json'));path=train/(tag+'_MEMORY.pt')
            assert c.sha(path)==recorded['checkpoint_sha256']
            net=o.initialize(recorded['seed']);net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            assert c.model_identity(net)['sha256']==recorded['final_state_sha256']
            heads[tag]=net.cuda().eval()
            for parameter in heads[tag].parameters():parameter.requires_grad_(False)
        head_initial={tag:c.model_identity(net)['sha256'] for tag,net in heads.items()}
        c.write(run/'METHOD_IDENTITY.json',dict(base_state_sha256=initial['sha256'],head_states=head_initial,
            protocol_sha256=c.sha(HERE/'CONTINUATION_PROTOCOL.json'),future_queries_used=False,
            prefill='Live causal Qwen features from first actual replay of each condition, shared within this model session after exact raw prefix checks. Not a KV or text-history bypass.',
            input_contract=['instruction','last2_RGB','last8_executed_motion_actions'],native_stop_preserved=True),True)
        window=c.Window()
        class Store:
            def get(self,index,t):return window.item()
        dataset=model.DecisionDataset([dict(record_idx=0,t=0,target=0,weight=1.)],Store(),policy.processor)
        collate=model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
        captured=[]
        hook=policy.action_head.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        cached_data=c.read(HERE/'DATA.json')
        cache_lookup={(r['instruction'],tuple(x[7:] for x in r['rgb_refs']),tuple(r['executed'])):i
            for i,r in enumerate(cached_data['features'])}
        cached_features=torch.load(HERE/'features_run_001/FEATURES.pt',map_location='cpu',weights_only=True)
        cached_inputs={r['index']:r['processed'] for r in c.records(HERE/'features_run_001/FEATURE_INPUTS.jsonl')}
        def forward():
            start=time.perf_counter();batch=collate([dataset[0]]);batch.pop('targets');batch.pop('weights')
            processed={k:c.tensor_identity(v) for k,v in batch.items()}
            prepared=time.perf_counter();logits=policy.forward_batch(**{k:v.cuda() for k,v in batch.items()});torch.cuda.synchronize()
            assert len(captured)==1;feature=captured.pop().float()
            assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(feature).all())
            timing=dict(preprocess_seconds=prepared-start,inference_seconds=time.perf_counter()-prepared)
            raw=c.raw_input(window);idx=cache_lookup.get((raw['instruction'],tuple(raw['rgb_sha256']),tuple(raw['executed_history'])))
            if idx is not None:
                current=feature[0].cpu();old=cached_features['features'][idx];native=logits[0].float().cpu();old_logits=cached_features['logits'][idx]
                timing['training_cache_comparison']=dict(index=idx,processed_inputs_equal=processed==cached_inputs[idx],
                    max_feature_delta=float((current-old).abs().max()),relative_feature_l2=float((current-old).norm()/old.norm().clamp_min(1e-12)),
                    max_logit_delta=float((native-old_logits).abs().max()),native_argmax_flip=int(native.argmax()!=old_logits.argmax()))
                if processed!=cached_inputs[idx]:
                    c.write(run/'CACHE_INPUT_DIVERGENCE.json',dict(raw=raw,current=processed,reference=cached_inputs[idx]),True)
                    raise c.PairError('TRAINING_CACHE_PREPROCESSED_INPUT_MISMATCH')
            return feature,logits.float(),processed,timing
        sock,child=socket.socketpair();sock.settimeout(180)
        env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
        log=(run/'simulator.log').open('x')
        proc=subprocess.Popen([str(c.LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'),'-I','-B',
            str(HERE/'continuation_service.py'),str(child.fileno()),str(run)],pass_fds=(child.fileno(),),
            cwd=c.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        child.close();stream=sock.makefile('rw')
        c.write(run/'CHILD_PROCESS.json',dict(pid=proc.pid,parent=os.getpid()),True)
        raw_config=c.read(c.LINE/protocol['raw_config']);families={f['family_id']:f for f in raw_config['families']}
        current_condition=None;prefix_features=[];prefix_inputs=[];contrast_rows={};forwards=0;audits=[]
        with torch.inference_mode():
            for rank,row in enumerate(protocol['rollouts']):
                folder=run/'rollouts'/f'{rank:03d}';folder.mkdir(parents=True)
                condition=protocol['conditions'][row['condition']];tag=row['model'];net=heads[tag]
                history=families[condition['family_id']]['candidate']['histories'][condition['history_id']]
                new_condition=current_condition!=row['condition']
                if new_condition:
                    current_condition=row['condition'];prefix_features=[];prefix_inputs=[];contrast_rows={}
                assert window.receive(call(dict(op='reset',rank=rank)))
                for t,action in enumerate(history):
                    raw=c.raw_input(window)
                    if new_condition:
                        feature,_,processed,timing=forward();forwards+=1
                        prefix_features.append(feature);prefix_inputs.append(raw)
                        c.append(folder/'PREFILL.jsonl',dict(step=t,raw=raw,processed=processed,feature=c.tensor_identity(feature),**timing))
                    else:assert raw==prefix_inputs[t],'PREFILL_RAW_INPUT_MISMATCH'
                    executed=c.ACTIONS['FLR'.index(action)]
                    assert window.receive(call(dict(op='action',action=executed)),executed=executed)
                    if t%20==0:progress(phase='PREFIX_REPLAY',rank=rank,step=t,forwards=forwards)
                features=torch.cat(prefix_features,0).unsqueeze(0)
                memory=net.encode(features)[0][:,-1]
                c.write(folder/'PREFILL_AUDIT.json',dict(prefix_steps=len(history),all_raw_inputs_matched=True,
                    live_feature_origin_rank=rank-rank%len(protocol['models']),features=c.tensor_identity(features),
                    memory_before_cutoff=c.tensor_identity(memory),new_observation_updates_once=True),True)
                rows=[];alive=True
                size,arm,seed=tag.split('_')
                references={other:True for other in contrast_rows
                    if other.split('_')[0]==size and other.split('_')[2]==seed and
                    ((arm=='Ours') != (other.split('_')[1]=='Ours'))}
                while alive:
                    raw=c.raw_input(window);feature,logits,processed,timing=forward();forwards+=1
                    tick=time.perf_counter();memory,_=net.update(feature,memory)
                    method=net.action_logits(memory,logits,feature)
                    assert bool(torch.isfinite(method).all())
                    values=logits[0].cpu().tolist();method_values=method[0].cpu().tolist()
                    native,action=runtime.selected_action(values,method_values)
                    decision=dict(decision=len(history)+len(rows),raw=raw,processed=processed,logits=values,
                        method_logits=method_values,native_action=native,executed_action=action,override=action!=native,
                        native_stop_guard_applied=native=='STOP' and method_values.index(max(method_values))!=3,
                        controller_seconds=time.perf_counter()-tick,**timing)
                    for other,matched in references.items():
                        if not matched:continue
                        previous=contrast_rows[other]
                        comparison=prefix_audit(previous[:len(rows)+1],rows+[decision])
                        if 'first_divergence' in comparison:
                            c.write(run/'FIRST_DIVERGENCE.json',dict(rank=rank,left=other,right=tag,**comparison),True)
                            raise c.PairError('INPUT_OR_NATIVE_ACTION_PREFIX_MISMATCH')
                        references[other]=comparison['first_method_action_difference'] is None
                    c.append(folder/'POLICY_STEPS.jsonl',decision);rows.append(decision)
                    alive=window.receive(call(dict(op='action',action=action)),executed=action)
                    progress(phase='AUTONOMOUS',rank=rank,step=decision['decision'],forwards=forwards)
                contrast_rows[tag]=rows
                task=c.read(folder/'TASK_RESULT.json')
                assert task['autonomous_decisions']==len(rows) and task['prefix_replayed_and_exact']
                c.write(folder/'ROLLOUT.json',dict(rank=rank,condition=condition,model=tag,task_result=task,
                    policy_log_sha256=c.sha(folder/'POLICY_STEPS.jsonl'),trace_sha256=c.sha(folder/'TRACE_PRIVILEGED.json')),True)
                completed.append(rank)
                print(json.dumps(dict(completed=len(completed),model=tag,label=task['label'],decisions=task['total_decisions'])),flush=True)
                if len(contrast_rows)==len(protocol['models']):
                    for size in ('S1','L3'):
                        for seed in (1209,1210,1211):
                            for control in ('B1','B2'):
                                left=f'{size}_{control}_{seed}';right=f'{size}_Ours_{seed}'
                                audit=prefix_audit(contrast_rows[left],contrast_rows[right])
                                entry=dict(condition=row['condition'],left=left,right=right,**audit);audits.append(entry)
                                if 'first_divergence' in audit:
                                    c.write(run/'FIRST_DIVERGENCE.json',entry,True)
                                    raise c.PairError('INPUT_OR_NATIVE_ACTION_PREFIX_MISMATCH')
                    c.write(run/f'CONDITION_{row["condition"]:03d}_AUDITS.json',audits[-12:],True)
        assert call(dict(op='close'))==dict(closed=True)
        assert proc.wait(timeout=30)==0,'SIMULATOR_SERVICE_NOT_CLOSED_CLEANLY'
        assert c.read(run/'CONTENT_AUDIT.json')['audit_pass']
        final=c.model_identity(policy);head_final={tag:c.model_identity(net)['sha256'] for tag,net in heads.items()}
        assert final['sha256']==initial['sha256'] and head_final==head_initial
        c.write(run/'STATE_SEAL.json',dict(base_unchanged=True,heads_unchanged=True,
            base_state_sha256=final['sha256'],head_states=head_final,completed_ranks=completed),True)
        c.write(run/'RESULT.json',dict(status='AUTONOMOUS_CONTINUATIONS_COMPLETE',complete_rollouts=len(completed),
            real_qwen_forwards=forwards,base_parameters_unchanged=True,head_parameters_unchanged=True,
            model_loads=1,optimizer_updates=0,audits=audits,publication_ready=False),True)
    finally:
        if completed and not (run/'STATE_SEAL.json').exists() and initial and len(head_initial)==18:
            try:
                final=c.model_identity(policy);head_final={tag:c.model_identity(net)['sha256'] for tag,net in heads.items()}
                c.write(run/'STATE_SEAL.json',dict(base_unchanged=final['sha256']==initial['sha256'],
                    heads_unchanged=head_final==head_initial,base_state_sha256=final['sha256'],
                    head_states=head_final,completed_ranks=completed),True)
            except BaseException as error:
                c.write(run/'STATE_SEAL_ERROR.json',dict(error=repr(error)),True)
        if hook:hook.remove()
        if proc:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        if stream:runtime.close_stream(stream)
        if sock:sock.close()
        if log:log.close()


if __name__=='__main__':
    run=Path(sys.argv[1])
    try:main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
