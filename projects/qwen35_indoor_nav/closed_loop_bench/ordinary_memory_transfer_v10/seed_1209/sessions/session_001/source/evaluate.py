"""Native/B2/Ours/B1/N0, one best4k process and five isolated simulator channels."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
V5=HERE.parent/'ordinary_cycle_pair_recovery_v5'
MEMORY=LINE/'research/continuation_memory_v1/contextual_readout_v10'
sys.path.insert(0,str(V5))
import common as c
base_evaluate=c.load('v10_base_evaluate',V5/'evaluate.py')
ARMS=('A','B','C','D','E')
CONTRASTS=(('A','B'),('A','C'),('A','D'),('A','E'),('B','C'),('D','C'),('E','C'),('D','B'),('E','B'),('E','D'))


def selected_action(base_values,method_values):
    native=c.ACTIONS[max(range(4),key=base_values.__getitem__)]
    chosen=native if native=='STOP' else c.ACTIONS[max(range(4),key=method_values.__getitem__)]
    return native,chosen


def compare_prefix(left,right):
    comparison=c.prefix_compare(left,right)
    # The current intentional action difference ends the matched prefix. Previous
    # actions have already been compared at every step, including this input.
    if not comparison['input_prefix_matched'] or comparison['argmax_flip_count']:
        raise c.PairError('BASE_INPUT_OR_NATIVE_ACTION_DIVERGENCE')
    return comparison,left['executed_action']!=right['executed_action']


def active_contrast(decisions,left,right):
    if (left in decisions)!=(right in decisions):
        raise c.PairError('PREFIX_TERMINATION_MISMATCH')
    return left in decisions


def main(session):
    p=c.read(session/'CONFIG.json')
    seed=p['method_seed']
    protocol=c.read(HERE/'PROTOCOL.json')
    model=policy=initial=None
    heads={};head_initial={};memories={}
    processes=[];streams={};sockets=[];logs=[];completed=[];sealed=set()
    active_rank=None;actions=0;stop=False
    def interrupted(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def progress(**extra):
        c.write(session/'PROGRESS.json',dict(unix=time.time(),group_rank=active_rank,completed=len(completed),total_actions=actions,**extra))
    def call(arm,value):
        streams[arm].write(json.dumps(value)+'\n');streams[arm].flush()
        line=streams[arm].readline()
        if not line:raise EOFError('SIMULATOR_EOF_'+arm)
        return json.loads(line)
    def seal():
        if policy is None or not set(completed)-sealed:return
        progress(phase='STATE_FINGERPRINT')
        final=c.model_identity(policy)
        head_final={arm:c.model_identity(net)['sha256'] for arm,net in heads.items()}
        assert all(net.no_memory==(protocol['memory_arms'][arm]=='N0') for arm,net in heads.items())
        unchanged=final['sha256']==initial['sha256'] and head_final==head_initial
        c.write(session/f'STATE_SEAL_{len(completed):03d}.json',dict(unchanged=unchanged,
            initial_sha256=initial['sha256'],final_sha256=final['sha256'],head_initial=head_initial,
            head_final=head_final,group_ranks=list(completed)),True)
        if not unchanged:raise c.PairError('PARAMETERS_CHANGED')
        sealed.update(completed)
    try:
        progress(phase='LOADING_MODEL')
        model,policy,initial=base_evaluate.load_policy(session,p)
        assert initial['sha256']==c.read(MEMORY/'run_001/FEATURE_REUSE.json')['natural']['base_state_sha256'],'FEATURE_ENCODER_DIFFERS_FROM_DEPLOYED_BASE'
        import torch
        mem_model=c.load('v10_navigation_memory',MEMORY/'model.py')
        data=c.read(LINE/protocol['data_source'])
        for arm,name in protocol['memory_arms'].items():
            checkpoint=MEMORY/'run_001'/f'{name}_{seed}_MEMORY.pt'
            assert c.sha(checkpoint)==p['memory_checkpoint_sha256'][name]
            net=mem_model.MemoryPolicy(2048,len(data['query_vocabulary']),8,64,.99,no_memory=name=='N0')
            net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
            net=net.cuda().eval()
            for parameter in net.parameters():parameter.requires_grad_(False)
            heads[arm]=net;head_initial[arm]=c.model_identity(net)['sha256']
        expected=c.read(MEMORY/'run_001/RESULT.json')
        for arm,name in protocol['memory_arms'].items():
            assert head_initial[arm]==expected['runs'][f'{name}_{seed}']['final_state_sha256']
        c.write(session/'METHOD_IDENTITY.json',dict(base_state_sha256=initial['sha256'],head_states=head_initial,
            memory_checkpoints=p['memory_checkpoint_sha256'],seed=seed,seed_selected_by_score=False,
            arms=protocol['arms'],no_memory_control={arm:net.no_memory for arm,net in heads.items()},
            protocol_sha256=c.sha(HERE/'PROTOCOL.json'),base_runtime_identity_is_reused_V5_loader=True,
            torch_cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            future_query_used_at_runtime=False,raw_truth_used_at_runtime=False,native_stop_preserved=True),True)
        collate=model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
        windows={arm:c.Window() for arm in ARMS}
        class Store:
            def get(self,index,t):return windows[ARMS[index]].item()
        dataset=model.DecisionDataset([dict(record_idx=i,t=0,target=0,weight=1.) for i in range(len(ARMS))],Store(),policy.processor)
        capture=[]
        handle=policy.action_head.register_forward_pre_hook(lambda module,args:capture.append(args[0]))
        simenv=os.environ.copy();simenv.pop('CUDA_VISIBLE_DEVICES',None)
        for arm in ARMS:
            parent,child=socket.socketpair();parent.settimeout(180)
            log=(session/f'simulator_{arm}.log').open('x')
            proc=subprocess.Popen([str(LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'),'-I','-B',str(V5/'executor.py'),
                str(child.fileno()),str(session),arm],pass_fds=(child.fileno(),),env=simenv,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT)
            child.close();processes.append(proc);sockets.append(parent);logs.append(log)
            streams[arm]=parent.makefile('rw')
            c.append(session/'CHILD_PROCESSES.jsonl',dict(pid=proc.pid,arm=arm))
        order=c.read(V5/'PAIR_ORDER.json')
        with torch.inference_mode():
            for rank in p['scheduled_ranks']:
                if stop:break
                active_rank=rank;row=order[rank]
                inference_order=ARMS[rank%len(ARMS):]+ARMS[:rank%len(ARMS)]
                folder=session/'pairs'/f'pair_{rank:03d}';folder.mkdir()
                for arm in ARMS:(folder/arm).mkdir()
                memories={arm:net.reset(1,'cuda:0') for arm,net in heads.items()}
                seen={arm:set() for arm in ARMS}
                alive={arm:windows[arm].receive(call(arm,dict(op='reset',index=row['index']))) for arm in inference_order}
                states=[c.read(folder/arm/'INITIAL_STATE.json') for arm in ARMS]
                if any(state!=states[0] for state in states[1:]):raise c.PairError('INITIAL_STATE_MISMATCH')
                audits={a+b:dict(input_prefix_matched=True,action_prefix_matched=True,logits_bitwise_equal=True,
                    max_logit_delta=0.,argmax_flip_count=0,prefix_decisions=0,first_action_difference=None) for a,b in CONTRASTS}
                step=0
                while any(alive.values()):
                    if stop:raise InterruptedError('GRACEFUL_RESOURCE_STOP')
                    step+=1;decisions={}
                    for arm in inference_order:
                        if not alive[arm]:continue
                        tick=time.perf_counter();raw=c.raw_input(windows[arm])
                        batch=collate([dataset[ARMS.index(arm)]])
                        batch.pop('targets');batch.pop('weights')
                        processed={key:c.tensor_identity(value) for key,value in batch.items()}
                        batch={key:value.to('cuda:0') for key,value in batch.items()}
                        pre_seconds=time.perf_counter()-tick;tick=time.perf_counter()
                        logits=policy.forward_batch(**batch);torch.cuda.synchronize()
                        assert len(capture)==1
                        feature=capture.pop()
                        values=logits.float()[0].cpu().tolist()
                        inference_seconds=time.perf_counter()-tick;tick=time.perf_counter()
                        method=logits
                        if arm in heads:
                            memories[arm],_=heads[arm].update(feature.float(),memories[arm])
                            method=heads[arm].action_logits(memories[arm],logits.float(),feature.float())
                        assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(method).all())
                        method_values=method[0].float().cpu().tolist()
                        native,action=selected_action(values,method_values)
                        method_seconds=time.perf_counter()-tick
                        repeated=raw['input_key'] in seen[arm];seen[arm].add(raw['input_key'])
                        decisions[arm]=dict(index=row['index'],step=step,unix=time.time(),raw=raw,processed=processed,
                            logits=values,method_logits=method_values,native_action=native,executed_action=action,
                            override=action!=native,repeated_input=repeated,inference_seconds=inference_seconds,
                            preprocess_seconds=pre_seconds,controller_seconds=method_seconds,
                            action_margin=sorted(method_values)[-1]-sorted(method_values)[-2],
                            stop_margin=method_values[3]-max(method_values[:3]),
                            memory_fingerprint=c.tensor_identity(memories[arm]) if arm in memories else None)
                    for left,right in CONTRASTS:
                        audit=audits[left+right]
                        if audit['first_action_difference'] is not None:continue
                        if not active_contrast(decisions,left,right):continue
                        try:comparison,different=compare_prefix(decisions[left],decisions[right])
                        except c.PairError:
                            c.write(folder/'FIRST_DIVERGENCE.json',dict(step=step,contrast=left+right,decisions=decisions),True)
                            raise
                        audit['logits_bitwise_equal'] &= comparison['logits_bitwise_equal']
                        audit['max_logit_delta']=max(audit['max_logit_delta'],comparison['max_logit_delta'])
                        audit['prefix_decisions']+=1
                        if different:audit['first_action_difference']=step
                    for arm in inference_order:
                        if arm not in decisions:continue
                        d=decisions[arm]
                        c.append(folder/arm/'POLICY_STEPS.jsonl',d)
                        alive[arm]=windows[arm].receive(call(arm,dict(op='action',action=d['executed_action'])),executed=d['executed_action'])
                        actions+=1
                    progress(phase='EVALUATING',step=step)
                episodes={arm:c.audit_episode(folder/arm,row['index']) for arm in ARMS}
                for left,right in CONTRASTS:
                    if audits[left+right]['first_action_difference'] is None:
                        keys=['positions','distances','steps','stopped','success','spl','ndtw','action_counts','collisions']
                        assert all(episodes[left][key]==episodes[right][key] for key in keys),'NO_DIFFERENCE_TERMINAL_MISMATCH'
                c.write(folder/'GROUP.json',dict(rank=rank,method_seed=seed,index=row['index'],episode_id=episodes['A']['episode_id'],
                    inference_order=inference_order,audits=audits,episodes=episodes,valid_behavioral_group=True,
                    protocol_sha256=c.sha(HERE/'PROTOCOL.json'),method_identity_sha256=c.sha(session/'METHOD_IDENTITY.json'),
                    logs={arm:{name:c.sha(folder/arm/name) for name in ('POLICY_STEPS.jsonl','STEPS_PRIVILEGED.jsonl')} for arm in ARMS}),True)
                completed.append(rank)
                print(json.dumps(dict(completed=len(completed),rank=rank,success={arm:episodes[arm]['success'] for arm in ARMS})),flush=True)
                if len(completed) in (5,20,100):
                    seal()
                    review=c.load('v10_milestone_review',HERE/'review.py')
                    c.write(session/f'MILESTONE_{len(completed):03d}.json',review.summarize(seed),True)
        for arm in streams:call(arm,dict(op='close'))
        seal();handle.remove()
        c.write(session/'WORKER_RESULT.json',dict(status='STOPPED' if stop else 'COMPLETE',completed=completed,total_actions=actions),True)
    except BaseException as exc:
        c.write(session/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),rank=active_rank,completed=completed),True)
        raise
    finally:
        try:seal()
        finally:
            for proc in processes:
                if proc.poll() is None:proc.terminate()
            for proc in processes:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            for stream in streams.values():stream.close()
            for sock in sockets:sock.close()
            for log in logs:log.close()
            metadata={}
            for path in (session/'cache/triton').rglob('*.json'):
                try:metadata[str(path.relative_to(session))]=c.read(path)
                except (ValueError,OSError):pass
            c.write(session/'TRITON_CACHE_METADATA.json',metadata,True)


if __name__=='__main__':main(Path(sys.argv[1]))
