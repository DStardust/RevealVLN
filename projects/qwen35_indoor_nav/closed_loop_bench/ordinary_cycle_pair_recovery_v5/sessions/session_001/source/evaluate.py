"""One frozen model, isolated paired environments, atomic whole-pair checkpoints."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as c
from cycle_policy import CycleRecovery


def load_policy(session, p):
    speed = c.LINE/'sft_acceptance/ordinary_speedup_10x_v1'
    sys.path.insert(0, str(speed/'official_einops_0_8_1/deps'))
    sys.path.insert(0, str(speed/'official_fla_0_5_2/deps'))
    import fla.ops.gated_delta_rule
    import torch
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(p['model_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    bound = modeling.torch_chunk_gated_delta_rule
    cells = dict(zip(bound.__code__.co_freevars, (x.cell_contents for x in bound.__closure__)))
    dispatch = getattr(cells.get('implementation'), '__module__', '')
    assert dispatch.startswith('fla.'), 'FLA_DISPATCH_CHANGED'
    model = c.load('v5_frozen_model', c.TRAIN/'model.py')
    assert c.sha(Path(p['checkpoint'])) == p['checkpoint_sha256'], 'CHECKPOINT_HASH'
    state = torch.load(p['checkpoint'], map_location='cpu', weights_only=True)
    assert state['binding']['protocol_sha256'] == p['training_protocol_sha256']
    assert state['binding']['sample_index_sha256'] == p['sample_index_sha256']
    assert state['cursor']['updates'] == 4000
    assert all(bool(torch.isfinite(t).all()) for t in state['trainable'].values())
    policy = model.build_policy(p['seed'])
    model.load_trainable(policy, state['trainable'])
    policy.eval()
    del state
    gc.collect()
    initial = c.model_identity(policy)
    c.write(session/'STATE_INITIAL.json', initial, True)
    import importlib.metadata
    versions = {}
    for package in ['torch','transformers','peft','triton','flash-linear-attention','numpy']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = 'unavailable'
    base_files = {str(x.relative_to(model.MODEL)): c.sha(x) for x in model.MODEL.iterdir()
                  if x.is_file() and x.suffix in ('.json','.safetensors','.jinja')}
    tokenizer = policy.processor.tokenizer.backend_tokenizer.to_str().encode()
    import hashlib
    identity = dict(session=session.name, pid=os.getpid(), gpu_uuid=p['gpu_uuid'], physical_gpu=p['gpu'],
        software=versions, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
        checkpoint_sha256=p['checkpoint_sha256'], base_files=base_files,
        model_source_sha256=c.sha(c.TRAIN/'model.py'), loaded_state_sha256=initial['sha256'],
        tokenizer_sha256=hashlib.sha256(tokenizer).hexdigest(), processor=policy.processor.to_dict(),
        image_processor=policy.processor.image_processor.to_dict(), attention=policy.base.config._attn_implementation,
        fla_dispatch=dispatch, triton_selected_config='unavailable before compilation; cache metadata saved at session end',
        dtype='unchanged BF16 base and FP32 action head', source_hashes=p['source_hashes'],
        protocol_sha256=c.sha(HERE/'PROTOCOL.json'), model_loads=1, optimizer_updates=0)
    c.write(session/'RUNTIME_IDENTITY.json', identity, True)
    return model, policy, initial


def main():
    session = Path(sys.argv[1])
    p = c.read(session/'CONFIG.json')
    model = policy = initial = None
    procs, streams, sockets, logs = [], {}, [], []
    completed = []
    sealed = set()
    began = time.monotonic()
    actions = 0
    active_rank = None
    stop = False
    def interrupted(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def progress(**extra):
        c.write(session/'PROGRESS.json', dict(unix=time.time(), pair_rank=active_rank,
            completed=len(completed), total_actions=actions, wall_seconds=time.monotonic()-began, **extra))
    def seal():
        if policy is None or not set(completed)-sealed:
            return
        progress(phase='STATE_FINGERPRINT')
        final = c.model_identity(policy)
        unchanged = final['sha256'] == initial['sha256']
        c.write(session/f'STATE_SEAL_{len(completed):03d}.json', dict(unchanged=unchanged,
            initial_sha256=initial['sha256'], final_sha256=final['sha256'], pair_ranks=list(completed)), True)
        if not unchanged:
            raise c.PairError('MODEL_STATE_CHANGED')
        sealed.update(completed)
    def call(arm, value):
        stream = streams[arm]
        stream.write(json.dumps(value)+'\n')
        stream.flush()
        raw = stream.readline()
        if not raw:
            raise EOFError('SIMULATOR_EOF_'+arm)
        return json.loads(raw)
    try:
        progress(phase='LOADING_MODEL')
        model, policy, initial = load_policy(session, p)
        import torch
        collate = model.make_collate(policy.processor.tokenizer.pad_token_id, policy.exec_sid,
                                     policy.query_sid, policy.base.config.image_token_id)
        windows = {a: c.Window() for a in ('A','B')}
        class Store:
            def get(self, index, t):
                return windows[('A','B')[index]].item()
        dataset = model.DecisionDataset([dict(record_idx=i,t=0,target=0,weight=1.) for i in range(2)], Store(), policy.processor)
        simenv = os.environ.copy()
        simenv.pop('CUDA_VISIBLE_DEVICES', None)
        for arm in ('A','B'):
            parent, child = socket.socketpair()
            parent.settimeout(180)
            log = (session/f'simulator_{arm}.log').open('x')
            proc = subprocess.Popen([str(c.LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'),'-I','-B',
                str(HERE/'executor.py'),str(child.fileno()),str(session),arm], pass_fds=(child.fileno(),),
                env=simenv,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT)
            child.close()
            procs.append(proc); sockets.append(parent); logs.append(log)
            streams[arm] = parent.makefile('rw')
            c.append(session/'CHILD_PROCESSES.jsonl',dict(pid=proc.pid,arm=arm))
        order = c.read(HERE/'PAIR_ORDER.json')
        prior_count = len(c.committed_pairs())
        with torch.inference_mode():
            for rank in p['scheduled_ranks']:
                if stop: break
                active_rank = rank
                row = order[rank]
                folder = session/'pairs'/f'pair_{rank:03d}'
                folder.mkdir()
                for arm in ('A','B'): (folder/arm).mkdir()
                guard = CycleRecovery()
                seen = {'A':set(),'B':set()}
                alive = {}
                for arm in row['inference_order']:
                    alive[arm] = windows[arm].receive(call(arm,dict(op='reset',index=row['index'])))
                if c.read(folder/'A/INITIAL_STATE.json') != c.read(folder/'B/INITIAL_STATE.json'):
                    c.write(folder/'FIRST_DIVERGENCE.json',dict(reason='INITIAL_STATE',A=c.read(folder/'A/INITIAL_STATE.json'),B=c.read(folder/'B/INITIAL_STATE.json')),True)
                    raise c.PairError('INITIAL_STATE_MISMATCH')
                audit = dict(input_prefix_matched=True,action_prefix_matched=True,logits_bitwise_equal=True,
                    max_logit_delta=0.,relative_max_logit_delta=0.,argmax_flip_count=0,prefix_decisions=0,first_override=None)
                step = 0
                while any(alive.values()):
                    if stop: raise InterruptedError('GRACEFUL_RESOURCE_STOP')
                    step += 1
                    decisions = {}
                    for arm in row['inference_order']:
                        if not alive[arm]: continue
                        tick = time.perf_counter()
                        raw = c.raw_input(windows[arm])
                        batch = collate([dataset[('A','B').index(arm)]])
                        batch.pop('targets'); batch.pop('weights')
                        processed = {k:c.tensor_identity(v) for k,v in batch.items()}
                        batch = {k:v.to('cuda:0') for k,v in batch.items()}
                        pre_seconds = time.perf_counter()-tick
                        tick = time.perf_counter()
                        logits = policy.forward_batch(**batch)
                        torch.cuda.synchronize()
                        values = logits.float().cpu()[0].tolist()
                        seconds = time.perf_counter()-tick
                        native = c.ACTIONS[max(range(4), key=values.__getitem__)]
                        tick = time.perf_counter()
                        action = native
                        override = False
                        if arm == 'B':
                            action, info = guard.choose(windows[arm].instruction, windows[arm].images, windows[arm].executed, values)
                            assert info['input_key'] == raw['input_key']
                            override = info['cycle_override']
                        repeated = raw['input_key'] in seen[arm]
                        seen[arm].add(raw['input_key'])
                        decisions[arm] = dict(index=row['index'],step=step,unix=time.time(),raw=raw,processed=processed,
                            logits=values,native_action=native,executed_action=action,override=override,
                            repeated_input=repeated,inference_seconds=seconds,preprocess_seconds=pre_seconds,
                            controller_seconds=time.perf_counter()-tick,action_margin=sorted(values)[-1]-sorted(values)[-2],
                            stop_margin=values[3]-max(values[:3]))
                    if audit['first_override'] is None:
                        if set(decisions) != {'A','B'}: raise c.PairError('PREFIX_TERMINATION_MISMATCH')
                        comparison = c.prefix_compare(decisions['A'],decisions['B'])
                        for key in ('input_prefix_matched','action_prefix_matched','logits_bitwise_equal'):
                            audit[key] &= comparison[key]
                        for key in ('max_logit_delta','relative_max_logit_delta'):
                            audit[key] = max(audit[key],comparison[key])
                        audit['argmax_flip_count'] += comparison['argmax_flip_count']
                        audit['prefix_decisions'] += 1
                        if not comparison['input_prefix_matched'] or not comparison['action_prefix_matched']:
                            c.write(folder/'FIRST_DIVERGENCE.json',dict(step=step,comparison=comparison,decisions=decisions),True)
                            raise c.PairError('PRE_INTERVENTION_DIVERGENCE')
                        if decisions['B']['override']: audit['first_override'] = step
                    for arm in row['inference_order']:
                        if arm not in decisions: continue
                        decision = decisions[arm]
                        c.append(folder/arm/'POLICY_STEPS.jsonl',decision)
                        alive[arm] = windows[arm].receive(call(arm,dict(op='action',action=decision['executed_action'])),executed=decision['executed_action'])
                        actions += 1
                    progress(phase='EVALUATING',step=step)
                episodes = {arm:c.audit_episode(folder/arm,row['index']) for arm in ('A','B')}
                if audit['first_override'] is None:
                    keys = ['positions','distances','steps','stopped','success','spl','ndtw','action_counts','collisions']
                    if any(episodes['A'][k] != episodes['B'][k] for k in keys):
                        raise c.PairError('NO_OVERRIDE_TERMINAL_MISMATCH')
                result = dict(row,session=session.name,protocol_sha256=c.sha(HERE/'PROTOCOL.json'),
                    runtime_identity_sha256=c.sha(session/'RUNTIME_IDENTITY.json'),valid_behavioral_pair=True,
                    audit=audit,episodes=episodes,parameter_seal='session STATE_SEAL covers this rank',
                    logs={arm:{f:c.sha(folder/arm/f) for f in ('POLICY_STEPS.jsonl','STEPS_PRIVILEGED.jsonl')} for arm in ('A','B')})
                c.write(folder/'PAIR.json',result,True)
                completed.append(rank)
                print(json.dumps(dict(pair_completed=rank,total_complete=prior_count+len(completed),
                    A=episodes['A']['success'],B=episodes['B']['success'],audit=audit)),flush=True)
                if prior_count+len(completed) in p['milestones']:
                    seal()
                    aggregate = c.load('v5_aggregate',HERE/'aggregate.py')
                    aggregate.main()
        for arm in streams: call(arm,dict(op='close'))
        seal()
        c.write(session/'WORKER_RESULT.json',dict(status='STOPPED' if stop else 'COMPLETE',completed=completed,total_actions=actions),True)
    except BaseException as exc:
        failure = dict(status='CORRECTNESS_ERROR' if isinstance(exc,(c.PairError,AssertionError,ValueError)) else 'INFRASTRUCTURE_ERROR',
            error=repr(exc),traceback=traceback.format_exc(),pair_rank=active_rank,completed=completed,total_actions=actions)
        c.write(session/'FAILURE.json',failure,True)
        print(json.dumps(failure),flush=True)
        raise
    finally:
        try: seal()
        finally:
            for proc in procs:
                if proc.poll() is None: proc.terminate()
            for proc in procs:
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill();proc.wait(timeout=10)
            for stream in streams.values(): stream.close()
            for sock in sockets: sock.close()
            for log in logs: log.close()
            metadata = {}
            for path in (session/'cache/triton').rglob('*.json'):
                try: metadata[str(path.relative_to(session))] = c.read(path)
                except (ValueError,OSError): pass
            c.write(session/'TRITON_CACHE_METADATA.json',metadata,True)


if __name__ == '__main__': main()
