"""Recapture every executed action-token feature from exact physical replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
sys.path[:0] = [str(HERE), str(BASE), str(BASE / 'recovery_action_v1')]
import common as u
from action_boundary_v2 import ActionBoundary, assistant_header
from replay import forced_scores
from contracts import validate_chunk, admit_trajectory


def main(run, output, ids):
    u.verify_sources(run)
    output.mkdir(parents=True, exist_ok=False)
    u.setup_imports(output)
    import numpy as np
    import torch
    import transformers
    import habitat_sim
    import streamvln_eval as official
    from memory_v2 import ExecutionMemory
    from capture_runtime_v3 import DenseRuntime, tensor_hash
    began = time.time()
    torch.set_num_threads(4)
    def seed():
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    seed()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = u.read(run / 'PROTOCOL.json')
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(u.MODEL), model_max_length=4096,
        padding_side='right', local_files_only=True)
    tokens = [tokenizer.encode(s, add_special_tokens=False) for s in ['STOP', '↑', '←', '→']]
    assert all(len(t) == 1 for t in tokens), 'ACTION_TOKEN_MAPPING'
    token_ids = [t[0] for t in tokens]
    header = assistant_header(tokenizer)
    cfg = transformers.AutoConfig.from_pretrained(str(u.MODEL), local_files_only=True)
    cfg.mm_vision_tower = str(u.MODEL / 'siglip-so400m-patch14-384')
    cfg.vision_tower = cfg.mm_vision_tower
    model, loading = official.StreamVLNForCausalLM.from_pretrained(str(u.MODEL), config=cfg,
        attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False, local_files_only=True, output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
    model.model.num_history = 8
    model.requires_grad_(False); model.to(0); model.eval(); model.reset(1)
    # Preserve the old dense encoder call path. Its random head never acts.
    adapter = ExecutionMemory(cfg.hidden_size).cuda().eval().requires_grad_(False)
    def base_hash():
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode()); digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            raw = value.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw))
        return digest.hexdigest()
    before = base_hash()
    assert before == protocol['expected_base_state_sha256'], 'LOADED_BASE_IDENTITY_CHANGED'
    identity = dict(model_loaded=True, base_state_sha256=before, base_updates=0,
        source_commit=protocol['source_commit'], action_token_ids=token_ids, assistant_header_ids=header,
        dtype='bfloat16_base/float32_action_scores', attention='flash_attention_2',
        versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                      habitat_sim=habitat_sim.__version__, cuda=torch.version.cuda),
        gpu=torch.cuda.get_device_name(0), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
        loaded_seconds=time.time() - began, hidden_size=cfg.hidden_size, loading_info=loading,
        protocol_sha256=u.sha(run / 'PROTOCOL.json'), source_lock_sha256=u.sha(run / 'SOURCE_LOCK.json'))
    u.write(output / 'RUNTIME_IDENTITY.json', identity)
    manifest = {e['id']: e for e in u.read(run / 'DATA_MANIFEST.json')['episodes']}
    plans = {}
    for plan in u.read(run / 'REQUESTS.json'):
        plans.setdefault(plan['trajectory_id'], {})[plan['query_start_step']] = plan
    active = {}
    completed = []
    real_env, generate = official.Env, model.generate
    def array_hash(value):
        value = np.ascontiguousarray(value)
        digest = hashlib.sha256(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
        return digest.hexdigest()
    def emit(value):
        active['log'].write(json.dumps(value, allow_nan=False) + '\n')
        active['log'].flush()
    def progress():
        u.write(output / 'PROGRESS.json', dict(status='CAPTURING', complete=completed, planned=ids,
            model_loaded=True, current_id=active.get('id'), environment_step=active.get('steps'),
            captured_queries=len(active.get('query_records', [])),
            captured_tokens=len(active.get('actors', [])),
            last_action=active.get('actual_actions', [None])[-1] if active.get('actual_actions') else None,
            unix=time.time()))
    def observe(rgb, previous):
        item = active['dense'].observe(rgb, previous)
        active['memory_features'].append(active['dense'].last_feature.clone())
        emit(dict(event='memory_write', environment_step=active['steps'], **item))
    class ReplayEnv(real_env):
        def reset(self, *args, **kwargs):
            obs = super().reset(*args, **kwargs)
            active['env'] = self; active['steps'] = 0
            active['rgb'] = array_hash(obs['rgb'])
            assert active['rgb'] == active['trace']['rgb_sha256'][0], 'REPLAY_START_MISMATCH'
            assert self.current_episode.instruction.instruction_text == active['entry']['instruction'], 'INSTRUCTION_MISMATCH'
            active['dense'] = DenseRuntime(model, tokenizer, active['helper'].image_processor,
                adapter, token_ids, active['entry']['instruction'])
            emit(dict(event='reset', rgb_sha256=active['rgb']))
            observe(obs['rgb'], None)
            return obs
        def step(self, action, *args, **kwargs):
            t = active['steps']; trace = active['trace']
            assert t < len(trace['actions']) and int(action) == trace['actions'][t], 'REPLAY_ACTION_MISMATCH'
            assert active['rgb'] == trace['rgb_sha256'][t], 'REPLAY_RGB_MISMATCH'
            active['actual_actions'].append(int(action)); active['actual_rgb'].append(active['rgb'])
            prior = active['rgb']
            obs = super().step(action, *args, **kwargs)
            active['steps'] += 1; active['rgb'] = array_hash(obs['rgb'])
            metrics = self.get_metrics()
            emit(dict(event='action', step=active['steps'], executed_action=int(action),
                before_rgb_sha256=prior, after_rgb_sha256=active['rgb'],
                distance=float(metrics['distance_to_goal']),
                collision=bool(metrics.get('collisions', {}).get('is_collision', False))))
            if int(action) != 0 and not self.episode_over:
                observe(obs['rgb'], int(action))
            if active['steps'] % 8 == 0:
                progress()
            return obs
    official.Env = ReplayEnv
    def feature_hook(module, args, result):
        active['last_feature'] = result.last_hidden_state[:, -1].detach().float()
    hook = model.model.register_forward_hook(feature_hook)
    class CaptureTokens(transformers.LogitsProcessor):
        def __init__(self):
            self.boundary = ActionBoundary(header)
        def __call__(self, input_ids, scores):
            offset = self.boundary.offset(input_ids)
            if offset is None:
                return scores
            assert active['steps'] == active['query_start'], 'PHYSICAL_STEP_DURING_GENERATION'
            assert active['dense'].writes == active['query_start'] + 1, 'FUTURE_MEMORY_IN_CHUNK'
            chunk = active['chunk_actions']
            if offset == len(chunk):
                return forced_scores(scores, tokenizer.eos_token_id)
            assert 0 <= offset < len(chunk), 'UNEXECUTED_ACTION_CAPTURE'
            assert offset == len(active['chunk_actors']), 'DUPLICATE_OR_SKIPPED_ACTION_TOKEN'
            assert scores.dtype == torch.float32, 'ACTION_SCORE_DTYPE_CHANGED'
            feature = active['last_feature'][0].cpu()
            logits = scores[0, token_ids].detach().float().cpu()
            assert torch.isfinite(feature).all() and torch.isfinite(logits).all(), 'NONFINITE_CAPTURE'
            active['chunk_actors'].append(feature)
            active['chunk_logits'].append(logits)
            return forced_scores(scores, token_ids[chunk[offset]])
    def capture_generate(*args, **kwargs):
        start = active['steps']; trace = active['trace']
        q = len(active['query_records'])
        assert q < len(trace['query_steps']) and trace['query_steps'][q] == start, 'QUERY_TIMING_CHANGED'
        end = trace['query_steps'][q + 1] if q + 1 < len(trace['query_steps']) else len(trace['actions'])
        evidence = {name: dict(shape=list(kwargs[name].shape), dtype=str(kwargs[name].dtype),
                              sha256=tensor_hash(kwargs[name]))
                    for name in ('inputs', 'images', 'depths', 'poses', 'intrinsics')}
        evidence.update(time_ids=kwargs['time_ids'], rgb_sha256=active['rgb'])
        assert evidence == active['old_queries'][q]['input'], 'PROCESSED_INPUT_CHANGED'
        active.update(query_start=start, chunk_actions=trace['actions'][start:end],
                      chunk_actors=[], chunk_logits=[])
        kwargs['logits_processor'] = transformers.LogitsProcessorList([CaptureTokens()])
        tick = time.time()
        value = generate(*args, **kwargs)
        torch.cuda.synchronize()
        actors, logits = active['chunk_actors'], active['chunk_logits']
        assert len(actors) == end - start, 'MISSING_ACTION_FEATURE'
        generated = value.sequences[0].cpu().tolist()
        expected = header + [token_ids[a] for a in trace['actions'][start:end]] + [tokenizer.eos_token_id]
        assert generated == expected, 'GENERATED_EXECUTED_CHUNK_MISMATCH'
        if start in active['plans']:
            certificate = validate_chunk(trace, active['plans'][start], generated,
                [x.tolist() for x in actors], [x.tolist() for x in logits],
                header_ids=header, action_token_ids=token_ids, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, context_query_start=start,
                memory_replay_exclusive_end=active['dense'].writes, actor_width=cfg.hidden_size)
            active['certificates'].append(certificate)
        old = active['old_cache']
        delta = logits[0] - old['base_logits'][q].float()
        parity = dict(query_index=q, max_logit_delta=float(delta.abs().max()),
            argmax_flip=bool(logits[0].argmax() != old['base_logits'][q].argmax()),
            max_actor_delta=float((actors[0] - old['actor_features'][q].float()).abs().max()),
            native_margin=float(torch.topk(logits[0], 2).values.diff().abs()[0]),
            processed_input_exact=True)
        active['parity'].append(parity)
        record = dict(event='generation', environment_step=start, input=evidence, generated_ids=generated,
            actor_feature_sha256=[tensor_hash(x) for x in actors],
            native_logits=[x.tolist() for x in logits], actions=trace['actions'][start:end],
            context_memory_index=start, seconds=time.time() - tick, anchor_parity=parity)
        active['query_records'].append(record)
        active['actors'].extend(actors); active['logits'].extend(logits)
        active['contexts'].extend([start] * len(actors))
        active['offsets'].extend(range(len(actors)))
        emit(record); progress()
        return value
    model.generate = capture_generate
    failure = None
    try:
        for index in ids:
            seed()
            entry = manifest[index]; trace = u.read(entry['trajectory'])
            assert u.sha(entry['old_cache']) == entry['old_cache_sha256'], 'OLD_CACHE_CHANGED'
            old_cache = torch.load(entry['old_cache'], map_location='cpu', weights_only=True)
            assert u.sha(entry['old_trace']) == old_cache['source_trace_sha256'], 'OLD_TRACE_CACHE_BINDING_CHANGED'
            old_queries = [json.loads(line) for line in Path(entry['old_trace']).read_text().splitlines()
                           if json.loads(line)['event'] == 'generation']
            assert len(old_queries) == len(trace['query_steps'])
            group = output / 'episodes' / str(index); group.mkdir(parents=True)
            with (group / 'TRACE.jsonl').open('x', buffering=1) as log:
                active.clear()
                active.update(id=index, entry=entry, trace=trace, plans=plans[index], old_cache=old_cache,
                    old_queries=old_queries, log=log, memory_features=[], actors=[], logits=[],
                    contexts=[], offsets=[], certificates=[], query_records=[], parity=[],
                    actual_actions=[], actual_rgb=[])
                helper = official.VLNEvaluator(entry['config'], split='train', env_num=1,
                    output_path=str(group), model=model, tokenizer=tokenizer, epoch=0,
                    args=argparse.Namespace(save_video=False, num_frames=32, num_future_steps=4, num_history=8))
                active['helper'] = helper
                try:
                    with torch.inference_mode():
                        helper.eval_action(0)
                except BaseException:
                    if 'env' in active:
                        active['env'].close()
                    raise
            receipt = admit_trajectory(trace, list(active['plans'].values()), active['certificates'],
                actual_actions=active['actual_actions'], actual_rgb_sha256=active['actual_rgb'])
            assert len(active['actors']) == len(active['memory_features']) == len(trace['actions'])
            outcomes = [json.loads(line) for line in (group / 'result.json').read_text().splitlines()]
            assert len(outcomes) == 1 and outcomes[0]['success'] == float(trace['success']), 'REPLAY_TERMINAL_CHANGED'
            assert outcomes[0]['steps'] == len(trace['actions']) <= 500, 'DECISION_BUDGET_CHANGED'
            known = [entry['kind'] == 'PRESERVATION' or t >= trace['cutoff'] for t in range(len(trace['actions']))]
            cache = group / 'CACHE.pt'; temp = cache.with_suffix('.tmp')
            torch.save(dict(memory_features=torch.stack(active['memory_features']),
                actor_features=torch.stack(active['actors']), base_logits=torch.stack(active['logits']),
                executed_actions=torch.tensor(trace['actions']), query_steps=torch.tensor(trace['query_steps']),
                actor_context_steps=torch.tensor(active['contexts']), action_token_offsets=torch.tensor(active['offsets']),
                supervision_region=torch.tensor(known), new_training_admission=False,
                id=index, house=entry['house'], partition=entry['partition'], kind=entry['kind'],
                cutoff=trace['cutoff'], source_trajectory_sha256=entry['trajectory_sha256'],
                representation='REAL_ALL_ACTION_TOKEN_CAPTURE_WITH_QUERY_START_MEMORY'), temp)
            temp.replace(cache)
            u.write(group / 'CHUNK_CERTIFICATES.json', active['certificates'])
            u.write(group / 'PARITY.json', dict(rows=active['parity'],
                raw_and_processed_inputs_exact=True,
                argmax_flips=sum(x['argmax_flip'] for x in active['parity']),
                interpretation='Old/live first-token numerical drift; not paired navigation or automatic label change.'))
            receipt.update(cache_path=str(cache), cache_sha256=u.sha(cache),
                captured_all_actor_rows=len(active['actors']), captured_all_queries=len(active['query_records']),
                runtime_identity_sha256=u.sha(output / 'RUNTIME_IDENTITY.json'),
                trace_sha256=u.sha(group / 'TRACE.jsonl'))
            u.write(group / 'COMPLETE.json', receipt)
            completed.append(index); progress()
    except BaseException as error:
        failure = repr(error)
        u.write(output / 'FAILURE.json', dict(error=failure, id=active.get('id'),
            step=active.get('steps'), completed=completed, traceback=traceback.format_exc()))
        raise
    finally:
        hook.remove(); official.Env = real_env; model.generate = generate
        after = base_hash()
        assert after == before, 'BASE_STATE_CHANGED'
        u.write(output / 'STATE_SEAL.json', dict(base_before=before, base_after=after,
            complete_ids=completed, failure=failure, source_lock_sha256=u.sha(run / 'SOURCE_LOCK.json')))
        u.write(output / 'RESULT.json', dict(status='COMPLETE' if failure is None else 'PARTIAL_FAILURE',
            complete=completed, planned=ids, model_loaded=True, base_updates=0,
            wall_seconds=time.time() - began, peak_gpu_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', required=True)
    args = parser.parse_args()
    main(args.run, args.output, [int(x) for x in args.ids.split(',')])
