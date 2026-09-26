"""Real StreamVLN/Habitat execution with the registered all-token residual.

One process loads the frozen base once and executes every registered head for
each condition. CPU contract tests do not constitute a live run of this worker.
"""
import argparse
import importlib.util
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import common as u

# A file alias keeps the upstream ``model`` package free for StreamVLN.
_spec = importlib.util.spec_from_file_location("q35n_execution_adaptation_head", HERE / "model.py")
_head = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_head)
ExecutionAdaptation = _head.ExecutionAdaptation


def load_registered_head(record, protocol, device):
    import torch
    path = Path(record['path'])
    if u.sha(path) != record['sha256']:
        raise ValueError('HEAD_FILE_CHANGED')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config_path = path.parent.parent / 'TRAINING_CONFIG.json'
    config = u.read(config_path)
    config_sha = u.sha(config_path)
    if config_sha != record['training_config_sha256']:
        raise ValueError('REGISTERED_TRAINING_CONFIG_CHANGED')
    expected = dict(config_sha256=config_sha, seed=record['seed'], mode=record['mode'])
    if saved['binding'] != expected or saved['step'] != protocol['steps']:
        raise ValueError('HEAD_TRAINING_BINDING_CHANGED')
    if config['debug_only'] or saved['step'] != config['steps']:
        raise ValueError('DEBUG_OR_NONFINAL_HEAD_REJECTED')
    if record['seed'] not in config['seeds'] or record['mode'] not in config['modes']:
        raise ValueError('HEAD_OUTSIDE_TRAINING_REGISTRY')
    if config['action_scope'] != 'ALL_ACTION_TOKENS_QUERY_START_MEMORY' or config['base_updates'] != 0:
        raise ValueError('HEAD_ACTION_SCOPE_MISMATCH')
    for source, sha in config['source_files'].items():
        if u.sha(source) != sha:
            raise ValueError('HEAD_TRAINING_SOURCE_CHANGED: ' + source)
    head = ExecutionAdaptation(config['width'], record['mode']).to(device).eval()
    head.load_state_dict(saved['model'])
    head.requires_grad_(False)
    return head


def main(run, output, ids):
    u.verify_sources(run)
    binding = u.read(run / 'EXECUTOR_BINDING.json')
    if binding['path'] != str(Path(__file__).resolve()) or binding['sha256'] != u.sha(__file__):
        raise ValueError('EXECUTOR_REVISION_CHANGED')
    protocol = u.read(run / 'PROTOCOL.json')
    manifest = {row['id']: row for row in u.read(run / 'DATA_MANIFEST.json')['episodes']}
    if not ids or len(set(ids)) != len(ids) or any(i not in manifest for i in ids):
        raise ValueError('INVALID_REGISTERED_EPISODES')
    output.mkdir(parents=True, exist_ok=False)
    # Resolve the new local interfaces before upstream paths are prepended.
    from runtime import LiveRuntime, ChunkActionProcessor, make_logits_processors, tensor_hash
    from audit import audit_pair
    from action_boundary_v2 import assistant_header
    # Upstream model/ is a namespace package; any later model.py would win.
    sys.path.remove(str(HERE))
    u.setup_imports(output)
    import os
    import numpy as np
    import torch
    import transformers
    import habitat
    import habitat_sim
    import streamvln_eval as official

    began = time.time()
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(u.MODEL), model_max_length=4096,
        padding_side='right', local_files_only=True)
    action_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in ('STOP', '↑', '←', '→')]
    if any(len(tokens) != 1 for tokens in action_tokens):
        raise ValueError('ACTION_TOKENIZATION_CHANGED')
    token_ids = [tokens[0] for tokens in action_tokens]
    token_to_action = {token: action for action, token in enumerate(token_ids)}
    cfg = transformers.AutoConfig.from_pretrained(str(u.MODEL), local_files_only=True)
    cfg.mm_vision_tower = str(u.MODEL / 'siglip-so400m-patch14-384')
    cfg.vision_tower = cfg.mm_vision_tower
    model, loading = official.StreamVLNForCausalLM.from_pretrained(str(u.MODEL), config=cfg,
        attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False, local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('BASE_LOADING_MISMATCH')
    model.model.num_history = 8
    model.requires_grad_(False)
    model.to(0).eval()
    model.reset(1)
    heads = {'NATIVE': ExecutionAdaptation(cfg.hidden_size, 'CURRENT').cuda().eval()}
    heads['NATIVE'].requires_grad_(False)
    for arm, record in protocol['heads'].items():
        if arm == 'NATIVE':
            raise ValueError('NATIVE_HEAD_MUST_NOT_BE_REPLACED')
        heads[arm] = load_registered_head(record, protocol, 'cuda:0')
        if heads[arm].writer.in_features != cfg.hidden_size:
            raise ValueError('HEAD_BACKBONE_WIDTH_MISMATCH')
    seed_pairs = {}
    for seed in protocol['seeds']:
        pair = {}
        for mode in ('CURRENT', 'DELTA'):
            matches = [arm for arm, item in protocol['heads'].items() if item['seed'] == seed and item['mode'] == mode]
            if len(matches) != 1:
                raise ValueError('INCOMPLETE_REGISTERED_SEED_PAIR')
            pair[mode] = matches[0]
        seed_pairs[str(seed)] = pair
    if len(protocol['heads']) != 2 * len(seed_pairs):
        raise ValueError('EXTRA_UNREGISTERED_HEAD')
    header = assistant_header(tokenizer)

    def base_hash():
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode()); digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            raw = value.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw))
        return digest.hexdigest()

    def head_hashes():
        return {arm: {key: tensor_hash(value) for key, value in head.state_dict().items()}
                for arm, head in heads.items()}

    frozen, frozen_heads = base_hash(), head_hashes()
    if frozen != protocol['expected_base_state_sha256']:
        raise ValueError('LOADED_BASE_IDENTITY_CHANGED')
    u.write(output / 'RUNTIME_IDENTITY.json', dict(model_loaded=True, base_state_sha256=frozen,
        executor_binding=binding, executor_binding_sha256=u.sha(run / 'EXECUTOR_BINDING.json'),
        source_lock_sha256=u.sha(run / 'SOURCE_LOCK.json'), protocol_sha256=u.sha(run / 'PROTOCOL.json'),
        source_commit=protocol['source_commit'], heads=protocol['heads'], seed_pairs=seed_pairs,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, habitat_sim=habitat_sim.__version__),
        imports={k: str(Path(m.__file__).resolve()) for k, m in [('torch', torch), ('habitat', habitat), ('official', official)]},
        dtype='bfloat16_base/float32_head_and_scores', attention='flash_attention_2',
        gpu=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unavailable')),
        torch_cuda_version=torch.version.cuda,
        action_token_ids=token_ids, loading_info=loading, loaded_seconds=time.time()-began,
        base_updates=0, action_scope='ALL_ACTION_TOKENS_QUERY_START_MEMORY',
        full_vocabulary_EOS_and_empty_output_fallback='UNCHANGED_UPSTREAM',
        teacher_or_privileged_policy_inputs=False))

    active = {}
    original_env, original_generate = official.Env, model.generate

    def array_hash(value):
        value = np.ascontiguousarray(value)
        digest = hashlib.sha256(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
        return digest.hexdigest()

    def emit(row):
        active['log'].write(json.dumps(row, allow_nan=False) + '\n')
        active['log'].flush()

    def observe(rgb, previous_action):
        tick = time.time()
        evidence = active['runtime'].observe(rgb, previous_action)
        torch.cuda.synchronize()
        emit(dict(event='memory_write', environment_step=active['steps'], seconds=time.time()-tick, **evidence))

    class LoggedEnv(original_env):
        def reset(self, *args, **kwargs):
            obs = super().reset(*args, **kwargs)
            active['env'] = self
            active['steps'] = 0
            active['rgb'] = array_hash(obs['rgb'])
            active['pending_actions'] = deque()
            if self.current_episode.instruction.instruction_text != active['instruction']:
                raise ValueError('INSTRUCTION_MISMATCH')
            active['runtime'] = LiveRuntime(model, tokenizer, active['helper'].image_processor,
                heads[active['arm']], token_ids, active['instruction'])
            emit(dict(event='reset', rgb_sha256=active['rgb'],
                instruction_sha256=hashlib.sha256(active['instruction'].encode()).hexdigest()))
            observe(obs['rgb'], None)
            return obs

        def step(self, action, *args, **kwargs):
            if active['steps'] >= 500 or not active['pending_actions']:
                raise ValueError('BUDGET_OR_GENERATION_EXECUTION_MISMATCH')
            expected, offset = active['pending_actions'].popleft()
            if int(action) != expected:
                raise ValueError('GENERATED_ACTION_NOT_EXECUTED')
            before = active['rgb']
            obs = super().step(action, *args, **kwargs)
            active['steps'] += 1
            active['rgb'] = array_hash(obs['rgb'])
            metrics = self.get_metrics()
            emit(dict(event='action', step=active['steps'], executed_action=int(action),
                before_rgb_sha256=before, after_rgb_sha256=active['rgb'],
                query_environment_step=active['query_start'], token_offset=offset,
                distance=float(metrics['distance_to_goal']),
                collision=bool(metrics.get('collisions', {}).get('is_collision', False))))
            if int(action) == 0:
                active['runtime'].mark_stopped()
            elif not self.episode_over:
                observe(obs['rgb'], int(action))
            return obs

    def feature_hook(module, args, result):
        active['last_feature'] = result.last_hidden_state[:, -1].detach().float()
        active['forward_serial'] = active.get('forward_serial', 0) + 1

    def real_actor_feature():
        serial = active['forward_serial']
        if serial <= active['consumed_actor_serial']:
            raise ValueError('NO_NEW_BACKBONE_FORWARD_FOR_ACTION_TOKEN')
        active['consumed_actor_serial'] = serial
        active['actor_forward_serials'].append(serial)
        return active['last_feature']

    def logged_generate(*args, **kwargs):
        if active['pending_actions']:
            raise ValueError('NEW_QUERY_BEFORE_GENERATED_ACTIONS_CONSUMED')
        runtime = active['runtime']
        query_start = active['steps']
        if runtime.writes != query_start + 1:
            raise ValueError('PHYSICAL_OBSERVATION_COUNT_MISMATCH')
        evidence = {name: dict(shape=list(kwargs[name].shape), dtype=str(kwargs[name].dtype),
                    sha256=tensor_hash(kwargs[name])) for name in ('inputs', 'images', 'depths', 'poses', 'intrinsics')}
        evidence.update(time_ids=kwargs['time_ids'], rgb_sha256=active['rgb'])
        runtime.begin_query()
        active['last_feature'] = None
        active.update(forward_serial=0, consumed_actor_serial=0, actor_forward_serials=[])
        processor = ChunkActionProcessor(runtime, header, token_ids,
            real_actor_feature, apply_residual=active['arm'] != 'NATIVE')
        kwargs['logits_processor'] = make_logits_processors(processor)
        tick = time.time()
        try:
            generated = original_generate(*args, **kwargs)
            torch.cuda.synchronize()
        finally:
            runtime.end_query()
        if active['steps'] != query_start:
            raise ValueError('PHYSICAL_STEP_DURING_GENERATION')
        tokens = generated.sequences[0].detach().cpu().tolist()
        if tokens[:len(header)] != header:
            raise ValueError('GENERATED_HEADER_CHANGED')
        body = tokens[len(header):]
        if len(body) != len(processor.records) or len(body) != len(active['actor_forward_serials']):
            raise ValueError('MISSING_GENERATED_TOKEN_EVIDENCE')
        for offset, token in enumerate(body):
            record = processor.records[offset]
            if token != record['method_token']:
                raise ValueError('METHOD_TOKEN_NOT_GENERATED')
            record['generated_token'] = token
            record['base_forward_serial'] = active['actor_forward_serials'][offset]
        action_tokens_actual = [(token_to_action[token], offset) for offset, token in enumerate(body) if token in token_to_action]
        if len(action_tokens_actual) > 4:
            raise ValueError('UNREGISTERED_FIFTH_ACTION_TOKEN')
        text = tokenizer.batch_decode(generated.sequences, skip_special_tokens=False)[0].strip()
        parsed = active['helper'].parse_actions(text)
        if parsed != [action for action, _ in action_tokens_actual]:
            raise ValueError('UPSTREAM_ACTION_PARSER_TOKEN_MISMATCH')
        fallback = not parsed
        active['pending_actions'] = deque(action_tokens_actual if parsed else [(0, None)])
        active['query_start'] = query_start
        emit(dict(event='generation', environment_step=query_start, input=evidence,
            header_ids=header, generated_ids=tokens, tokens=processor.records,
            parsed_actions=parsed, empty_action_fallback=fallback,
            seconds=time.time()-tick, memory_writes=runtime.writes,
            unused_generated_actions_are_not_executed_history=True))
        return generated

    official.Env = LoggedEnv
    model.generate = logged_generate
    hook = model.model.register_forward_hook(feature_hook)
    completed = []
    try:
        registered = list(protocol['heads'])
        for index in ids:
            episode = manifest[index]
            order = ['NATIVE'] + registered[index % len(registered):] + registered[:index % len(registered)]
            group = output / 'episodes' / str(index)
            group.mkdir(parents=True, exist_ok=False)
            outcomes, traces = {}, {}
            for arm in order:
                random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
                path = group / arm
                path.mkdir()
                with (path / 'TRACE.jsonl').open('x') as log:
                    active.clear()
                    active.update(arm=arm, log=log, instruction=episode['instruction'], steps=0)
                    evaluator = official.VLNEvaluator(episode['config'], split='val_unseen', env_num=1,
                        output_path=str(path), model=model, tokenizer=tokenizer, epoch=0,
                        args=argparse.Namespace(save_video=False, num_frames=32, num_future_steps=4, num_history=8))
                    active['helper'] = evaluator
                    try:
                        with torch.inference_mode():
                            evaluator.eval_action(0)
                    finally:
                        if 'env' in active:
                            active['env'].close()
                result_rows = [json.loads(line) for line in (path / 'result.json').read_text().splitlines()]
                if len(result_rows) != 1 or str(result_rows[0]['episode_id']) != str(episode['episode_id']):
                    raise ValueError('EVALUATOR_EPISODE_MISMATCH')
                if result_rows[0]['steps'] != active['steps'] or active['steps'] > 500:
                    raise ValueError('EVALUATOR_ACTION_BUDGET_MISMATCH')
                outcomes[arm] = result_rows[0]
                traces[arm] = u.sha(path / 'TRACE.jsonl')
                u.write(output / 'PROGRESS.json', dict(status='RUNNING', complete=completed, planned=ids,
                    current_id=index, current_arm=arm, model_loaded=True, action_scope='ALL_ACTION_TOKENS_QUERY_START_MEMORY'))
            logs = {arm: [json.loads(line) for line in (group / arm / 'TRACE.jsonl').read_text().splitlines()] for arm in order}
            audits, paired = {}, {}
            try:
                for arm in order[1:]:
                    audits[arm] = audit_pair(logs['NATIVE'], logs[arm], outcomes['NATIVE'], outcomes[arm],
                        action_token_ids=token_ids, eos_token_id=tokenizer.eos_token_id)
                for seed, pair in seed_pairs.items():
                    left, right = pair['CURRENT'], pair['DELTA']
                    paired[seed] = audit_pair(logs[left], logs[right], outcomes[left], outcomes[right],
                        action_token_ids=token_ids, eos_token_id=tokenizer.eos_token_id)
            except ValueError as error:
                u.write(group / 'FIRST_DIVERGENCE.json', dict(error=str(error), completed_arms=order,
                    trace_paths={arm: str(group / arm / 'TRACE.jsonl') for arm in order}))
                raise
            u.write(group / 'COMPLETE.json', dict(id=index, house=episode['house'], outcomes=outcomes,
                trace_hashes=traces, audits=audits, seed_pair_audits=paired,
                runtime_identity_sha256=u.sha(output / 'RUNTIME_IDENTITY.json'),
                action_scope='ALL_ACTION_TOKENS_QUERY_START_MEMORY'))
            completed.append(index)
            u.write(output / 'PROGRESS.json', dict(status='RUNNING', complete=completed, planned=ids, model_loaded=True))
        after, head_after = base_hash(), head_hashes()
        if after != frozen or head_after != frozen_heads:
            raise ValueError('BASE_OR_HEAD_STATE_CHANGED')
        u.write(output / 'STATE_SEAL.json', dict(base_before=frozen, base_after=after,
            heads_unchanged=True, head_states=head_after))
        u.write(output / 'RESULT.json', dict(status='COMPLETE', complete=completed, planned=ids,
            base_unchanged=True, heads_unchanged=True, base_updates=0, wall_seconds=time.time()-began,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    finally:
        hook.remove()
        official.Env = original_env
        model.generate = original_generate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', required=True)
    args = parser.parse_args()
    main(args.run, args.output, [int(value) for value in args.ids.split(',')])
