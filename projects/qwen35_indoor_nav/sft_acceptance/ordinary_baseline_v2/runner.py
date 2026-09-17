"""Preparation-only ordinary CE candidate. Import/preflight never imports torch.

The command line has no GPU probe or admission override. A separately reviewed
RUN_ADMISSION.json and RUN_ADMITTED protocol are mandatory before model loading.
No GPU transport, placeholder management, or production-queue mutation is here.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
OLD_POLICY = HERE.parent / 'v1/recovery_r1/policy.py'
ACTIONS = ['move_forward', 'turn_left', 'turn_right', 'STOP']


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def scoped(path, root=LINE):
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()), 'PATH_OUTSIDE_LINE')
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration(protocol):
    config = protocol['training']
    for key in ('epochs', 'chunk_steps', 'seed', 'gradient_accumulation_chunks', 'max_sequence_tokens'):
        require(type(config[key]) is int, 'INTEGER_TRAINING_CONFIG:' + key)
    require(config['epochs'] > 0 and config['chunk_steps'] == 4, 'INVALID_EPOCH_OR_CHUNK')
    require(config['gradient_accumulation_chunks'] > 0, 'INVALID_ACCUMULATION')
    require(config['max_sequence_tokens'] > 0, 'INVALID_SEQUENCE_LIMIT')
    require(config.get('loss') == 'unweighted_action_ce' and config.get('schedule') == 'constant', 'UNSUPPORTED_LOSS_OR_SCHEDULE')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip'):
        value = config[key]
        require(isinstance(value, (int, float)) and math.isfinite(value), 'NONFINITE_CONFIG')
        require(value > 0 if key != 'weight_decay' else value >= 0, 'INVALID_OPTIMIZER_CONFIG')
    return config


def verify_code(protocol):
    seal_path = scoped(protocol.get('code_seal_path', HERE / 'CODE_SEAL.json'))
    require(sha256(seal_path) == protocol.get('code_seal_sha256'), 'CODE_SEAL_HASH_MISMATCH')
    codes = read_json(seal_path)
    codes = codes.get('files', codes)
    required = {HERE / 'runner.py', HERE / 'data.py', OLD_POLICY,
                LINE / 'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json'}
    resolved = {scoped(HERE / key if not Path(key).is_absolute() else key): value
                for key, value in codes.items()}
    resolved.update({scoped(LINE / key): value for key, value in protocol.get('readonly_dependency_sha256', {}).items()})
    require(required.issubset(resolved), 'MISSING_CODE_LOCK')
    for path, expected in resolved.items():
        require(sha256(path) == expected, 'CODE_HASH_MISMATCH:' + str(path))
    return {str(path): digest for path, digest in resolved.items()}


def preflight(protocol_path, snapshot_dir, full_source_check=False):
    protocol_path = scoped(protocol_path)
    snapshot_dir = scoped(snapshot_dir)
    protocol = read_json(protocol_path)
    config = configuration(protocol)
    codes = verify_code(protocol)
    manifest_path = snapshot_dir / 'RESULT.json'
    manifest = read_json(manifest_path)
    seal_path = snapshot_dir / 'SEAL.json'
    seal = read_json(seal_path)
    require({'RESULT.json', 'TRAINING_INDEX.jsonl', 'SOURCE_HASHES.json', 'SPLIT.json'}.issubset(seal), 'SNAPSHOT_SEAL_INCOMPLETE')
    for name, digest in seal.items():
        require(sha256(scoped(snapshot_dir / name, snapshot_dir)) == digest, 'SNAPSHOT_SEAL_MISMATCH:' + name)
    index = snapshot_dir / 'TRAINING_INDEX.jsonl'
    expected = manifest.get('training_index_sha256')
    if expected is None:
        expected = manifest.get('files', {}).get('TRAINING_INDEX.jsonl')
        if isinstance(expected, dict):
            expected = expected.get('sha256')
    require(expected is not None and sha256(index) == expected, 'INDEX_HASH_MISMATCH')
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    require(rows, 'EMPTY_TRAINING_INDEX')
    split = read_json(snapshot_dir / 'SPLIT.json')
    fit = set(split['FIT'])
    dev = set(split['INTERNAL_DEV'])
    confirm = set(split['INTERNAL_CONFIRM'])
    require(not fit & (dev | confirm) and not dev & confirm, 'SNAPSHOT_SPLIT_OVERLAP')
    sources = read_json(snapshot_dir / 'SOURCE_HASHES.json')
    required = {'sourceRoot', 'policy_file', 'supervision_file', 'rgb_reference_root',
                'source', 'scene_group', 'physical_source_route_sha256', 'decisions',
                'record_id', 'policy_sha256', 'supervision_sha256'}
    seen = set()
    for row in rows:
        require(required.issubset(row), 'INCOMPLETE_INDEX_ROW')
        require(type(row['decisions']) is int and row['decisions'] > 0, 'INVALID_DECISIONS')
        require(row['record_id'] not in seen, 'DUPLICATE_RECORD_ID')
        seen.add(row['record_id'])
        scoped(ROOT / row['sourceRoot'])
        require(row.get('split') == 'FIT', 'NON_FIT_TRAINING_ROW')
        require(row['scene_group'] in fit, 'TRAINING_HOUSE_OUTSIDE_FIT')
        for field in ('policy', 'supervision'):
            source_path = scoped(ROOT / row['sourceRoot'] / row[field + '_file'])
            key = str(source_path.relative_to(ROOT))
            require(sources.get(key) == row[field + '_sha256'], 'ROW_SOURCE_HASH_BINDING:' + key)
    decisions = sum(row['decisions'] for row in rows)
    physical = {(row['scene_group'], row['physical_source_route_sha256']) for row in rows}
    counts = manifest['counts']
    require(counts['instruction_records'] == len(rows) and counts['instruction_conditioned_decisions'] == decisions
            and counts['strict_routes'] == len(physical), 'SNAPSHOT_COUNTS_MISMATCH')
    if full_source_check:
        for key, digest in sources.items():
            require(sha256(scoped(ROOT / key)) == digest, 'SOURCE_METADATA_HASH_MISMATCH:' + key)
    report = dict(status='CPU_PREFLIGHT_ONLY', protocol_sha256=sha256(protocol_path),
                  snapshot_result_sha256=sha256(manifest_path), snapshot_seal_sha256=sha256(seal_path), training_index_sha256=expected,
                  code_sha256=codes, records=len(rows), decisions_per_epoch=decisions,
                  epochs=config['epochs'], planned_decisions=decisions * config['epochs'],
                  updates_per_epoch=math.ceil(sum((r['decisions'] + 3) // 4 for r in rows) / config['gradient_accumulation_chunks']),
                  gpu_run_allowed=protocol.get('gpu_run_allowed') is True,
                  real_gpu_interface_verified=False, rgb_full_scan_performed=False)
    report['source_metadata_full_recheck_performed'] = full_source_check
    report['formal_fullscale_launch_ready'] = False
    return protocol, rows, report


def admission(protocol, report, admission_path):
    require(protocol.get('gpu_run_allowed') is True and protocol.get('status') == 'RUN_ADMITTED',
            'GPU_RUN_NOT_ADMITTED: preparation protocol forbids training')
    path = scoped(admission_path)
    require(path.name == 'RUN_ADMISSION.json', 'EXACT_ADMISSION_FILENAME_REQUIRED')
    permit = read_json(path)
    for key in ('protocol_sha256', 'snapshot_result_sha256', 'snapshot_seal_sha256', 'training_index_sha256', 'code_sha256'):
        require(permit.get(key) == report[key], 'ADMISSION_BINDING_MISMATCH:' + key)
    require(permit.get('status') == 'APPROVED', 'ADMISSION_NOT_APPROVED')
    uuid = permit.get('gpu_uuid', '')
    require(uuid.startswith('GPU-') and os.environ.get('CUDA_VISIBLE_DEVICES') == uuid,
            'EXACT_SINGLE_GPU_UUID_REQUIRED')
    budget = permit.get('budget', {})
    require(budget == protocol.get('resource_budget'), 'BUDGET_PROTOCOL_MISMATCH')
    for key in ('wall_seconds', 'max_forward_tokens', 'max_decisions', 'max_gpu_memory_bytes'):
        require(type(budget.get(key)) is int and budget[key] > 0, 'INVALID_BUDGET:' + key)
    require(budget['max_decisions'] >= report['planned_decisions'], 'BUDGET_CANNOT_COVER_EPOCHS')
    require(budget['max_forward_tokens'] >= configuration(protocol)['max_sequence_tokens'] * report['planned_decisions'], 'TOKEN_BUDGET_CANNOT_COVER_EPOCHS')
    evidence_ref = permit.get('interface_evidence', {})
    evidence_path = scoped(evidence_ref.get('path', LINE / 'MISSING_EVIDENCE'))
    require(sha256(evidence_path) == evidence_ref.get('sha256'), 'INTERFACE_EVIDENCE_HASH_MISMATCH')
    evidence = read_json(evidence_path)
    for key in ('protocol_sha256', 'snapshot_result_sha256', 'snapshot_seal_sha256', 'code_sha256'):
        require(evidence.get(key) == report[key], 'INTERFACE_EVIDENCE_BINDING_MISMATCH:' + key)
    require(evidence.get('gpu_uuid') == uuid, 'INTERFACE_GPU_MISMATCH')
    require(all(evidence.get(k) is True for k in (
        'real_rgb_forward_backward_pass', 'checkpoint_resume_pass', 'finite_gradients_pass',
        'token_budget_enforced_pass', 'sequence_limit_pass', 'resource_guard_pass', 'old_forward_equivalence_pass')),
        'REAL_GPU_INTERFACE_NOT_PASSED')
    throughput = evidence.get('measured_decisions_per_second')
    require(isinstance(throughput, (int, float)) and math.isfinite(throughput) and throughput > 0,
            'MISSING_REAL_THROUGHPUT')
    require(report['planned_decisions'] / throughput <= budget['wall_seconds'],
            'MEASURED_THROUGHPUT_EXCEEDS_WALL_BUDGET')
    model_assets = protocol.get('model_asset_sha256', {})
    model_root = LINE / 'runtime/models/Qwen3.5-2B_15852e8'
    require(any(name.endswith('.safetensors') for name in model_assets), 'MODEL_WEIGHT_HASH_REQUIRED')
    for relative, expected in model_assets.items():
        asset = scoped(LINE / relative, model_root)
        require(sha256(asset) == expected, 'MODEL_ASSET_CHANGED:' + relative)
    scoped(permit['output_dir'], HERE)
    return permit


def route_order(rows, seed, epoch):
    order = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(order)
    return order


def initial_cursor():
    return dict(epoch=0, route_position=0, step=0, updates=0, decisions=0, chunks=0)


def train_epochs(rows, config, backend, record_factory, cursor=None, memory=None, checkpoint=None, progress=None):
    """Full-route TBPTT, deterministic route shuffle, resumable at chunk boundaries.

    The backend owns tensors/optimizer. No route is replaced by a random prefix;
    a partial final chunk remains an ordinary optimizer update, including STOP.
    """
    cursor = dict(initial_cursor() if cursor is None else cursor)
    require(0 <= cursor['epoch'] <= config['epochs'], 'INVALID_RESUME_EPOCH')
    epoch_ce_sum = 0.0
    epoch_decisions = 0
    epoch_confusion = [[0] * 4 for _ in ACTIONS]
    ordered_epoch, order = None, None
    while cursor['epoch'] < config['epochs']:
        if ordered_epoch != cursor['epoch']:
            order = route_order(rows, config['seed'], cursor['epoch'])
            ordered_epoch = cursor['epoch']
        require(0 <= cursor['route_position'] < len(order), 'INVALID_RESUME_ROUTE')
        row = rows[order[cursor['route_position']]]
        require(0 <= cursor['step'] < row['decisions'], 'INVALID_RESUME_STEP')
        record = record_factory(row)
        if cursor['step'] == 0:
            memory = backend.zero_memory()
        else:
            require(memory is not None, 'RESUME_MEMORY_REQUIRED')
        while cursor['step'] < row['decisions']:
            start = cursor['step']
            end = min(start + config['chunk_steps'], row['decisions'])
            decisions = [record.decision(t, decode_rgb=True) for t in range(start, end)]
            for t, decision in zip(range(start, end), decisions):
                require(set(decision) == {'control', 'policy', 'supervision'}, 'DECISION_FIELDS')
                require(set(decision['policy']) == {'instruction', 'images', 'executed_actions'}, 'POLICY_WHITELIST')
                require(decision['control']['decision_step'] == t, 'NONCAUSAL_STEP')
                require(decision['control']['memory_reset'] is (t == 0), 'MEMORY_RESET_MISMATCH')
                require(decision['supervision']['target_action'] in ACTIONS, 'INVALID_ACTION')
                require(decision['supervision']['ce_mask'] is True or decision['supervision']['ce_mask'] == 1,
                        'ORDINARY_CE_MASK_REQUIRED')
                require((decision['supervision']['target_action'] == 'STOP') is (t == row['decisions'] - 1),
                        'TERMINAL_STOP_REQUIRED')
            end_of_epoch = end == row['decisions'] and cursor['route_position'] == len(order) - 1
            memory, metrics = backend.train_chunk(decisions, memory, finalize=end_of_epoch)
            epoch_decisions += end - start
            if 'ce' in metrics:
                epoch_ce_sum += metrics['ce'] * (end - start)
            if 'confusion' in metrics:
                for target in range(4):
                    for prediction in range(4):
                        epoch_confusion[target][prediction] += metrics['confusion'][target][prediction]
            memory = backend.detach_memory(memory)
            cursor['step'] = end
            cursor['updates'] += int(metrics['optimizer_step'])
            cursor['chunks'] += 1
            cursor['decisions'] += end - start
            if end == row['decisions']:
                cursor['step'] = 0
                cursor['route_position'] += 1
                memory = None
                if cursor['route_position'] == len(order):
                    cursor['route_position'] = 0
                    cursor['epoch'] += 1
            if checkpoint is not None and end_of_epoch:
                support = [sum(row) for row in epoch_confusion]
                measured = sum(support)
                recall = [epoch_confusion[i][i] / support[i] if support[i] else None for i in range(4)]
                stop_predictions = sum(row[3] for row in epoch_confusion)
                summary = dict(metrics, online_training_epoch_decisions=epoch_decisions,
                               online_training_ce=epoch_ce_sum / epoch_decisions if 'ce' in metrics else None,
                               online_training_confusion=epoch_confusion,
                               online_training_recall=recall,
                               online_training_accuracy=sum(epoch_confusion[i][i] for i in range(4)) / measured if measured else None,
                               online_training_stop_precision=epoch_confusion[3][3] / stop_predictions if stop_predictions else None,
                               evaluation_performed=False)
                checkpoint(dict(cursor), memory, summary)
            if progress is not None and (end_of_epoch or
                    (metrics['optimizer_step'] and (cursor['updates'] == 1 or cursor['updates'] % 100 == 0))):
                progress(dict(cursor), metrics)
            if end_of_epoch:
                epoch_ce_sum, epoch_decisions = 0.0, 0
                epoch_confusion = [[0] * 4 for _ in ACTIONS]
            if end == row['decisions']:
                break
    return cursor


class TorchBackend:
    """Unbenchmarked single-device candidate; constructed only after admission."""
    def __init__(self, config, permit):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.torch = torch
        self.config, self.budget = config, permit['budget']
        self.started = time.monotonic()
        self.elapsed_before_resume = 0.0
        self.decisions = 0
        self.pending_chunks = self.pending_decisions = 0
        require(torch.cuda.device_count() == 1, 'SINGLE_VISIBLE_GPU_REQUIRED')
        gpu = torch.cuda.get_device_properties(0)
        require(str(gpu.uuid) == permit['gpu_uuid'], 'LIVE_GPU_UUID_MISMATCH')
        require(self.budget['max_gpu_memory_bytes'] < gpu.total_memory, 'INVALID_GPU_MEMORY_FRACTION')
        torch.cuda.set_per_process_memory_fraction(self.budget['max_gpu_memory_bytes'] / gpu.total_memory)
        torch.set_num_threads(4)
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        old = load_module('q35n_readonly_recovery_policy', OLD_POLICY)
        self.old = old
        processor = AutoProcessor.from_pretrained(old.MODEL, local_files_only=True, trust_remote_code=False)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            old.MODEL, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0')
        self.policy = old.Policy(base, processor)
        self.policy.train()
        self.parameters = [p for p in self.policy.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.parameters, lr=config['learning_rate'],
                                    weight_decay=config['weight_decay'], betas=(.9, .999), eps=1e-8)

    def zero_memory(self):
        return self.old.zero_memory()

    def detach_memory(self, memory):
        return memory.detach()

    def elapsed(self):
        return self.elapsed_before_resume + time.monotonic() - self.started

    def step(self, inputs, memory):
        """Same old forward formula, explicit per-instance cumulative budget.

        This does not call or alter old.step's 14M hard-coded token counter.
        It has not yet passed the separately required real GPU equivalence gate.
        """
        torch, p = self.torch, self.policy
        require(self.elapsed() < self.budget['wall_seconds'], 'WALL_BUDGET_EXHAUSTED')
        require(self.decisions < self.budget['max_decisions'], 'DECISION_BUDGET_EXHAUSTED')
        images, executed = inputs['images'], inputs['executed_actions']
        require(1 <= len(images) <= 2 and all(im.size == (224, 224) for im in images), 'RGB_INTERFACE')
        require(len(executed) <= 8 and all(a in ACTIONS[:-1] for a in executed), 'EXECUTED_INTERFACE')
        require(tuple(memory.shape) == (1, 8, 2048), 'MEMORY_SHAPE')
        p.mm.rope_deltas = None
        text = p.processor.apply_chat_template([{'role': 'user', 'content': [
            *[{'type': 'image'} for _ in images], {'type': 'text', 'text': inputs['instruction']}]}],
            tokenize=False, add_generation_prompt=True)
        b = p.processor(text=[text], images=images, return_tensors='pt').to('cuda:0')
        base_ids, n = b['input_ids'], b['input_ids'].shape[1]
        positions = (base_ids != p.base.config.image_token_id) & (b['mm_token_type_ids'] == b['mm_token_type_ids'][0, 0])
        require(bool(positions.any()), 'MISSING_TEXT_POSITION')
        text_type = b['mm_token_type_ids'][positions][0].item()
        suffix = [p.sid['<NAV_OLD_MEMORY>']] * 8 + [p.sid[f'<EXEC_{a.upper()}_OK>'] for a in executed] + [p.sid['<NAV_WRITE_QUERY>']] * 8 + [p.sid['<NAV_ACTION_QUERY>']]
        ids = torch.cat([base_ids, torch.tensor([suffix], device='cuda:0')], 1)
        tokens = ids.shape[1]
        require(tokens <= self.config['max_sequence_tokens'], 'TOKEN_SEQUENCE_LIMIT_NO_TRUNCATION')
        require(p.forward_tokens + tokens <= self.budget['max_forward_tokens'], 'FORWARD_TOKEN_BUDGET')
        p.forward_tokens += tokens
        self.decisions += 1
        mask = torch.cat([b['attention_mask'], torch.ones((1, len(suffix)), device='cuda:0', dtype=b['attention_mask'].dtype)], 1)
        types = torch.cat([b['mm_token_type_ids'], torch.full((1, len(suffix)), text_type, device='cuda:0', dtype=b['mm_token_type_ids'].dtype)], 1)
        pos, _ = p.mm.get_rope_index(input_ids=ids, mm_token_type_ids=types, image_grid_thw=b['image_grid_thw'], attention_mask=mask)
        require(tuple(pos.shape) == (3, 1, ids.shape[1]), 'MROPE_SHAPE')
        emb = p.base.get_input_embeddings()(ids)
        with torch.no_grad():
            features = torch.cat(p.mm.get_image_features(b['pixel_values'], b['image_grid_thw'], return_dict=True).pooler_output, 0)
        image_mask, _ = p.mm.get_placeholder_mask(ids, inputs_embeds=emb, image_features=features)
        require(int((ids == p.base.config.image_token_id).sum()) == features.shape[0], 'VISUAL_TOKEN_COUNT')
        emb = emb.masked_scatter(image_mask, features.to(emb.dtype)).clone()
        write_start = n + 8 + len(executed)
        emb[:, n:n + 8] = memory + p.old_slot
        emb[:, write_start:write_start + 8] = p.write_query
        emb[:, -1:] = p.action_query
        out = p.mm.language_model(inputs_embeds=emb, attention_mask=mask, position_ids=pos,
                                  past_key_values=None, use_cache=False, return_dict=True)
        require(out.past_key_values is None and p.mm.rope_deltas is None, 'PERSISTENT_KV_FORBIDDEN')
        memory = p.writer(out.last_hidden_state[:, write_start:write_start + 8].float()).to(torch.bfloat16)
        logits = p.action_head(out.last_hidden_state[:, -1].float())
        require(bool(torch.isfinite(memory).all() and torch.isfinite(logits).all()), 'NONFINITE_FORWARD')
        return memory, logits

    def train_chunk(self, decisions, memory, finalize=False):
        torch = self.torch
        if self.pending_chunks == 0:
            self.opt.zero_grad(set_to_none=True)
        losses, confusion = [], [[0] * 4 for _ in ACTIONS]
        for decision in decisions:
            memory, logits = self.step(decision['policy'], memory)
            target = ACTIONS.index(decision['supervision']['target_action'])
            losses.append(torch.nn.functional.cross_entropy(logits, torch.tensor([target], device='cuda:0')))
            confusion[target][int(logits.detach().argmax(-1).item())] += 1
        loss = torch.stack(losses).sum()
        loss.backward()
        self.pending_chunks += 1
        self.pending_decisions += len(decisions)
        update = finalize or self.pending_chunks == self.config['gradient_accumulation_chunks']
        grad_norm = None
        if update:
            for p in self.parameters:
                if p.grad is not None:
                    p.grad.div_(self.pending_decisions)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config['gradient_clip'], error_if_nonfinite=True)
            require(float(grad_norm) > 0, 'ZERO_GRADIENT')
            self.opt.step()
            require(all(bool(torch.isfinite(p).all()) for p in self.parameters), 'NONFINITE_PARAMETER')
            self.pending_chunks = self.pending_decisions = 0
        require(torch.cuda.max_memory_reserved() <= self.budget['max_gpu_memory_bytes'], 'GPU_MEMORY_BUDGET')
        return memory, dict(ce=float(loss.detach()) / len(decisions), confusion=confusion,
                            optimizer_step=update, grad_norm=float(grad_norm) if grad_norm is not None else None,
                            forward_tokens=self.policy.forward_tokens, elapsed_seconds=self.elapsed())

    def state(self):
        require(self.pending_chunks == 0, 'CHECKPOINT_REQUIRES_FLUSHED_GRADIENTS')
        return dict(trainable=self.old.trainable_state(self.policy), optimizer=self.opt.state_dict(),
                    forward_tokens=self.policy.forward_tokens, decisions=self.decisions,
                    elapsed_seconds=self.elapsed(), torch_rng=self.torch.get_rng_state(),
                    cuda_rng=self.torch.cuda.get_rng_state(), python_rng=random.getstate())

    def restore(self, state):
        self.old.load_trainable(self.policy, state['trainable'])
        self.opt.load_state_dict(state['optimizer'])
        self.policy.forward_tokens = state['forward_tokens']
        self.decisions = state['decisions']
        self.elapsed_before_resume = state['elapsed_seconds']
        self.torch.set_rng_state(state['torch_rng'])
        self.torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--train', action='store_true')
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--full-source-check', action='store_true', help='CPU rehash all locked metadata; no RGB/model decoding')
    args = parser.parse_args(argv)
    protocol, rows, report = preflight(args.protocol, args.snapshot, args.full_source_check)
    if args.preflight:
        require(args.resume is None, 'PREFLIGHT_DOES_NOT_RESTORE_CHECKPOINT')
        print(json.dumps(report, indent=2))
        return report
    require(args.admission is not None, 'RUN_ADMISSION_REQUIRED')
    permit = admission(protocol, report, args.admission)
    out = scoped(permit['output_dir'], HERE)
    binding = dict(protocol_sha256=report['protocol_sha256'], snapshot_result_sha256=report['snapshot_result_sha256'],
                   snapshot_seal_sha256=report['snapshot_seal_sha256'],
                   training_index_sha256=report['training_index_sha256'], code_sha256=report['code_sha256'],
                   admission_sha256=sha256(args.admission))
    checkpoint_state = None
    if args.resume:
        require(out.is_dir(), 'RESUME_OUTPUT_MISSING')
        resume = scoped(args.resume, out)
        receipt = read_json(Path(str(resume) + '.json'))
        require(receipt['binding'] == binding and sha256(resume) == receipt['sha256'], 'CHECKPOINT_BINDING_OR_HASH')
    else:
        require(not out.exists(), 'OUTPUT_ALREADY_EXISTS_USE_REVIEWED_RESUME')
        out.mkdir(parents=True)
    backend = TorchBackend(configuration(protocol), permit)
    if args.resume:
        checkpoint_state = backend.torch.load(resume, map_location='cpu', weights_only=True)
        require(checkpoint_state['binding'] == binding, 'CHECKPOINT_BINDING')
        backend.restore(checkpoint_state['backend'])
    data = load_module('q35n_ordinary_v2_data', HERE / 'data.py')

    def checkpoint(cursor, memory, metrics):
        # Epoch checkpoints preserve failed/partial files. No rotating deletion.
        path = out / ('checkpoint_%09d.pt' % cursor['updates'])
        require(not path.exists(), 'CHECKPOINT_EXISTS')
        value = dict(binding=binding, cursor=cursor, memory=memory.cpu() if memory is not None else None,
                     backend=backend.state())
        with path.open('xb') as stream:
            backend.torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        with Path(str(path) + '.json').open('x') as stream:
            json.dump(dict(binding=binding, sha256=sha256(path), cursor=cursor, metrics=metrics), stream)

    state = checkpoint_state or {}
    memory = state.get('memory')
    if memory is not None:
        memory = memory.to('cuda:0')
    def progress(cursor, metrics):
        event = dict(unix=time.time(), cursor=cursor, last_chunk_metrics=metrics,
                     evaluation_performed=False)
        with (out / 'PROGRESS.jsonl').open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()
        print(json.dumps(event, allow_nan=False), flush=True)

    cursor = train_epochs(rows, configuration(protocol), backend, data.OrdinaryRecord,
                          state.get('cursor'), memory, checkpoint, progress)
    with (out / 'RESULT.json').open('x') as stream:
        json.dump(dict(status='TRAINING_EPOCHS_COMPLETED', binding=binding, cursor=cursor,
                       navigation_gain_verified=False, closed_loop_evaluation_performed=False), stream, indent=2)


if __name__ == '__main__':
    main()
