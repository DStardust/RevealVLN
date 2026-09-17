"""Ordinary baseline v3 training: real multi-sample batches, optional 4-rank DDP.

Loss reduction (registered): per-rank weighted CE sum scaled by world/W_global so
DDP gradient averaging equals the global weighted mean ((sum w*ce)/sum w). Inflection
weights come frozen from the sample index. No memory module; samples are causally
self-contained. Budgets are cumulative across resume, with an explicit legacy
tail reserve, and enforced at shared per-update boundaries.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
SPEED = LINE / 'sft_acceptance/ordinary_speedup_10x_v1'
FLA_DEPS = SPEED / 'official_fla_0_5_2/deps'
EINOPS_DEPS = SPEED / 'official_einops_0_8_1/deps'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value, exclusive=False):
    blob = json.dumps(value, indent=2, allow_nan=False)
    if exclusive:
        with Path(path).open('x') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('w') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)


def verify_code(protocol):
    for name, digest in protocol['code_sha256'].items():
        path = HERE / name
        require(path.is_file() and sha256(path) == digest, 'CODE_CHANGED:' + name)


def cosine_warmup(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    # Variable-length batches fragment the default caching allocator (measured:
    # reserved peaks + shape variance OOMed well below the 26 GiB rank budget).
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    protocol = json.loads(args.protocol.read_text())
    require(protocol['id'] == 'Q35N_ORDINARY_SYNC_RECOVERY_V1', 'PROTOCOL_ID')
    verify_code(protocol)
    require(os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1',
            'OFFLINE_FLAGS')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS_CONFIG')

    kernel = protocol['kernel']
    if kernel == 'fla':
        sys.path.insert(0, str(EINOPS_DEPS))
        sys.path.insert(0, str(FLA_DEPS))
        import fla.ops.gated_delta_rule  # noqa: F401  materialize internal path first
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    modeling = importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
    bound = modeling.torch_chunk_gated_delta_rule
    cells = dict(zip(bound.__code__.co_freevars, (c.cell_contents for c in bound.__closure__)))
    resolved = getattr(cells.get('implementation'), '__module__', '') or ''
    if kernel == 'fla':
        require(resolved.startswith('fla.'), 'FLA_NOT_RESOLVED:' + resolved)
    else:
        require(resolved.endswith('modeling_qwen3_5'), 'FLA_LEAKED:' + resolved)

    data = load_module('q35n_v3_data', HERE / 'data.py')
    model = load_module('q35n_v3_model', HERE / 'model.py')
    control = load_module('q35n_sync_control', HERE / 'control.py')

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    require(world == protocol['world_size'], 'WORLD_SIZE_PROTOCOL')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        torch.cuda.set_device(local_rank)
        import datetime
        torch.distributed.init_process_group('nccl', timeout=datetime.timedelta(seconds=180),
                                             init_method=Path(os.environ['Q35N_STORE']).as_uri(),
                                             rank=rank, world_size=world)
    device = torch.device('cuda', local_rank)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(
        protocol['budget']['max_gpu_memory_bytes_per_rank'] / total_memory, device)

    rows, report = data.load_rows()
    require(report['training_index_sha256'] == protocol['snapshot']['training_index_sha256'],
            'SNAPSHOT_BINDING')
    total_decisions = sum(r['decisions'] for r in rows)
    samples = data.load_sample_index(HERE.parent / 'ordinary_baseline_v3/SAMPLE_INDEX.jsonl',
                                     protocol['sample_index_sha256'], total_decisions)
    policy = model.build_policy(protocol['seed'])
    policy.to(device)
    policy.train()
    if world > 1:
        model.broadcast_params(policy, 0)
    # Manual gradient averaging replaces DDP hooks: the all-reduce runs only
    # after backward is fully complete (no hook/stream interleaving), and is
    # exactly the registered per-rank * world / W_global reduction.
    raw_policy = policy

    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=protocol['optimizer']['learning_rate'],
                            weight_decay=protocol['optimizer']['weight_decay'],
                            betas=tuple(protocol['optimizer']['betas']), eps=protocol['optimizer']['eps'])

    plans = [data.plan_epoch_batches(samples, protocol['batching']['max_tokens'], protocol['seed'],
                                     epoch, world)
             for epoch in range(protocol['epochs'])]
    for p in plans:
        require(len({len(x) for x in p}) == 1, 'UNEVEN_RANK_SHARDS')
    total_updates = sum(len(p[rank]) for p in plans)
    warmup = max(1, int(0.03 * total_updates))

    cursor = dict(epoch=0, position=0, updates=0, decisions=0)  # decisions: this rank
    if args.resume:
        receipt = json.loads(Path(str(args.resume) + '.json').read_text())
        require(sha256(args.resume) == receipt['sha256'], 'RESUME_RECEIPT_HASH')
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        require(state['cursor'] == receipt['cursor'], 'RECEIPT_CURSOR_MISMATCH')
        require(state['binding']['sample_index_sha256'] == protocol['sample_index_sha256'],
                'RESUME_DATA_BINDING')
        if 'resume_from' in protocol:
            reg = protocol['resume_from']
            if sha256(args.resume) != reg['sha256']:
                # Same-run continuation past the registered start: accept a
                # self-produced checkpoint with a valid receipt at a later cursor.
                receipt = json.loads(Path(str(args.resume) + '.json').read_text())
                require(sha256(args.resume) == receipt['sha256'], 'RESUME_RECEIPT_HASH')
                require(receipt['cursor']['updates'] >= reg['updates'],
                        'RESUME_BEFORE_REGISTERED_START')
        model.load_trainable(raw_policy, state['trainable'])
        opt.load_state_dict(copy.deepcopy(state['optimizer']))
        cursor.update(state['cursor'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng'])

    # Old checkpoints saved rank-0's local count; loading it on every rank
    # corrupts counters, not the common batch position. Rebuild from frozen plan.
    old_local_decisions = cursor['decisions']
    cursor['decisions'] = control.processed_by_rank(plans, cursor, rank)
    global_decisions = sum(control.processed_by_rank(plans, cursor, r) for r in range(world))
    start_global_decisions = global_decisions
    start_updates = cursor['updates']
    accounting = json.loads((args.run_dir / 'ATTEMPT_ACCOUNTING.json').read_text())
    require(accounting['resume_sha256'] == sha256(args.resume), 'ACCOUNTING_RESUME_BINDING')
    charge_base = accounting['charged_base_decisions']

    out = args.run_dir
    if not args.resume:
        if rank == 0:
            out.mkdir(parents=True, exist_ok=False)
        if world > 1:
            torch.distributed.barrier()
        require(out.is_dir(), 'RUN_DIR_MISSING')
    else:
        require(out.is_dir(), 'RESUME_OUTPUT_MISSING')
    store = data.SampleStore(rows)
    collate = model.make_collate(
        pad_id=raw_policy.processor.tokenizer.pad_token_id,
        exec_sid=raw_policy.exec_sid, query_sid=raw_policy.query_sid,
        image_token_id=raw_policy.base.config.image_token_id)
    budget = protocol['budget']
    started = time.monotonic()
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stop_requested.append('SIGINT'))
    stats = dict(weighted_ce=0.0, weight_sum=0.0, decisions=0,
                 confusion=[[0] * 4 for _ in range(4)])
    last_report = started
    latest_checkpoint = str(args.resume) if args.resume else None
    last_saved_updates = -1
    over_budget = []
    agreed_stop = 0
    print(json.dumps(dict(event='RESUME_READY', rank=rank, cursor=cursor,
                         old_rank0_decisions=old_local_decisions,
                         global_plan_decisions=global_decisions, charged_base=charge_base)), flush=True)

    def charged():
        return control.charged_decisions(charge_base, global_decisions - start_global_decisions)

    def sync_stop():
        mask = control.stop_mask(stop_requested, time.time(), accounting['deadline_unix'],
                                 cursor['updates'], budget['max_updates'], charged(),
                                 budget['max_decisions'])
        return control.agree_stop(torch, mask, device, world)

    def save_checkpoint():
        path = out / ('checkpoint_%09d.pt' % cursor['updates'])
        require(not path.exists(), 'CHECKPOINT_EXISTS')
        value = dict(binding=dict(protocol_sha256=sha256(args.protocol),
                                  sample_index_sha256=protocol['sample_index_sha256']),
                     cursor=dict(cursor), global_decisions=global_decisions,
                     charged_compute_decisions=charged(),
                     trainable=model.trainable_state(raw_policy),
                     optimizer=opt.state_dict(),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                     python_rng=random.getstate())
        tmp = path.with_suffix('.pt.tmp')
        with tmp.open('xb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        write_json(Path(str(path) + '.json'), dict(sha256=sha256(path), cursor=cursor,
                                                   global_decisions=global_decisions,
                                                   charged_compute_decisions=charged(),
                                                   unix=time.time()), exclusive=True)
        return str(path)

    def all_reduce_sum(value):
        tensor = torch.tensor([value], dtype=torch.float64, device=device)
        if world > 1:
            torch.distributed.all_reduce(tensor)
        return float(tensor.item())

    while cursor['epoch'] < protocol['epochs']:
        agreed_stop = sync_stop()
        if agreed_stop:
            break
        epoch, position = cursor['epoch'], cursor['position']
        shard = plans[epoch][rank][position:]
        dataset = model.DecisionDataset(samples, store, raw_policy.processor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=iter(shard), num_workers=protocol['data']['num_workers'],
            prefetch_factor=protocol['data']['prefetch_factor'], persistent_workers=False,
            pin_memory=True, collate_fn=collate)
        for batch in loader:
            agreed_stop = sync_stop()
            if agreed_stop:
                break
            batch = {k: (v.to(device, non_blocking=True) if hasattr(v, 'to') else v)
                     for k, v in batch.items()}
            logits = policy(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                            mm_token_type_ids=batch['mm_token_type_ids'],
                            image_grid_thw=batch['image_grid_thw'], pixel_values=batch['pixel_values'],
                            exec_index=batch['exec_index'], action_index=batch['action_index'])
            ce = torch.nn.functional.cross_entropy(logits, batch['targets'], reduction='none')
            totals = torch.tensor([float(batch['weights'].sum()), int(batch['targets'].shape[0])],
                                  dtype=torch.float64, device=device)
            if world > 1:
                torch.distributed.all_reduce(totals)
            weight_sum_global, batch_decisions_global = totals.tolist()
            loss = (ce * batch['weights']).sum() * world / weight_sum_global
            loss.backward()
            model.sync_grads(policy, world)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, protocol['optimizer']['gradient_clip'],
                                                       error_if_nonfinite=True)
            require(float(grad_norm) > 0, 'ZERO_GRADIENT')
            for group in opt.param_groups:
                group['lr'] = cosine_warmup(cursor['updates'], total_updates, warmup,
                                            protocol['optimizer']['learning_rate'])
            opt.step()
            opt.zero_grad(set_to_none=True)
            cursor['updates'] += 1
            batch_decisions = int(batch['targets'].shape[0])
            cursor['decisions'] += batch_decisions
            cursor['position'] += 1
            global_decisions += int(batch_decisions_global)
            require(torch.cuda.max_memory_reserved(device) <= budget['max_gpu_memory_bytes_per_rank'],
                    'GPU_MEMORY_BUDGET')
            with torch.no_grad():
                predictions = logits.argmax(-1)
                for t, p_ in zip(batch['targets'].tolist(), predictions.tolist()):
                    stats['confusion'][t][p_] += 1
            stats['weighted_ce'] += float((ce * batch['weights']).sum().detach())
            stats['weight_sum'] += float(batch['weights'].sum())
            stats['decisions'] += batch_decisions
            now = time.monotonic()
            agreed_stop = sync_stop()
            end_of_epoch = cursor['position'] == len(plans[epoch][rank])
            due = control.report_due(cursor['updates'], protocol['log_every_updates'],
                                     end_of_epoch, agreed_stop)
            if due:
                packed = torch.tensor(
                    [stats['weighted_ce'], stats['weight_sum'], stats['decisions'], cursor['decisions']],
                    dtype=torch.float64, device=device)
                if world > 1:
                    torch.distributed.all_reduce(packed)
                confusion_t = torch.tensor(stats['confusion'], dtype=torch.float64, device=device)
                if world > 1:
                    torch.distributed.all_reduce(confusion_t)
                gce, gw, window_decisions, reduced_plan_decisions = packed.tolist()
                require(int(reduced_plan_decisions) == global_decisions, 'RANK_COUNTER_MISMATCH')
                gconf = confusion_t.cpu().tolist()
                if rank == 0:
                    support = [sum(r) for r in gconf]
                    total = max(1.0, sum(support))
                    event = dict(status='STOPPING' if agreed_stop else 'TRAINING', unix=time.time(), cursor=dict(cursor),
                                 metrics=dict(mean_ce=gce / max(gw, 1e-12),
                                              accuracy=sum(gconf[i][i] for i in range(4)) / total,
                                              action_recall=[gconf[i][i] / support[i] if support[i] else None
                                                             for i in range(4)],
                                              action_target_counts=[int(s) for s in support],
                                              confusion=[[int(v) for v in r] for r in gconf],
                                              grad_norm=float(grad_norm),
                                              lr=opt.param_groups[0]['lr']),
                                 throughput=window_decisions / max(.001, now - last_report),
                                 session_throughput=(global_decisions - start_global_decisions) / max(.001, now - started),
                                 global_plan_decisions=global_decisions,
                                 budget=budget,
                                 cumulative_compute=dict(decisions=charged(),
                                     legacy_charge_includes_conservative_reserve=True,
                                     this_attempt_decisions=global_decisions-start_global_decisions),
                                 wall_deadline_unix=accounting['deadline_unix'],
                                 resume_updates=start_updates,
                                 latest_checkpoint=latest_checkpoint,
                                 evaluation_performed=False, navigation_gain_verified=False)
                    write_json(out / 'PROGRESS.json', event)
                    with (out / 'PROGRESS.jsonl').open('a') as stream:
                        stream.write(json.dumps(event, allow_nan=False) + '\n')
                    print(json.dumps(dict(updates=cursor['updates'],
                                          decisions=charged(),
                                          ce=event['metrics']['mean_ce'],
                                          throughput=round(event['throughput'], 3))), flush=True)
                last_report = now
                stats = dict(weighted_ce=0.0, weight_sum=0.0, decisions=0,
                             confusion=[[0] * 4 for _ in range(4)])
            if rank == 0 and cursor['updates'] % protocol['checkpoint_updates'] == 0 \
                    and last_saved_updates != cursor['updates']:
                latest_checkpoint = save_checkpoint()
                last_saved_updates = cursor['updates']
            if agreed_stop:
                break
        # Epoch boundary: advance deterministically (identical on all ranks);
        # without this the while loop re-creates empty loaders forever.
        if agreed_stop:
            break
        new_cursor = data.advance_epoch_boundary(cursor, len(plans[cursor['epoch']][rank]))
        if new_cursor['epoch'] != cursor['epoch']:
            cursor = new_cursor
            if rank == 0 and last_saved_updates != cursor['updates']:
                latest_checkpoint = save_checkpoint()
                last_saved_updates = cursor['updates']
    # graceful stop: all ranks reached this point via the collective stop flag
    if rank == 0 and cursor['updates'] and last_saved_updates != cursor['updates']:
        latest_checkpoint = save_checkpoint()
        last_saved_updates = cursor['updates']
    if rank == 0:
        status = 'EPOCHS_COMPLETED' if cursor['epoch'] >= protocol['epochs'] else 'STOPPED'
        result = dict(status=status, cursor=cursor,
                      global_decisions=global_decisions,
                      charged_compute_decisions=charged(),
                      latest_checkpoint=latest_checkpoint, unix=time.time(),
                      stop=control.stop_reasons(agreed_stop),
                      navigation_gain_verified=False)
        result_path = out / 'RESULT.json'
        if result_path.exists():
            k = 1
            while (out / ('RESULT_CONTINUATION_%d.json' % k)).exists():
                k += 1
            result_path = out / ('RESULT_CONTINUATION_%d.json' % k)
        with result_path.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    # Hard exit after all durable writes: DataLoader workers forked after CUDA
    # init can linger holding the context, and NCCL destroy teardown aborted
    # under the watchdog in acceptance (SIGABRT after results were safe).
    if world > 1:
        torch.distributed.barrier()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os._exit(0)


if __name__ == '__main__':
    main()
