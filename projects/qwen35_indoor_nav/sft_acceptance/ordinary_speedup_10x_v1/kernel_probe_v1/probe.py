"""Bounded official-FLA kernel probe for the Qwen3.5 Gated DeltaNet path.

Two modes, one process each, same fixed real decision set:
  reference: fla NOT importable -> transformers pure-Torch fallback (current training path)
  candidate: official fla-core 0.5.2 + einops 0.8.1 on sys.path BEFORE transformers import
             -> use_kernel_func_from_hub_with_fallback resolves the official FLA function

The forward/backward path is the sealed production one: ordinary_baseline_v2
runner.TorchBackend.step (1024-token limit, v6 budget envelope). Never a formal
training run: optimizer updates are diagnostic and discarded, no checkpoint is
saved, no navigation episode is executed. Hard caps: 512 forward decisions,
8 optimizer updates, 1000 s internal wall, 26 GiB GPU process cap.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
SPEED = HERE.parent
LINE = SPEED.parents[1]
REF = LINE / 'sft_acceptance/ordinary_baseline_v2'
V6 = LINE / 'sft_acceptance/ordinary_execution_v6'
FLA_DEPS = SPEED / 'official_fla_0_5_2/deps'
EINOPS_DEPS = SPEED / 'official_einops_0_8_1/deps'

DECISION_CAP = 512
UPDATE_CAP = 8
WALL_CAP_SECONDS = 1000.0
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['reference', 'candidate'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    require(args.mode in ('reference', 'candidate'), 'MODE')

    if args.mode == 'candidate':
        # Must precede the first transformers.models.qwen3_5 import: the kernel
        # binding is frozen at modeling-module import time by the decorator.
        # fla/__init__ does not import .ops; resolve_internal_import uses getattr
        # chains, so the internal path must be materialized by an explicit import.
        sys.path.insert(0, str(EINOPS_DEPS))
        sys.path.insert(0, str(FLA_DEPS))
        import fla.ops.gated_delta_rule  # noqa: F401  (exposes fla.ops.gated_delta_rule attrs)
    require(os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1',
            'OFFLINE_FLAGS_REQUIRED')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS_DETERMINISM_CONFIG')
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(uuid.startswith('GPU-'), 'EXACT_GPU_UUID_REQUIRED')

    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    runner = load_module('q35n_probe_reference_runner', REF / 'runner.py')
    data = load_module('q35n_probe_reference_data', REF / 'data.py')

    # Model assets verified against the registered baseline protocol hashes (read-only).
    ref_protocol = json.loads((REF / 'PROTOCOL.json').read_text())
    for relative, expected in ref_protocol['model_asset_sha256'].items():
        asset = LINE / relative
        require(sha256(asset) == expected, 'MODEL_ASSET_CHANGED:' + relative)

    protocol, rows, preflight = runner.preflight(REF / 'PROTOCOL.json', REF / 'snapshot_v1')

    import transformers
    modeling = importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
    fla_active = 'fla' in sys.modules
    # The kernel_wrapper decorator returns the closure `wrapped` itself; its
    # 'implementation' cell holds the resolved function (fla or torch fallback).
    bound = modeling.torch_chunk_gated_delta_rule
    resolved_impl = None
    if bound.__closure__:
        cells = dict(zip(bound.__code__.co_freevars, (c.cell_contents for c in bound.__closure__)))
        resolved_impl = cells.get('implementation')
    resolved_module = getattr(resolved_impl, '__module__', None) or ''
    if args.mode == 'candidate':
        require(fla_active and resolved_module.startswith('fla.'),
                'FLA_NOT_RESOLVED_IN_CANDIDATE_MODE:' + resolved_module)
    else:
        require(not fla_active and resolved_module.endswith('modeling_qwen3_5'),
                'FLA_LEAKED_INTO_REFERENCE_MODE:' + resolved_module)
    closure = [str(resolved_impl)[:200]]

    v6_protocol = json.loads((V6 / 'PROTOCOL.json').read_text())
    torch_uuid = str(torch.cuda.get_device_properties(0).uuid)
    canonical = torch_uuid if torch_uuid.startswith('GPU-') else 'GPU-' + torch_uuid
    require(canonical == uuid, 'LIVE_GPU_UUID_MISMATCH')
    backend = runner.TorchBackend(
        runner.configuration(protocol),
        dict(gpu_uuid=torch_uuid, budget=dict(v6_protocol['budget'])))

    counters = dict(decisions=0, updates=0)

    def clock():
        require(time.monotonic() - started < WALL_CAP_SECONDS, 'PROBE_WALL_CAP')
        return time.monotonic()

    def counted_step(inputs, memory):
        counters['decisions'] += 1
        require(counters['decisions'] <= DECISION_CAP, 'DECISION_CAP')
        return backend.step(inputs, memory)

    # Fixed decision set (deterministic; identical in both modes).
    audit = json.loads((REF / 'TOKEN_AUDIT.json').read_text())
    longest_id = audit['longest_records'][0]['record_id']
    longest_row = next(row for row in rows if row['record_id'] == longest_id)
    mid_row = next(row for row in rows if row['decisions'] >= 12 and row['record_id'] != rows[0]['record_id'])
    long_start = min(8, longest_row['decisions'] - 4)
    chunks = [
        ('short', rows[0], list(range(0, 4))),
        ('longest_mid', longest_row, list(range(long_start, long_start + 4))),
        ('longest_stop', longest_row, [longest_row['decisions'] - 1]),
        ('mid_a', mid_row, list(range(4, 8))),
        ('mid_b', mid_row, list(range(8, 12))),
    ]
    records = {}
    for name, row, steps in chunks:
        require(steps[-1] < row['decisions'], 'CHUNK_RANGE')
        records[name] = data.OrdinaryRecord(row)
    require(sum(len(s) for _, _, s in chunks) == 17, 'FIXED_SET_SIZE')

    trainable = [(n, p) for n, p in backend.policy.named_parameters() if p.requires_grad]
    trainable_names = [n for n, _ in trainable]

    logits_out = {}
    memories = {}
    ce_total = None
    for name, row, steps in chunks:
        memory = backend.zero_memory()
        chunk_loss = None
        for t in steps:
            clock()
            decision = records[name].decision(t)
            require(decision['control']['decision_step'] == t, 'NONCAUSAL_STEP')
            target = ACTIONS.index(decision['supervision']['target_action'])
            memory, logits = counted_step(decision['policy'], memory)
            ce = torch.nn.functional.cross_entropy(
                logits, torch.tensor([target], device=logits.device))
            chunk_loss = ce if chunk_loss is None else chunk_loss + ce
            logits_out['%s:%d' % (name, t)] = logits.detach().float().cpu()
        memories[name] = memory.detach().float().cpu()
        chunk_loss.backward()
        ce_total = float(chunk_loss.detach()) if ce_total is None else ce_total + float(chunk_loss.detach())
    grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
    require(float(grad_norm) > 0, 'ZERO_GRADIENT')
    require(all(p.grad is not None for _, p in trainable), 'MISSING_GRADIENT')
    grads = {n: p.grad.detach().float().cpu().clone() for n, p in trainable}
    before = {n: p.detach().float().cpu().clone() for n, p in trainable}
    backend.opt.step()
    counters['updates'] += 1
    require(counters['updates'] <= UPDATE_CAP, 'UPDATE_CAP')
    deltas = {n: (p.detach().float().cpu() - before[n]) for n, p in trainable}
    backend.opt.zero_grad(set_to_none=True)
    # Diagnostic update is discarded by construction: nothing is checkpointed.

    # Timing: fixed short record, warmup pass then measured pass, forward+backward
    # per 4-step chunk, no optimizer step (kernel-only throughput comparison).
    short = records['short']
    count = min(16, len(short))
    timing = []
    for phase in ('warmup', 'measured'):
        memory = backend.zero_memory()
        torch.cuda.synchronize()
        tick = clock()
        for begin in range(0, count, 4):
            loss = None
            for t in range(begin, min(begin + 4, count)):
                decision = short.decision(t)
                target = ACTIONS.index(decision['supervision']['target_action'])
                memory, logits = counted_step(decision['policy'], memory)
                ce = torch.nn.functional.cross_entropy(
                    logits, torch.tensor([target], device=logits.device))
                loss = ce if loss is None else loss + ce
            loss.backward()
            memory = memory.detach()
            backend.opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        timing.append(dict(phase=phase, decisions=count,
                           seconds=clock() - tick,
                           decisions_per_second=count / (clock() - tick)))

    dump = dict(
        mode=args.mode, unix=time.time(), elapsed_seconds=time.monotonic() - started,
        counters=counters, fla_active=fla_active,
        fla_modules=sorted(m for m in sys.modules if m == 'fla' or m.startswith('fla.'))[:8],
        kernel_closure_repr=closure,
        torch_version=str(torch.__version__), transformers_version=str(transformers.__version__),
        cuda_visible_devices=uuid, torch_gpu_uuid=canonical,
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        preflight_binding=dict(protocol_sha256=preflight['protocol_sha256'],
                               snapshot_result_sha256=preflight['snapshot_result_sha256'],
                               training_index_sha256=preflight['training_index_sha256']),
        decision_set=[dict(chunk=name, record_id=row['record_id'], steps=steps) for name, row, steps in chunks],
        logits=logits_out, memories=memories, ce_sum=ce_total,
        grad_norm=float(grad_norm), trainable=trainable_names,
        grads=grads, update_deltas=deltas, timing=timing,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        probe_updates_discarded=True, formal_training_updates=0, navigation_episodes=0,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ('PROBE_%s.pt' % args.mode.upper())
    with path.open('xb') as stream:
        torch.save(dump, stream)
        stream.flush()
        os.fsync(stream.fileno())
    summary = dict(mode=args.mode, counters=counters, fla_active=fla_active,
                   elapsed_seconds=dump['elapsed_seconds'], timing=timing,
                   ce_sum=ce_total, grad_norm=float(grad_norm),
                   peak_reserved_bytes=dump['peak_reserved_bytes'],
                   dump_sha256=sha256(path), unix=time.time())
    with (args.out / ('PROBE_%s.json' % args.mode.upper())).open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
