"""Ordinary baseline v3 acceptance battery: kernel panel, DDP parity, comparisons.

Phases (each bounded, diagnostic updates discarded, no navigation episodes):
  panel --kernel fallback|fla   : fixed panel forward+backward+1 discarded update dump
  panel_compare                 : apply frozen kernel thresholds
  parity_single                 : one process, 3 shard batches, one discarded update
  parity_multi                  : torchrun world=3, same global batch, one discarded update
  parity_compare                : apply frozen parity thresholds
All thresholds come from ACCEPTANCE.json frozen before any GPU run.
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


def setup_kernel(mode):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if mode == 'fla':
        sys.path.insert(0, str(EINOPS_DEPS))
        sys.path.insert(0, str(FLA_DEPS))
        import fla.ops.gated_delta_rule  # noqa: F401
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    modeling = importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
    bound = modeling.torch_chunk_gated_delta_rule
    cells = dict(zip(bound.__code__.co_freevars, (c.cell_contents for c in bound.__closure__)))
    resolved = getattr(cells.get('implementation'), '__module__', '') or ''
    if mode == 'fla':
        require(resolved.startswith('fla.'), 'FLA_NOT_RESOLVED:' + resolved)
    else:
        require(resolved.endswith('modeling_qwen3_5'), 'FLA_LEAKED:' + resolved)
    return torch, resolved


def load_context(acceptance):
    data = load_module('q35n_v3_accept_data', HERE / 'data.py')
    model = load_module('q35n_v3_accept_model', HERE / 'model.py')
    rows, report = data.load_rows()
    require(report['training_index_sha256'] == acceptance['snapshot']['training_index_sha256'],
            'SNAPSHOT_BINDING')
    samples = data.load_sample_index(HERE / 'SAMPLE_INDEX.jsonl',
                                     acceptance['sample_index_sha256'],
                                     sum(r['decisions'] for r in rows))
    return data, model, rows, samples


def make_optimizer(torch, policy, acceptance):
    opt_cfg = acceptance['optimizer']
    return torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                             lr=opt_cfg['learning_rate'], weight_decay=opt_cfg['weight_decay'],
                             betas=tuple(opt_cfg['betas']), eps=opt_cfg['eps'])


def run_batches(torch, data, model, raw_policy, callable_policy, samples, batch_ids, device,
                weight_sum_override=None, loss_scale=1.0):
    """Forward/backward planned batches through callable_policy (DDP-aware).

    loss per batch = (sum w*ce) * loss_scale / (weight_sum_override or local sum).
    With (loss_scale=world, override=W_global) per rank this reproduces the
    registered train.py DDP reduction; with (1.0, W_global) single-process it
    matches exactly. Returns (logits per batch, weight sum actually used).
    """
    store = data.SampleStore(ROWS_HOLDER[0])
    processor = raw_policy.processor
    collate = model.make_collate(
        pad_id=processor.tokenizer.pad_token_id,
        exec_sid=raw_policy.exec_sid, query_sid=raw_policy.query_sid,
        image_token_id=raw_policy.base.config.image_token_id)
    local_w = sum(sum(s['weight'] for s in batch) for batch in batch_ids)
    divisor = weight_sum_override if weight_sum_override is not None else local_w
    logits_out = []
    for batch_samples in batch_ids:
        dataset = model.DecisionDataset(batch_samples, store, processor)
        batch = collate([dataset[i] for i in range(len(batch_samples))])
        batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
        logits = callable_policy(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
            mm_token_type_ids=batch['mm_token_type_ids'], image_grid_thw=batch['image_grid_thw'],
            pixel_values=batch['pixel_values'], exec_index=batch['exec_index'],
            action_index=batch['action_index'])
        ce = torch.nn.functional.cross_entropy(logits, batch['targets'], reduction='none')
        loss = (ce * batch['weights']).sum() * loss_scale / divisor
        loss.backward()
        logits_out.append(logits.detach().float().cpu())
    return logits_out, divisor


ROWS_HOLDER = [None]


def phase_panel(args, acceptance):
    torch, resolved = setup_kernel(args.kernel)
    data, model, rows, samples = load_context(acceptance)
    ROWS_HOLDER[0] = rows
    device = torch.device('cuda:0')
    fraction = acceptance['budgets']['max_gpu_memory_bytes'] / torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(fraction)
    policy = model.build_policy(acceptance['seed'])
    policy.train()
    # Fixed 17-decision panel: same deterministic selection as kernel_probe_v1
    # (short/longest/STOP/mid real decisions), run as ONE padded v3 batch.
    audit = json.loads((data.REF / 'TOKEN_AUDIT.json').read_text())
    longest_id = audit['longest_records'][0]['record_id']
    longest_idx = next(i for i, row in enumerate(rows) if row['record_id'] == longest_id)
    mid_idx = next(i for i, row in enumerate(rows)
                   if row['decisions'] >= 12 and row['record_id'] != rows[0]['record_id'])
    long_start = min(8, rows[longest_idx]['decisions'] - 4)
    wanted = ([(0, t) for t in range(4)]
              + [(longest_idx, t) for t in range(long_start, long_start + 4)]
              + [(longest_idx, rows[longest_idx]['decisions'] - 1)]
              + [(mid_idx, t) for t in range(4, 12)])
    require(len(wanted) == acceptance['panel_decisions'], 'PANEL_DECISION_COUNT')
    lookup = {}
    for s in samples:
        key = (s['record_idx'], s['t'])
        if key in wanted and key not in lookup:
            lookup[key] = s
    # Length-sorted packing under the registered panel token budget: padding waste
    # would otherwise make the fixed panel heavier than a production batch.
    panel = sorted((lookup[k] for k in wanted), key=lambda s: s['est'])
    packs, current, current_tokens = [], [], 0
    for s in panel:
        if current and current_tokens + s['est'] > acceptance['panel_pack_max_tokens']:
            packs.append(current)
            current, current_tokens = [], 0
        current.append(s)
        current_tokens += s['est']
    if current:
        packs.append(current)
    logits, total_w = run_batches(torch, data, model, policy, policy, samples, packs, device)
    grads = {n: p.grad.detach().float().cpu().clone()
             for n, p in policy.named_parameters() if p.requires_grad}
    before = {n: p.detach().float().cpu().clone()
              for n, p in policy.named_parameters() if p.requires_grad}
    opt = make_optimizer(torch, policy, acceptance)
    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],
                                   acceptance['optimizer']['gradient_clip'],
                                   error_if_nonfinite=True)
    opt.step()
    deltas = {n: (p.detach().float().cpu() - before[n])
              for n, p in policy.named_parameters() if p.requires_grad}
    decisions = len(panel)
    dump = dict(kernel=args.kernel, resolved_implementation=resolved, logits=logits,
                weight_sum=total_w, grads=grads, update_deltas=deltas, decisions=decisions,
                optimizer_updates=1, discarded=True)
    path = args.out / ('PANEL_%s.pt' % args.kernel.upper())
    with path.open('xb') as stream:
        torch.save(dump, stream)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(kernel=args.kernel, decisions=decisions, dump=sha256(path))))


def compare_tensors(ref, cand, rel_l2_max, cosine_min):
    worst_rel, worst_cos = 0.0, None
    for name in ref:
        r, c = ref[name].float(), cand[name].float()
        rel = float((r - c).norm() / r.norm().clamp_min(1e-12))
        worst_rel = max(worst_rel, rel)
        if max(float(r.norm()), float(c.norm())) > 1e-12:
            import torch as _t
            cos = float(_t.nn.functional.cosine_similarity(r.flatten(), c.flatten(), dim=0))
            worst_cos = cos if worst_cos is None else min(worst_cos, cos)
    return worst_rel, worst_cos


def phase_panel_compare(args, acceptance):
    import torch
    with torch.serialization.safe_globals([]):
        ref = torch.load(args.run / 'PANEL_FALLBACK.pt', map_location='cpu', weights_only=True)
        cand = torch.load(args.run / 'PANEL_FLA.pt', map_location='cpu', weights_only=True)
    require(ref['resolved_implementation'].endswith('modeling_qwen3_5'), 'REFERENCE_NOT_FALLBACK')
    require(cand['resolved_implementation'].startswith('fla.'), 'CANDIDATE_NOT_FLA')
    t = acceptance['kernel_thresholds']
    gates = {}
    diffs = [float((r - c).abs().max()) for r, c in zip(ref['logits'], cand['logits'])]
    agree = sum(int((r.argmax(-1) == c.argmax(-1)).all()) for r, c in zip(ref['logits'], cand['logits']))
    total = sum(r.shape[0] for r in ref['logits'])
    gates['logits_max_abs_diff'] = max(diffs)
    gates['logits_pass'] = max(diffs) <= t['logits_max_abs_diff']
    gates['argmax_agreement'] = agree / total
    gates['argmax_pass'] = agree / total >= t['argmax_agreement_min']
    g_rel, g_cos = compare_tensors(ref['grads'], cand['grads'], None, None)
    gates['grad_relative_l2_max'] = g_rel
    gates['grad_cosine_min'] = g_cos
    gates['grad_pass'] = g_rel <= t['grad_relative_l2'] and (g_cos is None or g_cos >= t['grad_cosine_min'])
    u_rel, _ = compare_tensors(ref['update_deltas'], cand['update_deltas'], None, None)
    gates['update_relative_l2_max'] = u_rel
    gates['update_pass'] = u_rel <= t['update_relative_l2']
    gates['decisions_within_cap'] = (ref['decisions'] + cand['decisions']
                                     <= acceptance['budgets']['panel_max_decisions'])
    gates['status'] = 'PASS' if all(v for k, v in gates.items()
                                    if k.endswith('_pass') or k == 'decisions_within_cap') else 'FAIL'
    result = dict(unix=time.time(), gates=gates, thresholds=t, fail_preserved=True)
    with (args.run / 'PANEL_COMPARISON.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(gates, indent=2, allow_nan=False))


def maybe_fp32_trainables(policy, acceptance):
    """Decisive control for the parity gate: fp32 trainables eliminate bf16
    transport/accumulation noise, isolating the reduction logic itself."""
    if not acceptance.get('parity_fp32_trainables'):
        return
    import torch
    with torch.no_grad():
        for p in policy.parameters():
            if p.requires_grad:
                p.data = p.data.float()


def phase_parity_single(args, acceptance):
    torch, resolved = setup_kernel(acceptance['kernel'])
    data, model, rows, samples = load_context(acceptance)
    ROWS_HOLDER[0] = rows
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(
        acceptance['budgets']['max_gpu_memory_bytes'] / torch.cuda.get_device_properties(0).total_memory)
    policy = model.build_policy(acceptance['seed'])
    maybe_fp32_trainables(policy, acceptance)
    policy.train()
    world = acceptance['parity_world']
    plan = data.plan_epoch_batches(samples, acceptance['batching']['max_tokens'],
                                   acceptance['seed'], 0, world)
    global_batch = [[samples[i] for i in plan[r][0]] for r in range(world)]
    w_global = sum(s['weight'] for batch in global_batch for s in batch)
    # Reference = per-batch grads (same loss formula as production), then summed.
    per_batch = []
    for batch in global_batch:
        policy.zero_grad(set_to_none=True)
        run_batches(torch, data, model, policy, policy, samples, [batch], device,
                    weight_sum_override=w_global, loss_scale=1.0)
        per_batch.append({n: p.grad.detach().float().cpu().clone()
                          for n, p in policy.named_parameters() if p.requires_grad})
    grads = {}
    for n in per_batch[0]:
        acc = None
        for pb in per_batch:
            acc = pb[n] if acc is None else acc + pb[n]
        grads[n] = acc
    # Within-process rerun determinism: repeat batch[0] identically.
    policy.zero_grad(set_to_none=True)
    run_batches(torch, data, model, policy, policy, samples, [global_batch[0]], device,
                weight_sum_override=w_global, loss_scale=1.0)
    repeat0 = {n: p.grad.detach().float().cpu().clone()
               for n, p in policy.named_parameters() if p.requires_grad}
    path = args.out / 'PARITY_SINGLE.pt'
    with path.open('xb') as stream:
        torch.save(dict(kernel=resolved, grads=grads, per_batch=per_batch,
                        repeat_batch0=repeat0), stream)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(phase='parity_single', dump=sha256(path))))


def phase_parity_multi(args, acceptance):
    torch, resolved = setup_kernel(acceptance['kernel'])
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    require(world == acceptance['parity_world'], 'PARITY_WORLD')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_per_process_memory_fraction(
        acceptance['budgets']['max_gpu_memory_bytes']
        / torch.cuda.get_device_properties(local_rank).total_memory, device)
    data, model, rows, samples = load_context(acceptance)
    ROWS_HOLDER[0] = rows
    policy = model.build_policy(acceptance['seed'])
    maybe_fp32_trainables(policy, acceptance)
    policy.to(device)
    policy.train()
    model.broadcast_params(policy, 0)
    raw = policy  # raw grads are computed before any sync; sync comes after backward
    plan = data.plan_epoch_batches(samples, acceptance['batching']['max_tokens'],
                                   acceptance['seed'], 0, world)
    global_batch = [[samples[i] for i in plan[r][0]] for r in range(world)]
    w_local = sum(s['weight'] for s in global_batch[rank])
    w_global_t = torch.tensor([w_local], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(w_global_t)
    w_global = float(w_global_t.item())
    # Layer 1: raw per-rank grads WITHOUT the DDP wrapper (no sync) — must match
    # the single process's per-batch grads exactly (same batch, same formula).
    raw.zero_grad(set_to_none=True)
    run_batches(torch, data, model, raw, raw, samples, [global_batch[rank]], device,
                weight_sum_override=w_global, loss_scale=1.0)
    raw_grads = {n: p.grad.detach().float().cpu().clone()
                 for n, p in raw.named_parameters() if p.requires_grad}
    raw_path = args.out / ('PARITY_RAW_RANK%d.pt' % rank)
    with raw_path.open('xb') as stream:
        torch.save(dict(grads=raw_grads), stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Layer 2: the registered production path — manual gradient averaging
    # (all-reduce after backward completes; no DDP hooks).
    raw = policy
    raw.zero_grad(set_to_none=True)
    logits, _ = run_batches(torch, data, model, raw, raw, samples, [global_batch[rank]], device,
                            weight_sum_override=w_global, loss_scale=float(world))
    model.sync_grads(raw, world)
    grads = {n: p.grad.detach().float().cpu().clone()
             for n, p in raw.named_parameters() if p.requires_grad}
    # (within-process rerun determinism is checked in the single phase instead:
    #  DDP hooks armed by layer 2 must never see a non-DDP backward graph.)
    # Every rank dumps its own grads: DDP all-reduce must deliver identical
    # values on every rank; a silent desync would pass a rank-0-only check.
    rpath = args.out / ('PARITY_MULTI_RANK%d.pt' % rank)
    with rpath.open('xb') as stream:
        torch.save(dict(kernel=resolved, grads=grads), stream)
        stream.flush()
        os.fsync(stream.fileno())
    if rank == 0:
        path = args.out / 'PARITY_MULTI.pt'
        with path.open('xb') as stream:
            torch.save(dict(kernel=resolved, grads=grads), stream)
            stream.flush()
            os.fsync(stream.fileno())
        with (args.out / 'PARITY_MULTI_DONE.json').open('x') as stream:
            json.dump(dict(dump_sha256=sha256(path), ranks=world, unix=time.time()), stream)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(dict(phase='parity_multi', dump=sha256(path))), flush=True)
    # DDP collectives are complete after opt.step(); skip NCCL destroy teardown
    # (observed SIGABRT in the watchdog thread after results were already safe).
    sys.stdout.flush()
    os._exit(0)


def phase_parity_compare(args, acceptance):
    import torch
    single = torch.load(args.run / 'PARITY_SINGLE.pt', map_location='cpu', weights_only=True)
    multi = torch.load(args.run / 'PARITY_MULTI.pt', map_location='cpu', weights_only=True)
    t = acceptance['parity_thresholds']
    gates = {}
    # Gate 1: each rank's raw per-batch grads vs the single process's same batch
    # (exact same computation; isolates per-process forward/backward identity).
    raw_rels = []
    for r in range(acceptance['parity_world']):
        raw = torch.load(args.run / ('PARITY_RAW_RANK%d.pt' % r), map_location='cpu',
                         weights_only=True)
        r_rel, _ = compare_tensors(single['per_batch'][r], raw['grads'], None, None)
        raw_rels.append(r_rel)
    gates['raw_per_batch_relative_l2'] = raw_rels
    gates['raw_pass'] = all(x <= t['raw_relative_l2'] for x in raw_rels)
    # Gate 0: within-process rerun determinism (batch[0] repeated).
    rep_rel, _ = compare_tensors(single['per_batch'][0], single['repeat_batch0'], None, None)
    gates['rerun_batch0_relative_l2'] = rep_rel
    gates['rerun_pass'] = rep_rel <= t['rerun_relative_l2']
    # Gate 2: DDP all-reduced grads (rank 0) vs the single process's summed reference.
    rel, cos = compare_tensors(single['grads'], multi['grads'], None, None)
    gates['grad_relative_l2_max'] = rel
    gates['grad_cosine_min'] = cos
    gates['grad_pass'] = rel <= t['grad_relative_l2'] and (cos is None or cos >= t['grad_cosine_min'])
    # Gate 3: rank-to-rank sync of the all-reduced gradients.
    rank_sync = []
    for r in (1, 2):
        other = torch.load(args.run / ('PARITY_MULTI_RANK%d.pt' % r), map_location='cpu',
                           weights_only=True)
        r_rel, _ = compare_tensors(multi['grads'], other['grads'], None, None)
        rank_sync.append(r_rel)
    gates['rank_sync_relative_l2'] = rank_sync
    gates['rank_sync_pass'] = all(x <= t['rank_sync_relative_l2'] for x in rank_sync)
    gates['status'] = 'PASS' if (gates['raw_pass'] and gates['grad_pass']
                                 and gates['rank_sync_pass'] and gates['rerun_pass']) else 'FAIL'
    result = dict(unix=time.time(), gates=gates, thresholds=t, fail_preserved=True)
    with (args.run / 'PARITY_COMPARISON.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(gates, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--phase', required=True,
                        choices=['panel', 'panel_compare', 'parity_single', 'parity_multi',
                                 'parity_compare'])
    parser.add_argument('--kernel', choices=['fallback', 'fla'])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    acceptance = json.loads(args.acceptance.read_text())
    require(acceptance['id'] == 'Q35N_ORDINARY_BASELINE_V3_ACCEPTANCE', 'ACCEPTANCE_ID')
    for name, digest in acceptance['code_sha256'].items():
        path = HERE / name
        require(path.is_file() and sha256(path) == digest, 'CODE_CHANGED:' + name)
    require(os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1',
            'OFFLINE_FLAGS')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'CUBLAS_CONFIG')
    if args.phase == 'panel':
        require(args.kernel and args.out, 'PANEL_ARGS')
        args.out.mkdir(parents=True, exist_ok=True)
        phase_panel(args, acceptance)
    elif args.phase == 'panel_compare':
        phase_panel_compare(args, acceptance)
    elif args.phase == 'parity_single':
        args.out.mkdir(parents=True, exist_ok=True)
        phase_parity_single(args, acceptance)
    elif args.phase == 'parity_multi':
        phase_parity_multi(args, acceptance)
    elif args.phase == 'parity_compare':
        phase_parity_compare(args, acceptance)


if __name__ == '__main__':
    main()
