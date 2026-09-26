"""Matched all-token heads; frozen features, FIT-only loss, fixed final step."""
import argparse
from collections import Counter
import fcntl
import importlib.util
from pathlib import Path
import random
import sys
import time
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path[:0] = [str(HERE), str(BASE)]
import common as u
import numpy as np
import torch
import torch.nn.functional as F
from model import ExecutionAdaptation
from data import load_rows, row_logits
from runtime import selected_action
from preserve_objective_v3 import preservation_loss

# Reuse already tested atomic checkpoint and RNG helpers, never their trainer.
spec = importlib.util.spec_from_file_location('unit_control_training', BASE / 'parallel_improvements_v1/train_controls.py')
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)
MODES = ('CURRENT', 'DELTA')
SOURCE = BASE / 'recovery_confirmation_v2/runs/confirmation_001'


def objective(model, recovery, ordinary, weights, device):
    if recovery['partition'] != 'FIT' or ordinary['partition'] != 'FIT':
        raise ValueError('FIT_ONLY_OPTIMIZATION')
    if recovery['kind'] != 'RECOVERY' or ordinary['kind'] != 'PRESERVATION':
        raise ValueError('RECOVERY_AND_PRESERVATION_REQUIRED')
    mask = recovery['known'].to(device)
    if not bool(mask.any()) or not bool(ordinary['known'].all()):
        raise ValueError('INVALID_SUPERVISION_MASK')
    r = row_logits(model, recovery, device)
    target = recovery['targets'].to(device)
    recovery_ce = F.cross_entropy(r[mask], target[mask], weight=weights)
    o = row_logits(model, ordinary, device)
    ordinary_target = ordinary['targets'].to(device)
    keep, kl, margin = preservation_loss(o, ordinary['base_logits'].to(device), ordinary_target)
    ordinary_ce = F.cross_entropy(o, ordinary_target)
    total = recovery_ce + .5 * ordinary_ce + 5 * keep
    return total, dict(recovery_ce=float(recovery_ce.detach()), ordinary_ce=float(ordinary_ce.detach()),
        ordinary_kl=float(kl.detach()), ordinary_margin=float(margin.detach()),
        recovery_tokens=int(mask.sum()), ordinary_tokens=len(o),
        recovery_accuracy=float((selected_action(r[mask]) == target[mask]).float().mean()),
        ordinary_agreement=float((selected_action(o) == ordinary_target).float().mean()))


def sources():
    paths = [HERE / n for n in ('model.py', 'data.py', 'runtime.py', 'train.py')]
    paths += [BASE / n for n in ('parallel_improvements_v1/execution_memory.py',
        'parallel_improvements_v1/train_controls.py', 'parallel_improvements_v1/control_models.py',
        'recovery_confirmation_v2/confirmation_model.py', 'recovery_confirmation_v2/train.py',
        'recovery_action_v1/recovery_model.py', 'memory_v2.py', 'preserve_objective_v3.py', 'common.py',
        'action_boundary_v2.py')]
    return {str(p): u.sha(p) for p in paths}


def prepare(run, capture_run, device='cuda:0', seeds=(42, 43, 44), steps=3000):
    """No waiting or GPU loading: incomplete recapture returns a concrete error."""
    run, capture_run = Path(run).resolve(), Path(capture_run).resolve()
    if steps != 3000 or tuple(seeds) != (42, 43, 44):
        raise ValueError('FORMAL_FIXED_3000_STEPS_THREE_SEEDS')
    fit = load_rows(capture_run, 'FIT')
    dev = load_rows(capture_run, 'DEV')
    if {r['house'] for r in fit} & {r['house'] for r in dev}:
        raise ValueError('FIT_DEV_HOUSE_OVERLAP')
    rows = {r['id']: r for r in fit}
    if len(rows) != len(fit):
        raise ValueError('DUPLICATE_TRAINING_ID')
    counts = Counter(a for r in fit if r['kind'] == 'RECOVERY' for a in r['targets'][r['known']].tolist())
    if any(counts[a] == 0 for a in range(4)):
        raise ValueError('RECOVERY_CLASS_MISSING')
    weights = torch.tensor([counts[a] ** -.5 for a in range(4)], dtype=torch.float32)
    weights /= weights.mean()
    schedules, inputs, initial = {}, {}, {}
    for seed in seeds:
        folder = SOURCE / 'data' / str(seed)
        admission = u.read(folder / 'ADMISSION.json')
        for name, key in [('INITIAL.pt', 'initial_sha256'), ('SCHEDULE.json', 'schedule_sha256')]:
            path = folder / name
            actual = u.sha(path)
            if actual != admission[key]:
                raise ValueError('ORIGINAL_INITIAL_OR_SCHEDULE_CHANGED')
            inputs[str(path)] = actual
        schedule = u.read(folder / 'SCHEDULE.json')
        if len(schedule) != steps or any(len(pair) != 2 or any(i not in rows for i in pair) or
                [rows[i]['kind'] for i in pair] != ['RECOVERY', 'PRESERVATION'] for pair in schedule):
            raise ValueError('RECORDED_SCHEDULE_DATA_MISMATCH')
        schedules[str(seed)] = schedule
        torch.manual_seed(seed)
        model = ExecutionAdaptation(3584, 'DELTA')
        missing = model.load_state_dict(torch.load(folder / 'INITIAL.pt', weights_only=True, map_location='cpu'), strict=False)
        if set(missing.missing_keys) != {'change_writer.weight', 'executed_action.weight'} or missing.unexpected_keys:
            raise ValueError('INITIAL_ARCHITECTURE_MISMATCH')
        initial[str(seed)] = model.state_dict()
    for name in ('PROTOCOL.json', 'SOURCE_LOCK.json', 'DATA_MANIFEST.json', 'REQUESTS.json'):
        inputs[str(capture_run / name)] = u.sha(capture_run / name)
    # Bind the sealed bytes used in this data admission, not a mutable progress file.
    for session in sorted((capture_run / 'capture').glob('*')):
        seal = session / 'STATE_SEAL.json'
        if seal.exists():
            inputs[str(seal)] = u.sha(seal)
            for complete in sorted(session.glob('episodes/*/COMPLETE.json')):
                receipt = u.read(complete)
                inputs[str(complete)] = u.sha(complete)
                inputs[receipt['cache_path']] = receipt['cache_sha256']
    config = dict(version='EXECUTION_ADAPTATION_ALL_TOKENS_V1', width=3584, seeds=list(seeds),
        modes=list(MODES), steps=steps, checkpoint_every=200, device=str(device),
        optimizer=dict(lr=1e-4, weight_decay=.01, clip_grad_norm=1.),
        objective=dict(recovery_ce=1., ordinary_ce=.5, preservation_kl_plus_margin=5.),
        class_counts=dict(counts), class_weights=weights.tolist(), schedules=schedules,
        source_files=sources(), input_files=inputs, capture_run=str(capture_run),
        software=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version),
        fit_ids=sorted(rows), dev_ids=sorted(r['id'] for r in dev),
        action_scope='ALL_ACTION_TOKENS_QUERY_START_MEMORY', base_updates=0,
        final_evaluation_split='val_unseen', exposed_evaluation=True, debug_only=False)
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / 'TRAINING_CONFIG.json'
    if config_path.exists():
        registered = u.read(config_path)
        config['initial_files'] = registered['initial_files']
        if registered != json.loads(json.dumps(config)):
            raise ValueError('TRAINING_BINDING_CHANGED')
        for record in config['initial_files'].values():
            if u.sha(record['path']) != record['sha256']:
                raise ValueError('INITIAL_STATE_CHANGED')
    else:
        for seed, state in initial.items():
            path = run / f'INITIAL_s{seed}.pt'
            if path.exists():
                existing = torch.load(path, weights_only=True, map_location='cpu')
                if existing.keys() != state.keys() or any(not torch.equal(existing[k], state[k]) for k in state):
                    raise ValueError('PARTIAL_PREPARATION_INITIAL_CHANGED')
            else:
                tmp = path.with_suffix('.tmp'); torch.save(state, tmp); tmp.replace(path)
        config['initial_files'] = {seed: dict(path=str(run / f'INITIAL_s{seed}.pt'),
            sha256=u.sha(run / f'INITIAL_s{seed}.pt')) for seed in initial}
        u.write(config_path, config)
        u.write(run / 'DATA_ADMISSION.json', dict(status='SEALED_ALL_TOKEN_DATA_ADMITTED',
            original_new_training_admission_field_changed=False,
            fit_trajectories=len(fit), dev_trajectories=len(dev),
            train_tokens=sum(int(r['known'].sum()) for r in fit),
            known_mask_preserved=True, dev_or_unseen_in_loss=False))
    return u.read(config_path), rows


def train_one(run, config, rows, seed, mode, resume=False, stop_after=None):
    run = Path(run); device = torch.device(config['device'])
    if mode not in config['modes'] or seed not in config['seeds']:
        raise ValueError('UNREGISTERED_MODEL')
    expected_scope = 'OLD_QUERY_ONLY_CPU_SMOKE' if config.get('debug_only') else 'REAL_SEALED_ALL_TOKEN_RECAPTURE'
    if any(row.get('data_scope') != expected_scope for row in rows.values()):
        raise ValueError('TRAINING_DATA_SCOPE_MISMATCH')
    if config.get('debug_only') and device.type != 'cpu':
        raise ValueError('DEBUG_CPU_ONLY')
    if u.read(run / 'TRAINING_CONFIG.json') != config:
        raise ValueError('TRAINING_BINDING_CHANGED')
    for path, sha in config['source_files'].items():
        if u.sha(path) != sha:
            raise ValueError('TRAINING_SOURCE_CHANGED')
    for path, sha in config['input_files'].items():
        if u.sha(path) != sha:
            raise ValueError('TRAINING_INPUT_CHANGED')
    folder = run / f'{mode}_s{seed}'; folder.mkdir(exist_ok=True)
    binding = dict(config_sha256=u.sha(run / 'TRAINING_CONFIG.json'), seed=seed, mode=mode)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    record = config['initial_files'][str(seed)]
    if u.sha(record['path']) != record['sha256']:
        raise ValueError('INITIAL_STATE_CHANGED')
    initial = torch.load(record['path'], weights_only=True, map_location='cpu')
    model = ExecutionAdaptation(config['width'], mode).to(device)
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    saved = sorted(folder.glob('checkpoint_*.pt')); final = folder / 'FINAL.pt'; start = 0
    if saved or final.exists():
        if not resume:
            raise ValueError('RESUME_REQUIRED')
        state = torch.load(final if final.exists() else saved[-1], map_location='cpu', weights_only=False)
        if state['binding'] != binding or state['schedule_cursor'] != state['step']:
            raise ValueError('CHECKPOINT_BINDING_CHANGED')
        start = state['step']
        if not 0 <= start <= config['steps'] or (final.exists() and start != config['steps']):
            raise ValueError('CHECKPOINT_STEP_INVALID')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        checkpoint.restore_rng(state['rng'], device)
    began = time.time(); weights = torch.tensor(config['class_weights'], device=device)
    u.write(folder / f'RUNTIME_{time.time_ns()}.json', dict(
        device=str(device), dtype=str(next(model.parameters()).dtype),
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(device), 'uuid', 'unavailable')) if device.type == 'cuda' else None,
        torch=torch.__version__, cuda=torch.version.cuda, base_loaded=False, base_updates=0,
        parameters=sum(p.numel() for p in model.parameters()), binding=binding))
    end = config['steps'] if stop_after is None else min(config['steps'], start + stop_after)
    if end < start or (end == start and start != config['steps']):
        raise ValueError('EMPTY_UPDATE_REQUEST')
    if start < end:
        with (folder / f'STEPS_{time.time_ns()}.jsonl').open('x', buffering=1) as log:
            for step in range(start, end):
                ri, oi = config['schedules'][str(seed)][step]
                optimizer.zero_grad(set_to_none=True)
                loss, info = objective(model, rows[ri], rows[oi], weights, device)
                loss.backward()
                if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError('NONFINITE_TRAINING')
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad_norm'])
                gradients = {name: float(param.grad.norm()) for name, param in model.named_parameters()
                    if name in ('writer.weight', 'change_writer.weight', 'executed_action.weight') and param.grad is not None}
                optimizer.step()
                info.update(step=step + 1, loss=float(loss.detach()), gradient_norm=float(norm),
                    writer_gradients=gradients, schedule=[ri, oi], wall_seconds=time.time() - began)
                log.write(json.dumps(info, allow_nan=False) + '\n')
                if (step + 1) % config['checkpoint_every'] == 0 or step + 1 == end:
                    checkpoint.save_checkpoint(folder / f'checkpoint_{step + 1:06d}.pt', model, optimizer, step + 1, binding, device)
                if (step + 1) % 10 == 0 or step + 1 == end:
                    u.write(folder / 'PROGRESS.json', dict(status='TRAINING', planned=config['steps'], **info))
    if end == config['steps'] and not final.exists():
        checkpoint.save_checkpoint(final, model, optimizer, end, binding, device)
    result = dict(status='COMPLETE' if end == config['steps'] else 'PAUSED_AT_CHECKPOINT',
        completed_steps=end, planned_steps=config['steps'], base_updates=0, base_loaded=False,
        action_scope=config['action_scope'], debug_only=config.get('debug_only', False),
        changed_parameters=[k for k, v in model.state_dict().items() if not torch.equal(v.cpu(), initial[k])],
        final_sha256=u.sha(final) if final.exists() else None, navigation_efficacy_measured=False)
    if not result['changed_parameters']:
        raise ValueError('NO_PARAMETER_UPDATE')
    u.write(folder / 'RESULT.json', result); u.write(folder / 'PROGRESS.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--capture-run', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        config, rows = prepare(args.run, args.capture_run, args.device)
        if not args.prepare_only:
            for seed in config['seeds']:
                for mode in config['modes']:
                    print(json.dumps(train_one(args.run, config, rows, seed, mode, args.resume)), flush=True)


if __name__ == '__main__':
    main()
