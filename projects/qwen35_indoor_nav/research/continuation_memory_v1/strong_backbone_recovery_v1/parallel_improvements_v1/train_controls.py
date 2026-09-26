"""Matched unit-readout controls using recorded FIT features, no backbone load.

Only query-first action features are available. This does not repair later-token
STOP coverage, change inference, evaluate navigation, or establish a benefit.
"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = BASE / 'recovery_confirmation_v2/runs/confirmation_001'
sys.path[:0] = [str(HERE), str(BASE / 'recovery_confirmation_v2'), str(BASE)]
import common as u
import numpy as np
import torch
from control_models import UnitReadoutMemory

spec = importlib.util.spec_from_file_location('recorded_confirmation_train', BASE / 'recovery_confirmation_v2/train.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
objective = training.objective
MODES = ('LOCAL', 'EMA', 'RECURRENT')


def source_identity():
    paths = [Path(__file__), HERE / 'control_models.py', BASE / 'common.py',
             BASE / 'memory_v2.py', BASE / 'preserve_objective_v3.py',
             BASE / 'recovery_action_v1/recovery_model.py',
             BASE / 'recovery_confirmation_v2/confirmation_model.py',
             BASE / 'recovery_confirmation_v2/train.py']
    return {str(p): u.sha(p) for p in paths}


def validate_row(row, width):
    x, q = row['memory_features'], row['query_steps']
    if row['partition'] != 'FIT':
        raise ValueError('TRAINING_REQUIRES_FIT')
    if x.ndim != 2 or x.shape[1] != width or not 0 < len(x) <= 500:
        raise ValueError('BAD_CAUSAL_SEQUENCE')
    if q.ndim != 1 or not len(q) or q.dtype != torch.int64 or not bool((q[1:] > q[:-1]).all()):
        raise ValueError('BAD_QUERY_ORDER')
    if int(q[0]) < 0 or int(q[-1]) >= len(x):
        raise ValueError('QUERY_OUTSIDE_HISTORY')
    if row['actor_features'].shape != (len(q), width) or row['base_logits'].shape != (len(q), 4):
        raise ValueError('MISALIGNED_ACTOR_FEATURES')
    if row['known'].dtype != torch.bool or row['known'].shape != q.shape or row['targets'].shape != q.shape:
        raise ValueError('MISALIGNED_SUPERVISION')
    known = row['known']
    if not bool(known.any()) or not bool(((row['targets'][known] >= 0) & (row['targets'][known] < 4)).all()):
        raise ValueError('INVALID_KNOWN_ACTION')
    if row['kind'] == 'PRESERVATION' and not bool(known.all()):
        raise ValueError('ORDINARY_OBJECTIVE_REQUIRES_ALL_KNOWN')
    if not all(bool(torch.isfinite(row[k]).all()) for k in ('memory_features', 'actor_features', 'base_logits')):
        raise ValueError('NONFINITE_RECORDED_FEATURE')


def prepare(run, seeds, modes, steps, device, source=SOURCE, debug_two_rows=False):
    """Bind existing initializations/data; only TEST fixtures select two FIT rows."""
    source, run = Path(source).resolve(), Path(run).resolve()
    original = u.read(source / 'PROTOCOL.json')
    if not 0 < steps <= original['steps'] or any(s not in (42, 43, 44) for s in seeds):
        raise ValueError('UNREGISTERED_STEPS_OR_SEED')
    if not modes or any(m not in MODES for m in modes):
        raise ValueError('UNREGISTERED_CONTROL')
    if len(set(seeds)) != len(seeds) or len(set(modes)) != len(modes):
        raise ValueError('DUPLICATE_MODEL_REGISTRATION')
    if original['optimizer'] != dict(name='AdamW', lr=1e-4, weight_decay=.01, clip_grad_norm=1.):
        raise ValueError('RECORDED_OPTIMIZER_CHANGED')
    if debug_two_rows and str(device) != 'cpu':
        raise ValueError('DEBUG_FIXTURE_IS_CPU_ONLY')
    inputs, schedules, initial_paths = {}, {}, {}
    pool_path = None
    for seed in seeds:
        data = source / 'data' / str(seed)
        admission = u.read(data / 'ADMISSION.json')
        candidate = Path(admission['pools_path'])
        for path, expected in [(candidate, admission['pools_sha256']),
                               (data / 'INITIAL.pt', admission['initial_sha256']),
                               (data / 'SCHEDULE.json', admission['schedule_sha256'])]:
            if str(path) not in inputs:
                actual = u.sha(path)
                if actual != expected:
                    raise ValueError('SOURCE_ASSET_CHANGED: ' + str(path))
                inputs[str(path)] = actual
            elif inputs[str(path)] != expected:
                raise ValueError('SEED_ASSET_BINDINGS_DISAGREE')
        if pool_path is not None and pool_path != candidate:
            raise ValueError('SEEDS_DO_NOT_SHARE_DATA')
        pool_path = candidate
        initial_paths[str(seed)] = str(data / 'INITIAL.pt')
        schedules[str(seed)] = u.read(data / 'SCHEDULE.json')[:steps]
        inputs[str(data / 'ADMISSION.json')] = u.sha(data / 'ADMISSION.json')
    inputs[str(source / 'PROTOCOL.json')] = u.sha(source / 'PROTOCOL.json')
    pack = torch.load(pool_path, map_location='cpu', weights_only=True, mmap=True)
    if debug_two_rows:
        pair = [min((i for i, r in enumerate(pack['rows']) if r['partition'] == 'FIT' and r['kind'] == kind),
                    key=lambda i: pack['rows'][i]['id']) for kind in ('RECOVERY', 'PRESERVATION')]
        schedules = {str(s): [pair] * steps for s in seeds}
    indices = sorted({i for schedule in schedules.values() for pair in schedule for i in pair})
    for i in indices:
        validate_row(pack['rows'][i], original['feature_width'])
    for schedule in schedules.values():
        if len(schedule) != steps or any([pack['rows'][i]['kind'] for i in pair] != ['RECOVERY', 'PRESERVATION'] for pair in schedule):
            raise ValueError('BAD_MATCHED_SCHEDULE')
    config = dict(version='UNIT_READOUT_CONTROLS_V1', seeds=list(seeds), modes=list(modes), steps=steps,
                  device=str(device), width=original['feature_width'], optimizer=original['optimizer'],
                  checkpoint_every=original['checkpoint_every'], objective=original['objective'],
                  software=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version),
                  original_protocol_sha256=inputs[str(source / 'PROTOCOL.json')],
                  source_files=source_identity(), input_files=inputs, initial_paths=initial_paths,
                  schedules=schedules, pool_path=str(pool_path), debug_two_rows=debug_two_rows,
                  base_loaded=False, base_updates=0, evaluation_enabled=False,
                  action_scope='QUERY_FIRST_ONLY; later-token STOP coverage unchanged')
    run.mkdir(parents=True, exist_ok=True)
    path = run / 'TRAINING_CONFIG.json'
    if path.exists():
        if u.read(path) != config:
            raise ValueError('RUN_BINDING_CHANGED')
    else:
        u.write(path, config)
    return config, pack


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def save_checkpoint(path, model, optimizer, step, binding, device):
    if path.exists():
        raise ValueError('CHECKPOINT_ALREADY_EXISTS')
    temp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    schedule_cursor=step, binding=binding, rng=rng_state(device)), temp)
    temp.replace(path)


def train_one(run, config, pack, seed, mode, resume=False, stop_after=None):
    run, device = Path(run), torch.device(config['device'])
    folder = run / f'{mode}_s{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    binding = dict(config_sha256=u.sha(run / 'TRAINING_CONFIG.json'), seed=seed, mode=mode)
    if u.read(run / 'TRAINING_CONFIG.json') != config:
        raise ValueError('RUN_BINDING_CHANGED')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    initial = torch.load(config['initial_paths'][str(seed)], map_location='cpu', weights_only=True)
    model = UnitReadoutMemory(config['width'], mode)
    model.load_state_dict(initial)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    checkpoints = sorted(folder.glob('checkpoint_*.pt'))
    final = folder / 'FINAL.pt'
    if (checkpoints or final.exists()) and not resume:
        raise ValueError('EXISTING_MODEL_REQUIRES_RESUME')
    start = 0
    if checkpoints or final.exists():
        checkpoint = torch.load(final if final.exists() else checkpoints[-1], map_location='cpu', weights_only=False)
        if checkpoint['binding'] != binding or checkpoint['schedule_cursor'] != checkpoint['step']:
            raise ValueError('CHECKPOINT_BINDING_CHANGED')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step']
        if not 0 <= start <= config['steps'] or (final.exists() and start != config['steps']):
            raise ValueError('BAD_CHECKPOINT_CURSOR')
        restore_rng(checkpoint['rng'], device)
    if final.exists():
        # FINAL is the commit point. Rebuild a missing receipt after a crash
        # between atomic weight publication and the small JSON publication.
        result = result_record(model, initial, config, start)
        result['final_sha256'] = u.sha(final)
        u.write(folder / 'RESULT.json', result)
        u.write(folder / 'PROGRESS.json', result)
        return result
    schedule = config['schedules'][str(seed)]
    indices = {i for pair in schedule for i in pair}
    rows = {i: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in pack['rows'][i].items()} for i in indices}
    weights = pack['class_weights'].to(device)
    end = config['steps'] if stop_after is None else min(config['steps'], start + stop_after)
    if end <= start:
        raise ValueError('EMPTY_UPDATE_REQUEST')
    began = time.time()
    runtime = dict(torch=torch.__version__, numpy=np.__version__, python=sys.version, device=str(device),
                   gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   gpu_uuid=str(getattr(torch.cuda.get_device_properties(device), 'uuid', 'unavailable')) if device.type == 'cuda' else None,
                   torch_cuda_version=torch.version.cuda,
                   dtype=str(next(model.parameters()).dtype), base_loaded=False, base_updates=0,
                   parameter_count=sum(p.numel() for p in model.parameters()), binding=binding)
    u.write(folder / f'RUNTIME_{time.time_ns()}.json', runtime)
    with (folder / f'STEPS_{time.time_ns()}.jsonl').open('x', buffering=1) as log:
        for step in range(start, end):
            ri, oi = schedule[step]
            loss, info = objective(model, rows[ri], rows[oi], weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError('NONFINITE_TRAINING')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad_norm'])
            writer = float(model.writer.weight.grad.norm())
            optimizer.step()
            info.update(step=step + 1, schedule_cursor=step + 1, mode=mode, seed=seed,
                        schedule=[ri, oi], loss=float(loss.detach()), gradient_norm=float(norm), writer_gradient=writer,
                        memory_steps=len(rows[ri]['memory_features']) + len(rows[oi]['memory_features']),
                        wall_seconds=time.time() - began)
            log.write(json.dumps(info, allow_nan=False) + '\n')
            if (step + 1) % config['checkpoint_every'] == 0 or step + 1 == end:
                save_checkpoint(folder / f'checkpoint_{step + 1:06d}.pt', model, optimizer, step + 1, binding, device)
            if (step + 1) % 10 == 0 or step + 1 == end:
                u.write(folder / 'PROGRESS.json', dict(status='TRAINING', planned=config['steps'], **info))
    result = result_record(model, initial, config, end)
    if end == config['steps']:
        save_checkpoint(final, model, optimizer, end, binding, device)
        result['final_sha256'] = u.sha(final)
    u.write(folder / 'RESULT.json', result)
    u.write(folder / 'PROGRESS.json', result)
    return result


def result_record(model, initial, config, step):
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v.cpu(), initial[k])]
    if not changed:
        raise ValueError('NO_PARAMETER_UPDATE')
    return dict(status='COMPLETE' if step == config['steps'] else 'PAUSED_AT_CHECKPOINT', completed_steps=step,
                planned_steps=config['steps'], changed_parameters=changed, base_updates=0, base_loaded=False,
                full_causal_unroll=True, known_query_mask_preserved=True, query_first_only=True,
                navigation_efficacy_measured=False, debug_only=config['debug_two_rows'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--modes', choices=MODES, nargs='+', default=list(MODES))
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        config, pack = prepare(args.run, args.seeds, args.modes, args.steps, args.device, args.source)
        for seed in args.seeds:
            for mode in args.modes:
                result = train_one(args.run, config, pack, seed, mode, resume=args.resume)
                print(json.dumps(dict(seed=seed, mode=mode, **result)), flush=True)


if __name__ == '__main__':
    main()
