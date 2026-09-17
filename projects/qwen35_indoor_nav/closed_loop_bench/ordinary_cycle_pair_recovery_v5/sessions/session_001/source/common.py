"""V5 orchestration and pair audit; frozen policy/window/metric implementations."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
V3 = HERE.parent / 'ordinary_cycle_pair_gpu1_v3'
TRAIN = LINE / 'sft_acceptance/ordinary_sync_recovery_v1'
TINY = HERE.parent / 'r2r_ce_tiny_v1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load('v5_frozen_window', TINY / 'common.py')
Window, advance, ACTIONS = old.Window, old.advance, old.ACTIONS
sha, xyzw_to_wxyz = old.sha, old.xyzw_to_wxyz


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=False):
    """Publish a whole JSON object atomically; never replace a committed result."""
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('x') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if exclusive:
            os.link(tmp, path)
            tmp.unlink()
        else:
            os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def records(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def raw_input(window):
    rgb = [hashlib.sha256(x).hexdigest() for x in window.images]
    value = [window.instruction, rgb, list(window.executed)]
    key = hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return dict(instruction=window.instruction, rgb_sha256=rgb,
                executed_history=list(window.executed), input_key=key)


def tensor_identity(tensor):
    import torch
    cpu = tensor.detach().cpu().contiguous()
    return dict(shape=list(cpu.shape), dtype=str(cpu.dtype),
                sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())


def model_identity(policy):
    # state_dict includes all parameters (including frozen ones) and persistent buffers.
    tensors = {name: tensor_identity(t) for name, t in policy.state_dict().items()}
    digest = hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, tensors=tensors)


class PairError(RuntimeError):
    pass


def prefix_compare(a, b):
    raw = a['raw'] == b['raw']
    processed = a['processed'] == b['processed']
    va, vb = a['logits'], b['logits']
    if not all(math.isfinite(x) for x in va + vb):
        raise PairError('NONFINITE_LOGITS')
    delta = max(abs(x-y) for x, y in zip(va, vb))
    flip = a['native_action'] != b['native_action']
    action = not flip and (b['override'] or a['executed_action'] == b['executed_action'])
    return dict(input_prefix_matched=raw and processed, action_prefix_matched=action,
                logits_bitwise_equal=va == vb, max_logit_delta=delta,
                relative_max_logit_delta=delta/max(1e-12, max(abs(x) for x in va+vb)),
                argmax_flip_count=int(flip))


def audit_episode(folder, index):
    """Recompute frozen metric definitions and tie every executed action to the trace."""
    import gzip
    aggregate = load('v5_frozen_aggregate', V3 / 'aggregate.py')
    e = read(folder / f'episode_{index:02d}.json')
    policy = records(folder / 'POLICY_STEPS.jsonl')
    steps = records(folder / 'STEPS_PRIVILEGED.jsonl')
    if not (len(policy) == len(steps) == e['steps'] <= 500):
        raise PairError('TRACE_LENGTH')
    if not e['stopped'] and e['steps'] != 500:
        raise PairError('EARLY_NONSTOP_TERMINAL')
    interface = records(folder / 'INTERFACE.jsonl')[0]
    executed = []
    previous_rgb = interface['rgb_sha256']
    last_rgb = None
    for i, (a, s) in enumerate(zip(policy, steps), 1):
        expected_rgb = [previous_rgb] if i == 1 else [last_rgb, previous_rgb]
        if (a['step'] != i or s['step'] != i or a['executed_action'] != s['action']
                or a['raw']['executed_history'] != executed[-8:]
                or a['raw']['rgb_sha256'] != expected_rgb):
            raise PairError('ACTION_INPUT_TRANSPORT')
        if a['native_action'] == 'STOP' and a['executed_action'] != 'STOP':
            raise PairError('STOP_OVERRIDE')
        if s['action'] == 'STOP' and i != len(steps):
            raise PairError('POST_STOP_ACTION')
        if s['position'] != e['positions'][i] or s['distance_to_goal'] != e['distances'][i]:
            raise PairError('METRIC_TRACE_ALIGNMENT')
        executed.append(s['action'])
        last_rgb, previous_rgb = previous_rgb, s['rgb_sha256']
    success = float(e['stopped'] and e['distances'][-1] < 3.)
    distance = sum(math.dist(x,y) for x,y in zip(e['positions'],e['positions'][1:]))
    spl = success * e['distances'][0] / max(e['distances'][0], distance)
    gt = json.load(gzip.open(read(HERE/'PROTOCOL.json')['gt_path']))
    ndtw = aggregate.exact_ndtw_check(e['positions'], gt[str(e['episode_id'])]['locations'])
    if not (success == e['success'] and e['oracle_success'] == float(min(e['distances']) < 3.)
            and math.isclose(spl,e['spl'],rel_tol=1e-5,abs_tol=1e-5)
            and math.isclose(ndtw,e['ndtw'],rel_tol=1e-9,abs_tol=1e-9)):
        raise PairError('METRIC_RECOMPUTATION')
    return e


def committed_pairs():
    """Only complete pairs covered by a successful end-state seal are admissible."""
    found = {}
    for session in sorted((HERE/'sessions').glob('session_*')) if (HERE/'sessions').exists() else []:
        covered = set()
        for seal in session.glob('STATE_SEAL_*.json'):
            value = read(seal)
            if value['unchanged']:
                covered.update(value['pair_ranks'])
        for path in session.glob('pairs/pair_*/PAIR.json'):
            row = read(path)
            if row['rank'] in covered and row['valid_behavioral_pair']:
                if row['rank'] in found:
                    raise PairError('DUPLICATE_ADMITTED_PAIR')
                found[row['rank']] = dict(row, artifact=str(path.relative_to(HERE)))
    return found
