"""Versioned bounded ordinary training; old preparation artifacts stay sealed."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
import traceback

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
REF = HERE.parent / 'ordinary_baseline_v2'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=False):
    path = Path(path)
    blob = json.dumps(value, indent=2, allow_nan=False)
    if exclusive:
        with path.open('x') as stream:
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


def verify():
    ref = module('ordinary_reference_runner', REF / 'runner.py')
    protocol = read(HERE / 'PROTOCOL.json')
    seal = read(HERE / 'CODE_SEAL.json')
    for name, digest in seal.items():
        path = (HERE / name).resolve()
        assert path.is_relative_to(LINE) and ref.sha256(path) == digest, ('CODE_CHANGED', name)
    original, rows, preflight = ref.preflight(REF / 'PROTOCOL.json', REF / 'snapshot_v1')
    assert protocol['training'] == {k: original['training'][k] for k in protocol['training']}
    for name, expected in original['model_asset_sha256'].items():
        assert ref.sha256(LINE / name) == expected, ('MODEL_CHANGED', name)
    return ref, protocol, rows, dict(protocol_sha256=ref.sha256(HERE / 'PROTOCOL.json'),
        code_seal_sha256=ref.sha256(HERE / 'CODE_SEAL.json'), reference=preflight)


def tensor_equal(a, b, torch):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tensor_equal(a[k], b[k], torch) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(tensor_equal(x, y, torch) for x, y in zip(a, b))
    return a == b


def checkpoint(backend, out, binding, cursor, memory, stats):
    path = out / ('checkpoint_%09d.pt' % cursor['updates'])
    assert not path.exists(), 'CHECKPOINT_ALREADY_EXISTS'
    state = dict(binding=binding, cursor=copy.deepcopy(cursor),
                 memory=None if memory is None else memory.detach().cpu(),
                 stats=copy.deepcopy(stats), cumulative_compute=dict(backend.total_compute), backend=backend.state())
    with path.open('xb') as stream:
        backend.torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    write(Path(str(path) + '.json'), dict(sha256=backend.ref.sha256(path), binding=binding,
          cursor=cursor, unix=time.time()), exclusive=True)
    return str(path)


def next_cursor(cursor, end, route_length, route_count):
    cursor = dict(cursor)
    cursor['step'] = end
    if end == route_length:
        cursor['step'] = 0
        cursor['route_position'] += 1
        if cursor['route_position'] == route_count:
            cursor['route_position'] = 0
            cursor['epoch'] += 1
    return cursor


def install_cumulative_meter(backend):
    """Probe restores must never erase the actual compute resource ledger."""
    backend.total_compute = dict(decisions=0, tokens=0)
    original_step = backend.step
    def metered(inputs, memory):
        assert backend.total_compute['decisions'] < backend.budget['max_decisions'], 'CUMULATIVE_DECISION_BUDGET'
        assert backend.total_compute['tokens'] + backend.config['max_sequence_tokens'] <= backend.budget['max_forward_tokens'], 'CUMULATIVE_TOKEN_BUDGET'
        before = backend.policy.forward_tokens
        try:
            return original_step(inputs, memory)
        finally:
            difference = backend.policy.forward_tokens-before
            if difference > 0:
                backend.total_compute['decisions'] += 1
                backend.total_compute['tokens'] += difference
    backend.step = metered


def install_owned_restore(backend):
    reference_restore = backend.restore
    def restore_owned(state):
        # Optimizer.load_state_dict can alias CPU step tensors in the source.
        # Each restore owns its state; the saved replay fixture stays immutable.
        return reference_restore(copy.deepcopy(state))
    backend.restore = restore_owned


def probe(backend, rows, data, out, binding):
    torch = backend.torch
    started = time.monotonic()
    initial = copy.deepcopy(backend.state())
    short = data.OrdinaryRecord(rows[0])
    inputs = short.decision(0)['policy']
    with torch.no_grad():
        old_m, old_l, old_info = backend.policy.step(inputs['instruction'], inputs['images'], inputs['executed_actions'], backend.zero_memory())
        backend.total_compute['decisions'] += 1
        backend.total_compute['tokens'] += old_info['tokens']
        new_m, new_l = backend.step(inputs, backend.zero_memory())
    difference = max(float((old_m-new_m).abs().max()), float((old_l-new_l).abs().max()))
    assert difference <= 1e-6, ('OLD_FORWARD_DIFFERENCE', difference)

    # Genuine longest instruction, causal frames and executed history, four-step backward.
    longest_id = read(REF / 'TOKEN_AUDIT.json')['longest_records'][0]['record_id']
    longest_row = next(row for row in rows if row['record_id'] == longest_id)
    longest = data.OrdinaryRecord(longest_row)
    start = min(8, longest_row['decisions'] - 4)
    before = backend.policy.forward_tokens
    _, long_metrics = backend.train_chunk([longest.decision(t) for t in range(start, start+4)], backend.zero_memory(), finalize=True)
    long_tokens = (backend.policy.forward_tokens-before)//4
    assert long_tokens > 512 and long_metrics['grad_norm'] > 0

    # Terminal STOP gradient from actual terminal target; diagnostic is not a rollout.
    terminal = longest.decision(longest_row['decisions']-1)
    assert terminal['supervision']['target_action'] == 'STOP'
    _, stop_metrics = backend.train_chunk([terminal], backend.zero_memory(), finalize=True)
    stop_gradient = float(backend.policy.action_head.weight.grad[3].float().norm())
    assert stop_gradient > 0 and torch.isfinite(backend.policy.action_head.weight.grad).all()

    write(out/'PROBE_PARTIAL.json', dict(old_forward_max_abs=difference, longest_chunk_mean_tokens=long_tokens, longest_grad_norm=long_metrics['grad_norm'], real_STOP_head_row_grad_norm=stop_gradient, unix=time.time()))
    state = copy.deepcopy(backend.state())
    state_path = out / 'probe_checkpoint.pt'
    with state_path.open('xb') as stream:
        torch.save(state, stream)
        stream.flush(); os.fsync(stream.fileno())
    loaded = torch.load(state_path, map_location='cpu', weights_only=True)
    assert tensor_equal(state, loaded, torch), 'CHECKPOINT_SERIALIZATION'
    immutable_loaded = copy.deepcopy(loaded)
    with torch.no_grad():
        expected_m, expected_l = backend.step(inputs, backend.zero_memory())
        backend.policy.action_head.weight.add_(1)
    backend.restore(loaded)
    with torch.no_grad():
        actual_m, actual_l = backend.step(inputs, backend.zero_memory())
    assert torch.equal(expected_m, actual_m) and torch.equal(expected_l, actual_l), 'CHECKPOINT_FORWARD'
    backend.restore(loaded)
    # Reload optimizer+RNG then repeat actual update; compare exact trainables and optimizer.
    backend.train_chunk([short.decision(0)], backend.zero_memory(), finalize=True)
    first_update = copy.deepcopy(backend.state())
    assert tensor_equal(loaded, immutable_loaded, torch), 'SOURCE_OPTIMIZER_STATE_MUTATED'
    backend.restore(copy.deepcopy(loaded))
    backend.train_chunk([short.decision(0)], backend.zero_memory(), finalize=True)
    second_update = backend.state()
    assert tensor_equal(first_update['trainable'], second_update['trainable'], torch), 'RESUMED_UPDATE'
    assert tensor_equal(first_update['optimizer'], second_update['optimizer'], torch), 'RESUMED_OPTIMIZER'

    # Test the actual mid-route payload: memory+cursor, not merely model weights.
    resume_row = next(row for row in rows if row['decisions'] >= 12)
    resume_record = data.OrdinaryRecord(resume_row)
    route_memory, _ = backend.train_chunk([resume_record.decision(t) for t in range(4)], backend.zero_memory(), finalize=True)
    resume_cursor = dict(epoch=0, route_position=0, step=4, updates=1, decisions=4, chunks=1)
    payload = dict(backend=copy.deepcopy(backend.state()), memory=route_memory.detach().cpu(), cursor=resume_cursor)
    with (out/'probe_cursor_checkpoint.pt').open('xb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    restored_payload = torch.load(out/'probe_cursor_checkpoint.pt', map_location='cpu', weights_only=True)
    assert tensor_equal(payload, restored_payload, torch), 'CURSOR_MEMORY_SERIALIZATION'
    next_inputs = [resume_record.decision(t) for t in range(4,8)]
    first_memory, _ = backend.train_chunk(next_inputs, route_memory.detach(), finalize=True)
    first_state = copy.deepcopy(backend.state())
    backend.restore(restored_payload['backend'])
    second_memory, _ = backend.train_chunk(next_inputs, restored_payload['memory'].to('cuda:0'), finalize=True)
    assert torch.equal(first_memory,second_memory), 'MID_ROUTE_MEMORY_RESUME'
    assert tensor_equal(first_state['trainable'],backend.state()['trainable'],torch), 'MID_ROUTE_UPDATE_RESUME'
    assert tensor_equal(first_state['optimizer'],backend.state()['optimizer'],torch), 'MID_ROUTE_OPTIMIZER_RESUME'

    checks = {}
    def rejected(name, change, undo):
        change()
        try:
            with torch.no_grad():
                backend.step(inputs, backend.zero_memory())
        except (ValueError, AssertionError) as error:
            assert name in str(error), str(error)
            checks[name] = True
        else:
            raise AssertionError('MISSING_BUDGET_REJECTION:' + name)
        finally:
            undo()
    limit = backend.config['max_sequence_tokens']
    rejected('TOKEN_SEQUENCE_LIMIT', lambda: backend.config.update(max_sequence_tokens=1), lambda: backend.config.update(max_sequence_tokens=limit))
    tokens = backend.budget['max_forward_tokens']
    rejected('CUMULATIVE_TOKEN_BUDGET', lambda: backend.budget.update(max_forward_tokens=backend.total_compute['tokens']), lambda: backend.budget.update(max_forward_tokens=tokens))
    decisions = backend.budget['max_decisions']
    rejected('CUMULATIVE_DECISION_BUDGET', lambda: backend.budget.update(max_decisions=backend.total_compute['decisions']), lambda: backend.budget.update(max_decisions=decisions))
    wall = backend.budget['wall_seconds']
    rejected('WALL_BUDGET', lambda: backend.budget.update(wall_seconds=0), lambda: backend.budget.update(wall_seconds=wall))

    with torch.no_grad():
        _, baseline_logits = backend.step(inputs, backend.zero_memory())
        altered_text = dict(inputs, instruction='Stop now.')
        _, text_logits = backend.step(altered_text, backend.zero_memory())
        from PIL import Image
        altered_rgb = dict(inputs, images=[Image.new('RGB', (224,224)) for _ in inputs['images']])
        _, rgb_logits = backend.step(altered_rgb, backend.zero_memory())
    torch.cuda.synchronize()
    measured_start = time.monotonic()
    count = min(16, rows[0]['decisions'])
    memory = backend.zero_memory()
    for begin in range(0, count, 4):
        memory, _ = backend.train_chunk([short.decision(t) for t in range(begin,min(begin+4,count))], memory, finalize=begin+4>=count)
        memory = memory.detach()
    torch.cuda.synchronize()
    throughput = count/(time.monotonic()-measured_start)
    assert time.monotonic()-started < 1200
    result = dict(status='REAL_INTERFACE_PASS', binding=binding, unix=time.time(),
        old_forward_max_abs=difference, longest_record_id=longest_id, longest_chunk_mean_tokens=long_tokens,
        longest_grad_norm=long_metrics['grad_norm'], real_STOP_head_row_grad_norm=stop_gradient,
        checkpoint_reload_forward_and_optimizer_update_exact=True, mid_route_cursor_memory_chunk_resume_exact=True,
        budget_rejections=checks,
        instruction_logit_l2=float((baseline_logits-text_logits).norm()), rgb_logit_l2=float((baseline_logits-rgb_logits).norm()),
        input_interventions_are_diagnostics_not_navigation_evidence=True,
        measured_decisions_per_second=throughput, throughput_scope='up to16 decisions on one actual instruction; not full-dataset throughput',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.monotonic()-started, probe_compute=dict(backend.total_compute),
        probe_updates_discarded=True, navigation_gain_verified=False)
    backend.restore(initial)
    backend.opt.zero_grad(set_to_none=True)
    backend.pending_chunks = backend.pending_decisions = 0
    write(HERE / 'PROBE.json', result, exclusive=True)
    return result


def train(backend, rows, data, out, binding, protocol):
    config = protocol['training']
    cursor = backend.ref.initial_cursor()
    memory = None
    stats = dict(ce_sum=0.0, decisions=0, confusion=[[0]*4 for _ in range(4)])
    started = time.monotonic()
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stop_requested.append('SIGINT'))
    last_report = 0.0
    latest_checkpoint = checkpoint(backend, out, binding, cursor, memory, stats)
    order_epoch, order = None, None
    while cursor['epoch'] < config['epochs']:
        if order_epoch != cursor['epoch']:
            order = backend.ref.route_order(rows, config['seed'], cursor['epoch'])
            order_epoch = cursor['epoch']
        row = rows[order[cursor['route_position']]]
        record = data.OrdinaryRecord(row)
        if cursor['step'] == 0:
            memory = backend.zero_memory()
        while cursor['step'] < row['decisions']:
            start = cursor['step']
            end = min(start+4, row['decisions'])
            ending_epoch = end == row['decisions'] and cursor['route_position'] == len(rows)-1
            soft_stop = bool(stop_requested) or backend.elapsed() >= protocol['budget']['wall_seconds']-600 or cursor['decisions'] >= 320000
            decisions = [record.decision(t) for t in range(start, end)]
            memory, metrics = backend.train_chunk(decisions, memory, finalize=ending_epoch or soft_stop)
            memory = memory.detach()
            cursor['decisions'] += end-start
            cursor['chunks'] += 1
            cursor['updates'] += int(metrics['optimizer_step'])
            cursor = next_cursor(cursor, end, row['decisions'], len(rows))
            if cursor['step'] == 0:
                memory = None
            stats['ce_sum'] += metrics['ce']*(end-start)
            stats['decisions'] += end-start
            for i in range(4):
                for j in range(4):
                    stats['confusion'][i][j] += metrics['confusion'][i][j]
            stop = soft_stop or cursor['updates'] >= protocol['budget']['max_updates'] or cursor['epoch'] >= config['epochs']
            if metrics['optimizer_step'] and (cursor['updates'] % protocol['checkpoint_updates'] == 0 or stop or ending_epoch):
                latest_checkpoint = checkpoint(backend, out, binding, cursor, memory, stats)
            if time.monotonic()-last_report >= protocol['progress_seconds'] or stop:
                support = [sum(row) for row in stats['confusion']]
                event = dict(status='SEGMENT_COMPLETED' if stop else 'TRAINING', unix=time.time(), cursor=cursor,
                    metrics=dict(last_chunk=metrics, mean_ce=stats['ce_sum']/stats['decisions'],
                        accuracy=sum(stats['confusion'][i][i] for i in range(4))/stats['decisions'],
                        action_recall=[stats['confusion'][i][i]/support[i] if support[i] else None for i in range(4)],
                        action_target_counts=support, confusion=stats['confusion']),
                    throughput=cursor['decisions']/max(.001,time.monotonic()-started),
                    budget=protocol['budget'], cumulative_compute=dict(backend.total_compute), latest_checkpoint=latest_checkpoint,
                    evaluation_performed=False, navigation_gain_verified=False)
                write(out / 'PROGRESS.json', event)
                with (out / 'PROGRESS.jsonl').open('a') as stream:
                    stream.write(json.dumps(event, allow_nan=False)+'\n')
                print(json.dumps(event, allow_nan=False), flush=True)
                last_report = time.monotonic()
            if stop:
                write(out / 'RESULT.json', dict(status='BOUNDED_SEGMENT_COMPLETED', cursor=cursor,
                    latest_checkpoint=latest_checkpoint, binding=binding, unix=time.time(), stop_requested=stop_requested,
                    full_epochs_completed=cursor['epoch'], navigation_gain_verified=False), exclusive=True)
                return
            if end == row['decisions']:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true', required=True)
    args = parser.parse_args()
    out = HERE / 'run_0001'
    out.mkdir(exist_ok=False)
    try:
        write(out / 'PROGRESS.json', dict(status='VERIFYING_LOCKS_AND_MODEL', unix=time.time()))
        ref, protocol, rows, binding = verify()
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == protocol['gpu_uuid']
        assert os.environ.get('Q35N_SUPERVISED_RUN') == binding['protocol_sha256'], 'SUPERVISOR_REQUIRED'
        data = module('ordinary_sealed_data', REF / 'data.py')
        write(out / 'BINDING.json', binding, exclusive=True)
        write(out / 'PROGRESS.json', dict(status='LOADING_MODEL_FOR_REAL_PROBE', unix=time.time()))
        import torch
        torch_uuid = str(torch.cuda.get_device_properties(0).uuid)
        canonical_uuid = torch_uuid if torch_uuid.startswith('GPU-') else 'GPU-' + torch_uuid
        assert canonical_uuid == protocol['gpu_uuid'], 'CANONICAL_LIVE_GPU_UUID_MISMATCH'
        write(out/'GPU_UUID_BINDING.json', dict(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            torch_uuid=torch_uuid, canonical_uuid=canonical_uuid), exclusive=True)
        # Read-only backend compares Torch's own representation; physical admission uses canonical UUID.
        backend = ref.TorchBackend(copy.deepcopy(protocol['training']), dict(gpu_uuid=torch_uuid, budget=copy.deepcopy(protocol['budget'])))
        install_owned_restore(backend)
        backend.ref = ref
        install_cumulative_meter(backend)
        write(out / 'PROGRESS.json', dict(status='REAL_GPU_PROBE', unix=time.time()))
        evidence = probe(backend, rows, data, out, binding)
        write(out / 'RUN_ADMISSION.json', dict(status='BOUNDED_SEGMENT_ADMITTED', binding=binding,
            interface_evidence_sha256=ref.sha256(HERE / 'PROBE.json'), user_authorization=protocol['user_authorization'],
            budget=protocol['budget'], fullscale_admitted=False, unix=time.time()), exclusive=True)
        train(backend, rows, data, out, binding, protocol)
    except BaseException as error:
        write(out / 'RESULT.json', dict(status='FAILED_NO_AUTOMATIC_RETRY', error=repr(error),
            traceback=traceback.format_exc(), unix=time.time(), navigation_gain_verified=False), exclusive=True)
        raise


if __name__ == '__main__':
    main()
