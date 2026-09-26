"""Read sealed real recaptures; later chunk tokens keep query-start memory."""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import torch


def _check(condition, reason):
    if not condition:
        raise ValueError(reason)


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _physical_contract(trace):
    actions, queries = trace['actions'], trace['query_steps']
    _check(0 < len(actions) <= 500 and actions[-1] == 0 and 0 not in actions[:-1], 'STOP_OR_BUDGET_CHANGED')
    _check(all(action in range(4) for action in actions), 'INVALID_EXECUTED_ACTION')
    _check(len(trace['rgb_sha256']) == len(actions), 'STOP_EXTRA_OR_MISSING_OBSERVATION')
    _check(queries and queries[0] == 0 and queries == sorted(set(queries)) and queries[-1] < len(actions), 'INVALID_QUERY_STEPS')
    _check(all(1 <= end - start <= 4 for start, end in zip(queries, queries[1:] + [len(actions)])), 'INVALID_CHUNK_LENGTH')
    _check(trace['partition'] in ('FIT', 'DEV') and trace['kind'] in ('RECOVERY', 'PRESERVATION'), 'INVALID_SPLIT_OR_KIND')
    _check(0 <= trace['cutoff'] <= len(actions), 'INVALID_CUTOFF')
    if trace['kind'] == 'RECOVERY':
        _check(trace['cutoff'] in queries, 'CUTOFF_NOT_A_QUERY')


def _features(cache, count, actor_count, width):
    for name, shape in [('memory_features', (count, width)), ('actor_features', (actor_count, width)),
                        ('base_logits', (actor_count, 4))]:
        value = cache[name]
        _check(isinstance(value, torch.Tensor) and tuple(value.shape) == shape, 'INVALID_' + name.upper() + '_SHAPE')
        _check(value.dtype == torch.float32 and value.device.type == 'cpu', 'INVALID_' + name.upper() + '_DTYPE_DEVICE')
        _check(bool(torch.isfinite(value).all()), 'NONFINITE_' + name.upper())


def _int_vector(cache, name, expected):
    value = cache[name]
    _check(isinstance(value, torch.Tensor) and value.dtype == torch.int64
           and value.device.type == 'cpu' and value.ndim == 1, 'INVALID_' + name.upper())
    _check(value.tolist() == expected, 'MISALIGNED_' + name.upper())


def _validated_cache(cache, trace, entry, width):
    _physical_contract(trace)
    count = len(trace['actions'])
    _features(cache, count, count, width)
    for key in ('id', 'house', 'partition', 'kind'):
        _check(cache[key] == entry[key] == trace[key], 'CACHE_SOURCE_' + key.upper() + '_MISMATCH')
    _check(cache['cutoff'] == trace['cutoff'], 'CACHE_CUTOFF_MISMATCH')
    _check(cache['source_trajectory_sha256'] == entry['trajectory_sha256'], 'CACHE_SOURCE_SHA_MISMATCH')
    _check(cache['representation'] == 'REAL_ALL_ACTION_TOKEN_CAPTURE_WITH_QUERY_START_MEMORY', 'WRONG_CAPTURE_REPRESENTATION')
    _check(cache['new_training_admission'] is False, 'UNEXPECTED_TRAINING_ADMISSION_CHANGE')
    _int_vector(cache, 'executed_actions', trace['actions'])
    _int_vector(cache, 'query_steps', trace['query_steps'])
    contexts = [trace['query_steps'][bisect_right(trace['query_steps'], step) - 1] for step in range(count)]
    _int_vector(cache, 'actor_context_steps', contexts)
    _int_vector(cache, 'action_token_offsets', [step - context for step, context in enumerate(contexts)])
    known = cache['supervision_region']
    expected = [trace['kind'] == 'PRESERVATION' or step >= trace['cutoff'] for step in range(count)]
    _check(known.dtype == torch.bool and known.device.type == 'cpu' and known.ndim == 1
           and known.tolist() == expected, 'ORIGINAL_SUPERVISION_MASK_CHANGED')
    return dict(cache, known=known, targets=cache['executed_actions'],
                actor_environment_steps=torch.arange(count), data_scope='REAL_SEALED_ALL_TOKEN_RECAPTURE')


def load_rows(run, partition='FIT'):
    """Load one complete FIT/DEV partition from a fully sealed physical dataset.

    This validates integrity, not training admission. The stored
    ``new_training_admission=False`` remains unchanged for a separate decision.
    No incomplete subset or old-cache fallback is selected automatically.
    """
    _check(partition in ('FIT', 'DEV'), 'INVALID_REQUESTED_PARTITION')
    run = Path(run).resolve()
    protocol = _read(run / 'PROTOCOL.json')
    lock_path = run / 'SOURCE_LOCK.json'
    lock = _read(lock_path)['files']
    lock_sha = _sha(lock_path)
    protocol_sha = _sha(run / 'PROTOCOL.json')
    for path in (run / 'PROTOCOL.json', run / 'DATA_MANIFEST.json'):
        _check(str(path) in lock and _sha(path) == lock[str(path)], 'RUN_SOURCE_IDENTITY_CHANGED')
    entries = _read(run / 'DATA_MANIFEST.json')['episodes']
    manifest = {entry['id']: entry for entry in entries}
    _check(len(manifest) == len(entries) == protocol['planned_trajectories'], 'INVALID_MANIFEST_DENOMINATOR')
    houses = {split: {entry['house'] for entry in entries if entry['partition'] == split} for split in ('FIT', 'DEV')}
    _check(not houses['FIT'] & houses['DEV'], 'FIT_DEV_HOUSE_OVERLAP')
    _check(all(entry['partition'] in ('FIT', 'DEV') for entry in entries), 'UNREGISTERED_MANIFEST_SPLIT')
    receipts = {}
    for session in sorted((run / 'capture').glob('*')):
        _check((session / 'STATE_SEAL.json').is_file(), 'UNSEALED_CAPTURE_SESSION')
        seal = _read(session / 'STATE_SEAL.json')
        _check(seal['base_before'] == seal['base_after'] == protocol['expected_base_state_sha256'], 'BASE_IDENTITY_CHANGED')
        _check(seal['source_lock_sha256'] == lock_sha, 'SESSION_SOURCE_LOCK_CHANGED')
        identity_path = session / 'RUNTIME_IDENTITY.json'
        identity = _read(identity_path)
        identity_sha = _sha(identity_path)
        _check(identity['base_state_sha256'] == protocol['expected_base_state_sha256']
               and identity['base_updates'] == 0 and identity['source_lock_sha256'] == lock_sha
               and identity['protocol_sha256'] == protocol_sha, 'RUNTIME_IDENTITY_MISMATCH')
        for path in sorted(session.glob('episodes/*/COMPLETE.json')):
            receipt = _read(path)
            index = receipt['trajectory_id']
            _check(index in manifest and index not in receipts and index in seal['complete_ids'], 'UNSEALED_OR_DUPLICATE_TRAJECTORY')
            _check(receipt['status'] == 'TRAJECTORY_RECAPTURE_ADMITTED' and receipt['physical_replay_exact']
                   and receipt['active_stop'], 'PHYSICAL_REPLAY_NOT_ADMITTED')
            _check(receipt['split'] == manifest[index]['partition'] and receipt['kind'] == manifest[index]['kind'], 'RECEIPT_SPLIT_KIND_MISMATCH')
            _check(receipt['runtime_identity_sha256'] == identity_sha, 'RECEIPT_RUNTIME_CHANGED')
            _check(Path(receipt['cache_path']).resolve() == (path.parent / 'CACHE.pt').resolve(), 'CACHE_PATH_OUTSIDE_TRAJECTORY')
            _check(_sha(path.parent / 'TRACE.jsonl') == receipt['trace_sha256'], 'CAPTURE_TRACE_CHANGED')
            receipts[index] = (receipt, identity['hidden_size'])
    _check(set(receipts) == set(manifest), 'INCOMPLETE_REGISTERED_DATASET')
    rows = []
    for index, entry in sorted(manifest.items()):
        if entry['partition'] != partition:
            continue
        receipt, width = receipts[index]
        _check(_sha(entry['trajectory']) == entry['trajectory_sha256'] == lock[str(entry['trajectory'])], 'PHYSICAL_SOURCE_CHANGED')
        trace = _read(entry['trajectory'])
        _check(trace['admitted'] and trace['success'], 'SOURCE_TRAJECTORY_NOT_ADMITTED')
        _check(receipt['physical_actions'] == receipt['captured_all_actor_rows'] == len(trace['actions'])
               and receipt['captured_all_queries'] == len(trace['query_steps']), 'CAPTURE_COVERAGE_MISMATCH')
        _check(_sha(receipt['cache_path']) == receipt['cache_sha256'], 'CAPTURE_CACHE_CHANGED')
        cache = torch.load(receipt['cache_path'], map_location='cpu', weights_only=True, mmap=True)
        row = _validated_cache(cache, trace, entry, width)
        row.update(cache_path=receipt['cache_path'], cache_sha256=receipt['cache_sha256'])
        rows.append(row)
    _check(bool(rows), 'EMPTY_REQUESTED_PARTITION')
    return rows


def row_logits(head, row, device):
    """Unroll once, reading each chunk's distinct actor rows at its query state."""
    device = torch.device(device)
    if row.get('data_scope') == 'OLD_QUERY_ONLY_CPU_SMOKE':
        _check(device.type == 'cpu', 'OLD_QUERY_ADAPTER_IS_CPU_SMOKE_ONLY')
    features = row['memory_features'].to(device)
    actors = row['actor_features'].to(device)
    actions = row['executed_actions'].to(device)
    contexts = row['actor_context_steps'].tolist()
    lookup = {}
    for actor_index, context in enumerate(contexts):
        lookup.setdefault(context, []).append(actor_index)
    memory = head.reset(batch_size=1)
    _check(memory.device.type == device.type and (device.index is None or memory.device.index == device.index),
           'HEAD_DEVICE_MISMATCH')
    deltas = [None] * len(contexts)
    for step in range(len(features)):
        previous = features[step - 1:step] if step else None
        action = actions[step - 1:step] if step else torch.tensor([4], dtype=torch.long, device=device)
        memory = head.update(features[step:step + 1], memory, previous, action)
        if step in lookup:
            for actor_index in lookup[step]:
                deltas[actor_index] = head.action_delta(actors[actor_index:actor_index + 1], memory)[0]
    _check(all(delta is not None for delta in deltas), 'ACTOR_CONTEXT_OUTSIDE_REPLAY')
    return row['base_logits'].to(device) + torch.stack(deltas)


def adapt_old_query_cache_for_cpu_smoke(cache, trace):
    """Explicit in-memory adapter; it creates no absent token features or labels."""
    _physical_contract(trace)
    _check(trace['partition'] == cache['partition'] == 'FIT', 'OLD_SMOKE_MUST_USE_FIT')
    _check(cache['house'] == trace['house'], 'OLD_HOUSE_MISMATCH')
    _check(cache['kind'] == trace['kind'] and cache['cutoff'] == trace['cutoff'], 'OLD_SOURCE_SCOPE_MISMATCH')
    count = len(trace['actions'])
    queries = trace['query_steps']
    _features(cache, count, len(queries), cache['memory_features'].shape[1])
    _int_vector(cache, 'query_steps', queries)
    _int_vector(cache, 'targets', [trace['actions'][step] for step in queries])
    expected = [trace['kind'] == 'PRESERVATION' or step >= trace['cutoff'] for step in queries]
    _check(cache['known'].dtype == torch.bool and cache['known'].tolist() == expected, 'OLD_KNOWN_MASK_CHANGED')
    return dict(cache, id=trace['id'], executed_actions=torch.tensor(trace['actions'], dtype=torch.long),
        actor_context_steps=cache['query_steps'], actor_environment_steps=cache['query_steps'],
        action_token_offsets=torch.zeros(len(queries), dtype=torch.long), supervision_region=cache['known'],
        new_training_admission=False, data_scope='OLD_QUERY_ONLY_CPU_SMOKE',
        missing_actor_positions=count - len(queries))
