"""Index real action tokens and observation transitions without recapturing features."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def action_rows(trace, pool_row_index):
    actions, queries = trace['actions'], trace['query_steps']
    if not actions or actions[-1] != 0 or 0 in actions[:-1]:
        raise ValueError('Expected actual successful trajectory with a single terminal STOP')
    if any(a not in range(4) for a in actions) or len(actions) > 500:
        raise ValueError('Invalid actual action or original 500-decision budget')
    if not queries or queries[0] != 0 or queries != sorted(set(queries)):
        raise ValueError('Query steps must start at zero and increase strictly')
    if queries[-1] >= len(actions) or len(trace['rgb_sha256']) != len(actions):
        raise ValueError('Query or observation/action alignment mismatch')
    if trace['kind'] not in ('RECOVERY', 'PRESERVATION') or trace['partition'] not in ('FIT', 'DEV'):
        raise ValueError('Only registered FIT/DEV recovery and preservation are admitted')
    ends = queries[1:] + [len(actions)]
    if any(not 1 <= end - start <= 4 for start, end in zip(queries, ends)):
        raise ValueError('Unexpected four-action generation boundary')
    for step, action in enumerate(actions):
        query_index = bisect_right(queries, step) - 1
        query_start = queries[query_index]
        offset = step - query_start
        existing_actor = offset == 0
        supervised_region = trace['kind'] == 'PRESERVATION' or step >= trace['cutoff']
        yield dict(trajectory_id=trace['id'], pool_row_index=pool_row_index,
            split=trace['partition'], kind=trace['kind'], environment_step=step,
            executed_action=action, previous_executed_action=actions[step - 1] if step else None,
            query_index=query_index, query_start_step=query_start, action_token_offset=offset,
            chunk_length=ends[query_index] - query_start,
            actor_feature_index=query_index if existing_actor else None,
            base_logits_index=query_index if existing_actor else None,
            has_existing_actor_feature=existing_actor, requires_actor_recapture=not existing_actor,
            action_in_registered_supervised_region=supervised_region,
            existing_actor_supervision_known=existing_actor and supervised_region,
            memory_feature_index=step,
            next_memory_feature_index=step + 1 if action != 0 else None,
            has_observed_transition=action != 0,
            terminal_stop=action == 0,
            next_observation_status='TERMINAL_STOP_NO_NEW_OBSERVATION' if action == 0 else 'ACTUALLY_OBSERVED',
            rgb_sha256=trace['rgb_sha256'][step],
            next_rgb_sha256=trace['rgb_sha256'][step + 1] if action != 0 else None)


def build(base, output):
    began = time.time()
    run = base / 'recovery_action_v1/runs/action_001'
    admission_path = run / 'data/ADMISSION.json'
    audit_path = base / 'parallel_improvements_v1/DATA_ACTION_AUDIT.json'
    manifest_path = run / 'features/DATA_MANIFEST.json'
    lock_path = run / 'features/SOURCE_LOCK.json'
    admission, audit = read(admission_path), read(audit_path)
    manifest = {r['id']: r for r in read(manifest_path)['episodes']}
    lock = read(lock_path)['files']
    pool_path = run / 'data/POOLS.pt'
    pool_sha = sha(pool_path)
    if pool_sha != admission['pools_sha256'] or pool_sha != audit['pool_sha256_recorded']:
        raise ValueError('Frozen POOLS identity mismatch')
    if audit['status'] != 'CPU_DATA_AUDIT_COMPLETE' or set(manifest) != {r['id'] for r in audit['rows']}:
        raise ValueError('Prior actual tensor/trajectory audit is incomplete')
    source_paths = [Path(__file__), admission_path, audit_path, manifest_path, lock_path]
    source_hashes = {str(p): sha(p) for p in source_paths}
    source_hashes[str(pool_path)] = pool_sha
    cache_sources = {int(Path(path).parent.name): dict(path=path, sha256=digest)
                     for path, digest in admission['certificates'].items()}
    output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(Counter)
    trajectories = []
    token_count = 0
    with (output / 'ACTION_TOKENS.jsonl').open('x') as stream:
        # Prior data_audit preserves the actual order of pack['rows'].
        for pool_row_index, audited in enumerate(audit['rows']):
            meta = manifest[audited['id']]
            path = Path(meta['trajectory'])
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != lock[str(path)] or digest != audited['trajectory_sha256']:
                raise ValueError('Frozen physical trajectory identity mismatch')
            trace = json.loads(raw)
            if not trace['admitted'] or not trace['success']:
                raise ValueError('Unadmitted physical trajectory')
            for key, other in [('id', 'id'), ('partition', 'split'), ('kind', 'kind')]:
                if trace[key] != audited[other]:
                    raise ValueError('Audited tensor mapping does not match trajectory')
            if len(trace['actions']) != audited['physical_steps'] or len(trace['query_steps']) != audited['queries']:
                raise ValueError('Audited feature dimensions do not match trajectory')
            rows = list(action_rows(trace, pool_row_index))
            if sum(r['existing_actor_supervision_known'] for r in rows) != audited['known_queries']:
                raise ValueError('Known supervision boundary mismatch')
            counts = groups[trace['partition'] + ':' + trace['kind']]
            counts['trajectories'] += 1
            counts['actions'] += len(rows)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                for name in ('has_existing_actor_feature', 'requires_actor_recapture',
                             'existing_actor_supervision_known', 'action_in_registered_supervised_region',
                             'has_observed_transition', 'terminal_stop'):
                    counts[name] += row[name]
                counts['terminal_stop_requires_actor_recapture'] += row['terminal_stop'] and row['requires_actor_recapture']
                counts['token_offset_' + str(row['action_token_offset'])] += 1
            trajectories.append(dict(id=trace['id'], pool_row_index=pool_row_index,
                split=trace['partition'], kind=trace['kind'], house=meta['house'], route_family=meta['route_family'],
                trajectory_path=str(path), trajectory_sha256=digest, cutoff=trace['cutoff'],
                memory_features_count=len(trace['actions']), actor_features_count=len(trace['query_steps']),
                tokens_first_line_1based=token_count + 1, tokens_count=len(rows),
                original_cache_source=cache_sources[trace['id']]))
            token_count += len(rows)
            source_hashes[str(path)] = digest
    write(output / 'TRAJECTORIES.json', trajectories)
    result = dict(status='TOKEN_INDEX_COMPLETE', trajectories=len(trajectories), actions=token_count,
        by_split_kind={k: dict(v) for k, v in sorted(groups.items())},
        pool_path=str(pool_path), pool_sha256=pool_sha,
        pool_mapping_source='Order of rows in the prior real-tensor DATA_ACTION_AUDIT; pool bytes reverified now.',
        feature_semantics='Mixed normalized visual, instruction and previous executed-action embedding; not pure visual.',
        actor_semantics='Only offset zero has a captured actor feature. Later tokens require recapture; no feature copying.',
        observation_semantics='memory_features[t+1] follows actual non-STOP action[t]; STOP never creates an observation.',
        supervision_semantics='Actual executed actions are indexed. Only existing query labels retain original known mask.',
        token_offset_semantics='Action index within actual generated chunk, excluding the assistant header.',
        pool_tensors_loaded=0, gpu_hours=0, optimizer_updates=0, unseen_read=False,
        source_hashes=source_hashes,
        artifact_hashes={name: sha(output / name) for name in ('ACTION_TOKENS.jsonl', 'TRAJECTORIES.json')},
        wall_seconds=time.time() - began)
    write(output / 'RESULT.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'trajectories', 'actions', 'by_split_kind', 'wall_seconds')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'token_index')
    args = parser.parse_args()
    build(args.base, args.output)
