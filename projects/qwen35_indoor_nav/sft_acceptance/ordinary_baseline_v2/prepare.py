"""Freeze closed ordinary production indexes for CPU-only training preparation.

No simulator, model, GPU, downloads, or edits to source artifacts. This verifies
receipt/index bindings and JSON metadata; RGB pixels are verified on lazy decode.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
PIPE = LINE / 'data_pipeline'
SPLIT = PIPE / 'ordinary_scale_v1/SPLIT_FREEZE.json'
MERGES = [
    ('ordinary_parallel_v1/merge_receipt_v2/merge', 780, 167127),
    ('ordinary_fullscale_source_v1/salvage_rescue_v1/run_v1', 814, 235410),
    ('ordinary_fullscale_source_v1/runtime_v5/merge', 496, 101121),
    ('ordinary_fullscale_source_v1/runtime_v6/merge', 660, 192951),
    ('ordinary_fullscale_source_v1/runtime_v4/merge', 1209, 290570),
    ('ordinary_fullscale_source_v1/auto_generation_v1/salvage_old_v1/run_v1', 284, 68345),
    ('ordinary_fullscale_source_v1/auto_generation_v4/merge', 836, 171350),
]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_path(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(LINE), 'SOURCE_OUTSIDE_LINE')
    return path


class Reader:
    def __init__(self):
        self.hashes = {}
        self.bytes = 0
        self.start = time.monotonic()

    def read(self, path, kind='json'):
        path = source_path(path)
        require(time.monotonic() - self.start < 900, 'CPU_PREPARATION_TIME_BUDGET')
        size = path.stat().st_size
        require(size <= 64 * 1024**2, 'METADATA_FILE_SIZE_BUDGET')
        self.bytes += size
        require(self.bytes <= 4 * 1024**3, 'METADATA_TOTAL_READ_BUDGET')
        blob = path.read_bytes()
        key = str(path.relative_to(ROOT))
        h = digest(blob)
        require(self.hashes.setdefault(key, h) == h, 'SOURCE_CHANGED_DURING_READ')
        if kind == 'json':
            return json.loads(blob)
        if kind == 'jsonl':
            return [json.loads(x) for x in blob.splitlines() if x.strip()]
        return blob


def count_rows(rows):
    unique = {}
    for row in rows:
        key = (row['scene_group'], row['physical_source_route_sha256'])
        require(unique.setdefault(key, row['decisions']) == row['decisions'],
                'PHYSICAL_ROUTE_DECISION_CONFLICT')
    return dict(strict_routes=len(unique), instruction_records=len(rows),
                unique_route_decisions=sum(unique.values()),
                instruction_conditioned_decisions=sum(r['decisions'] for r in rows))


def require_counts(actual, expected):
    for key, value in expected.items():
        require(actual[key] == value, 'RECEIPT_COUNT_MISMATCH:' + key)


def closed_pools(reader):
    pools = []
    base = PIPE / 'ordinary_scale_v1'
    terminal = reader.read(base / 'recovery_v4/GENERATION_COMPLETE.json')
    require(terminal['all_jobs_terminal'] is True and terminal['completed'] == terminal['target'] == 1000,
            'FIRST_POOL_NOT_CLOSED')
    transport = reader.read(base / 'recovery_v4/run/RESULT.json')
    require(transport['returncode'] == 0 and transport['error'] is None
            and transport['holder_restored'] is True, 'FIRST_POOL_TRANSPORT')
    rows, evidence = [], []
    audits = sorted((base / 'shards').glob('shard_*.audit.json'))
    require(len(audits) == terminal['audited_shards'] == 20, 'FIRST_POOL_AUDIT_COUNT')
    for path in audits:
        receipt = reader.read(path)
        index = path.with_name(path.name.replace('.audit.json', '.jsonl'))
        sub = reader.read(index, 'jsonl')
        require(receipt['integrity_pass'] is True, 'SUBSHARD_NOT_STRICT')
        require(reader.hashes[str(index.relative_to(ROOT))] == receipt['index_sha256'], 'SUBSHARD_INDEX_HASH')
        require_counts(count_rows(sub), {
            'strict_routes': receipt['certified_routes'],
            'instruction_records': receipt['instruction_records'],
            'instruction_conditioned_decisions': receipt['instruction_conditioned_decisions']})
        quarantine = reader.read(base / receipt['quarantine_manifest'])
        require(not {r['job_id'] for r in sub} & {r['job_id'] for r in quarantine}, 'QUARANTINE_IN_INDEX')
        rows += [dict(r, sourceRoot=str(base.relative_to(ROOT))) for r in sub]
        evidence.append(str(path.relative_to(ROOT)))
    counts = count_rows(rows)
    require_counts(counts, dict(strict_routes=770, instruction_conditioned_decisions=167870))
    pools.append(('ordinary_scale_v1/recovery_v4', rows, counts, evidence))
    for name, n_routes, n_actions in MERGES:
        base = PIPE / name
        receipt = reader.read(base / 'RESULT.json')
        require(receipt.get('training_started') is False and receipt.get('scientific_pass') is False,
                'UNEXPECTED_PRODUCTION_GRADE')
        require(receipt['strict_routes'] == n_routes
                and receipt['instruction_conditioned_decisions'] == n_actions, 'FROZEN_POOL_COUNT_CHANGED')
        index = base / 'TRAINING_INDEX.jsonl'
        rows = reader.read(index, 'jsonl')
        require(reader.hashes[str(index.relative_to(ROOT))] ==
                receipt.get('index_sha256', receipt.get('training_index_sha256')), 'MERGE_INDEX_HASH')
        counts = count_rows(rows)
        require_counts(counts, {k: receipt[k] for k in counts})
        pools.append((name, rows, counts, [str((base / 'RESULT.json').relative_to(ROOT))]))
    return pools


def quantiles(values):
    values = sorted(values)
    return {name: values[round((len(values) - 1) * q)] for name, q in
            [('min', 0), ('p25', .25), ('p50', .5), ('p75', .75), ('p95', .95), ('max', 1)]}


def dev_sources(reader, split):
    """Only official-train source metadata; no reserved trajectory/image answers."""
    rows = reader.read(PIPE / 'ordinary_scale_v1/SOURCE_MANIFEST.jsonl', 'jsonl')
    grouped = collections.defaultdict(dict)
    for row in rows:
        if row['split'] != 'INTERNAL_DEV':
            continue
        require(row['scene_id'] in split['INTERNAL_DEV'], 'DEV_HOUSE_MISMATCH')
        key = (row['scene_id'], row['source'])
        physical = row['physical_source_route_sha256']
        current = grouped[key].get(physical)
        if current is None or str(row['episode_id']) < str(current['episode_id']):
            grouped[key][physical] = row
    chosen = []
    for house in split['INTERNAL_DEV']:
        for source in ('R2R', 'RxR'):
            pool = grouped[(house, source)]
            keys = sorted(pool, key=lambda k: digest(('Q35N_ORDINARY_DEV_V2:' + k).encode()))
            chosen += [dict(pool[k], usage='closed_loop_evaluation_only_not_training',
                            simulation_evaluated=False) for k in keys[:10]]
    require({row['scene_id'] for row in chosen} == set(split['INTERNAL_DEV']), 'DEV_HOUSE_WITHOUT_ANY_SOURCE')
    return chosen


def save_new(path, value, jsonl=False):
    with path.open('xb') as f:
        if jsonl:
            for row in value:
                f.write(canonical(row) + b'\n')
        else:
            f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n')


def main(output):
    # Import only CPU adapter, never a model or old trainer with import side effects.
    import sys
    sys.path.insert(0, str(HERE))
    import data
    output = Path(output).resolve()
    require(output.parent == HERE and output.name.startswith('snapshot_'), 'NEW_SNAPSHOT_NAMESPACE_REQUIRED')
    require(not output.exists(), 'SNAPSHOT_EXISTS_USE_NEW_VERSION')
    reader = Reader()
    split = reader.read(SPLIT)
    require(not set(split['FIT']) & (set(split['INTERNAL_DEV']) | set(split['INTERNAL_CONFIRM'])), 'SPLIT_OVERLAP')
    pools = closed_pools(reader)
    rows, duplicates, seen = [], [], set()
    route_rollouts = {}
    physical_action_counts = collections.Counter()
    conditioned_action_counts = collections.Counter()
    house_stats = collections.defaultdict(lambda: collections.Counter())
    source_stats = collections.defaultdict(lambda: collections.Counter())
    sup_cache = {}
    for pool_name, pool_rows, counts, evidence in pools:
        for row in pool_rows:
            require(row['split'] == 'FIT' and row['scene_group'] in split['FIT'], 'NON_FIT_TRAINING_ROW')
            require(row['source'] in ('R2R', 'RxR'), 'UNDECLARED_SOURCE')
            require(row['quality_tier'] == 'ORDINARY_REPLAY_AND_EXPORT_VERIFIED', 'NON_STRICT_ROW')
            base = source_path(ROOT / row['sourceRoot'])
            ppath = source_path(base / row['policy_file'])
            spath = source_path(base / row['supervision_file'])
            policy = reader.read(ppath)
            if spath not in sup_cache:
                sup = reader.read(spath)
                # Retain only the fields needed for metadata consistency checks.
                sup_cache[spath] = {k: sup[k] for k in
                                   ('actions', 'scene_id', 'physical_source_route_sha256')}
            sup = sup_cache[spath]
            require(set(policy) == {'instruction', 'rgb_sequence'}, 'POLICY_FIELD_LEAKAGE')
            require(isinstance(policy['instruction'], str) and policy['instruction'].strip(), 'EMPTY_INSTRUCTION')
            actions = [data.normalize_action(a) for a in sup['actions']]
            require(len(actions) == row['decisions'] == len(policy['rgb_sequence']), 'ACTION_FRAME_ALIGNMENT')
            require(actions and actions[-1] == 'STOP' and 'STOP' not in actions[:-1], 'TERMINAL_STOP')
            require(sup['scene_id'] == row['scene_group'] and
                    sup['physical_source_route_sha256'] == row['physical_source_route_sha256'], 'SOURCE_ROUTE_BINDING')
            data.validate_row(row)
            # References are validated structurally here; pixels stay on disk.
            require(all(isinstance(ref, str) and not Path(ref).is_absolute() and '..' not in Path(ref).parts
                        for ref in policy['rgb_sequence']), 'RGB_REFERENCE_ESCAPE')
            physical = (row['scene_group'], row['physical_source_route_sha256'])
            rollout_sha = digest(canonical(dict(rgb_sequence=policy['rgb_sequence'], actions=actions)))
            require(route_rollouts.setdefault(physical, rollout_sha) == rollout_sha, 'CONFLICTING_PHYSICAL_ROLLOUT')
            instruction_sha = digest(policy['instruction'].encode())
            record_id = digest(canonical([row['source'], physical, instruction_sha, rollout_sha]))
            if record_id in seen:
                duplicates.append(dict(record_id=record_id, pool=pool_name, policy_file=row['policy_file']))
                continue
            seen.add(record_id)
            out = dict(row, record_id=record_id, pool=pool_name, receipt_paths=evidence,
                       instruction_sha256=instruction_sha, rollout_sha256=rollout_sha,
                       policy_sha256=reader.hashes[str(ppath.relative_to(ROOT))],
                       supervision_sha256=reader.hashes[str(spath.relative_to(ROOT))],
                       instruction_provenance='official_human_instruction',
                       loss='ordinary_action_ce_only', action_count_by_class=dict(collections.Counter(actions)))
            rows.append(out)
            conditioned_action_counts.update(actions)
            house_stats[row['scene_group']].update(instructions=1, decisions=len(actions))
            source_stats[row['source']].update(instructions=1, decisions=len(actions))
        print(json.dumps(dict(pool=pool_name, receipt_counts=counts, cumulative_rows=len(rows)), ensure_ascii=False), flush=True)
    physical_seen = set()
    for row in rows:
        physical = (row['scene_group'], row['physical_source_route_sha256'])
        if physical not in physical_seen:
            physical_seen.add(physical)
            physical_action_counts.update(row['action_count_by_class'])
            house_stats[row['scene_group']]['physical_routes'] += 1
            source_stats[row['source']]['physical_routes'] += 1
    require(sum(c['instruction_conditioned_decisions'] for _, _, c, _ in pools) == 1394744, 'HANDOFF_TOTAL_MISMATCH')
    require(sum(c['strict_routes'] for _, _, c, _ in pools) == 5849, 'HANDOFF_ROUTE_TOTAL_MISMATCH')
    dev = dev_sources(reader, split)
    require(not physical_seen & {(r['scene_id'], r['physical_source_route_sha256']) for r in dev}, 'TRAIN_DEV_OVERLAP')
    rows.sort(key=lambda r: r['record_id'])
    result = dict(status='CPU_METADATA_SNAPSHOT_PREPARED_NOT_TRAINED', created_unix=time.time(),
                  counts=count_rows(rows), houses=len(house_stats), by_house=dict(house_stats), by_source=dict(source_stats),
                  action_counts_instruction_conditioned=dict(conditioned_action_counts),
                  action_counts_physical=dict(physical_action_counts),
                  instruction_decisions_quantiles=quantiles([r['decisions'] for r in rows]),
                  stop_fraction=conditioned_action_counts['STOP'] / sum(conditioned_action_counts.values()),
                  identical_instruction_rollout_duplicates_removed=len(duplicates),
                  input_receipt_counts=[dict(pool=n, **c) for n, _, c, _ in pools],
                  dev_source_episodes=len(dev), dev_houses=len(split['INTERNAL_DEV']),
                  dev_source_shortfalls=[dict(house=h, source=s, requested=10,
                      selected=sum(r['scene_id']==h and r['source']==s for r in dev))
                      for h in split['INTERNAL_DEV'] for s in ('R2R','RxR')
                      if sum(r['scene_id']==h and r['source']==s for r in dev)<10],
                  dev_trajectories_generated=0, full_rgb_reaudit_performed=False,
                  integrity_scope='closed_quality_receipt_and_index_hash_plus_policy_supervision_metadata; RGB verified on decode',
                  metadata_bytes_read=reader.bytes, metadata_wall_seconds=time.monotonic()-reader.start,
                  envdrop_included=False, special_families_included=False,
                  training_started=False, gpu_run_allowed=False, scientific_pass=False,
                  same_house_as_historical_fit=True, no_global_unexposed_claim=True)
    # Recheck receipts/indexes and policy/supervision bytes before freeze. Active
    # EnvDrop metadata is not an input. Do not recursively scan content stores.
    for relative in list(reader.hashes):
        reader.read(ROOT / relative, kind='bytes')
    result['metadata_bytes_read'] = reader.bytes
    result['metadata_wall_seconds'] = time.monotonic()-reader.start
    output.mkdir()
    save_new(output/'TRAINING_INDEX.jsonl', rows, jsonl=True)
    save_new(output/'DEV_SOURCE_INDEX.jsonl', dev, jsonl=True)
    save_new(output/'SPLIT.json', split)
    save_new(output/'DUPLICATES.json', duplicates)
    save_new(output/'SOURCE_HASHES.json', reader.hashes)
    result['training_index_sha256'] = digest((output/'TRAINING_INDEX.jsonl').read_bytes())
    result['source_hashes_sha256'] = digest((output/'SOURCE_HASHES.json').read_bytes())
    result['dev_source_index_sha256'] = digest((output/'DEV_SOURCE_INDEX.jsonl').read_bytes())
    save_new(output/'RESULT.json', result)
    seal = {p.name: digest(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    save_new(output/'SEAL.json', seal)
    print(json.dumps(dict(snapshot=str(output), **{k:result[k] for k in
          ('counts','houses','stop_fraction','dev_source_episodes','training_started')}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'snapshot_v1')
    main(parser.parse_args().output)
