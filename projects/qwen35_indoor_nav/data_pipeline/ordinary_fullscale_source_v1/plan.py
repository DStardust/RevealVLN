"""CPU-only full remaining English FIT source census; never generated supervision."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
BASE = HERE.parent / 'ordinary_scale_v1'
PARALLEL = HERE.parent / 'ordinary_parallel_v1'
PILOT = HERE.parent / 'ordinary_pilot_v1'
MAX_SHARD_ROUTES = 1000
TARGET = 3000000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    path = Path(path).resolve(strict=True)
    assert path.is_relative_to(ROOT), ('OUTSIDE_PROJECT', str(path))
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path).resolve(strict=True)
    assert path.is_relative_to(ROOT)
    return json.loads(path.read_text())


def save(name, value):
    path = HERE / name
    assert path.resolve().is_relative_to(HERE)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == content, ('NEW_VERSION_REQUIRED', name)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            handle.write(content)


def key(episode):
    return hashlib.sha256(canonical({k: episode[k] for k in
        ('scene_id', 'start_position', 'start_rotation', 'reference_path', 'goals')})).hexdigest()


def features(job):
    path = job['episode']['reference_path']
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    assert math.isfinite(length) and length >= 0
    return dict(reference_waypoints=len(path), reference_polyline_m=length,
                aliases=len(job['instruction_alias_episodes']))


def summary(values):
    values = sorted(values)
    if not values:
        return dict(n=0, minimum=None, q25=None, median=None, q75=None, maximum=None, mean=None, total=0)
    def q(fraction):
        index = (len(values) - 1) * fraction
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] * (hi-index) + values[hi] * (index-lo) if hi != lo else values[lo]
    return dict(n=len(values), minimum=values[0], q25=q(.25), median=q(.5), q75=q(.75),
                maximum=values[-1], mean=statistics.mean(values), total=sum(values))


def partition(jobs, limit=MAX_SHARD_ROUTES):
    """Preserve whole houses whenever <=limit; physical route and aliases indivisible."""
    assert isinstance(limit, int) and limit > 0
    houses = collections.defaultdict(list)
    for job in sorted(jobs, key=lambda j: (j['scene_id'], j['source'], j['physical_source_route_sha256'])):
        houses[job['scene_id']].append(job)
    groups = []
    for house in sorted(houses):
        rows = houses[house]
        groups.extend(rows[i:i+limit] for i in range(0, len(rows), limit))
    shards = []
    for group in sorted(groups, key=lambda g: (-len(g), g[0]['scene_id'], g[0]['physical_source_route_sha256'])):
        viable = [i for i, shard in enumerate(shards) if len(shard)+len(group) <= limit]
        if not viable:
            shards.append([])
            sid = len(shards)-1
        else:
            sid = min(viable, key=lambda i: (limit-len(shards[i])-len(group), i))
        shards[sid].extend(group)
    for shard in shards:
        shard.sort(key=lambda j: (j['scene_id'], j['source'], j['physical_source_route_sha256']))
    return shards


def validate(jobs, shards, excluded, fit):
    physical = [j['physical_source_route_sha256'] for j in jobs]
    assert len(physical) == len(set(physical))
    assert set(physical).isdisjoint(excluded)
    assert {j['scene_id'] for j in jobs} <= set(fit)
    assert all(j['split'] == 'FIT' for j in jobs)
    assert sorted(physical) == sorted(j['physical_source_route_sha256'] for shard in shards for j in shard)
    aliases = [(j['source'], str(e['episode_id'])) for j in jobs for e in j['instruction_alias_episodes']]
    assert len(aliases) == len(set(aliases))
    for shard in shards:
        assert 0 < len(shard) <= MAX_SHARD_ROUTES
    for j in jobs:
        assert key(j['episode']) == j['physical_source_route_sha256']
        assert j['instruction_alias_episodes']
        for e in j['instruction_alias_episodes']:
            assert key(e) == j['physical_source_route_sha256']
            assert e['instruction'].get('language', 'en').startswith('en')
            assert e['instruction']['instruction_text'].strip()


def source_pool(inventory, fit, inputs):
    pool = []
    for source in inventory['sources']:
        path = ROOT / source['path']
        assert sha(path) == source['sha256']
        inputs[source['path']] = source['sha256']
        with gzip.open(path, 'rt') as handle:
            episodes = json.load(handle)['episodes']
        grouped = collections.defaultdict(list)
        seen = set()
        for e in episodes:
            eid = str(e['episode_id'])
            assert eid not in seen
            seen.add(eid)
            if not e['instruction'].get('language', 'en').startswith('en'):
                continue
            scene = Path(e['scene_id']).parent.name
            assert Path(e['scene_id']).parts == ('mp3d', scene, scene+'.glb')
            if scene not in fit:
                continue
            assert e['reference_path'] and e['instruction']['instruction_text'].strip()
            grouped[key(e)].append(e)
        for physical, aliases in sorted(grouped.items()):
            aliases.sort(key=lambda e: int(e['episode_id']))
            for e in aliases:
                e['instruction'] = {k:v for k,v in e['instruction'].items()
                                    if k in ('instruction_text', 'language', 'instruction_id')}
            e = aliases[0]
            pool.append(dict(job_id=source['source'].lower()+'_'+physical[:20], source=source['source'],
                source_path=source['path'], source_sha256=source['sha256'],
                scene_id=Path(e['scene_id']).parent.name, split='FIT', physical_source_route_sha256=physical,
                episode=e, instruction_alias_episodes=aliases))
    # A duplicate across sources must not silently drop aliases or assign a second physical rollout.
    physical = [j['physical_source_route_sha256'] for j in pool]
    assert len(physical) == len(set(physical)), 'CROSS_SOURCE_DUPLICATE_REQUIRES_EXPLICIT_ALIAS_MERGE_VERSION'
    return sorted(pool, key=lambda j: (j['scene_id'], j['source'], j['physical_source_route_sha256']))


def closed_evidence(inputs):
    """Immutable indices only. No active parallel outcomes used to select candidates/fit model."""
    jobs = read(BASE/'JOBS.json')
    accepted = {}
    for path in sorted((BASE/'shards').glob('*.audit.json')):
        audit = read(path)
        index = path.with_name(path.name.replace('.audit.json', '.jsonl'))
        assert audit['integrity_pass'] and sha(index) == audit['index_sha256']
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        assert len(rows) == audit['instruction_records']
        assert sum(r['decisions'] for r in rows) == audit['instruction_conditioned_decisions']
        for row in rows:
            accepted.setdefault(row['job_id'], []).append(row)
        for p in (path, index, BASE/audit['quarantine_manifest']):
            inputs[str(p.relative_to(ROOT))] = sha(p)
    assert len(accepted) == 770 and sum(map(len, accepted.values())) == 2307
    by_source = {}
    for source in ('R2R', 'RxR'):
        attempted = [j for j in jobs if j['source'] == source]
        valid = [j for j in attempted if j['job_id'] in accepted]
        lengths, decisions, ratios = [], [], []
        for j in valid:
            rows = accepted[j['job_id']]
            assert len(rows) == len(j['instruction_alias_episodes'])
            assert len({r['decisions'] for r in rows}) == 1
            n = rows[0]['decisions']
            length = features(j)['reference_polyline_m']
            lengths.append(length); decisions.append(n)
            if length > 0:
                ratios.append((n-1)/length)
        slope = (sum(decisions)-len(valid))/sum(lengths)
        by_source[source] = dict(attempted_routes=len(attempted), strict_routes=len(valid),
            strict_route_yield=len(valid)/len(attempted),
            strict_instruction_records=sum(len(accepted[j['job_id']]) for j in valid),
            unique_route_decisions=sum(decisions),
            instruction_conditioned_decisions=sum(r['decisions'] for j in valid for r in accepted[j['job_id']]),
            decisions=summary(decisions), reference_polyline_m=summary(lengths),
            decisions_minus_stop_per_reference_meter=summary(ratios), pooled_slope_minus_stop=slope,
            attempted_reference_polyline_m=summary([features(j)['reference_polyline_m'] for j in attempted]))
    pilot_audit = read(PILOT/'AUDIT_RESULT.json')
    pilot_rows = [json.loads(line) for line in (PILOT/'TRAINING_INDEX.jsonl').read_text().splitlines()]
    assert len(pilot_rows) == pilot_audit['instruction_records'] == 298
    pilot_decisions = {}
    for relative in sorted({r['supervision_file'] for r in pilot_rows}):
        path = PILOT/relative
        pilot_decisions[relative] = len(read(path)['actions'])
        inputs[str(path.relative_to(ROOT))] = sha(path)
    assert sum(pilot_decisions.values()) == pilot_audit['unique_route_decisions']
    assert sum(pilot_decisions[r['supervision_file']] for r in pilot_rows) == pilot_audit['instruction_conditioned_supervised_decisions']
    for p in (PILOT/'AUDIT_RESULT.json', PILOT/'TRAINING_INDEX.jsonl'):
        inputs[str(p.relative_to(ROOT))] = sha(p)
    return dict(calibration='closed_oldscale_1000_only_no_active_outcome_conditioning', by_source=by_source,
        oldscale_instruction_conditioned_decisions=sum(v['instruction_conditioned_decisions'] for v in by_source.values()),
        pilot_instruction_conditioned_decisions=pilot_audit['instruction_conditioned_supervised_decisions'],
        known_closed_instruction_conditioned_decisions=pilot_audit['instruction_conditioned_supervised_decisions']+
            sum(v['instruction_conditioned_decisions'] for v in by_source.values()),
        pixel_audit_repeated_this_node=False, scope='published_strict_index_hash_and_accounting_readback')


def predict(jobs, evidence):
    out = {}
    for source in ('R2R', 'RxR'):
        selected = [j for j in jobs if j['source'] == source]
        observed = evidence['by_source'][source]
        scenarios = {}
        slopes = dict(pooled=observed['pooled_slope_minus_stop'],
                      ratio_q25=observed['decisions_minus_stop_per_reference_meter']['q25'],
                      ratio_q75=observed['decisions_minus_stop_per_reference_meter']['q75'])
        for name, slope in slopes.items():
            physical = sum(min(513, 1+slope*features(j)['reference_polyline_m']) for j in selected)
            conditioned = sum(min(513, 1+slope*features(j)['reference_polyline_m'])*features(j)['aliases'] for j in selected)
            scenarios[name] = dict(all_pass_physical_decisions=physical,
                all_pass_instruction_conditioned_decisions=conditioned,
                yield_adjusted_physical_decisions=physical*observed['strict_route_yield'],
                yield_adjusted_instruction_conditioned_decisions=conditioned*observed['strict_route_yield'])
        out[source] = dict(routes=len(selected), aliases=sum(features(j)['aliases'] for j in selected),
            houses=len({j['scene_id'] for j in selected}),
            source_reference_polyline_m=summary([features(j)['reference_polyline_m'] for j in selected]),
            source_reference_waypoints=summary([features(j)['reference_waypoints'] for j in selected]),
            formal_compiler_cap_instruction_decisions=513*sum(features(j)['aliases'] for j in selected),
            scenarios=scenarios)
    return out


def main():
    inputs = {}
    for path, expected in read(BASE/'INPUT_LOCK.json').items():
        assert sha(ROOT/path) == expected, path
        inputs[path] = expected
    for path, expected in read(PARALLEL/'INPUT_LOCK.json')['immutable'].items():
        assert sha(ROOT/path) == expected, path
        inputs[path] = expected
    pilot, old, parallel = (read(root/'JOBS.json') for root in (PILOT, BASE, PARALLEL))
    assert [len(pilot), len(old), len(parallel)] == [100, 1000, 1000]
    excluded_jobs = pilot+old+parallel
    excluded = {j['physical_source_route_sha256'] for j in excluded_jobs}
    assert len(excluded) == 2100
    for p in (PILOT/'JOBS.json', BASE/'JOBS.json', PARALLEL/'JOBS.json',
              BASE/'INPUT_LOCK.json', PARALLEL/'INPUT_LOCK.json'):
        inputs[str(p.relative_to(ROOT))] = sha(p)
    split = read(BASE/'SPLIT_FREEZE.json')
    assert len(split['FIT']) == 51
    assert set(split['FIT']).isdisjoint(split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
    inventory = read(BASE/'SOURCE_INVENTORY.json')
    all_jobs = source_pool(inventory, set(split['FIT']), inputs)
    assert excluded <= {j['physical_source_route_sha256'] for j in all_jobs}
    jobs = [j for j in all_jobs if j['physical_source_route_sha256'] not in excluded]
    shards = partition(jobs)
    validate(jobs, shards, excluded, split['FIT'])
    evidence = closed_evidence(inputs)
    forecasts = dict(full_fit_hypothetical=predict(all_jobs, evidence),
                     remaining_unattempted=predict(jobs, evidence),
                     active_assigned_1000=predict(parallel, evidence))
    # Previously failed/quarantined candidates remain excluded; only not-yet-terminal output is forecast.
    future = forecasts['remaining_unattempted'], forecasts['active_assigned_1000']
    projection = {}
    for scenario in ('pooled', 'ratio_q25', 'ratio_q75'):
        for yield_mode in ('all_pass', 'yield_adjusted'):
            total = evidence['known_closed_instruction_conditioned_decisions'] + sum(
                block[source]['scenarios'][scenario][yield_mode+'_instruction_conditioned_decisions']
                for block in future for source in ('R2R', 'RxR'))
            projection[scenario+'_'+yield_mode] = dict(projected_total_instruction_conditioned_decisions=total,
                projected_gap_to_3000000=max(0, TARGET-total), reaches_target=total >= TARGET)
    save('JOBS.json', jobs)
    save('SPLIT_FREEZE.json', split)
    save('SOURCE_INVENTORY.json', inventory)
    save('EXCLUSION_MANIFEST.json', dict(pilot=100, oldscale=1000, parallel_assigned=1000,
        policy='exclude_all_assigned_regardless_of_terminal_failure_quarantine_or_inflight',
        physical_route_keys=sorted(excluded)))
    save('CLOSED_EMPIRICAL_EVIDENCE.json', evidence)
    source_rows = []
    for j in all_jobs:
        source_rows.append(dict(source=j['source'], scene_id=j['scene_id'], job_id=j['job_id'],
            physical_source_route_sha256=j['physical_source_route_sha256'], split='FIT',
            assigned_previously=j['physical_source_route_sha256'] in excluded,
            alias_episode_ids=[str(e['episode_id']) for e in j['instruction_alias_episodes']], **features(j)))
    save('FULL_FIT_SOURCE_CENSUS.json', source_rows)
    shard_records = []
    for sid, shard in enumerate(shards):
        path = f'manifests/shard_{sid:04d}/JOBS.json'
        save(path, shard)
        shard_records.append(dict(shard_id=sid, jobs_path=path, jobs_sha256=sha(HERE/path),
            output_root=f'production/shard_{sid:04d}', routes=len(shard),
            aliases=sum(len(j['instruction_alias_episodes']) for j in shard),
            houses=sorted({j['scene_id'] for j in shard}),
            source_routes=dict(collections.Counter(j['source'] for j in shard)), executable=False))
    save('FORECAST.json', dict(status='MODEL_BASED_CPU_ESTIMATE_NOT_GENERATED', target=TARGET,
        forecasts=forecasts, projected_final_with_old_failures_preserved=projection,
        method='source-specific pooled (decisions-1)/polyline_m; +1 STOP; cap513; separate empirical source yield',
        caveats=['reference_polyline_m is Euclidean metadata, not actual navmesh path length',
                 'all_pass is a sensitivity scenario, not a proven realizable upper bound',
                 'q25/q75 ratios are sensitivity scenarios, not confidence intervals',
                 'source-specific yields may shift across houses, lengths and candidate ranks',
                 'no active parallel outcomes used in calibration or candidate selection',
                 'episodes, aliases, physical decisions and instruction-conditioned decisions differ',
                 'no repeated epochs or duplicated prefixes counted as new supervision'],
        generated_action_records_this_node=0, scientific_pass=False))
    save('PLAN.json', dict(status='CPU_FULL_FIT_MANIFESTS_READY_NOT_GENERATED', executable=False,
        full_fit_routes=len(all_jobs), full_fit_aliases=sum(len(j['instruction_alias_episodes']) for j in all_jobs),
        full_fit_houses=len({j['scene_id'] for j in all_jobs}), excluded_routes=len(excluded),
        remaining_routes=len(jobs), remaining_aliases=sum(len(j['instruction_alias_episodes']) for j in jobs),
        remaining_houses=len({j['scene_id'] for j in jobs}), shards=shard_records,
        generated_routes=0, generated_action_records=0, official_eval_content_read=False,
        no_gpu_launcher=True, next_runtime_requires_independent_approval_and_asset_hash_lock=True,
        schema_unchanged_from_ordinary_parallel=True, no_global_unexposed_claim=True))
    save('MANIFEST_SCHEMA.json', dict(schema_version='ORDINARY_FULLSCALE_SOURCE_V1',
        job_schema='unchanged ordinary_scale_v1 JOBS entry', required_job_fields=sorted(jobs[0]),
        key_fields=['scene_id','start_position','start_rotation','reference_path','goals'],
        source_split='official train, English guide only, frozen 51 FIT houses',
        partition_invariants=['all aliases of one physical route in exactly one shard',
            'each shard <=1000 routes', 'whole houses preserved when <=1000 routes',
            'no prior 2100 assigned routes', 'independent writable output roots'],
        forbidden_interpretation='manifest is neither a physical rollout nor valid supervision'))
    for name in ('plan.py', 'test_plan.py', 'monitor.py', 'AUGMENTATION_CANDIDATES.json'):
        inputs[str((HERE/name).relative_to(ROOT))] = sha(HERE/name)
    save('INPUT_LOCK.json', inputs)
    generated = ['JOBS.json','SPLIT_FREEZE.json','SOURCE_INVENTORY.json','EXCLUSION_MANIFEST.json',
        'CLOSED_EMPIRICAL_EVIDENCE.json','FULL_FIT_SOURCE_CENSUS.json','FORECAST.json','PLAN.json',
        'MANIFEST_SCHEMA.json','INPUT_LOCK.json'] + [s['jobs_path'] for s in shard_records]
    save('MANIFEST_HASHES.json', {p:sha(HERE/p) for p in generated})
    print(json.dumps(dict(routes=len(jobs), aliases=sum(len(j['instruction_alias_episodes']) for j in jobs),
        shards=[r['routes'] for r in shard_records], projection=projection, executable=False), indent=2))


if __name__ == '__main__':
    main()
