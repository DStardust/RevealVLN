"""CPU-only one-shot freeze; never creates a simulator or signals a process."""
import collections
import json
from pathlib import Path
import shlex
import sys
import time
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as c
import transport as t
import run

RESCUE_SHA = '2d20b5dbd903421d03365d5c4eaf71e1efb2a1fdda4f0e4ca9d969eb2ddc1731'

def partition(jobs):
    assert len(jobs) == 9661 and len({j['job_id'] for j in jobs}) == 9661
    assert len({j['physical_source_route_sha256'] for j in jobs}) == 9661
    houses = collections.defaultdict(list)
    for job in jobs:
        houses[job['scene_id']].append(job)
    assert len(houses) >= 3
    sentinel = []
    ordered = sorted(houses)
    for offset in range(3):
        for house in ordered[:3]:
            assert len(houses[house]) > offset
            sentinel.append(houses[house][offset])
    used = {j['job_id'] for j in sentinel}
    tail = [j for j in jobs if j['job_id'] not in used]
    shards = [sentinel[i:i+3] for i in range(0, 9, 3)] + [tail[i:i+250] for i in range(0, len(tail), 250)]
    assert len(shards) == 42
    assert {j['job_id'] for part in shards for j in part} == {j['job_id'] for j in jobs}
    assert sum(map(len, shards)) == 9661
    return shards

def source_checks():
    assert c.sha(t.REC / 'RESCUE_JOBS.json') == RESCUE_SHA
    jobs = c.read(t.REC / 'RESCUE_JOBS.json')
    original = c.read(t.MANIFEST / 'JOBS.json')
    byid = {j['job_id']: j for j in original}
    assert len(byid) == 20000
    excluded = set(c.read(t.MANIFEST / 'PHYSICAL_EXCLUSION.json')['physical_route_keys'])
    partial = {r['job_id'] for r in c.read(t.REC / 'PARTIAL_NOT_RETRIED.json')}
    fit = set(c.read(c.BASE / 'SPLIT_FREEZE.json')['FIT'])
    attempted = set()
    old_parent = HERE.parent / 'envdrop_production_v1'
    for shard in range(21):
        routes = old_parent / f'shard_{shard:04d}' / 'routes'
        if routes.exists():
            attempted.update(p.name for p in routes.iterdir())
    assert len(attempted) == 10339 and partial <= attempted
    assert attempted.isdisjoint(j['job_id'] for j in jobs)
    assert attempted | {j['job_id'] for j in jobs} == set(byid)
    for j in jobs:
        assert j == byid[j['job_id']]
        assert j['split'] == 'FIT' and j['scene_id'] in fit
        assert j['physical_source_route_sha256'] not in excluded
        assert j['source'] == 'ENVDROP_OFFICIAL_CE_TRAIN'
        assert j['instruction_alias_episodes'] == [j['episode']]
        assert j['episode']['instruction']['language'] == 'en'
        assert j['full_natural_language_semantics_certified'] is False
    return jobs, partition(jobs)

def identities_and_training():
    train = c.LINE / 'sft_acceptance/ordinary_sync_recovery_v1'
    result = c.read(train / 'formal/attempt_001/RESULT.json')
    assert result['status'] == 'STOPPED' and result['stop'] == ['SIGNAL']
    checkpoint = Path(result['latest_checkpoint'])
    receipt = c.read(checkpoint.with_suffix('.pt.json'))
    assert c.sha(checkpoint) == receipt['sha256']
    request = c.read(HERE / 'TRAIN_STOP_REQUEST.json')
    for old in request['training_tree']:
        assert not run.same_process_running(old['pid'], old['starttime_ticks']), 'TRAINING_STILL_ACTIVE'
    for old in request['protected_processes'][:2]:
        actual = run.process_identity(old['pid'])
        assert actual['starttime_ticks'] == old['starttime_ticks'] and actual['cmdline'] == old['argv'], 'EVALUATION_IDENTITY'
    restored = c.read(train / 'lease_v1/RESTORATION.json')['holders']
    assert len(restored) == 3 and all(r['restored'] for r in restored)
    runbook = c.read(train / 'RUNBOOK_FILESTORE.json')
    holders = []
    for target, closed in zip(runbook['holders'], restored):
        gpu = target['gpu_index']; assert gpu in c.LANES
        pid = int(run.call('tmux', 'display-message', '-p', '-t', target['pane'], '#{pane_pid}'))
        assert pid == closed['pid']
        ident = run.process_identity(pid)
        assert ident['cmdline'] == shlex.split(target['command']) and ident['cwd'] == str(c.ROOT) and ident['proc_uid'] == 0
        ident.update(gpu_device=gpu, gpu_uuid=target['gpu_uuid'], pane_target=target['pane'],
                     pane_id=target['pane_id'], project_cache_environment=run.project_environment(pid))
        run.verify_identity(ident)
        snap = run.gpu_snapshot(ident)
        assert snap['processes'].get(pid, 0) > 20000
        external = [v for p, v in snap['processes'].items() if p != pid]
        assert all(0 <= v <= 768 for v in external) and sum(external) < 1024
        holders.append(ident)
    return dict(holders=holders), dict(training_result=result, checkpoint_receipt=receipt,
        checkpoint_sha256_verified=True, training_tree_exited=True, holders_restored=restored,
        evaluation_preserved=True, time_unix=time.time())

def main():
    assert not (HERE / 'INPUT_LOCK.json').exists() and not t.DATA.exists(), 'FRESH_NODE_ONLY'
    tests = c.read(HERE / 'CPU_TESTS_FINAL.json'); assert tests['passed'] and tests['failures'] == tests['errors'] == 0
    for name, digest in tests['code_sha256'].items():
        assert c.sha(HERE / name) == digest, ('TESTED_CODE_CHANGED', name)
    jobs, shards = source_checks()
    identities, closed = identities_and_training()
    lock = c.read(t.OLD / 'INPUT_LOCK.json')
    for name, digest in lock.items():
        assert c.sha(c.ROOT / name) == digest, ('INHERITED_LOCK_CHANGED', name)
    print('ALL_INHERITED_INPUTS_VERIFIED', len(lock), flush=True)
    assets = c.read(t.OLD / 'ASSETS.json')
    for j in jobs:
        for suffix in ('.glb', '.navmesh', '.house', '_semantic.ply'):
            name = f"third_party/ETP-R1/data/scene_datasets/mp3d/{j['scene_id']}/{j['scene_id']}{suffix}"
            assert name in assets and lock[name] == assets[name]
    cfg = dict(node='ORDINARY_ENVDROP_UNATTEMPTED_GPU345_V2', runtime_allowed=False,
        executable=False, training_allowed=False, routes=9661, houses=len({j['scene_id'] for j in jobs}),
        gpu_to_shards={str(g): list(s) for g, s in c.LANES.items()},
        shard_seconds={str(s): 600 if s < 3 else 3000 for s in c.SELECTED_SHARDS},
        lane_seconds={str(g): 43200 for g in c.LANES}, cleanup_margin_seconds=120,
        per_shard_max_bytes=12*1024**3, metadata_and_merge_max_bytes=8*1024**3,
        total_max_bytes=512*1024**3, worker_ram_bytes=12*1024**3, own_gpu_mib=4096,
        sentinel_required_strict_routes_per_gpu=3, automatic_continuation_after_strict_gate=True,
        retry_failed_or_partial=False, restore_exact_holder=True, external_process_signals_allowed=False,
        original_quality_thresholds_unchanged=True, full_natural_language_semantics_certified=False,
        source_grade='OFFICIAL_SYNTHETIC_ENGLISH_CE_PORT_NOT_HUMAN',
        old_terminal_routes=10338, old_partial_not_retried=1, source_rescue_sha256=RESCUE_SHA,
        scientific_pass=False, new_training_automatic_restart=False)
    outputs = []
    for shard, rows in enumerate(shards):
        root = c.shard_root(shard); root.mkdir(parents=True, exist_ok=False); (root / 'quality').mkdir()
        for name, value in {'JOBS.json': rows, 'SPLIT_FREEZE.json': c.read(c.BASE / 'SPLIT_FREEZE.json'),
            'SOURCE_INVENTORY.json': c.read(t.MANIFEST / 'SOURCE_INVENTORY.json')}.items():
            c.save(root / name, value); outputs.append(root / name)
    auth = dict(approved=True, user_request='把训练停了开始扩产吧', scope='new ordinary data only; protected GPU1 evaluation; no training restart',
                project_root=str(c.ROOT), configuration=cfg)
    for name, value in {'IDENTITIES.json': identities, 'TRAIN_CLOSED.json': closed,
        'ASSETS.json': assets, 'PREPARED_CONFIG.json': cfg, 'AUTHORIZATION.json': auth,
        'JOBS.json': jobs}.items():
        c.save(HERE / name, value); outputs.append(HERE / name)
    paths = list(HERE.glob('*.py')) + [HERE / 'SPEC_ZH.md', HERE / 'CPU_TESTS_FINAL.json'] + outputs
    paths += [t.OLD / name for name in t.SEALED] + [t.OLD / 'INPUT_LOCK.json']
    paths += [t.REC / name for name in ('RESCUE_JOBS.json', 'INVENTORY.json', 'PARTIAL_NOT_RETRIED.json', 'INPUT_HASHES.json')]
    lock.update({str(p.relative_to(c.ROOT)): c.sha(p) for p in paths})
    c.save(HERE / 'INPUT_LOCK.json', lock)
    for shard in c.SELECTED_SHARDS:
        c.save(c.shard_root(shard) / 'INPUT_LOCK.json', lock)
    print(json.dumps(dict(status='CPU_PREPARED_AWAITING_MAIN_APPROVAL', routes=len(jobs), shards=len(shards),
        houses=cfg['houses'], input_bindings=len(lock), input_lock_sha256=c.sha(HERE / 'INPUT_LOCK.json'),
        lane_routes={g: sum(len(shards[s]) for s in lane) for g, lane in c.LANES.items()},
        sentinel_job_ids=[[j['job_id'] for j in rows] for rows in shards[:3]])), flush=True)

if __name__ == '__main__':
    main()
