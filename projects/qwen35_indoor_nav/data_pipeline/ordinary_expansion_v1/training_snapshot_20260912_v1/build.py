"""Versioned CPU-only final admission; unchanged strict auditor, no source writes."""
import collections
import concurrent.futures as futures
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import signal
import time
import traceback

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LINE = HERE.parents[2]
ROOT = LINE.parents[1]
REF = LINE / 'sft_acceptance/ordinary_baseline_v2'
RUN = HERE / 'run_001'
STRICT = EXP.parent / 'ordinary_scale_v1/audit.py'
STRICT_SHA = '37c3443c312cb635572b524cba4df2870ed835eea467cc8bcd5ed6aa029d97fe'


def scoped(path):
    path = Path(path).resolve(strict=True)
    assert path.is_relative_to(LINE), 'SOURCE_OUTSIDE_LINE'
    return path


def sha(path):
    h = hashlib.sha256()
    with scoped(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(scoped(path).read_text())


def rows(path):
    raw = scoped(path).read_bytes()
    assert raw and raw.endswith(b'\n'), 'PARTIAL_INDEX'
    return [json.loads(s) for s in raw.splitlines()]


def save(path, value):
    assert path.parent.resolve().is_relative_to(HERE)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, scoped(path))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def accept_group(group, fit, physical, aliases):
    assert group
    first = group[0]; key = first['physical_source_route_sha256']
    assert key not in physical, 'DUPLICATE_PHYSICAL_ROUTE'
    owner = (first['sourceRoot'], first['job_id'], first['scene_group'])
    local = set()
    for row in group:
        assert row['split'] == 'FIT' and row['scene_group'] in fit, 'HOUSE_LEAKAGE'
        assert row['physical_source_route_sha256'] == key
        assert (row['sourceRoot'], row['job_id'], row['scene_group']) == owner
        alias = (row['source'], Path(row['policy_file']).name)
        assert alias not in local and alias not in aliases, 'DUPLICATE_INSTRUCTION_ALIAS'
        local.add(alias)
    physical.add(key); aliases.update(local)


def inventory():
    hashes = {}; locks = []
    def bind(p): hashes[str(scoped(p).relative_to(ROOT))] = sha(p)
    def lock(p):
        f = scoped(p).open('rb'); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(f)
    for p in (HERE/'SPEC_ZH.md', HERE/'build.py', HERE/'test_build.py', STRICT): bind(p)
    assert sha(STRICT) == STRICT_SHA
    runner = load('snapshot_human_preflight', REF/'runner.py')
    _, human, report = runner.preflight(REF/'PROTOCOL.json', REF/'snapshot_v1', full_source_check=True)
    assert len(human) == 17548 and report['decisions_per_epoch'] == 1394744
    for name in ('SEAL.json', 'RESULT.json', 'TRAINING_INDEX.jsonl', 'SPLIT.json', 'SOURCE_HASHES.json'):
        bind(REF/'snapshot_v1'/name)
    hashes.update(read(REF/'snapshot_v1/SOURCE_HASHES.json'))
    split = read(REF/'snapshot_v1/SPLIT.json')
    fit = set(split['FIT'])
    assert fit.isdisjoint(split['INTERNAL_DEV'] + split['INTERNAL_CONFIRM'])
    sourcepath = EXP/'envdrop_source_v1/JOBS.json'; bind(sourcepath)
    source = read(sourcepath); jobs = {j['job_id']: j for j in source}
    assert len(jobs) == len(source) == 20000
    excluded_path = EXP/'envdrop_source_v1/PHYSICAL_EXCLUSION.json'; bind(excluded_path)
    excluded = set(read(excluded_path)['physical_route_keys'])
    allrows = []; pools = []
    for relative, count in [('recovery_inventory_20260911_v1/run_001', 10133),
                            ('runtime_gpu3_recovery_v1/old_salvage', 140),
                            ('runtime_gpu3_recovery_v1/merge', 3042)]:
        root = EXP/relative; result = read(root/'RESULT.json')
        index = root/'TRAINING_INDEX.jsonl'
        assert result['index_sha256'] == sha(index)
        part = rows(index); assert len(part) == count
        allrows.extend(part); pools.append(dict(pool=relative, routes=count, decisions=sum(r['decisions'] for r in part)))
        bind(index); bind(root/'RESULT.json')
    for runtime, gpu in [('runtime_v1', 6), ('runtime_v2', 3), ('runtime_v2', 4), ('runtime_v2', 5), ('runtime_gpu3_recovery_v1', 3)]:
        lane = EXP/runtime/'lanes'/f'gpu_{gpu}'
        lock(lane/'PRODUCER.lock')
        resultpath = lane/'attempt_000/RESULT.json'; result = read(resultpath); bind(resultpath)
        assert result['restoration']['restored'], 'UNFINISHED_LEASE'
        bind(EXP/runtime/'INPUT_LOCK.json')
        if runtime == 'runtime_v2' and gpu in (4, 5):
            assert result['error'] is None and len(result['workers']) == 14
            for worker in result['workers']: assert worker['returncode'] == 0
    for shard in range(42):
        if shard % 3 == 0: continue
        root = EXP/'envdrop_production_v2'/f'shard_{shard:04d}'
        assert read(root/'GENERATION_COMPLETE.json')['all_jobs_terminal']
        lock(root/'PRODUCER.lock')
        assigned = read(root/'JOBS.json'); terminal = rows(root/'LEDGER.jsonl')
        assert len(assigned) == len(terminal) == len({r['job_id'] for r in terminal})
        assert {j['job_id'] for j in assigned} == {r['job_id'] for r in terminal}
        replay = {r['job_id'] for r in terminal if r['status'] == 'CERTIFIED'}
        accepted = set(); quarantined = set()
        for path in sorted((root/'shards').glob('*.audit.json')):
            audit = read(path); index = path.with_name(path.name.replace('.audit.json', '.jsonl'))
            assert audit['integrity_pass'] and sha(index) == audit['index_sha256']
            part = rows(index); qp = root/audit['quarantine_manifest']; q = read(qp)
            assert len(q) == audit['quarantined_routes']
            assert all(r['status'] == 'QUARANTINED_NOT_TRAINING_DATA' for r in q)
            quarantined.update(r['job_id'] for r in q)
            assert len(part) == audit['instruction_records']
            assert sum(r['decisions'] for r in part) == audit['instruction_conditioned_decisions']
            accepted.update(r['job_id'] for r in part)
            allrows.extend(dict(r, sourceRoot=str(root.relative_to(ROOT))) for r in part)
            for p in (path, index, qp): bind(p)
        assert accepted.isdisjoint(quarantined) and accepted | quarantined == replay
        for name in ('GENERATION_COMPLETE.json', 'JOBS.json', 'LEDGER.jsonl', 'INPUT_LOCK.json'): bind(root/name)
    assert len(allrows) == 19566
    grouped = collections.defaultdict(list)
    for row in allrows: grouped[(row['sourceRoot'], row['job_id'])].append(row)
    assert len(grouped) == len(allrows), 'ENVDROP_SINGLE_ALIAS_REQUIRED'
    candidates = []; physical = set(); aliases = set(); jobmaps = {}
    human_groups = collections.defaultdict(list)
    for row in human: human_groups[row['physical_source_route_sha256']].append(row)
    for group in human_groups.values(): accept_group(group, fit, physical, aliases)
    for (rootstr, ident), part in sorted(grouped.items()):
        root = scoped(ROOT/rootstr); job = jobs[ident]
        assert job['physical_source_route_sha256'] not in excluded
        assert job['scene_id'] in fit and job['split'] == 'FIT'
        assert len(job['instruction_alias_episodes']) == len(part) == 1
        if rootstr not in jobmaps: jobmaps[rootstr] = {j['job_id']:j for j in read(root/'JOBS.json')}
        assert job == jobmaps[rootstr][ident]
        accept_group(part, fit, physical, aliases)
        for name in ('JOBS.json', 'INPUT_LOCK.json'): bind(root/name)
        for name in ('result.json', 'diagnostic.json', 'replay_certificate.json'):
            bind(root/'routes'/ident/name)
        for row in part:
            for field in ('policy', 'supervision'):
                p = scoped(root/row[field+'_file']); assert p.is_relative_to(root)
                digest = sha(p)
                if field+'_sha256' in row: assert row[field+'_sha256'] == digest, 'OLD_CONTENT_CHANGED'
                row[field+'_sha256'] = digest; bind(p)
        candidates.append(dict(root=rootstr, job=job, stored=part))
    assert len(physical) == 25415 and len(aliases) == 37114
    return dict(hashes=hashes, locks=locks, human=human, candidates=candidates, split=split, pools=pools)


def worker_init():
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))
    global strict
    assert sha(STRICT) == STRICT_SHA
    strict = load('original_strict_snapshot', STRICT)


def audit_one(item):
    root = scoped(ROOT/item['root'])
    fresh, refs, steps = strict.audit_route(root, item['job'])
    stored = item['stored']
    strip = lambda r: {k:v for k,v in r.items() if k not in ('sourceRoot', 'policy_sha256', 'supervision_sha256')}
    assert fresh == [strip(r) for r in stored], 'PREVIOUS_STRICT_INDEX_CHANGED'
    assert steps > 0 and len(fresh) == 1
    row = dict(stored[0])
    row.update(record_id=hashlib.sha256(('expanded-v1:'+row['sourceRoot']+':'+row['policy_file']).encode()).hexdigest(),
               instruction_provenance='official_synthetic_english_envdrop',
               natural_language_full_semantics_certified=False,
               loss='ordinary_inflection_weighted_action_ce')
    return row, len(refs)


def progress(**kw):
    p = RUN/'PROGRESS.pending'
    with p.open('w') as f: json.dump(dict(unix=time.time(), **kw), f, allow_nan=False)
    p.replace(RUN/'PROGRESS.json')


def main():
    RUN.mkdir(exist_ok=False); start = time.monotonic(); pool = None; inv = None
    def interrupt(signum, frame): raise RuntimeError('AUDIT_INTERRUPTED_'+str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM): signal.signal(sig, interrupt)
    signal.alarm(7200)
    try:
        progress(status='METADATA_PREFLIGHT', training_admitted=False)
        inv = inventory(); save(RUN/'INPUT_HASHES.json', inv['hashes'])
        save(RUN/'INVENTORY.json', dict(human_records=len(inv['human']), envdrop_routes=len(inv['candidates']), pools=inv['pools']))
        pool = futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'), initializer=worker_init)
        iterator = iter(inv['candidates']); pending = set(); accepted = []; rgb = 0; last = 0
        def submit():
            try: item = next(iterator)
            except StopIteration: return
            pending.add(pool.submit(audit_one, item))
        for _ in range(24): submit()
        while pending:
            done, pending = futures.wait(pending, timeout=5, return_when=futures.FIRST_COMPLETED)
            for f in done:
                row, n = f.result(); accepted.append(row); rgb += n; submit()
            if time.monotonic()-last > 5 or not pending:
                last = time.monotonic()
                progress(status='FULL_ENVDROP_PIXEL_REAUDIT', audited=len(accepted), total=19566,
                         checked_rgb_references=rgb, elapsed_seconds=last-start, training_admitted=False)
                print(json.dumps(dict(audited=len(accepted), total=19566)), flush=True)
        pool.shutdown(wait=True); pool = None
        assert len(accepted) == 19566
        progress(status='FINAL_METADATA_RECHECK', audited=19566, total=19566, training_admitted=False)
        for path, digest in inv['hashes'].items(): assert sha(ROOT/path) == digest, ('SOURCE_CHANGED', path)
        combined = sorted(inv['human']+accepted, key=lambda r:r['record_id'])
        assert len({r['record_id'] for r in combined}) == len(combined) == 37114
        assert sum(r['decisions'] for r in combined) == 2650347
        index = RUN/'TRAINING_INDEX.jsonl'
        with index.open('x') as f:
            for row in combined: f.write(json.dumps(row, sort_keys=True, ensure_ascii=False)+'\n')
            f.flush(); os.fsync(f.fileno())
        save(RUN/'SPLIT.json', inv['split'])
        source_hashes = {str((ROOT/r['sourceRoot']/r[field+'_file']).relative_to(ROOT)):r[field+'_sha256']
                         for r in combined for field in ('policy','supervision')}
        save(RUN/'SOURCE_HASHES.json', source_hashes)
        by_source = collections.defaultdict(collections.Counter)
        by_house = collections.defaultdict(collections.Counter)
        for row in combined:
            for counter in (by_source[row['source']], by_house[row['scene_group']]):
                counter.update(instructions=1, decisions=row['decisions'])
        result = dict(status='STRICT_MERGED_TRAINING_SNAPSHOT_READY', counts=dict(strict_routes=25415,
                      instruction_records=len(combined), instruction_conditioned_decisions=2650347,
                      unique_route_decisions=464877+1255603), houses=len(by_house),
                      by_source=dict(by_source), by_house=dict(by_house), training_index_sha256=sha(index),
                      full_envdrop_pixel_reaudit_routes=19566, checked_rgb_references=rgb,
                      human_snapshot_seal_reused_and_all_source_metadata_rechecked=True,
                      physical_duplicates=0, alias_duplicates=0, source_metadata_unchanged=True,
                      all_old_failures_and_quarantines_preserved=True, partial_routes_not_retried=2,
                      remaining_to_engineering_3000000=349653, source_grade='HUMAN_AND_SEPARATELY_MARKED_OFFICIAL_SYNTHETIC',
                      training_started=False, scientific_pass=False, wall_seconds=time.monotonic()-start)
        save(RUN/'RESULT.json', result)
        assert sum(p.stat().st_size for p in RUN.iterdir() if p.is_file()) < 2*1024**3
        save(RUN/'SEAL.json', {p.name:sha(p) for p in (RUN/'RESULT.json', index, RUN/'SPLIT.json', RUN/'SOURCE_HASHES.json', RUN/'INPUT_HASHES.json')})
        progress(status=result['status'], audited=19566, total=19566, training_admitted=True, counts=result['counts'])
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as exc:
        failure = dict(status='FAILED_NOT_ADMITTED', error=repr(exc), traceback=traceback.format_exc(), training_admitted=False)
        save(RUN/'FAILURE.json', failure); progress(**failure); raise
    finally:
        signal.alarm(0)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            for child in mp.active_children(): child.terminate()
            for child in mp.active_children():
                child.join(5)
                if child.is_alive(): child.kill(); child.join(5)
        if inv:
            for lock in inv['locks']: lock.close()


if __name__ == '__main__': main()
