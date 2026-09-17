"""Frozen four-shard transport, strict unchanged audit, no GPU side effects."""
import hashlib
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
PARALLEL=HERE.parent
LINE=PARALLEL.parents[1]
ROOT=LINE.parents[1]
BASE=PARALLEL.parent/'ordinary_scale_v1'
ENV=LINE/'.envs/q35n_habitat_v017_g0r'
IDENTITIES=LINE/'authorizations/ORDINARY_PARALLEL_GPU67_IDENTITIES_V1.json'
AUTH=LINE/'authorizations/ORDINARY_PARALLEL_PRODUCTION_V1.json'
LANES={6:(0,2),7:(1,3)}


def sha(path):
    path=Path(path).resolve(strict=True);assert path.is_relative_to(ROOT)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def read(path):return json.loads(Path(path).read_text())


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def exact(source,old,new):
    assert source.count(old)==1,('EXACT_TRANSPORT_COUNT',old)
    return source.replace(old,new)


def shard_root(shard):
    assert type(shard) is int and 0<=shard<4
    return PARALLEL/'production'/f'shard_{shard:04d}'


def lane_for(shard):return 6 if shard in (0,2) else 7


def immutable_verify():
    lock=read(HERE/'INPUT_LOCK.json')
    for name,h in lock.items():assert sha(ROOT/name)==h,name
    return lock


def approval_value(gpu):
    assert gpu in LANES
    return dict(approved=True,gpu=gpu,shards=list(LANES[gpu]),
        input_lock_sha256=sha(HERE/'INPUT_LOCK.json'),identity_sha256=sha(IDENTITIES))


def approved(gpu):
    lock=immutable_verify()
    assert read(HERE/f'MAIN_AGENT_APPROVAL_GPU{gpu}.json')==approval_value(gpu),'MAIN_APPROVAL_REQUIRED'
    return lock


def state(shard):
    root=shard_root(shard);jobs=read(root/'JOBS.json');byid={j['job_id']:j for j in jobs}
    assert len(byid)==len(jobs)
    ledger=root/'LEDGER.jsonl';rows=[]
    if ledger.exists():
        raw=ledger.read_bytes();assert raw.endswith(b'\n'),'PARTIAL_LEDGER_REQUIRES_RECONCILIATION'
        rows=[json.loads(x) for x in raw.splitlines()]
    seen=set()
    for r in rows:
        ident=r['job_id'];assert ident in byid and ident not in seen
        assert read(root/'routes'/ident/'result.json')==r,'TERMINAL_LEDGER_MISMATCH'
        seen.add(ident)
    if (root/'routes').exists():
        for folder in (root/'routes').iterdir():
            assert folder.is_dir() and folder.name in seen,'UNFINISHED_ROUTE_REQUIRES_VERSIONED_RECOVERY'
    audits=[]
    if (root/'shards').exists():
        for p in (root/'shards').iterdir():
            assert not p.name.endswith('.pending'),'UNFINISHED_INDEX_REQUIRES_RECONCILIATION'
        for p in sorted((root/'shards').glob('*.jsonl')):
            cert=p.with_suffix('.audit.json');assert cert.exists(),'INDEX_WITHOUT_AUDIT'
            a=read(cert);assert sha(p)==a['index_sha256'] and a['integrity_pass']
            assert (root/a['quarantine_manifest']).is_file()
            audits.append(a)
        assert len(list((root/'shards').glob('*.audit.json')))==len(audits),'AUDIT_WITHOUT_INDEX'
    complete=(root/'GENERATION_COMPLETE.json').exists()
    if complete:
        assert read(root/'GENERATION_COMPLETE.json')['all_jobs_terminal']
        assert len(rows)==len(jobs) and len(audits)==(len(jobs)+49)//50
    return dict(shard=shard,terminal_jobs=len(rows),total_jobs=len(jobs),complete=complete,
        ledger_sha256=sha(ledger) if ledger.exists() else None)


def worker_source(shard):
    root=shard_root(shard);source=(BASE/'worker.py').read_text()
    replacements=[('OUT=Path(__file__).resolve().parent',f'OUT=Path({str(root)!r})'),
        ('LINE=OUT.parents[1]',f'LINE=Path({str(LINE)!r})'),
        ('ROOT=LINE.parents[1]',f'ROOT=Path({str(ROOT)!r})'),
        ("OUT/'prepare.py'",f'Path({str(BASE/"prepare.py")!r})'),
        ("OUT/'audit.py'",f'Path({str(BASE/"audit.py")!r})'),
        ('cfg.gpu_device_id=2;',f'cfg.gpu_device_id={lane_for(shard)};'),
        ("asset_locks[scene]=records", "assert all(FROZEN_ASSETS[r['path']]==r['sha256'] for r in records),'ASSET_HASH_CHANGED'\n                    asset_locks[scene]=records")]
    for old,new in replacements:source=exact(source,old,new)
    return source


def worker_module(shard):
    module=types.ModuleType(f'ordinary_parallel_shard_{shard}')
    module.__file__=str(BASE/'worker.py')
    module.FROZEN_ASSETS=read(HERE/'ASSETS.json')
    exec(compile(worker_source(shard),module.__file__,'exec'),module.__dict__)
    strict=load(f'parallel_strict_quarantine_{shard}',BASE/'recovery_v1/audit.py')
    strict.HERE=shard_root(shard)/'quality'
    module.audit=strict
    assert module.OUT==shard_root(shard) and module.ROOT==ROOT and module.LINE==LINE
    return module
