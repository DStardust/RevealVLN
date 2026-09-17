"""CPU freeze of exact never-attempted GPU6 routes; no renderer/production."""
import fcntl
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
FULL=HERE.parent
ROOT=HERE.parents[4]
OLD=FULL/'runtime_v3/lanes/gpu_6/attempt_000'


def sha(p):
    p=p.resolve(strict=True);assert p.is_relative_to(ROOT)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text())


def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)


def classify(jobs,ledger,dirs):
    mapping={j['job_id']:j for j in jobs};assert len(mapping)==len(jobs)
    terminal={r['job_id']:r for r in ledger};assert len(terminal)==len(ledger)
    assert set(terminal)<=set(dirs)<=set(mapping)
    return [j for j in jobs if j['job_id'] not in set(terminal)|set(dirs)],sorted(set(dirs)-set(terminal))


def execute():
    out=HERE/'source_v1';assert not out.exists(),'NEW_SOURCE_VERSION_REQUIRED'
    hold=(OLD.parent/'PRODUCER.lock').open('rb');fcntl.flock(hold,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=read(OLD/'RESULT.json');restore=read(OLD/'RESTORATION.json')
    assert result['error']=="AssertionError('GPU_MEMORY_ACCOUNTING')"
    assert result['restoration']==restore and restore['restored'] is True
    assert len(result['workers'])==1 and result['workers'][0]['shard']==2 and result['workers'][0]['returncode']==1
    proc=read(OLD/'PROCESS_2.json');assert not Path('/proc',str(proc['pid'])).exists(),'OLD_WORKER_STILL_PRESENT'
    hashes={str((OLD/f).relative_to(ROOT)):sha(OLD/f) for f in ('RESULT.json','RESTORATION.json','PROCESS_2.json')}
    rows=[];parts=[];locks=[];initial_dirs={}
    for shard,expected in ((2,426),(4,998)):
        root=FULL/'production'/f'shard_{shard:04d}'
        if (root/'PRODUCER.lock').exists():
            h=(root/'PRODUCER.lock').open('rb');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
        jobs=read(root/'JOBS.json');assert jobs==read(FULL/'manifests'/f'shard_{shard:04d}/JOBS.json')
        raw=(root/'LEDGER.jsonl').read_bytes() if (root/'LEDGER.jsonl').exists() else b''
        assert not raw or raw.endswith(b'\n'),'PARTIAL_LEDGER_REQUIRES_NEW_RECONCILIATION'
        ledger=[json.loads(x) for x in raw.splitlines()]
        dirs={p.name for p in (root/'routes').iterdir()} if (root/'routes').exists() else set()
        subset,interrupted=classify(jobs,ledger,dirs);assert len(subset)==expected
        rows+=subset;initial_dirs[shard]=dirs
        parts.append(dict(old_shard=shard,assigned=len(jobs),terminal=len(ledger),interrupted=interrupted,unattempted=len(subset),unattempted_aliases=sum(len(j['instruction_alias_episodes']) for j in subset)))
        for name in ('JOBS.json','INPUT_LOCK.json','SPLIT_FREEZE.json','SOURCE_INVENTORY.json','LEDGER.jsonl'):
            if (root/name).exists():hashes[str((root/name).relative_to(ROOT))]=sha(root/name)
    keys={j['physical_source_route_sha256'] for j in rows};assert len(keys)==len(rows)==1424
    aliases=[(j['source'],str(e['episode_id'])) for j in rows for e in j['instruction_alias_episodes']]
    assert len(set(aliases))==len(aliases)==4272
    for p,h in hashes.items():assert sha(ROOT/p)==h,'SOURCE_CHANGED_DURING_PREPARE'
    for shard,dirs in initial_dirs.items():
        p=FULL/'production'/f'shard_{shard:04d}/routes'
        assert ({d.name for d in p.iterdir()} if p.exists() else set())==dirs
    out.mkdir()
    save(out/'RESCUE_JOBS.json',rows)
    save(out/'PLAN.json',dict(executable=False,runtime_allowed=False,source_scope='NEVER_ATTEMPTED_ONLY_AFTER_GPU6_ACCOUNTING_FAILURE',
        gpu_to_shards={'6':[2,4]},routes=1424,instruction_aliases=4272,shards=parts,
        rescue_jobs_sha256=sha(out/'RESCUE_JOBS.json'),original_error_preserved=result['error'],
        original_wall_seconds_preserved=result['wall_seconds'],old_failed_routes_retried=0,old_partial_routes_retried=0,
        requires_new_main_budget_and_exact_gpu_identity=True,scientific_pass=False))
    for p in (HERE/'source.py',HERE/'test_source.py',out/'RESCUE_JOBS.json',out/'PLAN.json'):
        hashes[str(p.relative_to(ROOT))]=sha(p)
    save(out/'INPUT_LOCK.json',hashes)
    print(json.dumps(read(out/'PLAN.json')))


if __name__=='__main__':execute()
