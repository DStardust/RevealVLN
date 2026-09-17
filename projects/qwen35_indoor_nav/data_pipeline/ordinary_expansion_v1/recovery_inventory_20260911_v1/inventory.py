"""Read-only old EnvDrop evidence inventory. New outputs only under HERE."""
import collections
import fcntl
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXPANSION=HERE.parent
LINE=HERE.parents[2]
ROOT=LINE.parents[1]
DATA=EXPANSION/'envdrop_production_v1'
RUNTIME=EXPANSION/'runtime_v1'
STRICT=EXPANSION.parent/'ordinary_scale_v1/audit.py'
STRICT_SHA='37c3443c312cb635572b524cba4df2870ed835eea467cc8bcd5ed6aa029d97fe'


def safe(path):
    p=Path(path).resolve(strict=True)
    assert p.is_relative_to(ROOT),'OUTSIDE_PROJECT'
    return p


def sha(path):
    h=hashlib.sha256()
    with safe(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(safe(path).read_text())


def save(path,value):
    assert path.parent.resolve().is_relative_to(HERE)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def ledger(raw,unique=True):
    lines=raw.splitlines(keepends=True);rows=[];tail=b''
    for i,line in enumerate(lines):
        if not line.endswith(b'\n'):
            assert i==len(lines)-1
            tail=line;break
        rows.append(json.loads(line))
    if unique:assert len({x['job_id'] for x in rows})==len(rows),'DUPLICATE_TERMINAL'
    return rows,dict(bytes=len(tail),sha256=hashlib.sha256(tail).hexdigest())


def partition(jobs,terminal,directories):
    ids={j['job_id'] for j in jobs};done={r['job_id'] for r in terminal}
    assert len(ids)==len(jobs) and len(done)==len(terminal)
    assert done<=directories<=ids,'DIRECTORY_LEDGER_MISMATCH'
    return dict(terminal=[j for j in jobs if j['job_id'] in done],
        partial=[j for j in jobs if j['job_id'] in directories-done],
        unattempted=[j for j in jobs if j['job_id'] not in directories])


def accept(job,rows,physical,aliases):
    key=job['physical_source_route_sha256']
    expected={(job['source'],str(e['episode_id'])) for e in job['instruction_alias_episodes']}
    assert key not in physical,'DUPLICATE_PHYSICAL'
    assert len(expected)==len(rows)==len(job['instruction_alias_episodes'])
    assert expected.isdisjoint(aliases),'DUPLICATE_ALIAS'
    assert {Path(r['policy_file']).name for r in rows}=={f"policy_{e['episode_id']}.json" for e in job['instruction_alias_episodes']}
    for r in rows:
        assert r['job_id']==job['job_id'] and r['source']==job['source'] and r['source_sha256']==job['source_sha256']
        assert r['split']=='FIT' and r['scene_group']==job['scene_id'] and r['physical_source_route_sha256']==key
    physical.add(key);aliases.update(expected)


def gather():
    hashes={};locks=[]
    def bind(path):
        path=safe(path);hashes[str(path.relative_to(ROOT))]=sha(path)
    assert sha(STRICT)==STRICT_SHA
    bind(STRICT);bind(RUNTIME/'INPUT_LOCK.json')
    for path,digest in read(RUNTIME/'INPUT_LOCK.json').items():assert sha(ROOT/path)==digest,('OLD_SOURCE_LOCK_CHANGED',path)
    lane=RUNTIME/'lanes/gpu_6';attempt=lane/'attempt_000'
    result=read(attempt/'RESULT.json');restoration=read(attempt/'RESTORATION.json')
    assert result['error']=="AssertionError('SHARD_WALL_BUDGET')"
    assert result['restoration']==restoration and restoration['restored']
    assert result['external_processes_stopped']==0
    for worker in result['workers']:
        p=attempt/f"PROCESS_{worker['shard']}.json";identity=read(p)
        assert not Path('/proc',str(identity['pid'])).exists(),'OLD_WORKER_PID_PRESENT'
        assert not Path('/proc',str(identity['supervisor_pid'])).exists(),'OLD_SUPERVISOR_PID_PRESENT'
        bind(p)
    for p in (attempt/'RESULT.json',attempt/'RESTORATION.json',attempt/'LEASE_BEFORE.json',attempt/'LEASE_ACTIVE.json'):bind(p)
    handle=safe(lane/'PRODUCER.lock').open('rb');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    source_jobs=read(EXPANSION/'envdrop_source_v1/JOBS.json')
    assert len(source_jobs)==20000 and len({j['job_id'] for j in source_jobs})==20000
    assert all(len(j['instruction_alias_episodes'])==1 for j in source_jobs),'ENVDROP_SINGLE_ALIAS_EXPECTED'
    source_map={j['job_id']:j for j in source_jobs};seen_jobs=set()
    excluded=set(read(EXPANSION/'envdrop_source_v1/PHYSICAL_EXCLUSION.json')['physical_route_keys'])
    split=read(LINE/'sft_acceptance/ordinary_baseline_v2/snapshot_v1/SPLIT.json')
    bind(LINE/'sft_acceptance/ordinary_baseline_v2/snapshot_v1/SPLIT.json')
    bind(EXPANSION/'envdrop_source_v1/JOBS.json');bind(EXPANSION/'envdrop_source_v1/PHYSICAL_EXCLUSION.json')
    candidates=[];rescue=[];partial=[];old_quarantine=[];generation_rejected=[];shards=[];stored_total=[]
    for shard in range(21):
        root=DATA/f'shard_{shard:04d}';jobs=read(root/'JOBS.json')
        for j in jobs:
            assert j==source_map[j['job_id']] and j['job_id'] not in seen_jobs
            assert j['split']=='FIT' and j['scene_id'] in split['FIT']
            assert j['physical_source_route_sha256'] not in excluded
            seen_jobs.add(j['job_id'])
        if (root/'PRODUCER.lock').exists():
            handle=safe(root/'PRODUCER.lock').open('rb');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        terminal,tail=ledger((root/'LEDGER.jsonl').read_bytes() if (root/'LEDGER.jsonl').exists() else b'')
        folders=list((root/'routes').iterdir()) if (root/'routes').exists() else []
        assert all(p.is_dir() and not p.is_symlink() for p in folders)
        parts=partition(jobs,terminal,{p.name for p in folders});term={r['job_id']:r for r in terminal}
        stored={};quarantined=set()
        for path in sorted((root/'shards').glob('*.audit.json')):
            a=read(path);index=path.with_name(path.name.replace('.audit.json','.jsonl'))
            assert a['integrity_pass'] and sha(index)==a['index_sha256']
            rows,last=ledger(index.read_bytes(),False);assert not last['bytes']
            assert len(rows)==a['instruction_records'] and sum(r['decisions'] for r in rows)==a['instruction_conditioned_decisions']
            assert len({r['job_id'] for r in rows})==a['certified_routes']
            qpath=root/a['quarantine_manifest'];q=read(qpath);assert len(q)==a['quarantined_routes']
            for r in rows:stored.setdefault(r['job_id'],[]).append(r)
            for x in q:
                assert x['status']=='QUARANTINED_NOT_TRAINING_DATA';quarantined.add(x['job_id'])
            for p in (path,index,qpath):bind(p)
        for path in sorted((root/'quality').glob('quarantine_*.json')):
            for x in read(path):
                assert x['status']=='QUARANTINED_NOT_TRAINING_DATA';quarantined.add(x['job_id'])
            bind(path)
        replay={r['job_id'] for r in terminal if r['status']=='CERTIFIED'}
        assert set(stored).isdisjoint(quarantined) and set(stored)|quarantined<=replay
        complete=(root/'GENERATION_COMPLETE.json').exists()
        if complete:
            assert read(root/'GENERATION_COMPLETE.json')['all_jobs_terminal'] and len(terminal)==len(jobs)
            assert set(stored)|quarantined==replay
        for j in parts['terminal']:
            folder=root/'routes'/j['job_id'];r=read(folder/'result.json')
            assert r==term[j['job_id']]
            for p in folder.iterdir():
                if p.is_file():bind(p)
            if j['job_id'] in quarantined:
                old_quarantine.append(dict(job_id=j['job_id'],shard=shard,status='PRIOR_QUARANTINE_PRESERVED'))
            elif r['status']=='CERTIFIED':candidates.append(dict(shard=shard,job=j,stored=stored.get(j['job_id'])))
            else:
                assert not (folder/'supervision_only.json').exists() and not list(folder.glob('policy_*.json'))
                generation_rejected.append(dict(shard=shard,terminal=r))
        rescue.extend(parts['unattempted'])
        partial.extend(dict(shard=shard,job_id=j['job_id'],physical_source_route_sha256=j['physical_source_route_sha256'],status='PARTIAL_NOT_TRAINING_NOT_RETRY') for j in parts['partial'])
        for name in ('JOBS.json','INPUT_LOCK.json','SPLIT_FREEZE.json','SOURCE_INVENTORY.json','LEDGER.jsonl','GENERATION_COMPLETE.json'):
            if (root/name).exists():bind(root/name)
        stored_rows=[r for group in stored.values() for r in group];stored_total.extend(stored_rows)
        shards.append(dict(shard=shard,assigned=len(jobs),terminal=len(terminal),replay_certified=len(replay),
            stored_strict_routes=len(stored),stored_conditioned_decisions=sum(r['decisions'] for r in stored_rows),
            complete=complete,partial=len(parts['partial']),unattempted=len(parts['unattempted']),prior_quarantine=len(quarantined),partial_ledger_tail=tail))
    assert seen_jobs==set(source_map)
    keys=[j['physical_source_route_sha256'] for j in source_jobs];assert len(keys)==len(set(keys))
    return dict(hashes=hashes,locks=locks,shards=shards,candidates=candidates,rescue=rescue,partial=partial,
        old_quarantine=old_quarantine,generation_rejected=generation_rejected,stored_rows=stored_total,
        old_result=result,excluded=excluded)
