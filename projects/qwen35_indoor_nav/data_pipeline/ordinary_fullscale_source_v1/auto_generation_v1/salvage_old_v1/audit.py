"""CPU-only salvage of old GPU6 shard2 terminal routes; no failed route replay.

Independent old-root lock permits concurrent NEW-root production. Does not
take the old lane lock held protectively by the new producer. No GPU calls.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
HERE=Path(__file__).resolve().parent
FULL=HERE.parents[1]
ROOT=FULL.parents[3]
OLDROOT=FULL/'production/shard_0002'
ATTEMPT=FULL/'runtime_v3/lanes/gpu_6/attempt_000'
STRICT=FULL.parent/'ordinary_scale_v1/audit.py'


def sha(p):
    p=p.resolve(strict=True);assert p.is_relative_to(ROOT)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    p=p.resolve(strict=True);assert p.is_relative_to(ROOT)
    return json.loads(p.read_text())


def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)


def terminal_partition(jobs,ledger,dirs):
    byid={j['job_id']:j for j in jobs};record={r['job_id']:r for r in ledger}
    assert len(byid)==len(jobs) and len(record)==len(ledger)
    assert set(record)<=set(dirs)<=set(byid)
    return [j for j in jobs if j['job_id'] in record],sorted(set(dirs)-set(record))


def execute():
    out=HERE/'run_v1';assert not out.exists(),'SALVAGE_VERSION_REQUIRED'
    hold=(OLDROOT/'PRODUCER.lock').open('rb');fcntl.flock(hold,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=read(ATTEMPT/'RESULT.json')
    assert result['error']=="AssertionError('GPU_MEMORY_ACCOUNTING')"
    assert result['restoration']==read(ATTEMPT/'RESTORATION.json') and result['restoration']['restored'] is True
    assert result['workers']==[dict(shard=2,returncode=1,wall_seconds=result['workers'][0]['wall_seconds'])]
    process=read(ATTEMPT/'PROCESS_2.json');assert not Path('/proc',str(process['pid'])).exists()
    assert sha(STRICT)=='37c3443c312cb635572b524cba4df2870ed835eea467cc8bcd5ed6aa029d97fe'
    spec=importlib.util.spec_from_file_location('unchanged_old_gpu6_strict',STRICT)
    strict=importlib.util.module_from_spec(spec);spec.loader.exec_module(strict)
    jobs=read(OLDROOT/'JOBS.json');raw=(OLDROOT/'LEDGER.jsonl').read_bytes();assert raw.endswith(b'\n')
    ledger=[json.loads(x) for x in raw.splitlines()];record={r['job_id']:r for r in ledger}
    terminal,partial=terminal_partition(jobs,ledger,{p.name for p in (OLDROOT/'routes').iterdir()})
    assert len(terminal)==573 and partial==['r2r_6b16e77f4bfa0dc39187']
    paths=[HERE/'audit.py',HERE/'test_audit.py',STRICT,OLDROOT/'JOBS.json',OLDROOT/'LEDGER.jsonl',ATTEMPT/'RESULT.json',ATTEMPT/'RESTORATION.json',ATTEMPT/'PROCESS_2.json']
    paths += [p for folder in ('routes','shards','quality') for p in (OLDROOT/folder).rglob('*') if p.is_file()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    stored={};oldq=set()
    for p in sorted((OLDROOT/'shards').glob('*.audit.json')):
        cert=read(p);index=p.with_name(p.name.replace('.audit.json','.jsonl'))
        assert cert['integrity_pass'] and sha(index)==cert['index_sha256']
        for row in map(json.loads,index.read_text().splitlines()):stored.setdefault(row['job_id'],[]).append(row)
    for p in (OLDROOT/'quality').glob('quarantine_*.json'):
        for r in read(p):
            assert r['status']=='QUARANTINED_NOT_TRAINING_DATA';oldq.add(r['job_id'])
    assert set(stored).isdisjoint(oldq) and set(stored)|oldq<=set(record)
    rows=[];quarantine=[];physical=set();aliases=set();steps=0;started=time.monotonic()
    for job in terminal:
        assert time.monotonic()-started<1800,'CPU_AUDIT_WALL_BUDGET'
        ident=job['job_id'];assert read(OLDROOT/'routes'/ident/'result.json')==record[ident]
        if ident in oldq:
            quarantine.append(dict(job_id=ident,status='PRIOR_QUARANTINE_PRESERVED'));continue
        try:fresh,_,n=strict.audit_route(OLDROOT,job)
        except AssertionError:
            assert ident not in stored,'PREVIOUS_ACCEPTED_NOW_FAILED'
            quarantine.append(dict(job_id=ident,status='NEW_STRICT_QUARANTINE_NOT_TRAINING',traceback=traceback.format_exc()));continue
        if ident in stored:assert fresh==stored[ident]
        if fresh:
            assert record[ident]['status']=='CERTIFIED'
            key=job['physical_source_route_sha256'];assert key not in physical
            aa={(job['source'],str(e['episode_id'])) for e in job['instruction_alias_episodes']}
            assert len(aa)==len(fresh)==len(job['instruction_alias_episodes']) and aliases.isdisjoint(aa)
            physical.add(key);aliases.update(aa);steps+=n
            rows += [dict(r,sourceRoot=str(OLDROOT.relative_to(ROOT))) for r in fresh]
    for p,h in hashes.items():assert sha(ROOT/p)==h,'OLD_INPUT_CHANGED_DURING_AUDIT'
    out.mkdir()
    with (out/'TRAINING_INDEX.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    save(out/'QUARANTINE.json',quarantine);save(out/'INPUT_HASHES.json',hashes)
    save(out/'RESULT.json',dict(status='STRICT_OLD_TERMINAL_ROUTES_SALVAGED_NOT_TRAINED',strict_routes=len(physical),instruction_records=len(rows),
        unique_route_decisions=steps,instruction_conditioned_decisions=sum(r['decisions'] for r in rows),
        old_terminal_routes=573,interrupted_excluded=partial,original_failure_preserved=True,sourceRoot=str(OLDROOT.relative_to(ROOT)),
        physical_duplicates=0,alias_duplicates=0,training_started=False,scientific_pass=False,index_sha256=sha(out/'TRAINING_INDEX.jsonl')))
    print(json.dumps(read(out/'RESULT.json')))


if __name__=='__main__':execute()
