"""Read-only strict salvage + prospective unattempted-only rescue manifests.

Requires closed controllers and restored holders. Never writes into a source root.
Unknown/partial routes remain excluded and are not automatically retried.
"""
import argparse
import collections
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
HERE=Path(__file__).resolve().parent
FULL=HERE.parent
LINE=FULL.parents[1]
ROOT=LINE.parents[1]
RUNTIME=FULL/'runtime_v2'
sys.path.insert(0,str(HERE))
import gate
RUNTIME_LOCK='6523fc0812df56141c8e10f8b6abf0c2b9bf900c92eaaec50175288988d7f870'
STRICT_SHA='37c3443c312cb635572b524cba4df2870ed835eea467cc8bcd5ed6aa029d97fe'
ERRORS={"AssertionError('SHARD_WALL_BUDGET')","AssertionError('LANE_WALL_BUDGET')"}


def safe(path):
    path=Path(path).resolve(strict=True)
    assert path.is_relative_to(ROOT),'PROJECT_PATH_REQUIRED'
    return path


def sha(path):
    h=hashlib.sha256()
    with safe(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(safe(path).read_text())


def write(path,value):
    assert path.parent.resolve(strict=True).is_relative_to(HERE)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def ledger_bytes(raw,unique_jobs=True):
    """Incomplete final bytes are preserved as a tail, never parsed as a label."""
    chunks=raw.splitlines(keepends=True);rows=[];tail=b''
    for i,line in enumerate(chunks):
        if not line.endswith(b'\n'):
            assert i==len(chunks)-1
            tail=line;break
        rows.append(json.loads(line))
    ids=[r['job_id'] for r in rows]
    if unique_jobs:assert len(ids)==len(set(ids)),'DUPLICATE_LEDGER'
    return rows,dict(bytes=len(tail),sha256=hashlib.sha256(tail).hexdigest())


def partition(jobs,ledger,directory_names):
    ids=[j['job_id'] for j in jobs]
    assert len(ids)==len(set(ids)),'DUPLICATE_JOB'
    known=set(ids);terminal={r['job_id']:r for r in ledger}
    assert len(terminal)==len(ledger),'DUPLICATE_LEDGER'
    assert set(terminal)<=known and set(directory_names)<=known,'UNREGISTERED_ROUTE'
    assert set(terminal)<=set(directory_names),'TERMINAL_WITHOUT_DIRECTORY'
    return dict(terminal=[j for j in jobs if j['job_id'] in terminal],
        interrupted=[j for j in jobs if j['job_id'] in directory_names and j['job_id'] not in terminal],
        unattempted=[j for j in jobs if j['job_id'] not in terminal and j['job_id'] not in directory_names])


def validate_workers(result,processes,gpu,exists=None):
    assert result['error'] in ERRORS,'NOT_REGISTERED_RESOURCE_CENSOR'
    assert len(result['workers'])==1 and result['workers'][0]['shard']==gpu-3
    worker=result['workers'][0];proc=processes[gpu-3]
    assert type(worker['returncode']) is int and worker['returncode'] in (0,1,-15,-9)
    assert proc['gpu']==gpu and proc['shard']==gpu-3 and type(proc['pid']) is int and proc['pid']>0
    if exists is None:exists=lambda pid:Path('/proc',str(pid)).exists()
    assert not exists(proc['pid']),'OWN_WORKER_STILL_PRESENT'


def freeze_files(root,hashes):
    """Metadata/route bytes, not a duplicate copy of source RGB content."""
    for p in sorted(root.rglob('*')):
        if p.is_file():hashes[str(safe(p).relative_to(ROOT))]=sha(p)


def closed_locks(gpus):
    locks=[];evidence=[];hashes={}
    assert sha(RUNTIME/'INPUT_LOCK.json')==RUNTIME_LOCK
    for p,h in read(RUNTIME/'INPUT_LOCK.json').items():assert sha(ROOT/p)==h,('SOURCE_LOCK_CHANGED',p)
    hashes[str((RUNTIME/'INPUT_LOCK.json').relative_to(ROOT))]=RUNTIME_LOCK
    for gpu in gpus:
        lane=RUNTIME/'lanes'/f'gpu_{gpu}'
        handle=safe(lane/'PRODUCER.lock').open('rb');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        attempts=sorted(lane.glob('attempt_*'));assert len(attempts)==1,'REGISTERED_ATTEMPT_000_ONLY'
        attempt=attempts[0];result=read(attempt/'RESULT.json')
        validate_workers(result,{gpu-3:read(attempt/f'PROCESS_{gpu-3}.json')},gpu)
        evidence.append(gate.verify(attempt,result,gpu))
        for name in ('RESULT.json','RESTORATION.json','LEASE_BEFORE.json','LEASE_ACTIVE.json',f'PROCESS_{gpu-3}.json'):
            p=attempt/name;hashes[str(p.relative_to(ROOT))]=sha(p)
        root=FULL/'production'/f'shard_{gpu-3:04d}'
        handle=safe(root/'PRODUCER.lock').open('rb');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    return locks,evidence,hashes


def accepted_rows(root,job,rows,physical,aliases):
    key=job['physical_source_route_sha256'];assert key not in physical,'DUPLICATE_PHYSICAL'
    expected={(job['source'],str(e['episode_id'])) for e in job['instruction_alias_episodes']}
    assert len(expected)==len(rows)==len(job['instruction_alias_episodes'])
    assert aliases.isdisjoint(expected),'DUPLICATE_ALIAS'
    assert {Path(r['policy_file']).name for r in rows}=={f"policy_{e['episode_id']}.json" for e in job['instruction_alias_episodes']}
    for r in rows:
        assert r['job_id']==job['job_id'] and r['physical_source_route_sha256']==key
        assert r['source']==job['source'] and r['source_sha256']==job['source_sha256']
        assert r['split']=='FIT' and r['scene_group']==job['scene_id']
        for field in ('policy_file','supervision_file'):assert safe(root/r[field]).is_relative_to(root)
    physical.add(key);aliases.update(expected)
    return [dict(r,sourceRoot=str(root.relative_to(ROOT))) for r in rows]


def execute(output,gpus):
    output=Path(output)
    assert output.parent.resolve(strict=True)==HERE and not output.exists(),'NEW_DIRECT_CHILD_OUTPUT_REQUIRED'
    assert gpus and len(set(gpus))==len(gpus) and set(gpus)<={3,4}
    locks,evidence,hashes=closed_locks(gpus)
    strict_path=FULL.parent/'ordinary_scale_v1/audit.py'
    assert sha(strict_path)==STRICT_SHA
    spec=importlib.util.spec_from_file_location('unchanged_ordinary_strict',strict_path)
    strict=importlib.util.module_from_spec(spec);spec.loader.exec_module(strict)
    physical=set();aliases=set();accepted=[];quarantine=[];interrupted=[];rescue=[];stats=[]
    for gpu in gpus:
        shard=gpu-3;root=safe(FULL/'production'/f'shard_{shard:04d}')
        jobs=read(root/'JOBS.json')
        assert jobs==read(FULL/'manifests'/f'shard_{shard:04d}'/'JOBS.json')
        ledger,tail=ledger_bytes((root/'LEDGER.jsonl').read_bytes() if (root/'LEDGER.jsonl').exists() else b'')
        folders=list((root/'routes').iterdir()) if (root/'routes').exists() else []
        assert all(p.is_dir() and not p.is_symlink() for p in folders),'UNKNOWN_ROUTE_ENTRY'
        parts=partition(jobs,ledger,{p.name for p in folders})
        terminal={r['job_id']:r for r in ledger};stored={};oldq=set()
        for path in sorted((root/'shards').glob('*.audit.json')):
            a=read(path);index=path.with_name(path.name.replace('.audit.json','.jsonl'))
            assert a['integrity_pass'] and sha(index)==a['index_sha256']
            rows,last=ledger_bytes(index.read_bytes(),unique_jobs=False);assert not last['bytes']
            assert len(rows)==a['instruction_records'] and sum(r['decisions'] for r in rows)==a['instruction_conditioned_decisions']
            q=read(root/a['quarantine_manifest']);assert len(q)==a['quarantined_routes']
            for r in rows:stored.setdefault(r['job_id'],[]).append(r)
            for r in q:
                assert r['status']=='QUARANTINED_NOT_TRAINING_DATA' and r['job_id'] not in oldq
                oldq.add(r['job_id'])
        # A quarantine file can be committed before its index/certificate. Keep
        # those actual exclusions too; an interrupted publication cannot erase it.
        for qpath in sorted((root/'quality').glob('quarantine_*.json')):
            for r in read(qpath):
                assert r['status']=='QUARANTINED_NOT_TRAINING_DATA'
                oldq.add(r['job_id'])
        replay={i for i,r in terminal.items() if r['status']=='CERTIFIED'}
        assert set(stored).isdisjoint(oldq) and (set(stored)|oldq)<=replay
        start=len(accepted);unique_steps=0;newq=0
        for job in parts['terminal']:
            ident=job['job_id'];r=read(root/'routes'/ident/'result.json')
            assert r==terminal[ident],'LEDGER_RESULT_MISMATCH'
            if ident in oldq:
                quarantine.append(dict(job_id=ident,sourceRoot=str(root.relative_to(ROOT)),status='PRIOR_QUARANTINE_PRESERVED_NOT_TRAINING'));continue
            try:rows,refs,steps=strict.audit_route(root,job)
            except AssertionError:
                assert ident not in stored,'PRIOR_ACCEPTED_NOW_STRICT_FAILED'
                quarantine.append(dict(job_id=ident,sourceRoot=str(root.relative_to(ROOT)),status='NEW_STRICT_QUARANTINE_NOT_TRAINING',traceback=traceback.format_exc()));newq+=1;continue
            if ident in stored:assert rows==stored[ident],'PRIOR_INDEX_MISMATCH'
            if rows:
                assert ident in replay
                accepted+=accepted_rows(root,job,rows,physical,aliases);unique_steps+=steps
        for job in parts['interrupted']:
            interrupted.append(dict(job_id=job['job_id'],sourceRoot=str(root.relative_to(ROOT)),status='DIRECTORY_WITHOUT_COMPLETE_LEDGER_NOT_TRAINING_NOT_RETRY',physical_source_route_sha256=job['physical_source_route_sha256']))
        rescue+=parts['unattempted']
        for name in ('JOBS.json','SPLIT_FREEZE.json','SOURCE_INVENTORY.json','INPUT_LOCK.json','LEDGER.jsonl'):
            p=root/name
            if p.exists():hashes[str(p.relative_to(ROOT))]=sha(p)
        for name in ('routes','quality','shards'):
            if (root/name).exists():freeze_files(root/name,hashes)
        stats.append(dict(gpu=gpu,shard=shard,assigned=len(jobs),terminal=len(ledger),interrupted=len(parts['interrupted']),unattempted=len(parts['unattempted']),partial_ledger_tail=tail,
            generation_rejected=sum(r['status']!='CERTIFIED' for r in ledger),prior_quarantine=len(oldq),new_quarantine=newq,
            strict_routes=len({r['job_id'] for r in accepted[start:]}),instruction_records=len(accepted)-start,
            unique_route_decisions=unique_steps,instruction_conditioned_decisions=sum(r['decisions'] for r in accepted[start:])))
    keys=[j['physical_source_route_sha256'] for j in rescue]
    assert len(keys)==len(set(keys)) and set(keys).isdisjoint(physical)
    for p,h in hashes.items():assert sha(ROOT/p)==h,'SNAPSHOT_CHANGED_DURING_AUDIT'
    output.mkdir()
    with (output/'TRAINING_INDEX.jsonl').open('x') as f:
        for r in accepted:f.write(json.dumps(r)+'\n')
    write(output/'RESCUE_JOBS.json',rescue)
    write(output/'QUARANTINE.json',quarantine)
    write(output/'INTERRUPTED.json',interrupted)
    write(output/'RESTORATION_EVIDENCE.json',evidence)
    hashes[str(strict_path.relative_to(ROOT))]=STRICT_SHA
    for p in (HERE/'snapshot.py',HERE/'gate.py',HERE/'test_snapshot.py'):hashes[str(p.relative_to(ROOT))]=sha(p)
    write(output/'INPUT_HASHES.json',hashes)
    write(output/'SPLIT_FREEZE.json',read(FULL/'SPLIT_FREEZE.json'))
    write(output/'SOURCE_INVENTORY.json',read(FULL/'SOURCE_INVENTORY.json'))
    write(output/'RESULT.json',dict(status='STRICT_SALVAGE_AND_UNATTEMPTED_ONLY_PROSPECTIVE_SOURCE',executable=False,runtime_allowed=False,scientific_pass=False,training_started=False,shards=stats,
        strict_routes=len(physical),instruction_records=len(accepted),unique_route_decisions=sum(s['unique_route_decisions'] for s in stats),instruction_conditioned_decisions=sum(s['instruction_conditioned_decisions'] for s in stats),
        rescue_routes=len(rescue),rescue_instruction_aliases=sum(len(j['instruction_alias_episodes']) for j in rescue),interrupted_not_retried=len(interrupted),
        training_index_sha256=sha(output/'TRAINING_INDEX.jsonl'),rescue_jobs_sha256=sha(output/'RESCUE_JOBS.json'),old_sources_unchanged=True))
    print(json.dumps(read(output/'RESULT.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',choices=[3,4],default=[3,4])
    a=p.parse_args();execute(a.output,a.gpus)
