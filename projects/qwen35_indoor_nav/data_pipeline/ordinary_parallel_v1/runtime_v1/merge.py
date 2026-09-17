"""CPU strict readback, complete alias ownership, explicit per-shard sourceRoot."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import safe_size


def rows(path):
    raw=path.read_bytes();assert not raw or raw.endswith(b'\n'),'PARTIAL_JSONL'
    return [json.loads(line) for line in raw.splitlines()]


def alias_keys(job):
    return {(job['source'],str(e['episode_id'])) for e in job['instruction_alias_episodes']}


def validate_accepted(root,job,stored,fresh,physical_seen,alias_seen):
    assert fresh==stored,'STRICT_INDEX_REAUDIT_MISMATCH'
    assert len(stored)==len(job['instruction_alias_episodes'])
    assert {Path(r['policy_file']).name for r in stored}=={f"policy_{e['episode_id']}.json" for e in job['instruction_alias_episodes']}
    key=job['physical_source_route_sha256'];assert key not in physical_seen,'DUPLICATE_PHYSICAL_ROUTE'
    aliases=alias_keys(job);assert len(aliases)==len(stored) and aliases.isdisjoint(alias_seen),'DUPLICATE_ALIAS'
    for row in stored:
        assert row['job_id']==job['job_id'] and row['physical_source_route_sha256']==key
        assert row['source']==job['source'] and row['source_sha256']==job['source_sha256']
        assert row['split']=='FIT' and row['scene_group']==job['scene_id']
        for field in ('policy_file','supervision_file'):
            path=(root/row[field]).resolve(strict=True);assert path.is_relative_to(root.resolve())
    physical_seen.add(key);alias_seen.update(aliases)
    return [dict(r,sourceRoot=str(root.relative_to(c.ROOT))) for r in stored]


def execute():
    c.immutable_verify();out=c.PARALLEL/'merge';assert not out.exists(),'MERGE_VERSION_REQUIRED'
    # No merge while a lane is running or its final cleanup receipt is absent.
    locks=[]
    for gpu in c.LANES:
        lane=HERE/'lanes'/f'gpu_{gpu}';assert lane.exists(),'LANE_NOT_ATTEMPTED'
        handle=(lane/'PRODUCER.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        for attempt in lane.glob('attempt_*'):
            assert (attempt/'RESULT.json').exists(),'UNACCOUNTED_ATTEMPT'
            result=c.read(attempt/'RESULT.json')
            assert result['restoration']['restored'],'HOLDER_NOT_RESTORED'
    strict=c.load('parallel_merge_original_strict_auditor',c.BASE/'audit.py')
    physical=set();aliases=set();accepted=[];summary=[];input_hashes={}
    expected_all={j['physical_source_route_sha256'] for j in c.read(c.PARALLEL/'JOBS.json')}
    excluded=set(c.read(c.PARALLEL/'EXCLUSION_MANIFEST.json')['unique_physical_route_keys_excluded'])
    assert len(expected_all)==1000 and expected_all.isdisjoint(excluded)
    total_disk=0
    for shard in range(4):
        root=c.shard_root(shard)
        handle=(root/'PRODUCER.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        state=c.state(shard)
        if not state['complete']:
            receipt=c.read(root/'CENSOR_RECEIPT.json')
            assert receipt==dict(approved=True,shard=shard,ledger_sha256=state['ledger_sha256'],
                status='EXPLICIT_RESOURCE_CENSOR_NO_UNATTEMPTED_NEGATIVE_LABELS')
        jobs=c.read(root/'JOBS.json');jobmap={j['job_id']:j for j in jobs}
        ledger=rows(root/'LEDGER.jsonl') if (root/'LEDGER.jsonl').exists() else []
        replay={r['job_id'] for r in ledger if r['status']=='CERTIFIED'}
        stored=[];quarantined=set();cert_count=0
        for path in sorted((root/'shards').glob('*.audit.json')):
            a=c.read(path);index=path.with_name(path.name.replace('.audit.json','.jsonl'))
            assert a['integrity_pass'] and c.sha(index)==a['index_sha256']
            part=rows(index);qpath=root/a['quarantine_manifest'];q=c.read(qpath)
            assert len(q)==a['quarantined_routes'] and all(x['status']=='QUARANTINED_NOT_TRAINING_DATA' for x in q)
            assert len(part)==a['instruction_records'] and sum(r['decisions'] for r in part)==a['instruction_conditioned_decisions']
            assert len({r['job_id'] for r in part})==a['certified_routes']
            quarantined.update(x['job_id'] for x in q);stored+=part;cert_count+=1
            for p in (path,index,qpath):input_hashes[str(p.relative_to(c.ROOT))]=c.sha(p)
        grouped={}
        for row in stored:grouped.setdefault(row['job_id'],[]).append(row)
        assert set(grouped).isdisjoint(quarantined) and set(grouped)|quarantined<=replay
        if state['complete']:assert set(grouped)|quarantined==replay
        unique_decisions=0;conditioned=0
        for ident,part in grouped.items():
            assert ident in jobmap
            fresh,_,steps=strict.audit_route(root,jobmap[ident])
            accepted+=validate_accepted(root,jobmap[ident],part,fresh,physical,aliases)
            unique_decisions+=steps;conditioned+=sum(r['decisions'] for r in part)
        summary.append(dict(shard=shard,sourceRoot=str(root.relative_to(c.ROOT)),terminal_jobs=len(ledger),
            assigned_routes=len(jobs),complete=state['complete'],unattempted_or_censored_routes=len(jobs)-len(ledger),
            replay_certified_routes=len(replay),strict_routes=len(grouped),quarantine_routes=len(quarantined),
            replay_certified_not_yet_strict_audited=len(replay-set(grouped)-quarantined),
            rejected_routes=sum(r['status']!='CERTIFIED' for r in ledger),strict_instruction_records=sum(len(v) for v in grouped.values()),
            unique_route_decisions=unique_decisions,instruction_conditioned_decisions=conditioned,audited_subshards=cert_count))
        total_disk+=safe_size.measure(root)['apparent_bytes_conservative']
    assert physical<=expected_all and physical.isdisjoint(excluded)
    assert total_disk+safe_size.measure(HERE)['apparent_bytes_conservative']<199*1024**3,'TOTAL_DISK_MARGIN'
    out.mkdir()
    with (out/'TRAINING_INDEX.jsonl').open('x') as f:
        for row in accepted:f.write(json.dumps(row)+'\n')
    c.save(out/'INPUT_HASHES.json',input_hashes)
    c.save(out/'RESULT.json',dict(status='STRICT_CPU_READBACK_MERGED_NOT_TRAINED',shards=summary,
        strict_routes=len(physical),instruction_records=len(accepted),
        unique_route_decisions=sum(r['unique_route_decisions'] for r in summary),
        instruction_conditioned_decisions=sum(r['instruction_conditioned_decisions'] for r in summary),
        physical_duplicates=0,alias_duplicates=0,complete_alias_ownership=True,
        sourceRoot_required=True,index_sha256=c.sha(out/'TRAINING_INDEX.jsonl'),training_started=False,scientific_pass=False))
    print(json.dumps(c.read(out/'RESULT.json')))


if __name__=='__main__':execute()
