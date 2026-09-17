"""CPU post-replay gate: all fixed three must satisfy the original strict auditor."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import merge

def inspect(root,jobs,strict):
    assert len(jobs)==3 and len({j['scene_id'] for j in jobs})==3,'FIXED_THREE_DIFFERENT_FIT_HOUSES'
    byid={j['job_id']:j for j in jobs};assert len(byid)==3
    ledger=merge.rows(root/'LEDGER.jsonl')
    assert len(ledger)==3 and {r['job_id'] for r in ledger}==set(byid),'EXACT_SENTINEL_TERMINALS'
    assert all(r['status']=='CERTIFIED' for r in ledger),'SENTINEL_REPLAY_REJECTED'
    hashes={};stored=[];audits=sorted((root/'shards').glob('*.audit.json'))
    assert len(audits)==1,'ONE_SENTINEL_SUBSHARD_REQUIRED'
    for p in audits:
        a=c.read(p);index=p.with_name(p.name.replace('.audit.json','.jsonl'))
        qpath=root/a['quarantine_manifest'];q=c.read(qpath)
        assert a['integrity_pass'] and c.sha(index)==a['index_sha256']
        assert a['certified_routes']==3 and a['quarantined_routes']==0 and q==[],'SENTINEL_STRICT_QUARANTINE'
        part=merge.rows(index)
        assert len(part)==a['instruction_records']==3
        assert sum(r['decisions'] for r in part)==a['instruction_conditioned_decisions']
        stored+=part
        for file in (p,index,qpath):hashes[str(file.relative_to(c.ROOT))]=c.sha(file)
    grouped={}
    for row in stored:grouped.setdefault(row['job_id'],[]).append(row)
    assert set(grouped)==set(byid),'SENTINEL_ALL_THREE_REQUIRED'
    physical=set();aliases=set();steps=0
    for ident,part in grouped.items():
        fresh,_,n=strict.audit_route(root,byid[ident])
        merge.validate_accepted(root,byid[ident],part,fresh,physical,aliases);steps+=n
    for p in (root/'LEDGER.jsonl',root/'GENERATION_COMPLETE.json',root/'JOBS.json'):
        hashes[str(p.relative_to(c.ROOT))]=c.sha(p)
    return dict(strict_pass=True,strict_routes=3,instruction_records=3,unique_route_decisions=steps,
        input_hashes=hashes,source_jobs_sha256=c.sha(root/'JOBS.json'),input_lock_sha256=c.sha(HERE/'INPUT_LOCK.json'),
        source_grade='OFFICIAL_SYNTHETIC_ENGLISH_CE_PORT_NOT_HUMAN',full_natural_language_semantics_certified=False,
        scientific_pass=False,training_started=False)

def execute(output):
    output=output.resolve();assert output.is_relative_to(HERE/'lanes') and not output.exists()
    result=None
    try:
        c.approved(6);root=c.shard_root(0)
        producer=(root/'PRODUCER.lock').open('a');fcntl.flock(producer,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert c.state(0)['complete'],'SENTINEL_NOT_CLOSED'
        strict=c.load('envdrop_sentinel_original_audit',c.BASE/'audit.py')
        result=inspect(root,c.read(root/'JOBS.json'),strict)
    except BaseException as exc:
        result=dict(strict_pass=False,error=repr(exc),strict_routes=0,scientific_pass=False)
        raise
    finally:
        if result is not None:c.save(output,result)
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);execute(p.parse_args().output)
