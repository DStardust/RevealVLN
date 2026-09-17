"""Closed-lane inventory, CPU salvage, prospective partition and freeze."""
import collections
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import transport as t
import run
import merge
OLD_DATA=HERE.parent/'envdrop_production_v2'
OLD_SHARDS=tuple(range(0,42,3))
OLD_LOCK_SHA='c693ff433c3a02557c5d125400ce77d877b396a4d26544ffd7420fba1189509f'

def partition(jobs):
    assert len(jobs)==len({j['job_id'] for j in jobs})==3112
    assert len({j['physical_source_route_sha256'] for j in jobs})==3112
    houses=collections.defaultdict(list)
    for j in jobs:houses[j['scene_id']].append(j)
    sentinel=[houses[h][0] for h in sorted(houses)[:3]]
    assert len(sentinel)==3
    used={j['job_id'] for j in sentinel};tail=[j for j in jobs if j['job_id'] not in used]
    shards=[sentinel]+[tail[i:i+50] for i in range(0,len(tail),50)]
    assert len(shards)==64 and sum(map(len,shards))==3112
    return shards

def inventory():
    assert c.sha(t.OLD/'INPUT_LOCK.json')==OLD_LOCK_SHA
    lane=t.OLD/'lanes/gpu_3/attempt_000'
    result=c.read(lane/'RESULT.json');restored=c.read(lane/'RESTORATION.json')
    assert result['error']=="AssertionError('SHARD_WALL_BUDGET')" and restored==result['restoration']
    assert restored['restored'] and result['external_processes_stopped']==0
    for pid in (1421591,1424442):assert not Path('/proc',str(pid)).exists(),'OLD_GPU3_WORKER_STILL_PRESENT'
    jobs=[];terminal=[];partials=[];candidates=[];hashes={}
    for shard in OLD_SHARDS:
        root=OLD_DATA/f'shard_{shard:04d}';rows=c.read(root/'JOBS.json');byid={j['job_id']:j for j in rows}
        assert len(byid)==len(rows)
        jobs+=rows;ledger=root/'LEDGER.jsonl'
        completed=merge.rows(ledger) if ledger.exists() else []
        terminal+=completed;seen=set()
        for row in completed:
            ident=row['job_id'];assert ident in byid and ident not in seen;seen.add(ident)
            folder=root/'routes'/ident
            assert c.read(folder/'result.json')==row
            assert row['status']=='CERTIFIED'
            candidates.append((shard,byid[ident]))
            for path in folder.iterdir():
                assert path.is_file()
                hashes[str(path.relative_to(c.ROOT))]=c.sha(path)
        if (root/'routes').exists():
            for folder in (root/'routes').iterdir():
                assert folder.is_dir() and folder.name in byid
                if folder.name not in seen:
                    partials.append(dict(shard=shard,job_id=folder.name,status='PARTIAL_NOT_RETRIED'))
                    for p in folder.iterdir():hashes[str(p.relative_to(c.ROOT))]=c.sha(p)
        paths=[root/'JOBS.json']+([ledger] if ledger.exists() else [])
        paths+=list((root/'shards').glob('*')) if (root/'shards').exists() else []
        paths+=list((root/'quality').glob('*.json'))
        for p in paths:
            assert p.is_file() and not p.name.endswith('.pending')
            hashes[str(p.relative_to(c.ROOT))]=c.sha(p)
    assert len(jobs)==3253 and len({j['job_id'] for j in jobs})==3253
    assert len(terminal)==140 and len(partials)==1 and partials[0]['job_id']=='envdrop_0b264fe96107cd01107c'
    excluded={r['job_id'] for r in terminal+partials}
    pending=[j for j in jobs if j['job_id'] not in excluded]
    assert len(pending)==3112
    other=[j for shard in range(42) if shard not in OLD_SHARDS for j in c.read(OLD_DATA/f'shard_{shard:04d}/JOBS.json')]
    assert {j['job_id'] for j in pending}.isdisjoint(j['job_id'] for j in other)
    assert {j['physical_source_route_sha256'] for j in pending}.isdisjoint(j['physical_source_route_sha256'] for j in other)
    original={j['job_id']:j for j in c.read(t.OLD/'JOBS.json')}
    fit=set(c.read(c.BASE/'SPLIT_FREEZE.json')['FIT'])
    for j in pending:
        assert j==original[j['job_id']] and j['scene_id'] in fit and j['split']=='FIT'
    for p in (lane/'RESULT.json',lane/'RESTORATION.json',lane/'SUPERVISOR_EXCEPTION.json'):
        hashes[str(p.relative_to(c.ROOT))]=c.sha(p)
    times=[r['elapsed_seconds'] for r in terminal]
    proof=dict(old_assigned=3253,old_terminal=140,old_partial=partials,unattempted=3112,
        distinct_fit_houses=len({j['scene_id'] for j in pending}),old_route_mean_seconds=statistics.mean(times),
        old_route_max_seconds=max(times),old_error=result['error'],other_lanes_disjoint=True,old_failure_preserved=True)
    return pending,candidates,hashes,proof

def salvage(candidates):
    out=HERE/'old_salvage';out.mkdir(exist_ok=False)
    strict=c.load('gpu3_old_strict_auditor',c.BASE/'audit.py')
    assert c.sha(c.BASE/'audit.py')=='37c3443c312cb635572b524cba4df2870ed835eea467cc8bcd5ed6aa029d97fe'
    rows=[];prior_count=0;physical=set();aliases=set()
    for shard in (0,3):
        root=OLD_DATA/f'shard_{shard:04d}';stored={}
        for p in sorted((root/'shards').glob('*.audit.json')):
            audit=c.read(p);index=p.with_name(p.name.replace('.audit.json','.jsonl'))
            assert c.sha(index)==audit['index_sha256'] and audit['integrity_pass']
            assert c.read(root/audit['quarantine_manifest'])==[]
            for row in merge.rows(index):stored.setdefault(row['job_id'],[]).append(row)
        prior_count+=len(stored)
        for source_shard,j in candidates:
            if source_shard!=shard:continue
            fresh,_,steps=strict.audit_route(root,j)
            assert len(fresh)==1 and steps==fresh[0]['decisions']
            if j['job_id'] in stored:assert fresh==stored[j['job_id']],'OLD_PUBLISHED_AUDIT_CHANGED'
            added=merge.validate_accepted(root,j,fresh,fresh,physical,aliases)
            for r in added:
                r['policy_sha256']=c.sha(root/r['policy_file'])
                r['supervision_sha256']=c.sha(root/r['supervision_file'])
                rows.append(r)
    assert len(rows)==140 and prior_count==103
    with (out/'TRAINING_INDEX.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
        f.flush()
        import os
        os.fsync(f.fileno())
    result=dict(status='STRICT_OLD_GPU3_SALVAGE_COMPLETE_NOT_TRAINED',strict_routes=len(rows),
        instruction_conditioned_decisions=sum(r['decisions'] for r in rows),previously_published=prior_count,
        recovered_unpublished=len(rows)-prior_count,index_sha256=c.sha(out/'TRAINING_INDEX.jsonl'),
        old_outputs_modified=False,training_started=False)
    c.save(out/'RESULT.json',result)
    return result

def main():
    assert not (HERE/'INPUT_LOCK.json').exists() and not t.DATA.exists(),'FRESH_NODE_ONLY'
    tests=c.read(HERE/'CPU_TESTS.json');assert tests['passed'] and tests['errors']==tests['failures']==0
    for name,digest in tests['code_sha256'].items():assert c.sha(HERE/name)==digest,'TESTED_CODE_CHANGED'
    locks=[]
    for p in [t.OLD/'lanes/gpu_3/PRODUCER.lock']+[OLD_DATA/f'shard_{s:04d}/PRODUCER.lock' for s in (0,3)]:
        handle=p.open('rb');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
    pending,candidates,hashes,proof=inventory();shards=partition(pending)
    lock=c.read(t.OLD/'INPUT_LOCK.json')
    for name,digest in lock.items():assert c.sha(c.ROOT/name)==digest,('INHERITED_LOCK_CHANGED',name)
    old_identity=next(i for i in c.read(t.OLD/'IDENTITIES.json')['holders'] if i['gpu_device']==3)
    restoration=c.read(t.OLD/'lanes/gpu_3/attempt_000/RESTORATION.json')
    ident=dict(old_identity,**restoration['new_identity'])
    run.verify_identity(ident)
    snapshot=run.gpu_snapshot(ident)
    assert snapshot['processes'].get(ident['pid'],0)>20000
    external=[v for p,v in snapshot['processes'].items() if p!=ident['pid']]
    assert all(0<=v<=768 for v in external) and sum(external)<1024
    protected=[run.process_identity(p) for p in (1421392,1421592,1421593)]
    recovered=salvage(candidates)
    for name,digest in hashes.items():assert c.sha(c.ROOT/name)==digest,'OLD_METADATA_CHANGED_DURING_AUDIT'
    cfg=dict(node='GPU3_UNATTEMPTED_CONTINUATION_50_ROUTE_V1',runtime_allowed=False,executable=False,
        training_allowed=False,routes=3112,gpu_to_shards={'3':list(range(64))},
        shard_seconds={str(s):600 if s==0 else 3600 for s in range(64)},lane_seconds={'3':43200},
        cleanup_margin_seconds=120,per_shard_max_bytes=4*1024**3,total_max_bytes=264*1024**3,
        metadata_and_merge_max_bytes=8*1024**3,worker_ram_bytes=12*1024**3,own_gpu_mib=4096,
        retry_failed_or_partial=False,restore_exact_holder=True,old_failure_preserved=True,
        auto_continue_after_strict_sentinel=True,original_quality_thresholds_unchanged=True)
    outputs=[]
    for shard,rows in enumerate(shards):
        root=c.shard_root(shard);root.mkdir(parents=True,exist_ok=False);(root/'quality').mkdir()
        for name,value in {'JOBS.json':rows,'SPLIT_FREEZE.json':c.read(c.BASE/'SPLIT_FREEZE.json'),
            'SOURCE_INVENTORY.json':c.read(t.MANIFEST/'SOURCE_INVENTORY.json')}.items():
            c.save(root/name,value);outputs.append(root/name)
    values={'JOBS.json':pending,'SOURCE_RECONCILIATION.json':proof,'OLD_INPUT_HASHES.json':hashes,
        'IDENTITIES.json':{'holders':[ident]},'PROTECTED_PRODUCERS.json':protected,
        'ASSETS.json':c.read(t.OLD/'ASSETS.json'),'PREPARED_CONFIG.json':cfg,
        'AUTHORIZATION.json':dict(approved=True,user_request='不能让gpu3那个继续生成吗',configuration=cfg),
        'PREPARE_RECEIPT.json':dict(unix=time.time(),strict_old_salvage=recovered,old_lane_lock_exclusive=True,
            old_shard_locks_exclusive=True,source_preflight=proof,holder_gpu_snapshot=snapshot)}
    for name,value in values.items():c.save(HERE/name,value);outputs.append(HERE/name)
    paths=list(HERE.glob('*.py'))+[HERE/'SPEC_ZH.md',HERE/'CPU_TESTS.json',t.OLD/'INPUT_LOCK.json',t.OLD/'transport.py']+outputs
    paths+=list((HERE/'old_salvage').iterdir())
    lock.update(hashes)
    lock.update({str(p.relative_to(c.ROOT)):c.sha(p) for p in paths})
    c.save(HERE/'INPUT_LOCK.json',lock)
    for s in c.SELECTED_SHARDS:c.save(c.shard_root(s)/'INPUT_LOCK.json',lock)
    print(json.dumps(dict(status='PREPARED_MAIN_APPROVAL_REQUIRED',routes=3112,shards=64,
        source=proof,salvage=recovered,input_lock_sha256=c.sha(HERE/'INPUT_LOCK.json'),immutable_bindings=len(lock))),flush=True)

if __name__=='__main__':main()
