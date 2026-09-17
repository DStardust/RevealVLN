"""All seven shards, strict final readback. Invoke only after every producer is closed."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import merge


def execute(waves):
    waves=[p.resolve(strict=True) for p in waves]
    assert len(waves)==len(set(waves)) and all(p.parent==c.PARALLEL for p in waves)
    locks=[];assigned=[];inputs={};wave_rows=[]
    for wave in waves:
        freeze=c.read(wave/'INPUT_LOCK.json')
        for path,h in freeze.items():assert c.sha(c.ROOT/path)==h,path
        inputs[str((wave/'INPUT_LOCK.json').relative_to(c.ROOT))]=c.sha(wave/'INPUT_LOCK.json')
        cfg=c.read(wave/'PREPARED_CONFIG.json');assigned+=cfg['selected_shards']
        for gpu,shards in cfg['gpu_to_shards'].items():
            lane=wave/'lanes'/f'gpu_{gpu}';assert lane.is_dir()
            handle=(lane/'PRODUCER.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
            attempts=sorted(lane.glob('attempt_*'));assert attempts,'MISSING_LANE_ATTEMPT'
            for attempt in attempts:
                result=c.read(attempt/'RESULT.json')
                assert result['restoration']['restored'],'HOLDER_NOT_RESTORED'
        result=c.read(wave/'merge/RESULT.json')
        assert result['selected_shards']==cfg['selected_shards']
        index=wave/'merge/TRAINING_INDEX.jsonl';assert c.sha(index)==result['index_sha256']
        rows=merge.rows(index)
        assert len(rows)==result['instruction_records']
        assert sum(r['decisions'] for r in rows)==result['instruction_conditioned_decisions']
        wave_rows+=rows
        inputs[str(index.relative_to(c.ROOT))]=c.sha(index)
    assert sorted(assigned)==list(range(7)),'ALL_SEVEN_SHARDS_EXACTLY_ONCE_REQUIRED'
    expected={j['physical_source_route_sha256'] for j in c.read(c.PARALLEL/'JOBS.json')}
    assert len(expected)==6642
    excluded=set(c.read(c.PARALLEL/'EXCLUSION_MANIFEST.json')['physical_route_keys'])
    assert expected.isdisjoint(excluded)
    strict=c.load('fullscale_final_strict',c.BASE/'audit.py')
    physical=set();aliases=set();accepted=[];summaries=[]
    for shard in range(7):
        root=c.shard_root(shard)
        handle=(root/'PRODUCER.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(handle)
        state=c.state(shard)
        if not state['complete']:
            assert c.read(root/'CENSOR_RECEIPT.json')==dict(approved=True,shard=shard,ledger_sha256=state['ledger_sha256'],
                status='EXPLICIT_RESOURCE_CENSOR_NO_UNATTEMPTED_NEGATIVE_LABELS')
        jobs={j['job_id']:j for j in c.read(root/'JOBS.json')}
        stored={};quarantine=set()
        for path in sorted((root/'shards').glob('*.audit.json')):
            a=c.read(path);index=path.with_name(path.name.replace('.audit.json','.jsonl'))
            assert c.sha(index)==a['index_sha256'] and a['integrity_pass']
            for row in merge.rows(index):stored.setdefault(row['job_id'],[]).append(row)
            q=c.read(root/a['quarantine_manifest'])
            assert len(q)==a['quarantined_routes']
            assert all(x['status']=='QUARANTINED_NOT_TRAINING_DATA' for x in q)
            quarantine.update(r['job_id'] for r in q)
        assert set(stored).isdisjoint(quarantine)
        decisions=0
        for ident,rows in stored.items():
            fresh,_,n=strict.audit_route(root,jobs[ident])
            accepted+=merge.validate_accepted(root,jobs[ident],rows,fresh,physical,aliases)
            decisions+=n
        summaries.append(dict(shard=shard,terminal_routes=state['terminal_jobs'],complete=state['complete'],
            strict_routes=len(stored),quarantine_routes=len(quarantine),unique_route_decisions=decisions))
    assert physical<=expected and physical.isdisjoint(excluded)
    ordering=lambda r:(r['sourceRoot'],r['job_id'],r['policy_file'])
    assert sorted(accepted,key=ordering)==sorted(wave_rows,key=ordering),'WAVE_VS_FINAL_INDEX_MISMATCH'
    out=c.PARALLEL/'merge_all';out.mkdir(exist_ok=False)
    with (out/'TRAINING_INDEX.jsonl').open('x') as f:
        for row in accepted:f.write(json.dumps(row)+'\n')
    c.save(out/'SOURCE_LOCKS.json',inputs)
    c.save(out/'RESULT.json',dict(status='ALL_SEVEN_STRICT_READBACK_NOT_TRAINED',shards=summaries,
        strict_routes=len(physical),instruction_records=len(accepted),
        unique_route_decisions=sum(s['unique_route_decisions'] for s in summaries),
        instruction_conditioned_decisions=sum(r['decisions'] for r in accepted),sourceRoot_required=True,
        duplicates=0,index_sha256=c.sha(out/'TRAINING_INDEX.jsonl'),training_started=False,scientific_pass=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--waves',type=Path,nargs='+',required=True)
    execute(p.parse_args().waves)
