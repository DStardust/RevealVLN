"""Freeze V2 never-attempted routes only; preserved terminal/partial are not retried."""
import fcntl
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
FULL=HERE.parent
ROOT=HERE.parents[4]
PREVIOUS=FULL/'auto_generation_v2'

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

def evidence():
    attempt=PREVIOUS/'lanes/gpu_6/attempt_000'
    result=read(attempt/'RESULT.json');restore=read(attempt/'RESTORATION.json')
    assert result['error']=="FileNotFoundError(2, 'No such file or directory')"
    assert result['restoration']==restore and restore['restored'] is True
    assert len(result['workers'])==1 and result['workers'][0]['shard']==2 and result['workers'][0]['returncode']==1
    process=read(attempt/'PROCESS_2.json');assert not Path('/proc',str(process['pid'])).exists()
    assert 'SUPERVISOR_STOP:15' in (attempt/'worker_2.log').read_text()
    paths=[attempt/p for p in ('RESULT.json','RESTORATION.json','PROCESS_2.json','worker_2.log')]
    previous=read(PREVIOUS/'PREVIOUS_FAILURES.json')
    paths += [PREVIOUS/'PREVIOUS_FAILURES.json',PREVIOUS/'INPUT_LOCK.json',PREVIOUS/'transport.py']
    prior_wall=previous['prior_wall_seconds']+result['wall_seconds']
    worker_seconds=dict(previous['prior_worker_seconds'])
    worker_seconds['2']+=result['workers'][0]['wall_seconds']
    rows=[];parts=[];dirs_before={}
    for shard in (2,4):
        root=PREVIOUS/'production'/f'shard_{shard:04d}'
        jobs=read(root/'JOBS.json')
        frozen=read(FULL/'auto_generation_v1/source_v1/RESCUE_JOBS.json')
        original_ids={j['job_id'] for j in read(FULL/'manifests'/f'shard_{shard:04d}/JOBS.json')}
        assert jobs==[j for j in frozen if j['job_id'] in original_ids]
        raw=(root/'LEDGER.jsonl').read_bytes() if (root/'LEDGER.jsonl').exists() else b''
        assert not raw or raw.endswith(b'\n'),'PARTIAL_LEDGER_UNRECONCILED'
        ledger=[json.loads(x) for x in raw.splitlines()]
        dirs={p.name for p in (root/'routes').iterdir()} if (root/'routes').exists() else set()
        subset,partial=classify(jobs,ledger,dirs);rows.extend(subset);dirs_before[str(shard)]=sorted(dirs)
        parts.append(dict(old_shard=shard,assigned=len(jobs),terminal=len(ledger),interrupted=partial,
                          unattempted=len(subset),unattempted_aliases=sum(len(j['instruction_alias_episodes']) for j in subset)))
        paths.extend(root/name for name in ('JOBS.json','INPUT_LOCK.json','SPLIT_FREEZE.json','SOURCE_INVENTORY.json','LEDGER.jsonl') if (root/name).exists())
    assert len({j['physical_source_route_sha256'] for j in rows})==len(rows)
    aliases=[(j['source'],str(e['episode_id'])) for j in rows for e in j['instruction_alias_episodes']]
    assert len(set(aliases))==len(aliases)
    return rows,dict(prior_wall_seconds=prior_wall,prior_worker_seconds=worker_seconds,
        immediate_previous_wall_seconds=result['wall_seconds'],immediate_previous_error=result['error'],
        failed_supervisor_path_unknown=True,previous_failure_evidence_preserved=True,
        shards=parts,old_route_directory_snapshot=dirs_before,routes=len(rows),instruction_aliases=len(aliases),
        prior_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths})

def execute():
    assert not (HERE/'source_v1').exists(),'NEW_SOURCE_VERSION_REQUIRED'
    handles=[]
    for p in (PREVIOUS/'lanes/gpu_6/PRODUCER.lock',PREVIOUS/'production/shard_0002/PRODUCER.lock',PREVIOUS/'production/shard_0004/PRODUCER.lock'):
        if p.exists():
            h=p.open('rb');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(h)
    rows,report=evidence();assert len(rows)==1363
    assert {str(s['old_shard']):s['unattempted'] for s in report['shards']}=={'2':365,'4':998}
    assert evidence()==(rows,report),'SOURCE_CHANGED_DURING_FREEZE'
    out=HERE/'source_v1';out.mkdir()
    save(out/'RESCUE_JOBS.json',rows)
    save(out/'PREVIOUS_FAILURES.json',report)
    save(out/'PLAN.json',dict(executable=False,runtime_allowed=False,scientific_pass=False,
        scope='NEVER_ATTEMPTED_AFTER_V2_SUPERVISOR_FAILURE',routes=len(rows),instruction_aliases=report['instruction_aliases'],
        shards=report['shards'],rescue_jobs_sha256=sha(out/'RESCUE_JOBS.json'),
        old_terminal_retried=0,old_partial_retried=0,prior_lane_seconds=report['prior_wall_seconds'],
        new_output_parent=str((FULL/'auto_generation_v3_production').relative_to(ROOT))))
    paths=[HERE/'source.py',HERE/'test_source.py',out/'RESCUE_JOBS.json',out/'PREVIOUS_FAILURES.json',out/'PLAN.json']
    lock=dict(report['prior_hashes']);lock.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    save(out/'INPUT_LOCK.json',lock);print(json.dumps(read(out/'PLAN.json')))
if __name__=='__main__':execute()
