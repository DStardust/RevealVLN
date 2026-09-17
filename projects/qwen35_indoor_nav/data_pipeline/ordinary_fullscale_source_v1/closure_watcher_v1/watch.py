"""One bounded CPU-only watcher for four ALREADY-RUNNING approved lanes.

Never starts/retries production, signals a GPU process, or edits an old receipt.
Only successful restored lanes reach their already-reviewed strict CPU merger.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
HERE=Path(__file__).resolve().parent
FULL=HERE.parent
ROOT=HERE.parents[4]
PYTHON=ROOT/'projects/qwen35_indoor_nav/.envs/q35n_habitat_v017_g0r/bin/python3'
LANES={3:('runtime_v5','production',[6],'runtime_v5/merge.py','runtime_v5/merge'),
       4:('runtime_v6','rescue_production',[0,1],'runtime_v6/merge.py','runtime_v6/merge'),
       6:('runtime_v3','production',[2,4],'runtime_v3_gpu6_merge_v1/merge.py','runtime_v3_gpu6_merge_v1/run_v1'),
       7:('runtime_v4','production',[3,5],'runtime_v4/merge.py','runtime_v4/merge')}
CAP_SECONDS=24000
POLL_SECONDS=45


def read(path):
    path=path.resolve(strict=True);assert path.is_relative_to(ROOT)
    return json.loads(path.read_text())


def sha(path):
    path=path.resolve(strict=True);assert path.is_relative_to(ROOT)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def new(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())


def append(out,value):
    with (out/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def ready(result,restoration,shards):
    assert result['error'] is None,'PRODUCTION_FAILED_NO_AUTOMATIC_RETRY'
    assert result['restoration']==restoration and restoration.get('restored') is True,'RESTORATION_NOT_PASSED'
    assert len(result['workers'])==len(shards)
    assert {w['shard'] for w in result['workers']}==set(shards)
    assert all(type(w['returncode']) is int and w['returncode']==0 for w in result['workers']),'WORKER_NOT_SUCCESS'
    assert 'new_identity' in restoration and 'gpu' in restoration,'ACTUAL_RESTORATION_EVIDENCE_REQUIRED'
    pid=restoration['new_identity']['pid'];gpu=restoration['gpu']
    assert gpu['processes'].get(str(pid),gpu['processes'].get(pid,0))>20000,'RESTORED_HOLDER_MEMORY_REQUIRED'


def merged(path):
    r=read(path/'RESULT.json')
    assert r['status']=='STRICT_CPU_READBACK_MERGED_NOT_TRAINED'
    assert not r['training_started'] and r['physical_duplicates']==r['alias_duplicates']==0
    assert sha(path/'TRAINING_INDEX.jsonl')==r['index_sha256']
    return r


def execute():
    lock=(HERE/'WATCHER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path,h in read(HERE/'INPUT_LOCK.json').items():assert sha(ROOT/path)==h,'WATCHER_FROZEN_INPUT_CHANGED'
    out=HERE/'run_v1';out.mkdir(exist_ok=False)
    start=time.monotonic();deadline=start+CAP_SECONDS
    pid=os.getpid();stat=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    new(out/'IDENTITY.json',dict(pid=pid,starttime_ticks=int(stat[19]),cwd=str(Path.cwd()),
        started_unix=time.time(),started_monotonic=start,deadline_monotonic=deadline,cap_seconds=CAP_SECONDS,
        poll_seconds=POLL_SECONDS,only_existing_lanes=LANES,production_launch_allowed=False,gpu_operations_allowed=False))
    states={g:'WATCHING' for g in LANES};outcomes={};decode_failures={g:0 for g in LANES}
    while any(s=='WATCHING' for s in states.values()) and time.monotonic()<deadline:
        for gpu,(version,namespace,shards,entry,output) in LANES.items():
            if states[gpu]!='WATCHING':continue
            attempt=FULL/version/'lanes'/f'gpu_{gpu}/attempt_000'
            try:
                progress=[]
                for shard in shards:
                    p=FULL/namespace/f'shard_{shard:04d}'/'PROGRESS.json'
                    if p.exists():
                        r=read(p);progress.append(dict(shard=shard,completed=r['completed'],target=r['target'],strict_routes=r['audited_routes'],conditional_actions=r['audited_instruction_conditioned_decisions'],age_seconds=time.time()-p.stat().st_mtime))
                append(out,dict(event='read_only_progress',gpu=gpu,monotonic=time.monotonic(),progress=progress,counts_provisional=True))
                if not (attempt/'RESULT.json').exists():continue
                result=read(attempt/'RESULT.json');restore=read(attempt/'RESTORATION.json')
                ready(result,restore,shards)
                evidence={str(p.relative_to(ROOT)):sha(p) for p in (attempt/'RESULT.json',attempt/'RESTORATION.json')}
                new(out/f'GPU_{gpu}_SUCCESS_BEFORE_MERGE.json',evidence)
                target=FULL/output
                if target.exists():
                    outcomes[gpu]=merged(target);states[gpu]='PREEXISTING_STRICT_MERGE_VERIFIED';continue
                remaining=deadline-time.monotonic()
                assert remaining>0,'WATCHER_DEADLINE_BEFORE_MERGE'
                append(out,dict(event='strict_cpu_merge_start',gpu=gpu,entry=entry,monotonic=time.monotonic()))
                with (out/f'GPU_{gpu}_MERGE.log').open('x') as log:
                    proc=subprocess.run([str(PYTHON),'-I','-B',str(FULL/entry)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=min(1800,remaining),check=False)
                assert proc.returncode==0,'STRICT_CPU_MERGE_FAILED_NO_RETRY'
                assert all(sha(ROOT/p)==h for p,h in evidence.items()),'PRODUCTION_RECEIPT_CHANGED'
                outcomes[gpu]=merged(target);states[gpu]='STRICT_MERGE_VERIFIED'
                append(out,dict(event='strict_cpu_merge_closed',gpu=gpu,result=outcomes[gpu],monotonic=time.monotonic()))
            except json.JSONDecodeError as e:
                decode_failures[gpu]+=1
                append(out,dict(event='partial_json_read',gpu=gpu,count=decode_failures[gpu],error=repr(e)))
                if decode_failures[gpu]>=3:states[gpu]='MALFORMED_METADATA_NO_MERGE'
            except Exception as e:
                states[gpu]='FAILED_OR_UNVERIFIED_NO_RETRY';outcomes[gpu]=dict(error=repr(e))
                append(out,dict(event='lane_stopped_without_retry',gpu=gpu,error=repr(e),monotonic=time.monotonic()))
        if any(s=='WATCHING' for s in states.values()):time.sleep(min(POLL_SECONDS,max(0,deadline-time.monotonic())))
    for gpu,status in states.items():
        if status=='WATCHING':states[gpu]='WATCHER_DEADLINE_PRODUCTION_NOT_TOUCHED'
    new(out/'RESULT.json',dict(states=states,outcomes=outcomes,elapsed_seconds=time.monotonic()-start,
        production_retries=0,gpu_operations=0,training_started=False,scientific_pass=False))
    print(json.dumps(read(out/'RESULT.json')),flush=True)


if __name__=='__main__':execute()
