"""Bounded, two-CPU-worker strict salvage. No simulation, CUDA, old-file edits."""
import collections
import concurrent.futures
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('inventory',HERE/'inventory.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
RUN=HERE/'run_001'


def worker_init():
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(7200,7200))
    global strict
    assert m.sha(m.STRICT)==m.STRICT_SHA
    s=importlib.util.spec_from_file_location('unchanged_strict',m.STRICT);strict=importlib.util.module_from_spec(s);s.loader.exec_module(strict)


def audit_one(item):
    root=m.DATA/f"shard_{item['shard']:04d}";job=item['job']
    try:
        rows,refs,steps=strict.audit_route(root,job)
    except AssertionError:
        assert item['stored'] is None,'PREVIOUSLY_ACCEPTED_NOW_STRICT_FAILED'
        return dict(shard=item['shard'],job_id=job['job_id'],accepted=False,status='NEW_STRICT_QUARANTINE',error=traceback.format_exc())
    assert rows and steps>0
    if item['stored'] is not None:assert rows==item['stored'],'PREVIOUS_INDEX_MISMATCH'
    bound=[]
    for row in rows:
        for name in ('policy_file','supervision_file'):assert m.safe(root/row[name]).is_relative_to(root.resolve())
        bound.append(dict(row,sourceRoot=str(root.relative_to(m.ROOT)),policy_sha256=m.sha(root/row['policy_file']),supervision_sha256=m.sha(root/row['supervision_file'])))
    return dict(shard=item['shard'],job_id=job['job_id'],accepted=True,rows=bound,steps=steps,
        rgb_references_checked=len(refs),previously_published=item['stored'] is not None)


def progress(value):
    temp=RUN/'PROGRESS.pending'
    with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False)
    temp.replace(RUN/'PROGRESS.json')


def append(path,value):
    with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    assert not RUN.exists(),'ONE_ATTEMPT_ONLY'
    RUN.mkdir();start=time.monotonic();pool=None;inventory=None
    def interrupted(signum,frame):raise RuntimeError('CPU_AUDIT_SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    signal.signal(signal.SIGALRM,interrupted);signal.alarm(7200)
    def budget():
        assert time.monotonic()-start<7200,'CPU_AUDIT_WALL_BUDGET'
        assert sum(p.stat().st_size for p in RUN.iterdir() if p.is_file())<2*1024**3,'CPU_AUDIT_OUTPUT_BUDGET'
    try:
        test=subprocess.run([str(m.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'test_inventory.py')],capture_output=True,text=True,timeout=60)
        m.save(RUN/'CPU_TEST_RESULT.json',dict(passed=test.returncode==0,output=test.stdout+test.stderr));assert test.returncode==0
        code_hashes={str(p.relative_to(m.ROOT)):m.sha(p) for p in (HERE/'SPEC_ZH.md',HERE/'inventory.py',HERE/'audit.py',HERE/'test_inventory.py')}
        m.save(RUN/'SOURCE_LOCK.json',code_hashes)
        progress(dict(status='CPU_METADATA_PREFLIGHT',unix=time.time(),training_admission=False))
        inventory=m.gather();budget()
        hashes=dict(inventory['hashes'],**code_hashes);m.save(RUN/'INPUT_HASHES.json',hashes)
        m.save(RUN/'RESCUE_JOBS.json',inventory['rescue']);m.save(RUN/'PARTIAL_NOT_RETRIED.json',inventory['partial'])
        m.save(RUN/'PRIOR_QUARANTINE.json',inventory['old_quarantine']);m.save(RUN/'GENERATION_REJECTED.json',inventory['generation_rejected'])
        overview=dict(shards=inventory['shards'],planned=20000,terminal=sum(x['terminal'] for x in inventory['shards']),
            partial=len(inventory['partial']),unattempted=len(inventory['rescue']),strict_reaudit_candidates=len(inventory['candidates']),
            prior_published_routes=len({r['job_id'] for r in inventory['stored_rows']}),
            prior_published_decisions=sum(r['decisions'] for r in inventory['stored_rows']),
            old_production_failure_preserved=True,training_admission=False)
        m.save(RUN/'INVENTORY.json',overview);print(json.dumps(overview,ensure_ascii=False),flush=True)
        pool=concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=worker_init)
        items=iter(inventory['candidates']);pending={};physical=set();aliases=set();done=0;accepted=0;decisions=0;quarantined=0
        by_house=collections.Counter();by_shard=collections.defaultdict(collections.Counter);last=0
        def submit():
            try:item=next(items)
            except StopIteration:return False
            pending[pool.submit(audit_one,item)]=item;return True
        for _ in range(8):
            if not submit():break
        while pending:
            budget()
            completed,_=concurrent.futures.wait(pending,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                item=pending.pop(future);r=future.result();done+=1
                if r['accepted']:
                    job=item['job'];m.accept(job,r['rows'],physical,aliases)
                    assert job['physical_source_route_sha256'] not in inventory['excluded']
                    for row in r['rows']:append(RUN/'CANDIDATE_INDEX_NOT_ADMITTED.jsonl',row)
                    accepted+=1;decisions+=sum(x['decisions'] for x in r['rows']);by_house[job['scene_id']]+=1
                    by_shard[str(item['shard'])].update(routes=1,decisions=r['steps'],recovered_unpublished=int(not r['previously_published']))
                else:quarantined+=1;append(RUN/'NEW_QUARANTINE.jsonl',r)
                append(RUN/'AUDIT_EVENTS.jsonl',{k:v for k,v in r.items() if k!='rows'});submit()
            if time.monotonic()-last>5 or not pending:
                last=time.monotonic()
                progress(dict(status='CPU_STRICT_REAUDIT',unix=time.time(),audited=done,total=len(inventory['candidates']),accepted_routes=accepted,
                    candidate_decisions=decisions,new_quarantined=quarantined,elapsed_seconds=time.monotonic()-start,training_admission=False))
        pool.shutdown(wait=True);pool=None
        assert done==len(inventory['candidates']) and accepted+quarantined==done
        progress(dict(status='VERIFYING_FINAL_INPUT_HASHES',unix=time.time(),audited=done,total=done,training_admission=False))
        for path,digest in hashes.items():budget();assert m.sha(m.ROOT/path)==digest,('INPUT_CHANGED',path)
        index=RUN/'CANDIDATE_INDEX_NOT_ADMITTED.jsonl';final=RUN/'TRAINING_INDEX.jsonl';assert not final.exists()
        # Rename only this run's newly generated candidate index after all gates.
        index.rename(final)
        result=dict(status='STRICT_CPU_SALVAGE_COMPLETE_NOT_TRAINED',unix=time.time(),strict_routes=accepted,instruction_records=len(aliases),
            instruction_conditioned_decisions=decisions,unique_route_decisions=decisions,source='ENVDROP_OFFICIAL_SYNTHETIC_ENGLISH',
            new_quarantined=quarantined,prior_quarantine_preserved=len(inventory['old_quarantine']),
            unattempted_routes=len(inventory['rescue']),partial_not_retried=len(inventory['partial']),by_house=dict(by_house),by_shard=dict(by_shard),
            index_sha256=m.sha(final),source_metadata_unchanged=True,unchanged_strict_auditor_sha256=m.STRICT_SHA,
            prior_base_decisions=1394744,combined_with_prior_base_decisions=1394744+decisions,remaining_to_3000000=max(0,3000000-1394744-decisions),
            physical_duplicates=0,alias_duplicates=0,training_started=False,active_snapshot_modified=False,new_simulator_rollouts=0,
            old_failure_preserved=True,wall_seconds=time.monotonic()-start,gpu_processes_signalled=0,scientific_pass=False)
        m.save(RUN/'RESULT.json',result);progress(dict(result,audited=done,total=done));print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        failure=dict(status='FAILED_OR_CENSORED_NOT_ADMITTED',unix=time.time(),error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start,training_admission=False)
        m.save(RUN/'FAILURE.json',failure);progress(failure);raise
    finally:
        signal.alarm(0)
        if pool is not None:
            pool.shutdown(wait=False,cancel_futures=True)
            for child in mp.active_children():child.terminate()
            for child in mp.active_children():
                child.join(timeout=5)
                if child.is_alive():child.kill();child.join(timeout=5)
        if inventory:
            for handle in inventory['locks']:handle.close()


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    main()
