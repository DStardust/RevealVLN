"""Native StreamVLN unseen subset; same frozen worker and semantics as native_001."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u


def devices():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
    return {int(a):dict(uuid=b.strip(),free_mib=int(c)) for a,b,c in (line.split(',') for line in text.splitlines())}


def completed(run,phase):
    found={}
    for session in (run/phase).glob('*'):
        if not (session/'RESULT.json').exists():continue
        assert u.read(session/'RESULT.json')['base_unchanged']
        for path in (session/'episodes').glob('*/COMPLETE.json'):
            item=u.read(path)
            assert item['id'] not in found,'DUPLICATE_COMPLETE_EPISODE'
            found[item['id']]=dict(item,artifact=str(path))
    return found


def main(run):
    lock=(run/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    u.verify_sources(run);protocol=u.read(run/'PROTOCOL.json');started=time.time();workers=[];used_hours=0.
    # Include all earlier attempts, regardless of exit code or measured score.
    for receipt in (run/'attempts').glob('*.json'):
        record=u.read(receipt)
        used_hours+=record.get('wall_seconds',0)/3600
    status=dict(status='RUNNING',phase='ASSET_VERIFY',started_unix=started,completed_native=len(completed(run,'baseline')),
        planned_native=protocol['native_episodes'],method_training_started=False,method_benefit='UNKNOWN',gpu_hours=used_hours)
    u.write(run/'STATUS.json',status)
    def interrupted(signum,frame):raise InterruptedError(f'SIGNAL_{signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def persist():
        status.update(unix=time.time(),gpu_hours=used_hours+sum((time.time()-w['start'])/3600 for w in workers),
            workers=[dict(pid=w['proc'].pid,gpu=w['gpu'],phase=w['phase'],output=str(w['output']),
                progress=u.read(w['output']/'PROGRESS.json') if (w['output']/'PROGRESS.json').exists() else None) for w in workers])
        u.write(run/'STATUS.json',status)
    try:
        if not (run/'ASSET_VERIFY.json').exists():
            verified=[]
            for item in u.read(run/'ASSET_PROVENANCE.json')['files']:
                path=u.ASSETS/item['path'];actual=u.sha(path)
                assert actual==item['sha256'],'PUBLIC_ASSET_CHANGED: '+str(path)
                verified.append(dict(path=str(path),sha256=actual,bytes=path.stat().st_size))
                status['asset_files_verified']=len(verified);persist()
            u.write(run/'ASSET_VERIFY.json',dict(files=verified,status='VERIFIED',source='Existing upstream-pinned public asset manifest'))
        for phase,ids in [('baseline',list(range(protocol['native_episodes'])))]:
            status['phase']=phase.upper();done=completed(run,phase)
            pending=[i for i in ids if i not in done]
            if not pending:continue
            available=devices();registered=protocol['gpu_indices']
            free=[g for g in registered if available[g]['free_mib']>=26000]
            if phase=='identity':free=free[:1]
            if not free:raise RuntimeError('NO_REGISTERED_GPU_WITH_26_GIB_FREE')
            for rank,gpu in enumerate(free):
                shard=pending[rank::len(free)]
                if not shard:continue
                attempt=f'{phase}_gpu{gpu}_{time.time_ns()}';output=run/phase/attempt
                logpath=run/'attempts'/(attempt+'.log');logpath.parent.mkdir(parents=True,exist_ok=True)
                log=logpath.open('x')
                cmd=[str(u.PYTHON),'-I','-B',str(u.HERE/'native_worker.py'),'--run',str(run),'--output',str(output),'--ids',','.join(map(str,shard))]
                if phase=='identity':cmd.append('--identity-pairs')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=available[gpu]['uuid'],PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                proc=subprocess.Popen(cmd,env=env,cwd=u.REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                workers.append(dict(proc=proc,start=time.time(),gpu=gpu,uuid=available[gpu]['uuid'],phase=phase,output=output,log=log,
                    receipt=logpath.with_suffix('.json'),command=cmd,ids=shard))
            while workers:
                for w in list(workers):
                    code=w['proc'].poll()
                    if code is None:continue
                    elapsed=time.time()-w['start'];used_hours+=elapsed/3600;w['log'].close()
                    u.write(w['receipt'],dict(pid=w['proc'].pid,gpu=w['gpu'],uuid=w['uuid'],phase=w['phase'],ids=w['ids'],
                        command=w['command'],returncode=code,wall_seconds=elapsed,output=str(w['output'])))
                    workers.remove(w)
                    if code:raise RuntimeError(f"WORKER_FAILED:{w['output'].name}:exit{code}")
                if time.time()-started>protocol['wall_hours_limit']*3600 or used_hours+sum((time.time()-w['start'])/3600 for w in workers)>protocol['gpu_session_hours_limit']:
                    raise RuntimeError('REGISTERED_RESOURCE_BUDGET_REACHED')
                status['completed_native']=len(completed(run,'baseline'))
                # Show progress before the worker-wide final model-state seal.
                status['recorded_native']=len(list((run/'baseline').glob('*/episodes/*/COMPLETE.json')))
                persist()
                if workers:time.sleep(5)
            assert set(completed(run,phase))==set(ids),'MISSING_COMPLETE_REGISTERED_EPISODES'
        rows=list(completed(run,'baseline').values());outcomes=[r['outcomes']['NATIVE'] for r in rows];count=protocol['native_episodes']
        by_house={}
        for house in sorted({r['house'] for r in rows}):
            selected=[r['outcomes']['NATIVE'] for r in rows if r['house']==house]
            by_house[house]=dict(n=len(selected),successes=sum(x['success'] for x in selected),sr=sum(x['success'] for x in selected)/len(selected))
        result=dict(status='NATIVE_UNSEEN_SUBSET_COMPLETE',complete=count,planned=count,by_house=by_house,
            sr=sum(x['success'] for x in outcomes)/count,spl=sum(x['spl'] for x in outcomes)/count,
            os=sum(x['os'] for x in outcomes)/count,mean_steps=sum(x['steps'] for x in outcomes)/count,
            zero_adapter_identity_pairs=0,previous_identity_evidence=protocol['identity_evidence'],backbone=protocol['backbone'],exposed_internal_dev=False,public_unseen_subset=True,public_split_previously_exposed=True,full_1839_result=False,
            method_training_completed=True,trained_memory_heads_evaluated=False,method_benefit='NOT_YET_MEASURED',adopted=False,gpu_hours=used_hours)
        u.write(run/'RESULT.json',result)
        status.update(status='NATIVE_UNSEEN_SUBSET_COMPLETE',phase='BASELINE_COMPLETE',completed_native=count)
        persist()
    except BaseException as error:
        status.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error),traceback=traceback.format_exc())
        u.write(run/f'FAILURE_{time.time_ns()}.json',status)
        raise
    finally:
        # Only process groups created here, with their original Popen identity.
        for w in workers:
            proc=w['proc']
            if proc.poll() is None and os.getpgid(proc.pid)==proc.pid:os.killpg(proc.pid,signal.SIGTERM)
        for w in workers:
            try:w['proc'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                if w['proc'].poll() is None and os.getpgid(w['proc'].pid)==w['proc'].pid:os.killpg(w['proc'].pid,signal.SIGKILL)
                w['proc'].wait(timeout=10)
            w['log'].close();elapsed=time.time()-w['start'];used_hours+=elapsed/3600
            u.write(w['receipt'],dict(pid=w['proc'].pid,gpu=w['gpu'],uuid=w['uuid'],phase=w['phase'],ids=w['ids'],command=w['command'],
                returncode=w['proc'].returncode,wall_seconds=elapsed,output=str(w['output']),aborted_by_own_pipeline=True))
        workers.clear();status.update(finished_unix=time.time());persist();lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    run=u.HERE/'runs'/a.run_id
    if not a.resume and (run/'STARTED.json').exists():raise RuntimeError('USE_RESUME_FOR_EXISTING_RUN')
    u.write(run/'STARTED.json',dict(started_unix=time.time(),resume=a.resume))
    main(run)
