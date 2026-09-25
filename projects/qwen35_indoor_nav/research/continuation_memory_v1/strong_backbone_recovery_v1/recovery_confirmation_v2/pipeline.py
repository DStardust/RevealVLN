"""Independent six-model confirmation using the immutable previous training pool."""
import argparse
from collections import Counter, deque
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import common as u
from transfer_pipeline import placeholder, records
from unseen_pipeline import devices


def freeze(run, extra=()):
    files={}
    for p in list(HERE.glob('*.py'))+[HERE.parent/n for n in ('common.py','action_boundary_v2.py','memory_v2.py',
            'capture_runtime_v3.py','transfer_runtime.py','transfer_audit.py','transfer_pipeline.py','unseen_pipeline.py',
            'preserve_objective_v3.py','train_memory_v2.py')]+list(extra):files[str(p)]=u.sha(p)
    for p in (run/'PROTOCOL.json',run/'DATA_MANIFEST.json'):files[str(p)]=u.sha(p)
    for p in (HERE.parent/'recovery_action_v1/recovery_model.py',HERE.parent/'recovery_action_v1/replay.py'):
        files[str(p)]=u.sha(p)
    for e in u.read(run/'DATA_MANIFEST.json')['episodes']:
        if 'prefix_path' in e:files[e['prefix_path']]=u.sha(e['prefix_path'])
    for p,h in u.read(run/'PROTOCOL.json')['shared_source_files'].items():
        assert u.sha(p)==h,'SHARED_SOURCE_CHANGED'
        files[p]=h
    for e in u.read(run/'DATA_MANIFEST.json')['episodes']:
        config=Path(e['config']);digest=u.sha(config)
        if 'config_sha256' in e:assert digest==e['config_sha256'],'FIXTURE_CONFIG_CHANGED'
        files[str(config)]=digest
        # These pinned single-episode YAML fixtures have one literal data_path.
        datasets=[line.split('data_path:',1)[1].strip() for line in config.read_text().splitlines() if 'data_path:' in line]
        assert len(datasets)==1 and Path(datasets[0]).is_file()
        files[datasets[0]]=u.sha(datasets[0])
    upstream=u.read(HERE.parent/'intervention_v2/runs/recovery_001/train/SOURCE_LOCK.json')['files']
    files.update({p:h for p,h in upstream.items() if Path(p).is_relative_to(u.CODE)})
    lock=run/'SOURCE_LOCK.json'
    if lock.exists():assert u.read(lock)['files']==files,'SOURCE_LOCK_CHANGED'
    else:u.write(lock,dict(files=files))


def evaluation(run, phase):
    out=run/phase
    if (out/'PROTOCOL.json').exists():
        freeze(out);return out
    p=u.read(run/'PROTOCOL.json');heads={}
    for arm,entry in p['models'].items():
        f=run/'training'/arm/'FINAL.pt';r=u.read(f.parent/'RESULT.json')
        assert r['actual_steps']==p['steps'] and u.sha(f)==r['final_sha256']
        heads[arm]=dict(path=str(f),sha256=r['final_sha256'],**entry)
    origin=Path(p['previous_run'])/phase
    manifest=u.read(origin/'DATA_MANIFEST.json')
    p.update(heads=heads,planned_groups=len(manifest['episodes']),split=u.read(origin/'PROTOCOL.json')['split'],capture_only=False)
    u.write(out/'PROTOCOL.json',p);u.write(out/'DATA_MANIFEST.json',manifest)
    freeze(out);return out

from confirmation_review import summary, final_report


def main(run):
    lock=(run/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);u.verify_sources(run)
    p=u.read(run/'PROTOCOL.json');began=time.time();workers=[];leased=False
    hours=sum(u.read(f)['wall_seconds']/3600 for f in (run/'attempts').glob('*.json') if u.read(f)['gpu'] is not None)
    hours+=u.read(run/'PREFLIGHT.json').get('probe_gpu_hours',0)
    wall_before=sum(u.read(f)['wall_seconds'] for f in (run/'sessions').glob('*.json'))
    state=dict(status='RUNNING',phase='TRAIN',gpu_hours=hours);phase_run=run;phase_folder='evaluation'
    def save():
        state.update(unix=time.time(),gpu_hours=hours+sum((time.time()-w['start'])/3600 for w in workers if w['gpu'] is not None),
            recorded_groups=len(records(phase_run,phase_folder,False)),sealed_groups=len(records(phase_run,phase_folder,True)),
            workers=[dict(pid=w['proc'].pid,gpu=w['gpu'],output=str(w['out']),
                progress=u.read(w['out']/'PROGRESS.json') if (w['out']/'PROGRESS.json').exists() else None) for w in workers])
        u.write(run/'STATUS.json',state)
    def spawn(command,gpu,out):
        path=run/'attempts'/f'{state["phase"]}_{gpu}_{time.time_ns()}.log';path.parent.mkdir(exist_ok=True)
        log=path.open('x');inventory=devices();uuid=inventory[gpu]['uuid'] if gpu is not None else ''
        if gpu is not None and inventory[gpu]['free_mib']<26000:raise RuntimeError('INSUFFICIENT_GPU_MEMORY')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        proc=subprocess.Popen(command,cwd=u.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(proc=proc,gpu=gpu,uuid=uuid,start=time.time(),out=out,log=log,path=path,command=command))
    def finish(w):
        nonlocal hours
        elapsed=time.time()-w['start'];hours+=elapsed/3600 if w['gpu'] is not None else 0;w['log'].close()
        u.write(w['path'].with_suffix('.json'),dict(gpu=w['gpu'],uuid=w['uuid'],pid=w['proc'].pid,returncode=w['proc'].returncode,
            wall_seconds=elapsed,command=w['command'],output=str(w['out'])))
    def wait(refill=None):
        last_space=0
        while workers:
            for w in list(workers):
                if w['proc'].poll() is not None:
                    finish(w);workers.remove(w)
                    if w['proc'].returncode:raise RuntimeError('WORKER_FAILED:'+str(w['path']))
                    if refill:refill(w['gpu'])
            save()
            if state['gpu_hours']>p['gpu_session_hours_limit'] or wall_before+time.time()-began>p['wall_hours_limit']*3600:raise RuntimeError('BUDGET_REACHED')
            if time.time()-last_space>60:
                state['artifact_bytes']=sum(f.stat().st_size for f in run.rglob('*') if f.is_file());last_space=time.time()
                if state['artifact_bytes']>p['artifact_limit_gib']*1024**3:raise RuntimeError('ARTIFACT_LIMIT')
            if workers:time.sleep(5)
    def execute(root,folder,script):
        nonlocal phase_run,phase_folder
        phase_run,phase_folder=root,folder;u.verify_sources(root)
        for session in (root/folder).glob('*'):
            if not (session/'STATE_SEAL.json').exists():
                failed=root/'failed_attempts';failed.mkdir(exist_ok=True);session.rename(failed/(session.name+'_'+str(time.time_ns())))
        manifest=u.read(root/'DATA_MANIFEST.json')['episodes'];done=records(root,folder,True)
        pending=deque(e['id'] for e in manifest if e['id'] not in done);state['planned_groups']=len(manifest)
        def refill(gpu):
            if not pending:return
            ids=[pending.popleft() for _ in range(min(p['chunk_pairs'],len(pending)))];out=root/folder/f'gpu{gpu}_{time.time_ns()}'
            spawn([str(u.PYTHON),'-I','-B',str(HERE/script),'--run',str(root),'--output',str(out),'--ids',','.join(map(str,ids))],gpu,out)
            if folder=='evaluation' and not u.read(root/'PROTOCOL.json').get('capture_only'):u.write(root/'LIVE_RESULT.json',summary(root))
        available=[g for g in p['gpu_indices'] if devices()[g]['free_mib']>=26000]
        if pending and not available:raise RuntimeError('NO_USABLE_GPU')
        for gpu in available:refill(gpu)
        wait(refill);save();assert set(records(root,folder,True))=={e['id'] for e in manifest}
        if folder=='evaluation' and not u.read(root/'PROTOCOL.json').get('capture_only'):
            result=summary(root);u.write(root/'RESULT.json',result);u.write(root/'LIVE_RESULT.json',result)
    def interrupted(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        before=placeholder('status')
        if before['leases'] or before['manual_paused'] or before['external_pids']:raise RuntimeError('RESOURCE_BUSY')
        acquired=placeholder('acquire');leased=True;u.write(run/f'RESOURCE_LEASE_{time.time_ns()}.json',dict(before=before,acquired=acquired))
        state['phase']='TRAIN';save()
        pending_models=deque(arm for arm in p['models'] if not (run/'training'/arm/'RESULT.json').exists())
        def train_next(gpu):
            if pending_models:
                arm=pending_models.popleft()
                spawn([str(u.PYTHON),'-I','-B',str(HERE/'train.py'),'--run',str(run),'--arm',arm],gpu,run/'training'/arm)
        available=[g for g in p['gpu_indices'] if devices()[g]['free_mib']>=26000]
        if pending_models and not available:raise RuntimeError('NO_USABLE_GPU')
        for gpu in available:train_next(gpu)
        wait(train_next)
        state['phase']='DEV';execute(evaluation(run,'dev'),'evaluation','worker.py')
        state['phase']='UNSEEN';execute(evaluation(run,'unseen'),'evaluation','worker.py')
        final_report(run,hours);state.update(status='COMPLETE',phase='REVIEW_COMPLETE')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error))
        u.write(run/f'FAILURE_{time.time_ns()}.json',dict(state,traceback=traceback.format_exc()));raise
    finally:
        for w in workers:
            if w['proc'].poll() is None and os.getpgid(w['proc'].pid)==w['proc'].pid:os.killpg(w['proc'].pid,signal.SIGTERM)
        for w in workers:
            try:w['proc'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                if w['proc'].poll() is None and os.getpgid(w['proc'].pid)==w['proc'].pid:os.killpg(w['proc'].pid,signal.SIGKILL)
                w['proc'].wait(timeout=10)
            finish(w)
        workers.clear()
        if leased:u.write(run/f'RESOURCE_RELEASE_{time.time_ns()}.json',dict(result=placeholder('release')))
        u.write(run/'sessions'/f'{time.time_ns()}.json',dict(status=state['status'],wall_seconds=time.time()-began))
        save();lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if (a.run/'STARTED.json').exists() and not a.resume:raise ValueError('RESUME_REQUIRED')
    if u.read(a.run/'STATUS.json')['status']=='COMPLETE':raise ValueError('CLOSED_RUN')
    u.write(a.run/'STARTED.json',dict(unix=time.time(),resume=a.resume));main(a.run)
