"""CPU mechanism check, then frozen seven-arm evaluation on all new routes."""
import argparse
from collections import deque
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import common as u
from transfer_pipeline import placeholder, records
from unseen_pipeline import devices
from review import summary, review


def main(run):
    lock = (run/'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    u.verify_sources(run)
    p = u.read(run/'PROTOCOL.json')
    started = time.time()
    workers = []
    leased = False
    hours = sum(u.read(f)['wall_seconds']/3600 for f in (run/'attempts').glob('*.json') if u.read(f)['gpu'] is not None)
    previous_wall = sum(u.read(f)['wall_seconds'] for f in (run/'sessions').glob('*.json'))
    state = dict(status='RUNNING', phase='CPU_MECHANISM', planned_groups=p['planned_groups'], planned_executions=p['planned_executions'])
    def save():
        state.update(unix=time.time(), gpu_hours=hours+sum((time.time()-w['start'])/3600 for w in workers if w['gpu'] is not None),
            recorded_groups=len(records(run,'evaluation',False)), sealed_groups=len(records(run,'evaluation',True)),
            workers=[dict(pid=w['proc'].pid, gpu=w['gpu'], output=str(w['out']),
                progress=u.read(w['out']/'PROGRESS.json') if (w['out']/'PROGRESS.json').exists() else None) for w in workers])
        u.write(run/'STATUS.json', state)
    def spawn(command, gpu, out):
        path = run/'attempts'/f'{state["phase"]}_{gpu}_{time.time_ns()}.log'
        path.parent.mkdir(exist_ok=True)
        inventory = devices() if gpu is not None else {}
        uuid = inventory[gpu]['uuid'] if gpu is not None else ''
        if gpu is not None and inventory[gpu]['free_mib']<26000:
            raise RuntimeError('INSUFFICIENT_GPU_MEMORY')
        log = path.open('x')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        proc = subprocess.Popen(command, cwd=u.REPO, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        workers.append(dict(proc=proc,gpu=gpu,uuid=uuid,start=time.time(),out=out,log=log,path=path,command=command))
    def finish(w):
        nonlocal hours
        elapsed = time.time()-w['start']
        hours += elapsed/3600 if w['gpu'] is not None else 0
        w['log'].close()
        u.write(w['path'].with_suffix('.json'), dict(gpu=w['gpu'], uuid=w['uuid'], pid=w['proc'].pid,
            returncode=w['proc'].returncode, wall_seconds=elapsed, command=w['command'], output=str(w['out'])))
    def wait(refill=None):
        last_update = 0
        while workers:
            for w in list(workers):
                if w['proc'].poll() is not None:
                    finish(w)
                    workers.remove(w)
                    if w['proc'].returncode:
                        raise RuntimeError('WORKER_FAILED:'+str(w['path']))
                    if refill:
                        refill(w['gpu'])
            save()
            if state['gpu_hours']>p['gpu_session_hours_limit'] or previous_wall+time.time()-started>p['wall_hours_limit']*3600:
                raise RuntimeError('BUDGET_REACHED')
            if time.time()-last_update>30:
                u.write(run/'LIVE_RESULT.json', summary(run))
                state['artifact_bytes'] = sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
                if state['artifact_bytes']>p['artifact_limit_gib']*1024**3:
                    raise RuntimeError('ARTIFACT_LIMIT')
                last_update = time.time()
            if workers:
                time.sleep(5)
    def interrupted(signum, frame):
        raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        save()
        if not (run/'mechanism/RESULT.json').exists():
            spawn([str(u.PYTHON),'-I','-B',str(HERE/'mechanism.py'),'--run',str(run)],None,run/'mechanism')
            wait()
        assert u.read(run/'mechanism/RESULT.json')['status']=='COMPLETE'
        state['phase'] = 'FROZEN_UNSEEN_EXTENSION'
        # Quarantine incomplete sessions; only whole, state-sealed groups count.
        for session in (run/'evaluation').glob('*'):
            if not (session/'STATE_SEAL.json').exists():
                failed = run/'failed_attempts'
                failed.mkdir(exist_ok=True)
                session.rename(failed/(session.name+'_'+str(time.time_ns())))
        done = records(run,'evaluation',True)
        pending = deque(e['id'] for e in u.read(run/'DATA_MANIFEST.json')['episodes'] if e['id'] not in done)
        if pending:
            before = placeholder('status')
            if before['leases'] or before['manual_paused'] or before['external_pids']:
                raise RuntimeError('RESOURCE_BUSY')
            acquired = placeholder('acquire')
            leased = True
            u.write(run/f'RESOURCE_LEASE_{time.time_ns()}.json',dict(before=before,acquired=acquired))
        def refill(gpu):
            if not pending:
                return
            ids = [pending.popleft() for _ in range(min(p['chunk_pairs'],len(pending)))]
            out = run/'evaluation'/f'gpu{gpu}_{time.time_ns()}'
            spawn([str(u.PYTHON),'-I','-B',str(HERE.parent/'recovery_confirmation_v2/worker.py'),
                   '--run',str(run),'--output',str(out),'--ids',','.join(map(str,ids))],gpu,out)
        inventory = devices()
        available = [g for g in p['gpu_indices'] if inventory[g]['free_mib']>=26000]
        if pending and not available:
            raise RuntimeError('NO_USABLE_GPU')
        for gpu in available:
            refill(gpu)
        wait(refill)
        state['phase'] = 'REVIEW'
        save()
        review(run,hours)
        state.update(status='COMPLETE',phase='REVIEW_COMPLETE')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error))
        u.write(run/f'FAILURE_{time.time_ns()}.json',dict(state,traceback=traceback.format_exc()))
        raise
    finally:
        for w in workers:
            if w['proc'].poll() is None and os.getpgid(w['proc'].pid)==w['proc'].pid:
                os.killpg(w['proc'].pid,signal.SIGTERM)
        for w in workers:
            try:
                w['proc'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                if w['proc'].poll() is None and os.getpgid(w['proc'].pid)==w['proc'].pid:
                    os.killpg(w['proc'].pid,signal.SIGKILL)
                w['proc'].wait(timeout=10)
            finish(w)
        workers.clear()
        if leased:
            u.write(run/f'RESOURCE_RELEASE_{time.time_ns()}.json',dict(result=placeholder('release')))
        u.write(run/'sessions'/f'{time.time_ns()}.json',dict(status=state['status'],wall_seconds=time.time()-started))
        save()
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if (args.run/'STARTED.json').exists() and not args.resume:
        raise ValueError('RESUME_REQUIRED')
    if u.read(args.run/'STATUS.json')['status']=='COMPLETE':
        raise ValueError('CLOSED_RUN')
    u.write(args.run/'STARTED.json',dict(unix=time.time(),resume=args.resume))
    main(args.run)
