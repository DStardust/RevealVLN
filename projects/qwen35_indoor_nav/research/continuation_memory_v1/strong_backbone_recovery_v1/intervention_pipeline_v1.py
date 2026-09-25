"""Complete frozen TRAIN pairs, then fit CPU intervention selectors independently."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
from transfer_pipeline import records,placeholder,summarize
from unseen_pipeline import devices


def main(run):
    lock=(run/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    u.verify_sources(run);p=u.read(run/'PROTOCOL.json');workers=[];leased=False;started=time.time()
    hours=sum(u.read(x)['wall_seconds']/3600 for x in (run/'attempts').glob('*.json') if u.read(x).get('gpu') is not None)
    state=dict(status='RUNNING',phase='COLLECT_TRAIN_PAIRS',planned_groups=200,planned_episodes=800,started_unix=started)
    def save():
        state.update(unix=time.time(),gpu_hours=hours+sum((time.time()-w['start'])/3600 for w in workers if w['gpu'] is not None),
            recorded_groups=len(records(run,'evaluation',False)),sealed_groups=len(records(run,'evaluation',True)),
            workers=[dict(pid=w['p'].pid,gpu=w['gpu'],output=str(w['output']),progress=u.read(w['output']/'PROGRESS.json') if (w['output']/'PROGRESS.json').exists() else None) for w in workers])
        u.write(run/'STATUS.json',state)
    def finish(w):
        nonlocal hours
        seconds=time.time()-w['start'];hours+=seconds/3600 if w['gpu'] is not None else 0;w['log'].close()
        u.write(w['receipt'],dict(gpu=w['gpu'],pid=w['p'].pid,output=str(w['output']),command=w['command'],returncode=w['p'].returncode,wall_seconds=seconds))
    def spawn(command,gpu,out,name):
        logpath=run/'attempts'/(name+'_'+str(time.time_ns())+'.log');logpath.parent.mkdir(exist_ok=True);log=logpath.open('x')
        uuid=devices()[gpu]['uuid'] if gpu is not None else ''
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        process=subprocess.Popen(command,cwd=u.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(p=process,gpu=gpu,output=out,command=command,log=log,receipt=logpath.with_suffix('.json'),start=time.time()))
    def wait():
        while workers:
            for w in list(workers):
                if w['p'].poll() is not None:
                    finish(w);workers.remove(w)
                    if w['p'].returncode:raise RuntimeError('WORKER_FAILED:'+str(w['receipt']))
            save()
            if state['gpu_hours']>p['gpu_session_hours_limit'] or time.time()-started>p['wall_hours_limit']*3600:raise RuntimeError('REGISTERED_BUDGET')
            if workers:time.sleep(5)
    def interrupted(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for session in (run/'evaluation').glob('*'):
            if not (session/'STATE_SEAL.json').exists():
                dest=run/'failed_attempts';dest.mkdir(exist_ok=True);session.rename(dest/session.name)
        pending=[i for i in range(200) if i not in records(run,'evaluation',True)]
        if pending:
            before=placeholder('status')
            if before['leases'] or before['external_pids'] or before['manual_paused']:raise RuntimeError('PLACEHOLDER_BUSY')
            acquired=placeholder('acquire');leased=True;u.write(run/'RESOURCE_LEASE.json',dict(before=before,acquired=acquired,pid=os.getpid()))
            inventory=devices();free=[g for g in p['gpu_indices'] if inventory[g]['free_mib']>=26000]
            if not free:raise RuntimeError('NO_AUTHORIZED_GPU_MEMORY')
            for rank,gpu in enumerate(free):
                shard=pending[rank::len(free)]
                if not shard:continue
                out=run/'evaluation'/('gpu'+str(gpu)+'_'+str(time.time_ns()))
                spawn([str(u.PYTHON),'-I','-B',str(u.HERE/'transfer_worker_v2.py'),'--run',str(run),'--output',str(out),'--ids',','.join(map(str,shard))],gpu,out,'collect_gpu'+str(gpu))
            wait()
            u.write(run/'PLACEHOLDER_RELEASE.json',dict(result=placeholder('release'),unix=time.time()));leased=False
        assert len(records(run,'evaluation',True))==200
        result=summarize(run);result.update(purpose='TRAIN split causal calibration pairs, not unseen method efficacy',split='OFFICIAL_TRAIN',public_split_previously_exposed=True)
        u.write(run/'COLLECTION_RESULT.json',result)
        state['phase']='CPU_INTERVENTION_SELECTOR_TRAINING';save()
        if not (run/'GATE_REVIEW.json').exists():
            spawn([str(u.PYTHON),'-I','-B',str(u.HERE/'intervention_gate_v1.py'),'--run',str(run)],None,run,'gate_cpu');wait()
        state.update(status='COMPLETE',phase='CALIBRATION_COMPLETE');save()
        review=u.read(run/'GATE_REVIEW.json')
        lines=['CALIBRATION_COMPLETE','','200 条新官方 TRAIN 路线，800 次四臂真实执行；160 路线拟合 / 40 路线按房屋留出。',
            '原生与记忆头保持冻结；只在首次真实分歧处学习一次性的后续策略选择。标签描述整段后续策略成败，不是每个动作的局部因果真值。',
            'DEV 是选择已真实执行的完整后续分支，没有新运行在线门控策略；不称为新的 unseen SR。','']
        for arm,r in review['results'].items():lines.append(arm+': '+str({k:v for k,v in r.items() if k!='decisions'}))
        (run/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error))
        u.write(run/('FAILURE_'+str(time.time_ns())+'.json'),dict(state,traceback=traceback.format_exc()));raise
    finally:
        for w in workers:
            if w['p'].poll() is None and os.getpgid(w['p'].pid)==w['p'].pid:os.killpg(w['p'].pid,signal.SIGTERM)
        for w in workers:
            try:w['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                if w['p'].poll() is None and os.getpgid(w['p'].pid)==w['p'].pid:os.killpg(w['p'].pid,signal.SIGKILL)
                w['p'].wait(timeout=10)
            finish(w)
        workers.clear()
        if leased:u.write(run/'PLACEHOLDER_RELEASE.json',dict(result=placeholder('release'),unix=time.time()))
        state['finished_unix']=time.time();save();lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    root=u.HERE/'intervention_runs'/a.run_id
    if (root/'STARTED.json').exists() and not a.resume:raise ValueError('RESUME_REQUIRED')
    if (root/'STATUS.json').exists() and u.read(root/'STATUS.json')['status']=='COMPLETE':raise ValueError('COMPLETED_READ_ONLY')
    u.write(root/'STARTED.json',dict(unix=time.time(),resume=a.resume));main(root)
