"""Independent, resumable complete-group ordinary navigation comparison."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
from unseen_pipeline import devices


def placeholder(command):
    # Existing user-owned controller: never signal its workers ourselves.
    path='/run/gpu-placeholder/control.sock'
    if os.stat(path).st_uid!=os.getuid():raise RuntimeError('PLACEHOLDER_OWNER_MISMATCH')
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(25);client.connect(path)
        client.sendall((json.dumps(dict(command=command,pid=os.getpid()))+'\n').encode())
        response=json.loads(client.recv(65536))
    if not response['ok']:raise RuntimeError(str(response))
    return response['result']


def records(run,phase,sealed):
    result={}
    for session in (run/phase).glob('*'):
        if sealed and not (session/'STATE_SEAL.json').exists():continue
        for path in session.glob('episodes/*/COMPLETE.json'):
            item=u.read(path)
            if item['id'] in result:raise ValueError('DUPLICATE_COMPLETE_GROUP')
            result[item['id']]=dict(item,path=str(path))
    return result


def summarize(run):
    rows=records(run,'evaluation',True);planned=200
    arms={}
    for arm in ['NATIVE','BC','B2','OURS']:
        values=[x['outcomes'][arm] for x in rows.values()]
        successes=sum(v['success'] for v in values)
        arms[arm]=dict(complete=len(values),planned=planned,successes=successes,
            sr_completed=successes/len(values) if values else None,
            sr_full=successes/planned if len(values)==planned else None,
            identification_bounds=[successes/planned,(successes+planned-len(values))/planned],
            spl=sum(v['spl'] for v in values)/len(values) if values else None,
            osr=sum(v['os'] for v in values)/len(values) if values else None,
            mean_steps=sum(v['steps'] for v in values)/len(values) if values else None)
    paired={}
    for arm in ['BC','B2','OURS']:
        wins=[i for i,r in rows.items() if r['outcomes'][arm]['success']>r['outcomes']['NATIVE']['success']]
        losses=[i for i,r in rows.items() if r['outcomes'][arm]['success']<r['outcomes']['NATIVE']['success']]
        paired[arm]=dict(wins=wins,losses=losses,delta_sr=(len(wins)-len(losses))/planned if len(rows)==planned else None)
    houses={}
    for house in sorted({r['house'] for r in rows.values()}):
        selected=[r for r in rows.values() if r['house']==house]
        houses[house]={arm:dict(n=len(selected),successes=sum(r['outcomes'][arm]['success'] for r in selected)) for arm in arms}
    return dict(complete_groups=len(rows),planned_groups=planned,complete_episodes=4*len(rows),planned_episodes=800,
        arms=arms,paired=paired,by_house=houses,evaluation_complete=len(rows)==planned,
        purpose='Ordinary navigation preservation; no history-task recovery efficacy conclusion',
        adoption='NOT_AUTOMATIC',full_1839=False,public_split_previously_exposed=True)


def main(run):
    lock=(run/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    u.verify_sources(run);protocol=u.read(run/'PROTOCOL.json')
    start=time.time();workers=[];leased=False;hours=sum(u.read(p)['wall_seconds']/3600 for p in (run/'attempts').glob('*.json'))
    state=dict(status='RUNNING',phase='IDENTITY',started_unix=start,planned_groups=200,planned_episodes=800)
    def save():
        state.update(unix=time.time(),gpu_hours=hours+sum((time.time()-w['start'])/3600 for w in workers),
            recorded_groups=len(records(run,'evaluation',False)),sealed_groups=len(records(run,'evaluation',True)),
            workers=[dict(pid=w['p'].pid,gpu=w['gpu'],output=str(w['output']),
                progress=u.read(w['output']/'PROGRESS.json') if (w['output']/'PROGRESS.json').exists() else None) for w in workers])
        u.write(run/'STATUS.json',state);u.write(run/'LIVE_RESULT.json',summarize(run))
    def interrupted(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def finish(w,aborted=False):
        nonlocal hours
        duration=time.time()-w['start'];hours+=duration/3600
        w['log'].close()
        u.write(w['receipt'],dict(gpu=w['gpu'],uuid=w['uuid'],pid=w['p'].pid,ids=w['ids'],
            output=str(w['output']),returncode=w['p'].returncode,wall_seconds=duration,aborted=aborted))
    try:
        for phase,ids in [('identity',[0,1]),('evaluation',list(range(200)))]:
            state['phase']=phase.upper();pending=[i for i in ids if i not in records(run,phase,True)]
            if not pending:continue
            if phase=='evaluation':
                before=placeholder('status')
                if before['manual_paused'] or before['leases'] or before['external_pids']:
                    raise RuntimeError('PLACEHOLDER_NOT_AVAILABLE_FOR_THIS_TASK:'+str(before))
                leased=True
                u.write(run/'PLACEHOLDER_LEASE.json',dict(before=before,acquired=placeholder('acquire'),owner_pid=os.getpid()))
            inventory=devices()
            registered=[0] if phase=='identity' else protocol['gpu_indices']
            free=[g for g in registered if inventory[g]['free_mib']>=26000]
            if not free:raise RuntimeError('NO_REGISTERED_GPU_WITH_SUFFICIENT_MEMORY')
            for rank,gpu in enumerate(free):
                shard=pending[rank::len(free)]
                if not shard:continue
                name=f'{phase}_gpu{gpu}_{time.time_ns()}';out=run/phase/name
                logfile=run/'attempts'/(name+'.log');logfile.parent.mkdir(parents=True,exist_ok=True);log=logfile.open('x')
                command=[str(u.PYTHON),'-I','-B',str(u.HERE/'transfer_worker.py'),'--run',str(run),'--output',str(out),'--ids',','.join(map(str,shard))]
                if phase=='identity':command.append('--identity-pairs')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=inventory[gpu]['uuid'],OMP_NUM_THREADS='4',
                    OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                p=subprocess.Popen(command,env=env,cwd=u.REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                workers.append(dict(p=p,log=log,receipt=logfile.with_suffix('.json'),output=out,
                    gpu=gpu,uuid=inventory[gpu]['uuid'],ids=shard,start=time.time()))
            while workers:
                for w in list(workers):
                    code=w['p'].poll()
                    if code is not None:
                        finish(w);workers.remove(w)
                        if code:raise RuntimeError('WORKER_FAILED:'+str(w['output']))
                save()
                if state['gpu_hours']>protocol['gpu_session_hours_limit'] or time.time()-start>protocol['wall_hours_limit']*3600:
                    raise RuntimeError('REGISTERED_BUDGET_REACHED')
                if workers:time.sleep(5)
            if set(records(run,phase,True))!=set(ids):raise RuntimeError('MISSING_SEALED_GROUPS')
        result=summarize(run);result.update(status='COMPLETE',gpu_hours=hours)
        u.write(run/'RESULT.json',result)
        state.update(status='COMPLETE',phase='REVIEW_COMPLETE')
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,InterruptedError) else 'FAILED',error=str(error))
        u.write(run/f'FAILURE_{time.time_ns()}.json',dict(state,traceback=traceback.format_exc()))
        raise
    finally:
        for w in workers:
            if w['p'].poll() is None and os.getpgid(w['p'].pid)==w['p'].pid:os.killpg(w['p'].pid,signal.SIGTERM)
        for w in workers:
            try:w['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:
                if w['p'].poll() is None and os.getpgid(w['p'].pid)==w['p'].pid:os.killpg(w['p'].pid,signal.SIGKILL)
                w['p'].wait(timeout=10)
            finish(w,True)
        workers.clear()
        if leased:
            try:u.write(run/'PLACEHOLDER_RELEASE.json',dict(released=placeholder('release'),unix=time.time()))
            except Exception as error:state['placeholder_release_error']=str(error)
        state['finished_unix']=time.time();save();lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    run=u.HERE/'transfer_runs'/a.run_id
    if (run/'STARTED.json').exists() and not a.resume:raise ValueError('EXISTING_RUN_REQUIRES_RESUME')
    # Completed groups only; failed partial groups stay in their attempt directory.
    u.write(run/'STARTED.json',dict(resume=a.resume,unix=time.time()));main(run)

