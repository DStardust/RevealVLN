"""Eight-GPU independent coordinator; full model/group recovery and owned cleanup."""
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
from shared import *
resource=load('b2_parallel_resource',PILOT/'parallel_eval_v1/coordinator.py')
monitor=resource.monitor
sys.modules.pop('evaluate_continuations',None)
sys.path.insert(0,str(HERE))


def execute(run,cfg,stage,assignments,gpu=True):
    workers=[];start=time.monotonic();prior_hours=sum(r['gpu_hours'] for r in c.records(run/'RESOURCES.jsonl')) if (run/'RESOURCES.jsonl').exists() else 0
    errors=None;last_disk=0
    try:
        for d,tags in assignments:
            device=dict(cfg,**d);attempts=run/'attempts';attempts.mkdir(exist_ok=True)
            prefix=stage+'_gpu'+str(d['gpu'])+'_';folder=attempts/f'{prefix}{len(list(attempts.glob(prefix+"*")))+1:03d}';folder.mkdir()
            env=resource.env_for(device);env.update(B2_DEVICE=str(run/'devices'/f"gpu_{d['gpu']}.json"),B2_GPU=str(d['gpu']),B2_TRAIN_TAGS=','.join(tags))
            if not gpu:env['CUDA_VISIBLE_DEVICES']=''
            if gpu and monitor.gpu_snapshot(device)['free_mib']<cfg['min_start_gpu_gib']*1024:raise RuntimeError('DEVICE_HEADROOM')
            command=[cfg['torch_python'],'-I','-B',str(HERE/(stage+'.py')),str(run)]
            log=(folder/'stdout.log').open('x');began=time.monotonic()
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            w=dict(proc=proc,owner=monitor.process_identity(proc.pid),folder=folder,cfg=device,began=began,log=log,ended=None,members=[])
            workers.append(w);write(folder/'OWNER.json',w['owner'],True)
            write(folder/'COMMAND.json',dict(argv=command,device=d,training_tags=tags,gpu_enabled=gpu),True)
        while True:
            now=time.monotonic();views=[];hours=prior_hours
            for w in workers:
                code=w['proc'].poll()
                if code is not None and w['ended'] is None:w['ended']=now
                elapsed=(w['ended'] or now)-w['began']
                if gpu:hours+=elapsed/3600
                state=monitor.resources(w['proc'],w['owner'])
                for pid in state['owned_pids']:
                    try:
                        who=monitor.process_identity(pid)
                        if who not in w['members']:w['members'].append(who)
                    except (OSError,ValueError):pass
                snapshot=monitor.gpu_snapshot(w['cfg']) if gpu else None
                views.append(dict(gpu=w['cfg']['gpu'],returncode=code,seconds=elapsed,resource=state,device=snapshot))
                if code not in (None,0):raise RuntimeError(f"FAILED:{stage}:GPU{w['cfg']['gpu']}:{w['folder']}")
                if code is None and (elapsed>cfg['max_session_hours']*3600 or state['rss_bytes']>cfg['process_rss_gib']*2**30):raise RuntimeError('PROCESS_BUDGET_LIMIT')
                if code is None and gpu and snapshot['free_mib']<cfg['min_free_gpu_gib']*1024:raise RuntimeError('GPU_HEADROOM_LOST')
            write(run/'STATUS.json',dict(status='RUNNING',stage=stage,workers=views,gpu_hours=hours,unix=time.time()))
            append(run/'MONITOR.jsonl',dict(stage=stage,workers=views,gpu_hours=hours,unix=time.time()))
            if hours>cfg['gpu_session_hours']:raise RuntimeError('AGGREGATE_GPU_BUDGET')
            if now-last_disk>120:
                scan=subprocess.run(['du','-sb',str(run)],capture_output=True,text=True,timeout=60)
                if scan.returncode and not all('.tmp' in l and 'No such file' in l for l in scan.stderr.splitlines()):raise RuntimeError('DISK_SCAN_FAILED')
                if int(scan.stdout.split()[0])>cfg['artifact_gib']*2**30:raise RuntimeError('ARTIFACT_LIMIT')
                last_disk=now
            if all(w['ended'] is not None for w in workers):break
            time.sleep(5)
    except BaseException as exc:errors=repr(exc);raise
    finally:
        for w in workers:
            resource.stop_owned(w);w['log'].close()
            write(w['folder']/'EXIT.json',dict(returncode=w['proc'].poll(),seconds=(w['ended'] or time.monotonic())-w['began'],error=errors),True)
        append(run/'RESOURCES.jsonl',dict(stage=stage,gpu_hours=sum(((w['ended'] or time.monotonic())-w['began'])/3600 for w in workers) if gpu else 0,wall_seconds=time.monotonic()-start,error=errors))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--run-id',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=read(args.config)
    if not args.run_id.replace('_','').isalnum():raise ValueError('RUN_ID')
    run=HERE/'runs'/args.run_id
    if run.exists() and not args.resume:raise FileExistsError('USE_RESUME')
    run.mkdir(parents=True,exist_ok=True);lock=(run/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    immutable(run/'PROTOCOL.json',cfg);sources=read(HERE/'SOURCE_LOCK.json');verify_lock(sources);immutable(run/'SOURCE_LOCK.json',sources)
    (run/'devices').mkdir(exist_ok=True)
    tags=[f'{a}_{s}' for s in cfg['seeds'] for a in cfg['arms']]
    for i,d in enumerate(cfg['devices']):immutable(run/'devices'/f"gpu_{d['gpu']}.json",dict(d,conditions=list(range(128))[i::8]))
    acquired=False;error=None
    def signal_stop(signum,frame):raise InterruptedError(f'SIGNAL_{signum}')
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    try:
        if sha(Path(cfg['checkpoint']))!=cfg['checkpoint_sha256']:raise ValueError('CHECKPOINT_CHANGED')
        execute(run,cfg,'prepare',[(cfg['devices'][0],[])],gpu=False)
        if not (run/'features/FEATURE_RESULT.json').exists():execute(run,cfg,'features',[(cfg['devices'][1],[])])
        if not (run/'RESULT.json').exists():
            write(run/f'PLACEHOLDER_BEFORE_{time.time_ns()}.json',resource.verify_placeholders(),True)
            write(run/f'LEASE_ACQUIRED_{time.time_ns()}.json',resource.lease('acquire',pid=os.getpid()),True);acquired=True
            deadline=time.monotonic()+60
            while any(monitor.gpu_snapshot(d)['free_mib']<cfg['min_start_gpu_gib']*1024 for d in cfg['devices']):
                if time.monotonic()>deadline:raise RuntimeError('RELEASED_DEVICE_HEADROOM_UNAVAILABLE')
                time.sleep(2)
            while not all((run/'train'/tag/'RESULT.json').exists() for tag in tags):
                assignments=[(d,[t for t in tags[i::8] if not (run/'train'/t/'RESULT.json').exists()]) for i,d in enumerate(cfg['devices'])]
                execute(run,cfg,'train',[(d,t) for d,t in assignments if t])
            results=[read(run/'train'/t/'RESULT.json') for t in tags]
            if any(r['updates']!=1200 or r['base_updates'] for r in results):raise ValueError('TRAINING_INCOMPLETE')
            immutable(run/'train/RESULT.json',dict(models=results,total_updates=10800,base_updates=0,final_step=1200))
            if not (run/'DIAGNOSTICS_COMPLETE.json').exists():execute(run,cfg,'diagnose',[(cfg['devices'][0],[])])
            from evaluate_continuations import registry,admitted
            reg=registry(run)
            while len(admitted(run,reg))<len(reg['conditions']):
                complete=set(admitted(run,reg));assignments=[(d,[]) for i,d in enumerate(cfg['devices']) if set(range(128)[i::8])-complete]
                execute(run,cfg,'evaluate_continuations',assignments)
    except BaseException as exc:error=dict(reason=repr(exc),traceback=traceback.format_exc())
    finally:
        if acquired:
            write(run/f'LEASE_RELEASED_{time.time_ns()}.json',resource.lease('release',pid=os.getpid()),True)
            deadline=time.monotonic()+120;restored=False
            while time.monotonic()<deadline:
                state=resource.lease('status')
                if state['state']=='occupying' and len(state['worker_pids'])==6:
                    restored=all(monitor.gpu_snapshot(d)['used_mib']>30*1024 for d in cfg['devices'][2:])
                    if restored:break
                if state['external_pids'] or state['manual_paused']:break
                time.sleep(2)
            write(run/f'PLACEHOLDER_RESTORATION_{time.time_ns()}.json',dict(restored=restored,state=state),True)
    if error:
        write(run/'STATUS.json',dict(status='STOPPED',**error));raise RuntimeError(error['reason'])
    try:
        if not (run/'RESULT.json').exists():execute(run,cfg,'review',[(cfg['devices'][0],[])],gpu=False)
    except BaseException as exc:
        write(run/'STATUS.json',dict(status='STOPPED',stage='review',reason=repr(exc),traceback=traceback.format_exc()));raise
    write(run/'STATUS.json',dict(status='COMPLETE',models=9,rollouts=1152,result=str(run/'RESULT.json')))


if __name__=='__main__':main()
