"""Whole-condition parallel evaluation; original placeholder PID lease is always released."""
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

HERE=Path(__file__).resolve().parent
PILOT=HERE.parent
sys.path.insert(0,str(PILOT))
from shared import read,write,append,sha,immutable,load,c,V16,ROOT
monitor=load('parallel_owned_process_helpers',V16/'pipeline.py')
sys.path.insert(0,str(PILOT))
from evaluate_continuations import admitted

RUN=PILOT/'runs/pilot_001'
OLD_JOB=PILOT/'standalone_jobs/ident-pilot-20260921-01'
OLD_UNIT='q35n-ident-pilot-20260921-01.service'
SOCKET='/run/gpu-placeholder/control.sock'
HOLDER=Path('/mnt/data_nas/deeprobotics/daiyang/gpu_placeholder/gpu_placeholder.py')


def lease(command, **fields):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
        stream.settimeout(30);stream.connect(SOCKET)
        stream.sendall((json.dumps(dict(command=command,**fields))+'\n').encode())
        data=b''
        while not data.endswith(b'\n'):
            part=stream.recv(65536)
            if not part:break
            data+=part
    result=json.loads(data)
    if not result.get('ok'):raise RuntimeError(result)
    return result['result']


def process(pid):
    p=Path('/proc')/str(pid)
    return dict(monitor.process_identity(pid),argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
        cgroup=(p/'cgroup').read_text(),cwd=str((p/'cwd').resolve()))


def verify_placeholders():
    state=lease('status')
    if state['manual_paused'] or state['leases'] or state['external_pids'] or state['device_count']!=6:
        raise RuntimeError('PLACEHOLDER_STATE_CHANGED')
    workers=[]
    for logical,pid in sorted(state['worker_pids'].items()):
        who=process(pid);argv=who['argv']
        if who['uid']!=os.getuid() or str(HOLDER) not in argv or 'worker' not in argv or argv[-2:]!=['--device',logical] or '/system.slice/gpu-placeholder.service' not in who['cgroup']:
            raise RuntimeError('NOT_VERIFIED_PLACEHOLDER')
        workers.append(who)
    if len(workers)!=6:raise RuntimeError('PLACEHOLDER_WORKER_COUNT')
    return dict(state=state,workers=workers,script_sha256=sha(HOLDER))


def split_conditions(conditions, devices):
    ordered=sorted(conditions)
    return {d['gpu']:ordered[i::len(devices)] for i,d in enumerate(devices)}


def handoff(reg):
    job=read(OLD_JOB/'JOB.json');worker=read(OLD_JOB/'STATUS.json')
    if job['uid']!=os.getuid() or job['unit']!=OLD_UNIT or str(PILOT/'pipeline.py') not in job['command']:
        raise RuntimeError('ORIGINAL_JOB_IDENTITY')
    if worker['status']!='RUNNING':
        return dict(already_stopped=True,groups=sorted(admitted(RUN,reg)))
    identity=process(worker['pid'])
    if identity['uid']!=os.getuid() or '/system.slice/'+OLD_UNIT not in identity['cgroup']:
        raise RuntimeError('ORIGINAL_SERVICE_IDENTITY')
    initial=len(admitted(RUN,reg));began=time.monotonic()
    while time.monotonic()-began<600:
        write(HERE/'STATUS.json',dict(status='HANDOFF_WAIT',groups=initial,reason='Waiting for next complete nine-model group',unix=time.time()))
        current=len(list((RUN/'evaluate').glob('session_*/STATE_SEAL_???.json')))
        if current>initial or read(OLD_JOB/'STATUS.json')['status']!='RUNNING':break
        time.sleep(2)
    status=read(RUN/'STATUS.json')
    if process(worker['pid'])!=identity:raise RuntimeError('ORIGINAL_SERVICE_IDENTITY_CHANGED')
    before=sorted(admitted(RUN,reg))
    # Stop exactly our registered service, including its owned Habitat process.
    subprocess.run(['systemctl','stop',OLD_UNIT],check=True,timeout=100)
    after=sorted(admitted(RUN,reg))
    if not set(before)<=set(after):raise RuntimeError('COMPLETED_GROUP_LOST')
    return dict(identity=identity,groups=after,status_before=status,stopped_unix=time.time(),
        boundary_wait_seconds=time.monotonic()-began,reason='User-authorized parallel resource handoff; partial attempts retained')


def env_for(cfg):
    env=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX'):env.pop(name,None)
    env.update(V16_ASSET_LINE_ROOT=cfg['asset_line_root'],CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    return env


def stop_owned(worker):
    proc=worker['proc'];owner=worker['owner']
    try:
        if proc.poll() is None:monitor.cleanup(proc,owner)
    except (ProcessLookupError,FileNotFoundError):pass
    # The leader may have failed before its Habitat child; only retained identities qualify.
    for identity in worker.get('members',[]):
        try:
            if monitor.process_identity(identity['pid'])==identity and identity['pgid']==owner['pgid'] and identity['uid']==os.getuid():
                os.kill(identity['pid'],signal.SIGTERM)
        except ProcessLookupError:pass
        except FileNotFoundError:pass


def main():
    amendment=read(HERE/'AMENDMENT.json');cfg=read(RUN/'PROTOCOL.json');reg=read(RUN/'EVALUATION_REGISTRY.json')
    if sha(RUN/'PROTOCOL.json')!=amendment['original_protocol_sha256'] or sha(HERE/'evaluate.py')!=amendment['runtime_sha256']:
        raise RuntimeError('AMENDMENT_BINDING')
    with (HERE/'COORDINATOR.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'PLAN.json').exists():raise RuntimeError('ALREADY_DISPATCHED_NO_IMPLICIT_RETRY')
        immutable(HERE/'PLACEHOLDER_BEFORE.json',verify_placeholders())
        change=handoff(reg);immutable(HERE/'HANDOFF.json',change)
        remaining=sorted(set(range(len(reg['conditions'])))-set(change['groups']))
        split=split_conditions(remaining,amendment['devices'])
        immutable(HERE/'PLAN.json',dict(assignments=split,preserved_groups=change['groups'],planned_groups=64,planned_rollouts=576,registry_sha256=sha(RUN/'EVALUATION_REGISTRY.json')))
        completed_resources=c.records(RUN/'RESOURCES.jsonl')
        old_hours=sum(r['seconds']/3600 for r in completed_resources if r['gpu'])
        # SIGTERM may precede the original launcher's final resource write. Preserve a conservative estimate.
        interrupted=not any(r['stage']=='evaluate_continuations' for r in completed_resources)
        if interrupted and 'status_before' in change:old_hours+=(change['status_before'].get('elapsed',0)+30)/3600
        immutable(HERE/'RESOURCE_BASELINE.json',dict(gpu_hours=old_hours,interrupted_session_estimated=interrupted,estimate_margin_seconds=30))
        workers=[];acquired=False;error=None;all_done=False;lastdisk=0
        def interrupted_signal(signum,frame):raise InterruptedError(f'SIGNAL_{signum}')
        signal.signal(signal.SIGTERM,interrupted_signal);signal.signal(signal.SIGINT,interrupted_signal)
        try:
            write(HERE/'LEASE_ACQUIRED.json',lease('acquire',pid=os.getpid()),True);acquired=True
            for device in amendment['devices']:
                conditions=split[device['gpu']]
                if not conditions:continue
                folder=HERE/f"gpu_{device['gpu']}";folder.mkdir()
                assignment=dict(device,conditions=conditions,protocol_sha256=sha(RUN/'PROTOCOL.json'),
                    registry_sha256=sha(RUN/'EVALUATION_REGISTRY.json'),runtime_sha256=sha(HERE/'evaluate.py'),amendment_sha256=sha(HERE/'AMENDMENT.json'))
                write(folder/'ASSIGNMENT.json',assignment,True)
                devicecfg=dict(cfg,**device);snapshot=monitor.gpu_snapshot(devicecfg)
                if snapshot['free_mib']<12*1024:raise RuntimeError('INSUFFICIENT_RELEASED_HEADROOM')
                command=[cfg['torch_python'],'-I','-B',str(HERE/'evaluate.py'),str(RUN),str(folder/'ASSIGNMENT.json')]
                log=(folder/'stdout.log').open('x');began=time.monotonic()
                proc=subprocess.Popen(command,cwd=ROOT,env=env_for(devicecfg),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                worker=dict(proc=proc,owner=monitor.process_identity(proc.pid),folder=folder,cfg=devicecfg,began=began,log=log,ended=None,members=[])
                workers.append(worker);write(folder/'OWNER.json',worker['owner'],True)
                write(folder/'COMMAND.json',dict(argv=command,gpu=snapshot),True)
            while True:
                now=time.monotonic();views=[];hours=old_hours
                for w in workers:
                    code=w['proc'].poll()
                    if code is not None and w['ended'] is None:
                        w['ended']=now
                        write(w['folder']/'EXIT.json',dict(returncode=code,wall_seconds=now-w['began'],gpu=w['cfg']['gpu']),True)
                    elapsed=(w['ended'] or now)-w['began'];hours+=elapsed/3600
                    resource=monitor.resources(w['proc'],w['owner'])
                    for pid in resource['owned_pids']:
                        try:
                            identity=monitor.process_identity(pid)
                            if identity not in w['members']:w['members'].append(identity)
                        except (FileNotFoundError,ProcessLookupError):pass
                    gpu=monitor.gpu_snapshot(w['cfg']);progress=read(w['folder']/'PROGRESS.json') if (w['folder']/'PROGRESS.json').exists() else None
                    views.append(dict(gpu_index=w['cfg']['gpu'],state='RUNNING' if code is None else 'COMPLETE' if code==0 else 'FAILED',returncode=code,elapsed_seconds=elapsed,gpu=gpu,progress=progress,**resource))
                    if code not in (None,0):raise RuntimeError(f"GPU_{w['cfg']['gpu']}_WORKER_FAILED")
                    if code is None and (elapsed>cfg['max_session_hours']*3600 or resource['rss_bytes']>cfg['process_rss_gib']*2**30 or gpu['free_mib']<cfg['min_free_gpu_gib']*1024):
                        raise RuntimeError('WORKER_RESOURCE_LIMIT')
                write(HERE/'STATUS.json',dict(status='RUNNING',stage='parallel_evaluate',workers=views,gpu_hours=hours,gpu_budget_hours=cfg['gpu_session_hours'],unix=time.time()))
                append(HERE/'MONITOR.jsonl',dict(unix=time.time(),gpu_hours=hours,workers=views))
                if hours>cfg['gpu_session_hours']:raise RuntimeError('CUMULATIVE_GPU_BUDGET')
                if now-lastdisk>60:
                    size=int(subprocess.check_output(['du','-sb',str(RUN)],text=True).split()[0]);lastdisk=now
                    if size>cfg['artifact_gib']*2**30:raise RuntimeError('ARTIFACT_LIMIT')
                if all(w['ended'] is not None for w in workers):break
                time.sleep(5)
            all_done=len(admitted(RUN,reg))==64
            if not all_done:raise RuntimeError('UNFINISHED_REGISTERED_GROUPS')
        except BaseException as exc:
            error=dict(reason=repr(exc),traceback=traceback.format_exc())
        finally:
            for w in workers:
                try:stop_owned(w)
                except Exception as exc:
                    error=error or dict(reason='OWNED_CLEANUP_FAILED:'+repr(exc),traceback=traceback.format_exc())
                finally:w['log'].close()
            append(HERE/'RESOURCES.jsonl',dict(workers=[dict(gpu=w['cfg']['gpu'],wall_seconds=(w['ended'] or time.monotonic())-w['began'],returncode=w['proc'].poll()) for w in workers],baseline_gpu_hours=old_hours))
            if acquired:
                write(HERE/'LEASE_RELEASED.json',lease('release',pid=os.getpid()),True)
                deadline=time.monotonic()+120;restored=False
                while time.monotonic()<deadline:
                    state=lease('status')
                    write(HERE/'STATUS.json',dict(status='RESTORING_PLACEHOLDERS',evaluation_complete=all_done,error=error,placeholder=state,unix=time.time()))
                    if state['state']=='occupying' and len(state['worker_pids'])==6:
                        snapshots=[monitor.gpu_snapshot(d) for d in amendment['devices']]
                        if all(x['used_mib']>30*1024 for x in snapshots):restored=True;break
                    if state['external_pids'] or state['manual_paused']:break
                    time.sleep(2)
                write(HERE/'PLACEHOLDER_RESTORATION.json',dict(restored=restored,state=state,unix=time.time()),True)
        if error:
            write(HERE/'STATUS.json',dict(status='STOPPED',**error,unix=time.time()))
            raise RuntimeError(error['reason'])
        write(HERE/'STATUS.json',dict(status='REVIEWING',unix=time.time()))
        env=env_for(cfg);env['CUDA_VISIBLE_DEVICES']=''
        with (HERE/'review.log').open('x') as log:
            result=subprocess.run([cfg['torch_python'],'-I','-B',str(PILOT/'review.py'),str(RUN)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if result.returncode:raise RuntimeError('CPU_REVIEW_FAILED')
        groups=admitted(RUN,reg);blocks=[]
        for condition,session in sorted(groups.items()):
            identity=read(session/'RUNTIME_IDENTITY.json')
            blocks.append(dict(condition=condition,session=session.name,gpu_uuid=identity['gpu_uuid'],physical_gpu=identity['physical_gpu'],source='parallel' if 'parallel' in session.name else 'original'))
        write(HERE/'DEVICE_BLOCKS.json',dict(rows=blocks),True)
        write(HERE/'STATUS.json',dict(status='COMPLETE',groups=len(groups),rollouts=len(groups)*9,placeholder_restored=read(HERE/'PLACEHOLDER_RESTORATION.json')['restored'],unix=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(HERE/'STATUS.json',dict(status='STOPPED',reason=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
        raise
