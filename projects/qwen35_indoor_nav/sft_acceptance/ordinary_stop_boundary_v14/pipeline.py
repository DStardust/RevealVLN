"""Independent bounded training -> three-GPU paired evaluation -> review."""
import argparse,fcntl,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
review=u.load('stop14_review',u.HERE/'review.py')
resources=u.load('stop14_resource',u.V16/'pipeline.py')

def lease(command,**fields):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
        s.settimeout(30);s.connect('/run/gpu-placeholder/control.sock');s.sendall((json.dumps(dict(command=command,**fields))+'\n').encode());stream=s.makefile('r');value=json.loads(stream.readline())
    if not value.get('ok'):raise RuntimeError(value)
    return value['result']
def environment(cfg,device=None):
    e=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX'):e.pop(k,None)
    e.update(PYTHONUNBUFFERED='1',V16_ASSET_LINE_ROOT=str(u.ASSET),CUDA_VISIBLE_DEVICES=device['gpu_uuid'] if device else '',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return e
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert a.run_id.replace('_','').isalnum()
    run=u.HERE/'runs'/a.run_id
    if run.exists() and not a.resume:raise FileExistsError('USE_RESUME')
    run.mkdir(parents=True,exist_ok=True);guard=(run/'RUN.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    u.verify();cfg=u.read(u.HERE/'PROTOCOL.json')
    for name in ('PROTOCOL.json','SOURCE_LOCK.json'):
        if (run/name).exists():assert u.read(run/name)==u.read(u.HERE/name),'RESUME_BINDING_CHANGED'
        else:u.write(run/name,u.read(u.HERE/name),True)
    workers=[];leased=False;start=time.monotonic();gpu_hours=0.;error=None
    def interrupt(signum,frame):raise InterruptedError('SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def spawn(script,target,device=None):
        log=(target/(script+'.log')).open('x');cmd=[cfg['torch_python'],'-I','-B',str(u.HERE/(script+'.py')),str(target)]
        proc=subprocess.Popen(cmd,cwd=u.ROOT,env=environment(cfg,device),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        w=dict(proc=proc,owner=resources.process_identity(proc.pid),log=log,folder=target,device=device,start=time.monotonic(),done=False)
        workers.append(w);u.write(target/(script+'_PROCESS.json'),dict(identity=w['owner'],command=cmd,device=device),True);return w
    try:
        u.write(run/'STATUS.json',dict(status='TRAINING_STOP_ROW',unix=time.time(),gpu_hours=0))
        if not (run/'TRAIN_RESULT.json').exists():
            w=spawn('train',run)
            while w['proc'].poll() is None:
                if time.monotonic()-w['start']>1800:raise TimeoutError('CPU_TRAIN_TIMEOUT')
                time.sleep(3)
            w['log'].close();w['done']=True
            if w['proc'].returncode:raise RuntimeError('TRAIN_FAILED_SEE_LOG')
        state=lease('status')
        if state['manual_paused']:raise RuntimeError('PLACEHOLDERS_MANUALLY_PAUSED')
        for logical,pid in state['worker_pids'].items():
            who=resources.process_identity(pid);argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');cg=Path(f'/proc/{pid}/cgroup').read_text()
            assert who['uid']==os.getuid() and b'worker' in argv and argv[-3:-1]==[b'--device',logical.encode()] and '/gpu-placeholder.service' in cg,'NONOWNED_PLACEHOLDER'
        u.write(run/f'LEASE_{time.time_ns()}.json',dict(before=state,after=lease('acquire',pid=os.getpid()),exclusive=False),True);leased=True
        devices=cfg['devices'];remaining=review.summarize(run)['missing_ranks']
        if any(run.glob('sessions/*/FAILURE.json')):raise RuntimeError('PRIOR_FAILURE_REQUIRES_DIAGNOSIS_NOT_BLIND_REROLL')
        for i,device in enumerate(devices):
            ranks=remaining[i::len(devices)]
            if not ranks:continue
            snapshot=resources.gpu_snapshot(device)
            if snapshot['free_mib']<cfg['start_free_mib']:raise RuntimeError('INSUFFICIENT_SHARED_HEADROOM')
            target=run/'sessions'/f'gpu{device["gpu"]}_{time.time_ns()}';target.mkdir(parents=True);(target/'pairs').mkdir();(target/'frames').mkdir()
            config=dict(cfg,**device,run=str(run),scheduled_ranks=ranks,candidate_sha256=u.sha(run/'CANDIDATE.pt'))
            u.write(target/'CONFIG.json',config,True);u.write(target/'RESOURCE_INITIAL.json',snapshot,True);spawn('evaluate',target,device)
        while any(not w['done'] for w in workers):
            active=[]
            for w in workers:
                if w['done']:continue
                code=w['proc'].poll();elapsed=time.monotonic()-w['start']
                if code is not None:
                    w['done']=True;w['log'].close();gpu_hours+=elapsed/3600
                    u.write(w['folder']/'EXIT.json',dict(returncode=code,wall_seconds=elapsed,gpu_hours=elapsed/3600),True)
                    if code:raise RuntimeError('EVALUATION_FAILED:'+str(w['folder']))
                    continue
                g=resources.gpu_snapshot(w['device']);r=resources.resources(w['proc'],w['owner'])
                active.append(dict(gpu=w['device']['gpu'],seconds=elapsed,gpu_snapshot=g,resource=r))
                if g['free_mib']<cfg['minimum_free_mib']:raise RuntimeError('RESOURCE_HEADROOM_EXHAUSTED')
                if r['rss_bytes']>cfg['cpu_memory_gib']*2**30:raise RuntimeError('OWN_RSS_LIMIT')
                progress=w['folder']/'PROGRESS.json'
                if progress.exists() and time.time()-progress.stat().st_mtime>cfg['stall_seconds']:raise TimeoutError('WORKER_NO_PROGRESS')
            total_hours=gpu_hours+sum(x['seconds']/3600 for x in active)
            result=review.summarize(run);u.write(run/'LIVE_RESULT.json',result)
            status=dict(status='EVALUATING' if active else 'REVIEWING',unix=time.time(),complete_pairs=result['complete_pairs'],planned_pairs=100,gpu_hours=total_hours,workers=active)
            u.write(run/'STATUS.json',status);u.append(run/'RESOURCES.jsonl',status)
            if total_hours>cfg['gpu_hours_limit'] or time.monotonic()-start>cfg['wall_seconds']:raise TimeoutError('BOUNDED_RESOURCE_LIMIT')
            time.sleep(10)
        result=review.main(run)
        if result['complete_pairs']!=100:raise RuntimeError('INCOMPLETE_DENOMINATOR')
        u.write(run/'STATUS.json',dict(status='COMPLETE',unix=time.time(),complete_pairs=100,gpu_hours=gpu_hours,delta_sr=result['delta_sr'],adoption=result['adoption']))
    except BaseException as exc:
        error=repr(exc);u.write(run/f'FAILURE_{time.time_ns()}.json',dict(error=error,traceback=traceback.format_exc()),True)
        u.write(run/'STATUS.json',dict(status='STOPPED_WITH_EVIDENCE',reason=error,unix=time.time(),complete_pairs=review.summarize(run)['complete_pairs'],gpu_hours=gpu_hours))
        raise
    finally:
        for w in workers:
            if w['proc'].poll() is None:resources.cleanup(w['proc'],w['owner'])
            if not w['log'].closed:w['log'].close()
            if w['device'] and not (w['folder']/'EXIT.json').exists():
                elapsed=time.monotonic()-w['start'];gpu_hours+=elapsed/3600
                u.write(w['folder']/'EXIT.json',dict(returncode=w['proc'].returncode,wall_seconds=elapsed,gpu_hours=elapsed/3600,cleanup=True),True)
        status=u.read(run/'STATUS.json');status['gpu_hours']=gpu_hours;u.write(run/'STATUS.json',status)
        if leased:u.write(run/f'LEASE_RELEASE_{time.time_ns()}.json',dict(after=lease('release',pid=os.getpid())),True)
        if error:review.main(run)

if __name__=='__main__':main()
