"""Independent shared-data motion comparison, then separate FIT recovery collection."""
import argparse,fcntl,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import common as u
review=u.load('long100k_review',u.HERE/'review.py')
resources=u.load('long100k_resources',u.V16/'pipeline.py')
OPS=Path(__file__).resolve().parent
disk=u.load('long100k_async_disk',OPS/'disk_monitor.py')
def lease(command,**fields):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
        s.settimeout(30);s.connect('/run/gpu-placeholder/control.sock');s.sendall((json.dumps(dict(command=command,**fields))+'\n').encode());r=json.loads(s.makefile('r').readline())
    assert r.get('ok'),r
    return r['result']
def env(device=None):
    e=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX'):e.pop(k,None)
    e.update(PYTHONUNBUFFERED='1',V16_ASSET_LINE_ROOT=str(u.ASSET),CUDA_VISIBLE_DEVICES=device['gpu_uuid'] if device else '',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    return e
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args();assert args.run_id.replace('_','').isalnum()
    for name,expected in u.read(OPS/'SOURCE_LOCK.json')['files'].items():assert u.sha(OPS/name)==expected,'RECOVERY_SOURCE_CHANGED'
    run=u.HERE/'runs'/args.run_id
    if run.exists() and not args.resume:raise FileExistsError('USE_RESUME')
    run.mkdir(parents=True,exist_ok=True);lock=(run/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);u.verify()
    for name in ('PROTOCOL.json','SOURCE_LOCK.json'):
        if (run/name).exists():assert u.read(run/name)==u.read(u.HERE/name),'RESUME_BINDING_CHANGED'
        else:u.write(run/name,u.read(u.HERE/name),True)
    cfg=u.read(run/'PROTOCOL.json');up=Path(cfg['upstream_run']);workers=[];phase='PREFLIGHT';leased=False;began=time.monotonic();gpu_began=None;last_disk=0;artifact=0
    if (run/'RESULT.json').exists():print('ALREADY_COMPLETE');return
    def interrupt(s,f):raise InterruptedError('SIGNAL_'+str(s))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    prior_gpu_hours=sum(u.read(p).get('gpu_hours',0) for p in run.glob('UNSEEN/sessions/*/attempts/*/EXIT.json'))
    accounting=disk.Monitor(run,resources,sys.executable)
    def hours():return disk.charged_gpu_hours(prior_gpu_hours,workers,time.monotonic())
    def status(**extra):
        u.write(run/'STATUS.json',dict(status='RUNNING',phase=phase,unix=time.time(),gpu_hours=hours(),**extra))
    def wait_for(filename):
        while not (up/filename).exists():
            current=u.read(up/'STATUS.json');status(upstream=current)
            if current['status']=='STOPPED_WITH_EVIDENCE':raise RuntimeError('UPSTREAM_STOPPED:'+current.get('reason',''))
            if time.monotonic()-began>cfg['wait_seconds']:raise TimeoutError('DEPENDENCY_WAIT_LIMIT')
            time.sleep(20)
    def spawn(script,folder,gpu=None):
        folder.mkdir(parents=True,exist_ok=True);attempt=folder/'attempts'/str(time.time_ns());attempt.mkdir(parents=True);log=(attempt/'stdout.log').open('x');cmd=[cfg['torch_python'],'-I','-B',str(u.HERE/(script+'.py')),str(folder)]
        proc=subprocess.Popen(cmd,cwd=u.ROOT,env=env(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owner=resources.process_identity(proc.pid);w=dict(proc=proc,owner=owner,log=log,attempt=attempt,start=time.monotonic(),gpu=gpu,done=False,folder=folder,script=script);workers.append(w);u.write(attempt/'PROCESS.json',dict(owner=owner,command=cmd,device=gpu,started_unix=time.time()),True);return w
    def poll(active):
        nonlocal last_disk,artifact
        while any(not w['done'] for w in active):
            detail=[]
            for w in active:
                if w['done']:continue
                code=w['proc'].poll();elapsed=time.monotonic()-w['start']
                if code is not None:
                    w['charged_hours']=elapsed/3600 if w['gpu'] else 0;w['done']=True;w['log'].close();u.write(w['attempt']/'EXIT.json',dict(returncode=code,wall_seconds=elapsed,gpu_hours=elapsed/3600 if w['gpu'] else 0),True)
                    if code:raise RuntimeError('WORKER_FAILED:'+str(w['attempt']))
                    continue
                usage=resources.resources(w['proc'],w['owner']);g=resources.gpu_snapshot(w['gpu']) if w['gpu'] else None
                if usage['rss_bytes']>cfg['cpu_memory_gib']*2**30:raise RuntimeError('OWN_RSS_LIMIT')
                if g and g['free_mib']<cfg['minimum_free_mib']:raise RuntimeError('GPU_HEADROOM_LIMIT')
                progress=w['folder']/'PROGRESS.json'
                if g and progress.exists() and time.time()-progress.stat().st_mtime>cfg['stall_seconds']:raise TimeoutError('NO_PROGRESS')
                detail.append(dict(role=w['script'],gpu=w['gpu'],elapsed=elapsed,rss_bytes=usage['rss_bytes'],device=g))
            inventory=accounting.poll();artifact=inventory['known_apparent_bytes']
            if artifact>cfg['artifact_gib']*2**30 or hours()>cfg['gpu_hours_limit'] or (gpu_began and time.monotonic()-gpu_began>cfg['wall_seconds']):raise TimeoutError('REGISTERED_RESOURCE_LIMIT')
            extra=dict(workers=detail,artifact_bytes=artifact,artifact_scan=inventory)
            if phase in ('DEV','UNSEEN'):
                r=review.summarize(run/phase);u.write(run/phase/'LIVE_RESULT.json',r);extra.update(complete=r['complete_pairs'],planned=r['planned_pairs'])
            if phase=='RECOVERY':extra.update(complete=len(review.completed_collect(run/'recovery')),planned=u.read(run/'recovery/MANIFEST.json')['selected'])
            status(**extra);u.append(run/'RESOURCES.jsonl',u.read(run/'STATUS.json'));time.sleep(10)
    def evaluate(tag):
        stage=run/tag;stage.mkdir(exist_ok=True);spec=dict(phase=tag,episodes_path=str(Path(cfg['manifests'])/(tag+'_EPISODES.json')),order_path=str(Path(cfg['manifests'])/(tag+'_ORDER.json')),geometry_path=str(up/(tag+'_GEOMETRY.json')),gt_path=cfg['unseen_gt'] if tag=='UNSEEN' else cfg['train_gt'],planned_pairs=cfg['eval_counts'][tag])
        if (stage/'STAGE.json').exists():assert u.read(stage/'STAGE.json')==spec
        else:u.write(stage/'STAGE.json',spec,True)
        remaining=review.summarize(stage)['missing_ranks'];active=[]
        for i,gpu in enumerate(cfg['devices']):
            rr=remaining[i::len(cfg['devices'])]
            if not rr:continue
            assert resources.gpu_snapshot(gpu)['free_mib']>=cfg['start_free_mib'],'SHARED_GPU_HEADROOM'
            session=stage/'sessions'/f'gpu{gpu["gpu"]}_{time.time_ns()}';session.mkdir(parents=True);(session/'frames').mkdir();(session/'pairs').mkdir()
            p=dict(cfg,**gpu,**spec,run=str(run),scheduled_ranks=rr,candidate_sha256=cfg['memory_checkpoint_sha256'],reference_checkpoint_sha256=u.sha(Path(cfg['reference_checkpoint'])),runtime_executor=str(u.HERE/'executor.py'),v16_protocol_sha256=u.sha(run/'PROTOCOL.json'))
            u.write(session/'CONFIG.json',p,True);active.append(spawn('evaluate',session,gpu))
        poll(active);r=review.main(stage);assert r['complete_pairs']==cfg['eval_counts'][tag]
    try:
        u.write(run/f'RESOURCE_RUNTIME_{time.time_ns()}.json',dict(revision=str(OPS),source_sha256=u.sha(OPS/'SOURCE_LOCK.json'),semantic_changes=False,prior_gpu_hours=prior_gpu_hours),True)
        status();assert u.read(up/'RESULT.json')['status']=='COMPLETE';phase='PREFLIGHT'
        # Existing complete low-scoring results are never retried. Any unresolved worker failure blocks explicit resume.
        for failure in run.glob('UNSEEN/sessions/*/FAILURE.json'):raise RuntimeError('UNRESOLVED_WORKER_FAILURE:'+str(failure))
        assert u.sha(Path(cfg['memory_checkpoint']))==cfg['memory_checkpoint_sha256'],'PRETRAINED_CHECKPOINT_CHANGED'
        u.write(run/'DEPENDENCY.json',dict(memory_checkpoint=cfg['memory_checkpoint'],memory_checkpoint_sha256=cfg['memory_checkpoint_sha256'],memory_state_sha256=cfg['memory_state_sha256'],new_optimizer_updates=0))
        state=lease('status');assert not state['manual_paused']
        for logical,pid in state['worker_pids'].items():
            who=resources.process_identity(pid);argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');assert who['uid']==os.getuid() and b'worker' in argv and argv[-3:-1]==[b'--device',logical.encode()] and '/gpu-placeholder.service' in Path(f'/proc/{pid}/cgroup').read_text(),'NONOWNED_HOLDER'
        u.write(run/f'LEASE_{time.time_ns()}.json',dict(before=state,after=lease('acquire',pid=os.getpid()),exclusive=False),True);leased=True;gpu_began=time.monotonic()
        phase='UNSEEN';evaluate(phase);result=u.read(run/'UNSEEN/REVIEW.json')
        u.write(run/'RESULT.json',dict(status='COMPLETE',evaluations={'UNSEEN':result},gpu_hours=hours(),new_optimizer_updates=0,automatic_adoption=False,exposure=u.read(u.HERE/'EXPOSURE_AUDIT.json')),True)
        report='# 100000步记忆模型的普通导航迁移\n\n'
        report+=f"完整 {result['complete_pairs']}/{result['planned_pairs']} 对：V13 SR={result['arms']['A']['sr']}，100k记忆策略 SR={result['arms']['B']['sr']}，ΔSR={result['delta_sr']}。\n\n"
        report+='固定最终100000步checkpoint，不从20个checkpoint按本次分数选择。底模仍是best4k，新增训练0步；运行真实因果记忆、原始自然指令和四动作argmax。\n\n'
        report+='这是公开val_unseen1839上的跨任务迁移评测。记忆训练与其中6屋重合，另有开发暴露屋；暴露分层见REVIEW.json，不能声称全部场景训练未见或盲测泛化。没有更换目标/STOP/500步定义，也未把100k记忆与V13 STOP头混合。\n'
        (run/'REPORT_ZH.md').write_text(report);u.write(run/'STATUS.json',dict(status='COMPLETE',phase='DONE',unix=time.time(),gpu_hours=hours()))
    except BaseException as e:
        u.write(run/f'FAILURE_{time.time_ns()}.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc()),True);u.write(run/'STATUS.json',dict(status='STOPPED_WITH_EVIDENCE',phase=phase,reason=repr(e),unix=time.time(),gpu_hours=hours()));raise
    finally:
        for w in workers:
            if w['proc'].poll() is None:resources.cleanup(w['proc'],w['owner'])
            if not w['log'].closed:w['log'].close()
            if not (w['attempt']/'EXIT.json').exists():
                elapsed=time.monotonic()-w['start'];u.write(w['attempt']/'EXIT.json',dict(returncode=w['proc'].returncode,wall_seconds=elapsed,gpu_hours=elapsed/3600 if w['gpu'] else 0,cleanup=True),True)
            if not w['done']:w['charged_hours']=(time.monotonic()-w['start'])/3600 if w['gpu'] else 0
            w['done']=True
        accounting.close()
        current=u.read(run/'STATUS.json');current['gpu_hours']=hours();u.write(run/'STATUS.json',current)
        if leased:u.write(run/f'LEASE_RELEASE_{time.time_ns()}.json',dict(after=lease('release',pid=os.getpid())),True)
if __name__=='__main__':main()
