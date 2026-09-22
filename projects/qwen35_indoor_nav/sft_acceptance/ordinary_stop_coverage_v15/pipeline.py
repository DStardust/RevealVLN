"""Independent real collection -> training -> DEV -> complete val_unseen, with resume."""
import argparse,fcntl,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
review=u.load('coverage15_review',u.HERE/'review.py')
resources=u.load('coverage15_resources',u.V16/'pipeline.py')
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
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args();assert args.run_id.replace('_','').isalnum()
    run=u.HERE/'runs'/args.run_id
    if run.exists() and not args.resume:raise FileExistsError('USE_RESUME')
    run.mkdir(parents=True,exist_ok=True);guard=(run/'RUN.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);u.verify()
    for name in ('PROTOCOL.json','SOURCE_LOCK.json'):
        if (run/name).exists():assert u.read(run/name)==u.read(u.HERE/name),'RESUME_SOURCE_CHANGED'
        else:u.write(run/name,u.read(u.HERE/name),True)
    if (run/'RESULT.json').exists():
        assert u.read(run/'RESULT.json')['status']=='COMPLETE'
        for tag in u.read(run/'PROTOCOL.json')['eval_phases']:review.main(run/tag)
        print('ALREADY_COMPLETE_VERIFIED');return
    cfg=u.read(run/'PROTOCOL.json');split=u.read(Path(cfg['manifests'])/'SPLIT.json');cfg['training_houses_list']=split['all_training_houses'];workers=[];leased=False;phase='PREFLIGHT';start=time.monotonic();last_disk=0.;disk_bytes=0
    def stop(signum,frame):raise InterruptedError('SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def hours():
        prior=sum(u.read(p).get('gpu_hours',0) for p in run.glob('**/EXIT.json'))
        return prior+sum((time.monotonic()-w['start'])/3600 for w in workers if w['device'] and not w['done'])
    def spawn(script,folder,device=None,python=None):
        folder.mkdir(parents=True,exist_ok=True);attempt=folder/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
        command=[python or cfg['torch_python'],'-I','-B',str(u.HERE/(script+'.py')),str(folder)];log=(attempt/'stdout.log').open('x');proc=subprocess.Popen(command,cwd=u.ROOT,env=env(device),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        w=dict(proc=proc,owner=resources.process_identity(proc.pid),log=log,attempt=attempt,device=device,start=time.monotonic(),done=False,folder=folder,script=script);workers.append(w)
        u.write(attempt/'PROCESS.json',dict(owner=w['owner'],command=command,device=device,started_unix=time.time()),True);return w
    def poll(active):
        nonlocal last_disk,disk_bytes
        while any(not w['done'] for w in active):
            current=[]
            for w in active:
                if w['done']:continue
                code=w['proc'].poll();elapsed=time.monotonic()-w['start']
                if code is not None:
                    w['done']=True;w['log'].close();u.write(w['attempt']/'EXIT.json',dict(returncode=code,wall_seconds=elapsed,gpu_hours=elapsed/3600 if w['device'] else 0),True)
                    if code:raise RuntimeError('WORKER_FAILED:'+str(w['attempt']))
                    continue
                usage=resources.resources(w['proc'],w['owner']);snapshot=resources.gpu_snapshot(w['device']) if w['device'] else None
                if usage['rss_bytes']>cfg['cpu_memory_gib']*2**30:raise RuntimeError('OWN_RSS_LIMIT')
                if snapshot and snapshot['free_mib']<cfg['minimum_free_mib']:raise RuntimeError('GPU_HEADROOM_LIMIT')
                progress=w['folder']/'PROGRESS.json'
                if w['device'] and progress.exists() and time.time()-progress.stat().st_mtime>cfg['stall_seconds']:raise TimeoutError('NO_PROGRESS')
                current.append(dict(script=w['script'],device=w['device'],elapsed=elapsed,gpu=snapshot,rss_bytes=usage['rss_bytes']))
            total=hours()
            if time.monotonic()-last_disk>300:
                result=subprocess.run(['du','-sb',str(run)],capture_output=True,text=True,timeout=120,check=True);disk_bytes=int(result.stdout.split()[0]);last_disk=time.monotonic()
            if disk_bytes>cfg['artifact_gib']*2**30 or total>cfg['gpu_hours_limit'] or time.monotonic()-start>cfg['wall_seconds']:raise TimeoutError('REGISTERED_RESOURCE_LIMIT')
            status=dict(status='RUNNING',phase=phase,unix=time.time(),gpu_hours=total,artifact_bytes=disk_bytes,workers=current)
            if phase=='COLLECT':status.update(complete=len(review.completed_collect(run/'collect')),planned=640)
            elif phase in ('DEV','UNSEEN'):
                value=review.summarize(run/phase);u.write(run/phase/'LIVE_RESULT.json',value);status.update(complete=value['complete_pairs'],planned=value['planned_pairs'])
            u.write(run/'STATUS.json',status);u.append(run/'RESOURCES.jsonl',status);time.sleep(10)
    def batch(tag,script):
        stage=run/('collect' if tag=='COLLECT' else tag);stage.mkdir(exist_ok=True)
        label='FIT' if tag=='COLLECT' else tag;count=640 if tag=='COLLECT' else cfg['eval_counts'][tag]
        spec=dict(phase=tag,episodes_path=str(Path(cfg['manifests'])/(label+'_EPISODES.json')),order_path=str(Path(cfg['manifests'])/(label+'_ORDER.json')),geometry_path=str(run/(label+'_GEOMETRY.json')),gt_path=cfg['unseen_gt'] if tag=='UNSEEN' else cfg['train_gt'],planned_pairs=count)
        if (stage/'STAGE.json').exists():assert u.read(stage/'STAGE.json')==spec,'STAGE_CONFIG_CHANGED'
        else:u.write(stage/'STAGE.json',spec,True)
        if tag=='COLLECT':remaining=sorted(set(range(count))-set(review.completed_collect(stage)))
        else:remaining=review.summarize(stage)['missing_ranks']
        # Only explicit infrastructure failures may resume; numerical/data errors remain blocked.
        for failure in stage.glob('sessions/*/FAILURE.json'):
            error=u.read(failure)['error']
            if not any(x in error for x in ('EOF','ConnectionReset','Timeout','out of memory')):raise RuntimeError('UNRESOLVED_CORRECTNESS_FAILURE:'+str(failure))
        active=[]
        for i,device in enumerate(cfg['devices']):
            ranks=remaining[i::len(cfg['devices'])]
            if not ranks:continue
            if resources.gpu_snapshot(device)['free_mib']<cfg['start_free_mib']:raise RuntimeError('INSUFFICIENT_SHARED_HEADROOM')
            session=stage/'sessions'/f'gpu{device["gpu"]}_{time.time_ns()}';session.mkdir(parents=True);(session/'frames').mkdir();(session/'pairs').mkdir()
            p=dict(cfg,**device,**spec,run=str(run),scheduled_ranks=ranks,v16_protocol_sha256=u.sha(run/'PROTOCOL.json'))
            if tag!='COLLECT':p['candidate_sha256']=u.sha(run/'CANDIDATE.pt')
            u.write(session/'CONFIG.json',p,True);active.append(spawn(script,session,device))
        poll(active)
        if tag=='COLLECT':assert len(review.completed_collect(stage))==640,'INCOMPLETE_COLLECTION'
        else:
            result=review.main(stage);assert result['complete_pairs']==count,'INCOMPLETE_EVALUATION'
    try:
        u.write(run/'STATUS.json',dict(status='RUNNING',phase=phase,unix=time.time(),gpu_hours=hours()))
        if not (run/'GEOMETRY_RESULT.json').exists():poll([spawn('geometry',run,python=cfg['habitat_python'])])
        state=lease('status');assert not state['manual_paused'],'HOLDERS_MANUALLY_PAUSED'
        for logical,pid in state['worker_pids'].items():
            who=resources.process_identity(pid);argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');assert who['uid']==os.getuid() and b'worker' in argv and argv[-3:-1]==[b'--device',logical.encode()] and '/gpu-placeholder.service' in Path(f'/proc/{pid}/cgroup').read_text(),'NONOWNED_HOLDER'
        u.write(run/f'LEASE_{time.time_ns()}.json',dict(before=state,after=lease('acquire',pid=os.getpid()),exclusive=False),True);leased=True
        phase='COLLECT';batch(phase,'collect')
        phase='TRAIN'
        if not (run/'TRAIN_RESULT.json').exists():poll([spawn('train',run)])
        assert u.sha(run/'CANDIDATE.pt')==u.read(run/'TRAIN_RESULT.json')['checkpoint_sha256'],'TRAIN_CHECKPOINT_BINDING'
        for phase in cfg['eval_phases']:batch(phase,'evaluate')
        results={tag:u.read(run/tag/'REVIEW.json') for tag in cfg['eval_phases']}
        u.write(run/'RESULT.json',dict(status='COMPLETE',gpu_hours=hours(),evaluations=results,automatic_adoption=False),True)
        report='# 普通导航真实覆盖扩充与完整unseen验证\n\n'
        for tag,r in results.items():report+=f"{tag}: {r['complete_pairs']}/{r['planned_pairs']} 对，V13 SR={r['arms']['A']['sr']}，新候选 SR={r['arms']['B']['sr']}，ΔSR={r['delta_sr']}。\n\n"
        report+='训练仅使用40个训练房屋的新真实轨迹，底模与运动行冻结。完整val_unseen是已有公开暴露验证集，不是盲测。没有自动部署。\n';(run/'REPORT_ZH.md').write_text(report)
        u.write(run/'STATUS.json',dict(status='COMPLETE',phase='DONE',unix=time.time(),gpu_hours=hours()))
    except BaseException as error:
        u.write(run/f'FAILURE_{time.time_ns()}.json',dict(error=repr(error),phase=phase,traceback=traceback.format_exc()),True);u.write(run/'STATUS.json',dict(status='STOPPED_WITH_EVIDENCE',phase=phase,reason=repr(error),unix=time.time(),gpu_hours=hours()));raise
    finally:
        for w in workers:
            if w['proc'].poll() is None:resources.cleanup(w['proc'],w['owner'])
            if not w['log'].closed:w['log'].close()
            if not (w['attempt']/'EXIT.json').exists():
                elapsed=time.monotonic()-w['start'];u.write(w['attempt']/'EXIT.json',dict(returncode=w['proc'].returncode,wall_seconds=elapsed,gpu_hours=elapsed/3600 if w['device'] else 0,cleanup=True),True)
            w['done']=True
        status=u.read(run/'STATUS.json');status['gpu_hours']=hours();u.write(run/'STATUS.json',status)
        if leased:u.write(run/f'LEASE_RELEASE_{time.time_ns()}.json',dict(after=lease('release',pid=os.getpid())),True)
if __name__=='__main__':main()
