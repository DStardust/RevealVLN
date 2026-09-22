"""Independent EXPANDED-only long training plus asynchronous fixed checkpoint evaluation."""
import argparse,fcntl,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
resource=load('long_owned_resources',PILOT/'parallel_eval_v1/coordinator.py')
monitor=resource.monitor
DiskWatch=load('long_disk_watch',PARENT/'event_data_scale20_v2/disk_watch_r2.py').DiskWatch
sys.modules.pop('evaluate_continuations',None)
sys.path.insert(0,str(HERE))


def completed_checkpoint(run,step):
    p=run/'checkpoints'/f'STEP_{step:06d}'
    if not (p/'COMMIT.json').exists():return False
    value=read(p/'COMMIT.json')
    if value['step']!=step:raise ValueError('CHECKPOINT_STEP_CHANGED')
    for name,h in value['files'].items():
        if sha(p/name)!=h:raise ValueError('COMMITTED_CHECKPOINT_CHANGED')
    return True


def copy_bound(src,dst):
    if dst.exists():
        if sha(src)!=sha(dst):raise ValueError('COPIED_INPUT_CHANGED')
    else:
        dst.parent.mkdir(exist_ok=True,parents=True);tmp=dst.with_suffix(dst.suffix+'.tmp');shutil.copyfile(src,tmp);os.replace(tmp,dst)


def prepare_eval(run,step,cfg):
    if not completed_checkpoint(run,step):raise ValueError('UNCOMMITTED_CHECKPOINT')
    child=run/'evaluations'/f'STEP_{step:06d}';child.mkdir(parents=True,exist_ok=True)
    source=Path(cfg['source_run']);cp=run/'checkpoints'/child.name
    immutable(child/'PROTOCOL.json',dict(cfg,evaluated_step=step,planned_rollouts=256,source_checkpoint_sha256=sha(cp/'HEAD.pt')))
    immutable(child/'SOURCE_LOCK.json',read(run/'SOURCE_LOCK.json'))
    for name in ('DATA.json','WARMUP_DATA.json'):copy_bound(source/name,child/name)
    for src,dst in ((cp/'HEAD.pt',child/'train/EXPANDED_1209/FINAL.pt'),(cp/'HEAD_RESULT.json',child/'train/EXPANDED_1209/RESULT.json')):copy_bound(src,dst)
    ev=local_module('evaluate_continuations');reg=ev.registry_value(read(child/'DATA.json')['raw_families'],cfg)
    if len(reg['slots'])!=256 or reg['models']!=['EXPANDED_1209'] or reg['expected_prefix_comparisons']!=0:raise ValueError('SINGLE_ARM_REGISTRY')
    immutable(child/'EVALUATION_REGISTRY.json',reg)
    immutable(child/'DATA_AUDIT.json',dict(data_sha256=sha(child/'DATA.json'),registry_sha256=sha(child/'EVALUATION_REGISTRY.json'),source_audit_sha256=sha(source/'DATA_AUDIT.json'),scope='Exposed development, one checkpoint, single seed'))
    return child,reg


def acquire(run):
    state=resource.lease('status');verified=[]
    if state['device_count']!=6 or state['manual_paused']:raise RuntimeError('PLACEHOLDER_MANUALLY_PAUSED_OR_CONFIG_CHANGED')
    for logical,pid in state['worker_pids'].items():
        who=resource.process(pid)
        if who['uid']!=os.getuid() or str(resource.HOLDER) not in who['argv'] or 'worker' not in who['argv'] or who['argv'][-2:]!=['--device',logical] or '/system.slice/gpu-placeholder.service' not in who['cgroup']:raise RuntimeError('NOT_OWN_PLACEHOLDER')
        verified.append(who)
    owners=[]
    for pid in state['leases']:
        who=resource.process(int(pid))
        # Existing leases are recorded, never revoked. The controller supports concurrent owners.
        owners.append(who)
    immutable(run/f'RESOURCE_LEASE_{time.time_ns()}.json',dict(before=state,verified_holders=verified,other_owners=owners,
        after=resource.lease('acquire',pid=os.getpid()),exclusive=False,authorization='User authorized shared GPUs; only own holder controller lease is acquired.'))


def spawn(run,cfg,role,target,device=None):
    folder=run/'attempts'/f'{role}_{time.time_ns()}';folder.mkdir(parents=True)
    env=resource.env_for(dict(cfg,**(device or cfg['devices'][0])))
    if device:
        if monitor.gpu_snapshot(device)['free_mib']<cfg['min_start_gpu_gib']*1024:raise RuntimeError('DEVICE_HEADROOM')
        assignment=dict(device)
        if role=='evaluate':assignment['conditions']=target[1]
        immutable(folder/'ASSIGNMENT.json',assignment)
        env.update(B2_DEVICE=str(folder/'ASSIGNMENT.json'),B2_GPU=str(device['gpu']))
    else:env['CUDA_VISIBLE_DEVICES']=''
    actual=target[0] if role=='evaluate' else target
    script='evaluate_continuations' if role=='evaluate' else role
    cmd=[cfg['torch_python'],'-I','-B',str(HERE/(script+'.py')),str(actual)]
    log=(folder/'stdout.log').open('x');proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    w=dict(proc=proc,owner=monitor.process_identity(proc.pid),folder=folder,cfg=dict(cfg,**(device or {})),began=time.monotonic(),log=log,ended=None,members=[],role=role,gpu=device,closed=False,target=str(actual))
    immutable(folder/'OWNER.json',w['owner']);immutable(folder/'COMMAND.json',dict(argv=cmd,gpu=device,role=role))
    return w


def view(w,run):
    code=w['proc'].poll();now=time.monotonic()
    if code is not None and w['ended'] is None:
        w['ended']=now;w['log'].close();w['closed']=True
        immutable(w['folder']/'EXIT.json',dict(returncode=code,seconds=now-w['began'],gpu=w['gpu'],role=w['role']))
        append(run/'RESOURCES.jsonl',dict(role=w['role'],folder=str(w['folder']),seconds=now-w['began'],gpu_hours=(now-w['began'])/3600 if w['gpu'] else 0))
    rs=monitor.resources(w['proc'],w['owner'])
    for pid in rs['owned_pids']:
        try:
            ident=monitor.process_identity(pid)
            if ident not in w['members']:w['members'].append(ident)
        except (FileNotFoundError,ProcessLookupError):pass
    gpu=monitor.gpu_snapshot(w['gpu']) if w['gpu'] else None
    return dict(role=w['role'],returncode=code,gpu=w['gpu'],device=gpu,seconds=(w['ended'] or now)-w['began'],resource=rs,attempt=str(w['folder']),target=w['target'])


def retryable(w):
    text=(w['folder']/'stdout.log').read_text(errors='replace')[-20000:]
    if any(x in text for x in ('NONFINITE','CHANGED','MISMATCH','NONFIT','RESUME_CURSOR','IDENTITY','FORBIDDEN')):return False
    return any(x in text for x in ('CUDA out of memory','SIMULATOR_EOF','TimeoutError','timed out','ConnectionResetError','OSError: [Errno 5]'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():raise ValueError('RUN_ID')
    run=HERE/'runs'/a.run_id
    if run.exists() and not a.resume:raise FileExistsError('USE_RESUME')
    run.mkdir(parents=True,exist_ok=True);guard=(run/'RUN.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(a.config);immutable(run/'PROTOCOL.json',cfg);immutable(run/'SOURCE_LOCK.json',read(HERE/'SOURCE_LOCK.json'));verify_lock(read(run/'SOURCE_LOCK.json'))
    cfg=config(run);local_module('prepare').main(run)
    immutable(run/'CHECKPOINT_PLAN.json',dict(steps=cfg['checkpoint_steps'],arm='EXPANDED',seed=1209,total_updates=100000,new_updates=98800,planned_per_checkpoint=256,main_N=128,control_N=128,score_adaptation=False))
    workers=[];active=None;train=None;review=None;leased=False;error=None;finished=False
    retries=len(c.records(run/'RETRIES.jsonl')) if (run/'RETRIES.jsonl').exists() else 0
    disk=DiskWatch(run,monitor,write,append,cfg['artifact_gib']*2**30)
    def stop(signum,frame):raise InterruptedError('SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        acquire(run);leased=True
        deadline=time.monotonic()+60
        while monitor.gpu_snapshot(cfg['devices'][0])['free_mib']<cfg['min_start_gpu_gib']*1024:
            if time.monotonic()>deadline:raise RuntimeError('DEVICE_HEADROOM')
            time.sleep(2)
        if not (run/'train/RESULT.json').exists():
            train=spawn(run,cfg,'train',run,cfg['devices'][0]);workers.append(train)
        ev=local_module('evaluate_continuations')
        while True:
            views=[view(w,run) for w in workers if w['ended'] is None]
            for v in views:
                if v['returncode'] is None:
                    if v['resource']['rss_bytes']>cfg['process_rss_gib']*2**30:raise RuntimeError('OWN_PROCESS_RSS_LIMIT')
                    if v['device'] and v['device']['free_mib']<cfg['min_free_gpu_gib']*1024:raise RuntimeError('GPU_HEADROOM_LOST')
            if train and train['ended'] is not None:
                if train['proc'].returncode:
                    if not retryable(train) or retries>=cfg['infra_retries']:raise RuntimeError('TRAIN_FAILED:'+str(train['folder']))
                    resource.stop_owned(train);retries+=1;append(run/'RETRIES.jsonl',dict(role='train',attempt=str(train['folder']),reason='Infrastructure-only',unix=time.time()))
                    train=spawn(run,cfg,'train',run,cfg['devices'][0]);workers.append(train)
                else:
                    if not (run/'train/RESULT.json').exists():raise ValueError('TRAIN_EXIT_WITHOUT_COMPLETION')
                    train=None
            progress=read(run/'train/PROGRESS.json') if (run/'train/PROGRESS.json').exists() else {}
            if train and progress and time.monotonic()-train['began']>600 and time.time()-progress['unix']>cfg['stall_seconds']:raise RuntimeError('TRAIN_NO_PROGRESS')
            if active is not None and review is None:
                ew=[w for w in workers if w['role']=='evaluate' and w['target']==str(active[0])]
                failed=[w for w in ew if w['ended'] is not None and w['proc'].returncode and not w.get('handled')]
                for w in failed:
                    if not retryable(w) or retries>=cfg['infra_retries']:raise RuntimeError('EVALUATION_FAILED:'+str(w['folder']))
                    resource.stop_owned(w);w['handled']=True;retries+=1;append(run/'RETRIES.jsonl',dict(role='evaluate',attempt=str(w['folder']),unix=time.time()))
                if all(w['ended'] is not None for w in ew):
                    done=ev.admitted(active[0],active[1]);remaining={i for i,x in enumerate(active[1]['conditions']) if x['available']}-set(done)
                    if remaining:
                        if not failed and len(done)<=active[2]:raise RuntimeError('EVALUATION_NO_PROGRESS')
                        active=(active[0],active[1],len(done));launch_eval(active,remaining,cfg,run,workers)
                    else:review=spawn(run,cfg,'review',active[0]);workers.append(review)
            if review and review['ended'] is not None:
                if review['proc'].returncode:raise RuntimeError('REVIEW_FAILED:'+str(review['folder']))
                result=read(active[0]/'RESULT.json');append(run/'LEARNING_CURVE.jsonl',result)
                review=None;active=None
            if active is None:
                pending=[step for step in cfg['checkpoint_steps'] if not (run/'evaluations'/f'STEP_{step:06d}'/'RESULT.json').exists()]
                if not pending and train is None:finished=True;break
                if pending and completed_checkpoint(run,pending[0]):
                    child,reg=prepare_eval(run,pending[0],cfg);done=ev.admitted(child,reg)
                    remaining={i for i,x in enumerate(reg['conditions']) if x['available']}-set(done)
                    active=(child,reg,len(done))
                    if remaining:launch_eval(active,remaining,cfg,run,workers)
                    else:review=spawn(run,cfg,'review',child);workers.append(review)
            prior=sum(x['gpu_hours'] for x in c.records(run/'RESOURCES.jsonl')) if (run/'RESOURCES.jsonl').exists() else 0
            hours=prior+sum((time.monotonic()-w['began'])/3600 for w in workers if w['gpu'] and w['ended'] is None)
            sealed=[step for step in cfg['checkpoint_steps'] if (run/'checkpoints'/f'STEP_{step:06d}'/'COMMIT.json').exists()]
            done=[step for step in cfg['checkpoint_steps'] if (run/'evaluations'/f'STEP_{step:06d}'/'RESULT.json').exists()]
            status=dict(status='RUNNING',stage='train_and_checkpoint_evaluation',training=progress,checkpoints_saved=sealed,evaluations_complete=done,active_evaluation=active[0].name if active else None,workers=views,gpu_hours=hours,unix=time.time())
            write(run/'STATUS.json',status);append(run/'MONITOR.jsonl',status);disk.tick();time.sleep(10)
    except BaseException as exc:error=dict(reason=repr(exc),traceback=traceback.format_exc())
    finally:
        disk.stop()
        for w in workers:
            resource.stop_owned(w)
            if not w['closed']:view(w,run)
        if leased:
            after=resource.lease('release',pid=os.getpid());write(run/f'LEASE_RELEASED_{time.time_ns()}.json',dict(state=after,restoration='Controller restores holders when no other lease or external work remains.'),True)
    if error:
        write(run/'STATUS.json',dict(status='STOPPED',**error,unix=time.time()));raise RuntimeError(error['reason'])
    if not finished:raise RuntimeError('INCOMPLETE')
    result=dict(status='COMPLETE_DEVELOPMENT_LEARNING_CURVE',updates=100000,new_updates=98800,base_updates=0,
        checkpoints=20,planned=5120,complete=sum(read(run/'evaluations'/f'STEP_{step:06d}'/'RESULT.json')['complete'] for step in cfg['checkpoint_steps']),adopted=False)
    immutable(run/'RESULT.json',result);write(run/'STATUS.json',result)
    local_module('report').main(run)
    try:local_module('publish').main(run)
    except Exception as exc:write(run/'UPLOAD_ERROR.json',dict(error=repr(exc),unix=time.time()))


def launch_eval(active,remaining,cfg,run,workers):
    devices=[d for d in cfg['devices'] if d['gpu'] in cfg['eval_gpus']]
    for i,d in enumerate(devices):
        ranks=sorted(remaining)[i::len(devices)]
        if ranks:workers.append(spawn(run,cfg,'evaluate',(active[0],ranks),d))

if __name__=='__main__':main()
