"""Autonomous eight-GPU real-data generation with complete-family resume."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from aggregate_counts import counts
from coexist import acquire_preflight,protected_devices,available
from disk_watch_r2 import DiskWatch
from resume_r2 import preserve_complete_metadata,quarantine_array_temporaries
resource=load('scale_resource',PILOT/'parallel_eval_v1/coordinator.py')
monitor=resource.monitor

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():raise ValueError('RUN_ID')
    cfg=read(a.config);run=HERE/'runs'/a.run_id
    if run.exists() and not a.resume:raise FileExistsError('USE_RESUME')
    run.mkdir(parents=True,exist_ok=True)
    guard=(run/'RUN.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    immutable(run/'PROTOCOL.json',cfg);lock=read(HERE/'SOURCE_LOCK.json');verify_lock(lock);immutable(run/'SOURCE_LOCK.json',lock)
    if (run/'RESULT.json').exists():raise ValueError('ALREADY_COMPLETE_NO_REPLAY')
    for path,expected in read(HERE/'HOUSE_MANIFEST.json')['prior_parent_files'].items():
        if sha(LINE/path)!=expected:raise ValueError('PRIOR_PARENT_CHANGED')
    preserve_complete_metadata(run)
    (run/'attempts').mkdir(exist_ok=True);(run/'assignments').mkdir(exist_ok=True)
    began=time.monotonic();acquired=False;workers=[];error=None;last_disk=0;last_counts=0;stats={}
    disk=DiskWatch(run,monitor,write,append,cfg['artifact_gib']*2**30)
    previous=sum(v['gpu_hours'] for v in c.records(run/'RESOURCES.jsonl')) if (run/'RESOURCES.jsonl').exists() else 0
    def interrupted(signum,frame):raise InterruptedError('SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def launch(house,device):
        quarantine_array_temporaries(run,house)
        attempt=len(list((run/'attempts').glob(house+'_*')))+1
        folder=run/'attempts'/f'{house}_{attempt:03d}';folder.mkdir()
        assignment=dict(device,house=house)
        af=run/'assignments'/f'{house}_{attempt:03d}.json';immutable(af,assignment)
        env=resource.env_for(dict(cfg,**device));env.pop('CUDA_VISIBLE_DEVICES',None);env['B2_DEVICE']=str(af)
        command=[cfg['sim_python'],'-I','-B',str(HERE/'collect.py'),str(run)]
        log=(folder/'stdout.log').open('x')
        proc=subprocess.Popen(command,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        w=dict(proc=proc,owner=monitor.process_identity(proc.pid),members=[],log=log,folder=folder,
               house=house,device=device,began=time.monotonic(),ended=None,recorded=False,attempt=attempt)
        immutable(folder/'OWNER.json',w['owner']);immutable(folder/'COMMAND.json',dict(argv=command,assignment=assignment));workers.append(w)
    try:
        immutable(run/'RUNTIME_IDENTITY.json',dict(python=sys.executable,uid=os.getuid(),gid=os.getgid(),devices=cfg['devices'],
            scene_manifest_sha256=sha(HERE/'HOUSE_MANIFEST.json'),model_loaded=False,new_optimizer_updates=0,protocol_sha256=sha(run/'PROTOCOL.json')))
        write(run/'STATUS.json',dict(status='PREFLIGHT',stage='resource_release',unix=time.time()))
        immutable(run/f'PLACEHOLDER_BEFORE_{time.time_ns()}.json',acquire_preflight(cfg,resource))
        immutable(run/f'LEASE_ACQUIRED_{time.time_ns()}.json',resource.lease('acquire',pid=os.getpid()));acquired=True
        deadline=time.monotonic()+60
        while any(monitor.gpu_snapshot(d)['free_mib']<cfg['min_start_gpu_gib']*1024 for d in cfg['devices']):
            if time.monotonic()>deadline:raise RuntimeError('RELEASED_DEVICE_HEADROOM_UNAVAILABLE')
            time.sleep(2)
        skipped=read(run/'NOT_NEEDED_AFTER_TARGET.json')['houses'] if (run/'NOT_NEEDED_AFTER_TARGET.json').exists() else []
        pending=[h for h in cfg['houses'] if h not in skipped and not (run/'collect'/h/'DATASET.json').exists()]
        while pending or any(w['ended'] is None for w in workers) or not (Path(cfg['prior_run'])/'DATASET.json').exists():
            now=time.monotonic();views=[];hours=previous
            for w in list(workers):
                code=w['proc'].poll()
                if code is not None and w['ended'] is None:
                    w['ended']=now;w['log'].close()
                    immutable(w['folder']/'EXIT.json',dict(returncode=code,wall_seconds=now-w['began'],gpu=w['device']['gpu']))
                    resource.stop_owned(w)
                    if code!=0:
                        tail=(w['folder']/'stdout.log').read_text(errors='replace')[-8000:]
                        # Only known process/transport failures qualify. Data/source/audit failures never retry.
                        retryable=code in (-9,-11,137) or any(s in tail for s in ('OSError: [Errno 5]','COLLECTION_SESSION_LIMIT','ConnectionResetError'))
                        if retryable and w['attempt']<=cfg['infra_retries']:pending.insert(0,w['house'])
                        else:raise RuntimeError('WORKER_FAILED:'+w['house']+':'+str(w['folder']))
                elapsed=(w['ended'] or now)-w['began'];hours+=elapsed/3600
                if w['ended'] is not None:continue
                state=monitor.resources(w['proc'],w['owner'])
                for pid in state['owned_pids']:
                    try:
                        identity=monitor.process_identity(pid)
                        if identity not in w['members']:w['members'].append(identity)
                    except (FileNotFoundError,ProcessLookupError):pass
                gpu=monitor.gpu_snapshot(w['device'])
                views.append(dict(house=w['house'],gpu=w['device']['gpu'],pid=w['proc'].pid,seconds=elapsed,
                                  rss_bytes=state['rss_bytes'],device=gpu,attempt=w['attempt']))
                if elapsed>cfg['max_session_hours']*3600 or state['rss_bytes']>cfg['process_rss_gib']*2**30:raise RuntimeError('WORKER_RESOURCE_LIMIT')
                if gpu['free_mib']<cfg['min_free_gpu_gib']*1024:raise RuntimeError('GPU_HEADROOM_LOST')
            if hours>cfg['gpu_session_hours'] or now-began>cfg['max_wall_hours']*3600:raise RuntimeError('GENERATION_BUDGET')
            if now-last_counts>15:
                stats,_=counts(run);write(run/'COUNTS.json',stats);last_counts=now
            if stats['new_parents']>=cfg['target_combined_parents'] and pending:
                immutable(run/'NOT_NEEDED_AFTER_TARGET.json',dict(houses=pending,combined_parents=stats['new_parents'],selection='No new house starts after quantity target. Existing houses finish. No policy score used.'))
                pending=[]
            protected=protected_devices(cfg,monitor)
            active={w['device']['gpu'] for w in workers if w['ended'] is None}
            for device in available(cfg['devices'],protected,active):
                if pending and monitor.gpu_snapshot(device)['free_mib']>=cfg['min_start_gpu_gib']*1024:
                    launch(pending.pop(0),device)
            prior_status=read(Path(cfg['prior_run'])/'STATUS.json')
            if not (Path(cfg['prior_run'])/'DATASET.json').exists() and prior_status['status'] not in ('RUNNING','PREFLIGHT'):
                raise RuntimeError('PRIOR_BATCH_STOPPED_REQUIRES_REVIEW')
            write(run/'STATUS.json',dict(status='RUNNING',runtime_revision='V2_SUPPLEMENT_SHARED_RESOURCE',stage='real_physical_collection',unix=time.time(),workers=views,
                 pending_houses=pending,gpu_hours=hours,counts=stats,training=False,prior_protected_gpus=sorted(protected)))
            append(run/'MONITOR.jsonl',dict(unix=time.time(),workers=views,gpu_hours=hours))
            disk.tick()
            time.sleep(10)
    except BaseException as exc:
        error=dict(reason=repr(exc),traceback=traceback.format_exc())
        immutable(run/f'FAILURE_{time.time_ns()}.json',error)
    finally:
        disk.stop()
        for w in workers:
            resource.stop_owned(w)
            if not w['log'].closed:w['log'].close()
        append(run/'RESOURCES.jsonl',dict(gpu_hours=sum(((w['ended'] or time.monotonic())-w['began'])/3600 for w in workers),
               wall_seconds=time.monotonic()-began,error=error))
        if acquired:
            immutable(run/f'LEASE_RELEASED_{time.time_ns()}.json',resource.lease('release',pid=os.getpid()))
            deadline=time.monotonic()+120;restored=False
            while time.monotonic()<deadline:
                state=resource.lease('status')
                if state['state']=='occupying' and len(state['worker_pids'])==6:
                    restored=all(monitor.gpu_snapshot(d)['used_mib']>30*1024 for d in cfg['devices'] if d['gpu']>=2)
                    if restored:break
                if state['external_pids'] or state['manual_paused'] or state['leases']:break
                time.sleep(2)
            immutable(run/f'PLACEHOLDER_RESTORED_{time.time_ns()}.json',dict(restored=restored,state=state))
    preserve_complete_metadata(run)
    stats,families=counts(run);write(run/'COUNTS.json',stats)
    write(run/'DATASET_PARTIAL.json',dict(raw_families=families,scope=cfg['scope'],training_started=False))
    if not error:
        result=dict(status='DATA_SCALE_TARGET_COMPLETE' if stats['new_parents']>=cfg['target_combined_parents'] else 'PHYSICAL_SHORTFALL',
                    **stats,training_started=False,method_benefit='NOT_TESTED')
        immutable(run/'RESULT.json',result);immutable(run/'DATASET.json',dict(raw_families=families,scope=cfg['scope'],training_started=False))
    else:result=dict(status='STOPPED_WITH_PRESERVED_DATA',**stats,error=error,training_started=False)
    write(run/'STATUS.json',dict(**result,unix=time.time()))
    report=(f"# {result['status']}\n\n新增已核验父族 {stats['new_parents']}/800；变体 {stats['new_variants']}；"
            f"完整物理交叉执行 {stats['certified_physical_cross_executions']}。\n\n"
            "这批是受控 SEE2 数据扩充，尚未训练，未产生新的方法收益或普通 VLN-CE 成绩。"
            "派生变体、同屋空间近邻与多个标签有相关性，不能当作独立泛化样本。\n\n"
            f"停止/失败项：{error}\n")
    (run/'REPORT_ZH.md').write_text(report)
    subprocess.run([cfg['standalone_python'],'-I','-B',str(HERE/'publish.py'),str(run)],cwd=ROOT,timeout=1800,check=False)
    if error:raise RuntimeError(error['reason'])

if __name__=='__main__':main()
