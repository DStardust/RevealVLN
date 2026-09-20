"""Independent V16 stage runner. All work, limits and resumes belong to this process."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *

def gpu_snapshot(config):
    command=['nvidia-smi','-i',config['gpu_uuid'],'--query-gpu=uuid,name,memory.total,memory.used,memory.free,driver_version','--format=csv,noheader,nounits']
    r=subprocess.run(command,capture_output=True,text=True,timeout=20)
    if r.returncode:raise RuntimeError('GPU_DRIVER_UNAVAILABLE: '+r.stderr.strip())
    fields=[x.strip() for x in r.stdout.strip().split(',')]
    if fields[0]!=config['gpu_uuid']:raise RuntimeError('GPU_UUID_MISMATCH')
    return dict(uuid=fields[0],name=fields[1],total_mib=int(fields[2]),used_mib=int(fields[3]),free_mib=int(fields[4]),driver=fields[5])

def process_identity(pid):
    p=Path('/proc')/str(pid)
    raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,uid=p.stat().st_uid,pgid=int(fields[2]),starttime=fields[19])

def cleanup(proc,identity):
    if proc.poll() is not None:return
    if process_identity(proc.pid)!=identity or identity['pgid']!=proc.pid or identity['uid']!=os.getuid():
        raise RuntimeError('REFUSE_NONOWNED_PROCESS_GROUP')
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if process_identity(proc.pid)!=identity:raise RuntimeError('PROCESS_IDENTITY_CHANGED')
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)

def resources(proc,identity):
    rss=0;members=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            who=process_identity(int(p.name))
            if who['pgid']!=identity['pgid'] or who['uid']!=identity['uid']:continue
            members.append(who['pid'])
            for line in (p/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
        except (OSError,ValueError,IndexError):continue
    return dict(owned_pids=members,rss_bytes=rss)

def spent(run):
    return sum(r['wall_seconds'] for p in (HERE/'runs').glob('*/RESOURCE_SESSIONS.jsonl') for r in c.records(p))

def execute(run,config,phase,args,python, gpu=True):
    gpu_lock=None
    if gpu:
        gpu_lock=(HERE/('GPU_'+config['gpu_uuid']+'.lock')).open('a')
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempts=run/'attempts';attempts.mkdir(exist_ok=True)
    number=len(list(attempts.glob(phase+'_*')))+1
    folder=attempts/(phase+f'_{number:03d}');folder.mkdir()
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX'):env.pop(key,None)
    env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key,relative in dict(HF_HOME='cache/hf',XDG_CACHE_HOME='cache/xdg',TORCH_HOME='cache/torch',
        TRITON_CACHE_DIR='cache/triton',CUDA_CACHE_PATH='cache/cuda',NUMBA_CACHE_DIR='cache/numba',
        MPLCONFIGDIR='cache/mpl',TMPDIR='tmp',TMP='tmp',TEMP='tmp').items():
        (folder/relative).mkdir(parents=True,exist_ok=True);env[key]=str(folder/relative)
    if phase.startswith('collect'):env.pop('CUDA_VISIBLE_DEVICES',None)
    else:env['CUDA_VISIBLE_DEVICES']=config['gpu_uuid'] if gpu else ''
    before=gpu_snapshot(config) if gpu else None
    if gpu and before['free_mib']<config['min_free_gpu_gib']*1024:raise RuntimeError('RESOURCE_NO_HEADROOM')
    command=[str(python),'-I','-B',*map(str,args)]
    numerical_env={k:env[k] for k in ('CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF','TRITON_CACHE_DIR','CUDA_CACHE_PATH')}
    write(folder/'COMMAND.json',dict(command=command,gpu=before,unix=time.time(),phase=phase,numerical_environment=numerical_env),True)
    began=time.monotonic();proc=None;identity=None;code=None;error=None
    disk_proc=None;disk_output=None;disk_error=None;disk_started=None;last_disk_check=-300;disk_bytes=None
    try:
        with (folder/'stdout.log').open('x') as out:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            identity=process_identity(proc.pid);write(folder/'OWNER.json',identity,True)
            while proc.poll() is None:
                elapsed=time.monotonic()-began
                state=dict(phase=phase,attempt=number,elapsed_seconds=elapsed,unix=time.time(),**resources(proc,identity))
                if gpu:
                    state['gpu']=gpu_snapshot(config)
                    tasks=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
                    state['device_process_snapshot']=tasks.stdout
                    if elapsed>config['max_session_hours']*3600 or spent(run)+elapsed>config['gpu_session_hours']*3600:raise RuntimeError('RESOURCE_TIME_LIMIT')
                    if state['gpu']['free_mib']<config['min_free_gpu_gib']*1024:raise RuntimeError('RESOURCE_HEADROOM_LOST')
                if state['rss_bytes']>config['process_rss_gib']*2**30:raise RuntimeError('RESOURCE_RSS_LIMIT')
                if disk_proc is not None and disk_proc.poll() is not None:
                    disk_output.close();disk_error.close()
                    disk_stdout=(folder/'DISK_SCAN.stdout').read_text();disk_stderr=(folder/'DISK_SCAN.stderr').read_text()
                    # Atomic array/JSON writers may remove a temporary name between
                    # directory enumeration and stat. Preserve that observed race.
                    transient=disk_proc.returncode==1 and disk_stderr and all('.tmp' in line and 'No such file or directory' in line for line in disk_stderr.splitlines())
                    if disk_proc.returncode and not transient:raise RuntimeError('ARTIFACT_AUDIT_FAILED: '+disk_stderr)
                    disk_bytes=int(disk_stdout.split()[0]);state['artifact_snapshot_temporary_race']=bool(transient)
                    append(folder/'DISK_SCANS.jsonl',dict(bytes=disk_bytes,seconds=time.monotonic()-disk_started,unix=time.time(),temporary_name_race=bool(transient)))
                    disk_proc=None;last_disk_check=elapsed
                    if disk_bytes>config['artifact_gib']*2**30:raise RuntimeError('RESOURCE_ARTIFACT_LIMIT')
                if disk_proc is not None and time.monotonic()-disk_started>300:raise RuntimeError('ARTIFACT_SCAN_TIMEOUT_300_SECONDS')
                if disk_proc is None and elapsed-last_disk_check>=300:
                    disk_output=(folder/'DISK_SCAN.stdout').open('w');disk_error=(folder/'DISK_SCAN.stderr').open('w')
                    disk_proc=subprocess.Popen(['du','-sb',str(HERE)],stdin=subprocess.DEVNULL,stdout=disk_output,stderr=disk_error)
                    disk_started=time.monotonic()
                state['v16_artifact_bytes_last_measured']=disk_bytes
                state['artifact_scan_running']=disk_proc is not None
                append(folder/'RESOURCES.jsonl',state);write(run/'STATUS.json',dict(status='RUNNING',**state))
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            code=proc.returncode
        if code:raise RuntimeError(f'STAGE_PROCESS_FAILED:{phase}:{code}:{folder}')
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        if proc and identity:cleanup(proc,identity)
        if disk_proc is not None and disk_proc.poll() is None:
            disk_proc.terminate();disk_proc.wait(timeout=10)
        if disk_output:disk_output.close()
        if disk_error:disk_error.close()
        row=dict(phase=phase,attempt=number,wall_seconds=time.monotonic()-began,returncode=code,error=error,gpu_session=gpu)
        write(folder/'EXIT.json',row,True)
        if gpu:append(run/'RESOURCE_SESSIONS.jsonl',row)
        if gpu_lock:gpu_lock.close()

def preflight(run,config):
    verify_lock(read(run/'SOURCE_LOCK.json'))
    if sha(Path(config['checkpoint']))!=config['checkpoint_sha256']:raise ValueError('WRONG_CHECKPOINT')
    if sha(LINE/'sft_acceptance/ordinary_sync_recovery_v1/model.py')!=config['model_source_sha256']:raise ValueError('WRONG_MODEL_IMPLEMENTATION')
    paths=[ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3',LINE/'.envs/q35n_qwen_g2_v1/bin/python3',LINE/'.envs/q35n_habitat_v017_g0r/bin/python3']
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
    gpu=gpu_snapshot(config)
    write(run/'PREFLIGHT.json',dict(status='LOCAL_ASSETS_AND_GPU_ACCESS_CHECKED',gpu=gpu,paths=list(map(str,paths)),
        no_dedicated_lease_claim=True,base_loaded=False,navigation_started=False,source_lock_sha256=sha(run/'SOURCE_LOCK.json')))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--phase',choices=('preflight','collect','features','train','evaluate','review','all'),required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not re.fullmatch('[a-zA-Z0-9_-]{1,80}',args.run_id):raise ValueError('RUN_ID')
    config=read(args.config);run=HERE/'runs'/args.run_id
    if run.exists() and not args.resume:raise FileExistsError('USE_RESUME_FOR_EXISTING_RUN')
    run.mkdir(parents=True,exist_ok=True)
    with (run/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable(run/'PROTOCOL.json',config)
        if (run/'SOURCE_LOCK.json').exists():verify_lock(read(run/'SOURCE_LOCK.json'))
        else:write(run/'SOURCE_LOCK.json',source_lock(),True)
        torchpy=LINE/'.envs/q35n_qwen_g2_v1/bin/python3';simpy=LINE/'.envs/q35n_habitat_v017_g0r/bin/python3'
        phases=['preflight','collect','features','train','evaluate','review'] if args.phase=='all' else [args.phase]
        try:
            for phase in phases:
                if (run/(phase.upper()+'_COMPLETE.json')).exists():continue
                if phase=='preflight':preflight(run,config)
                elif phase=='collect':
                    execute(run,config,'collect_fit_dev',[HERE/'collect.py',run,'FIT','DEV'],simpy)
                    # Selection is sealed using only physical feasibility, before model TEST outputs exist.
                    immutable(run/'TEST_FREEZE.json',dict(manifest_sha256=config['data_manifest_sha256'],
                        test_houses=[r for r in read(HERE/'DATA_MANIFEST.json')['houses'] if r['split']=='TEST'],method_scores_read=False))
                    execute(run,config,'collect_test',[HERE/'collect.py',run,'TEST'],simpy)
                    from build_data import build
                    build(run)
                    execute(run,config,'raw_audit',[HERE/'audit_data.py',run],torchpy,gpu=False)
                    from evaluate_continuations import registry
                    registry(run)
                elif phase in ('features','train','evaluate'):
                    from evaluate_continuations import registry,admitted
                    script={'features':'extract_features.py','train':'train.py','evaluate':'evaluate_continuations.py'}[phase]
                    retries=0
                    while True:
                        try:execute(run,config,phase,[HERE/script,run],torchpy)
                        except RuntimeError as error:
                            # Only resource/service infrastructure faults may retry. Numerical/data
                            # correctness errors and complete low-scoring groups never retry.
                            attempts=sorted((run/'attempts').glob(phase+'_*'))
                            output=(attempts[-1]/'stdout.log').read_text() if attempts and (attempts[-1]/'stdout.log').exists() else ''
                            infra=any(key in output for key in ('CUDA out of memory','SIMULATOR_EOF','ConnectionResetError'))
                            if not infra or retries>=config['infra_retries'] or spent(run)>=config['gpu_session_hours']*3600:raise
                            retries+=1;append(run/'INFRA_RETRIES.jsonl',dict(phase=phase,retry=retries,error=repr(error),attempt=str(attempts[-1]),score_based=False))
                            continue
                        finished=(run/'features/FEATURE_RESULT.json').exists() if phase=='features' else (run/'train/RESULT.json').exists() if phase=='train' else len(admitted(run,registry(run)))==80
                        if finished:break
                        if spent(run)>=config['gpu_session_hours']*3600:raise RuntimeError('RESOURCE_TOTAL_BUDGET')
                elif phase=='review':
                    execute(run,config,'diagnose',[HERE/'diagnose.py',run],torchpy,gpu=False)
                    from review import main as review
                    review(run)
                write(run/(phase.upper()+'_COMPLETE.json'),dict(phase=phase,completed_unix=time.time()),True)
            write(run/'STATUS.json',dict(status='REQUESTED_PHASES_COMPLETE',phases=phases,gpu_session_hours=spent(run)/3600))
        except BaseException as exc:
            entry=dict(status='STOPPED',error=repr(exc),traceback=traceback.format_exc(),gpu_session_hours=spent(run)/3600,unix=time.time())
            append(run/'FAILURES.jsonl',entry);write(run/'STATUS.json',entry)
            from review import main as review
            try:review(run)
            except BaseException as review_error:append(run/'FAILURES.jsonl',dict(status='REVIEW_FAILED',error=repr(review_error)))
            raise

if __name__=='__main__':main()
