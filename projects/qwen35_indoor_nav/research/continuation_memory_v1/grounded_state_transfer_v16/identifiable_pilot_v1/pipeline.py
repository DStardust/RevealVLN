"""Independent bounded prepare/features/train/diagnose/evaluate/review pipeline."""
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
monitor=load('ident_owned_resource_helpers',V16/'pipeline.py')
sys.path.insert(0,str(HERE))

def sources():
    paths=list(HERE.glob('*.py'))
    paths += [V16/n for n in ('encoder.py','objective.py','select_action.py','v16_common.py','evaluator_v16.py','collect.py','pipeline.py','kernel_metadata.py')]
    paths += [V16.parent/n/'model.py' for n in ('query_semantics_v11','contextual_readout_v10','query_reader_repair_v2','pilot')]
    paths += [LINE/'data_pipeline/mechanism_runtime_v1/habitat_backend.py',LINE/'data_pipeline/mechanism_factory_v2/compiler.py',
              LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5/common.py',LINE/'closed_loop_bench/r2r_ce_tiny_v1/common.py',
              LINE/'sft_acceptance/ordinary_sync_recovery_v1/model.py',LINE/'sft_acceptance/ordinary_sync_recovery_v1/data.py',
              LINE/'sft_acceptance/ordinary_baseline_v2/data.py']
    return dict(files={str(p.relative_to(LINE)):sha(p) for p in paths})

def execute(run,cfg,stage):
    gpu=stage in ('features','train','diagnose','evaluate_continuations')
    gpu_hours=sum(x['seconds']/3600 for x in c.records(run/'RESOURCES.jsonl') if x['gpu']) if (run/'RESOURCES.jsonl').exists() else 0
    if gpu and gpu_hours>=cfg['gpu_session_hours']:raise RuntimeError('CUMULATIVE_GPU_BUDGET')
    parent=run/'attempts';parent.mkdir(exist_ok=True)
    folder=parent/f'{stage}_{len(list(parent.glob(stage+"_*")))+1:03d}';folder.mkdir()
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX'):env.pop(key,None)
    env.update(V16_ASSET_LINE_ROOT=cfg['asset_line_root'],CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'] if gpu else '',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    command=[cfg['torch_python'],'-I','-B',str(HERE/(stage+'.py')),str(run)]
    before=monitor.gpu_snapshot(cfg) if gpu else None
    if gpu and before['free_mib']<cfg['min_free_gpu_gib']*1024:raise RuntimeError('GPU_HEADROOM')
    write(folder/'COMMAND.json',dict(argv=command,gpu=before),True)
    began=time.monotonic();proc=None;owner=None;code=None;error=None;lastdisk=0
    try:
        with (folder/'stdout.log').open('x') as log:
            proc=subprocess.Popen(command,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            owner=monitor.process_identity(proc.pid);write(folder/'OWNER.json',owner,True)
            while proc.poll() is None:
                elapsed=time.monotonic()-began;resource=monitor.resources(proc,owner);device=monitor.gpu_snapshot(cfg) if gpu else None
                snapshot=dict(stage=stage,elapsed=elapsed,**resource,gpu=device)
                write(run/'STATUS.json',dict(status='RUNNING',**snapshot));append(folder/'MONITOR.jsonl',snapshot)
                if elapsed>cfg['max_session_hours']*3600 or (gpu and gpu_hours+elapsed/3600>cfg['gpu_session_hours']):raise RuntimeError('RESOURCE_TIME_LIMIT')
                if resource['rss_bytes']>cfg['process_rss_gib']*2**30 or (gpu and device['free_mib']<cfg['min_free_gpu_gib']*1024):raise RuntimeError('RESOURCE_MEMORY_LIMIT')
                if elapsed-lastdisk>300:
                    size=int(subprocess.run(['du','-sb',str(run)],capture_output=True,text=True,timeout=60,check=True).stdout.split()[0]);lastdisk=elapsed
                    if size>cfg['artifact_gib']*2**30:raise RuntimeError('ARTIFACT_LIMIT')
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            code=proc.returncode
        if code:raise RuntimeError(f'STAGE_FAILED:{stage}:{folder}')
    except BaseException as exc:error=repr(exc);raise
    finally:
        if proc and owner:monitor.cleanup(proc,owner)
        record=dict(stage=stage,gpu=gpu,seconds=time.monotonic()-began,returncode=code,error=error)
        write(folder/'EXIT.json',record,True);append(run/'RESOURCES.jsonl',record)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not a.run_id.replace('_','').isalnum():raise ValueError('RUN_ID')
    cfg=read(a.config);run=HERE/'runs'/a.run_id
    if run.exists() and not a.resume:raise FileExistsError('USE_RESUME')
    run.mkdir(parents=True,exist_ok=True)
    with (run/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable(run/'PROTOCOL.json',cfg);immutable(run/'SOURCE_LOCK.json',sources())
        try:
            from prepare import main as prepare
            prepare(run)
            if sha(Path(cfg['checkpoint']))!=cfg['checkpoint_sha256']:raise ValueError('CHECKPOINT_CHANGED')
            write(run/'PREFLIGHT.json',dict(gpu=monitor.gpu_snapshot(cfg),data_sha256=sha(run/'DATA.json'),source_sha256=sha(run/'SOURCE_LOCK.json'),base_loaded=False))
            if not (run/'features/FEATURE_RESULT.json').exists():execute(run,cfg,'features')
            while not (run/'train/RESULT.json').exists():execute(run,cfg,'train')
            if not (run/'DIAGNOSTICS_COMPLETE.json').exists():execute(run,cfg,'diagnose')
            from evaluate_continuations import registry, admitted
            reg=registry(run)
            while len(admitted(run,reg))<64:execute(run,cfg,'evaluate_continuations')
            if not (run/'RESULT.json').exists():execute(run,cfg,'review')
            write(run/'STATUS.json',dict(status='COMPLETE',models=9,rollouts=576,result=str(run/'RESULT.json')))
        except BaseException as exc:
            write(run/'STATUS.json',dict(status='STOPPED',reason=repr(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
