"""Bounded per-arm three-rank training; exact owned tree cleanup, no foreign signals."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
r=load('paired_runtime',HERE/'runtime.py')
own=load('paired_owner',HERE.parent/'ordinary_prefix_history8_v1/owned_process_r1.py')
scan=load('paired_live_tree_size',LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2/tree_size.py')
GPUS=['GPU-a62dba8b-285b-57e6-b6d2-cf6e5788864a','GPU-e458147b-4739-d22a-e764-50743cff4a11','GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b']
def tree(pid):
    result=set();pending=[pid]
    while pending:
        current=pending.pop()
        if current in result:continue
        result.add(current)
        try:
            with os.scandir(Path('/proc')/str(current)/'task') as stream:tasks=list(stream)
        except FileNotFoundError:continue
        for task in tasks:
            try:pending.extend(int(x) for x in (Path(task.path)/'children').read_text().split())
            except FileNotFoundError:pass
    return result
def gpu_info():
    root=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=15))
    rows=[]
    for uuid in GPUS:
        x=next(g for g in root.findall('gpu') if g.findtext('uuid')==uuid)
        rows.append(dict(uuid=uuid,memory_mib=float(x.findtext('fb_memory_usage/used').split()[0]),
            pids=[int(v.findtext('pid')) for v in x.findall('processes/process_info')]))
    return rows
def rss(pid):
    try:return next((int(x.split()[1])*1024 for x in (Path('/proc')/str(pid)/'status').read_text().splitlines() if x.startswith('VmRSS:')),0)
    except FileNotFoundError:return 0
def signal_known(identity,sig):
    now=own.inspect(identity['pid'])
    if now is None or now['state'] in ('Z','X'):return False
    assert own.stable(now)==identity,'DESCENDANT_IDENTITY_CHANGED'
    fd=own.pidfd_open(now['pid'])
    try:
        again=own.inspect(now['pid'])
        if again is None or again['state'] in ('Z','X'):return False
        assert own.stable(again)==identity
        try:own.pidfd_send(fd,sig)
        except ProcessLookupError:return False
        return True
    finally:os.close(fd)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['control_recent2','treatment_prefix8'],required=True);args=parser.parse_args()
    p=json.loads((HERE/'PROTOCOL.json').read_text());assert p['runtime_allowed']
    for path,digest in p['code_sha256'].items():assert r.sha(Path(path))==digest,path
    root=HERE/args.arm;root.mkdir(exist_ok=False);out=root/'run_001';out.mkdir()
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tmp=LINE/'.th8p';assert tmp.is_dir()
    initial=gpu_info();assert all(not x['pids'] and x['memory_mib']<1024 for x in initial),'TRAINING_CARDS_NOT_EMPTY'
    started=time.monotonic();deadline=time.time()+p['wall_seconds_per_arm']
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX'):env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(GPUS),CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',TMPDIR=str(tmp),TMP=str(tmp),TEMP=str(tmp),
        NCCL_SOCKET_IFNAME='lo',GLOO_SOCKET_IFNAME='lo',NCCL_IB_DISABLE='1',
        Q35N_STORE=str(out/'rendezvous_file'),Q35N_DEADLINE=str(deadline),
        HF_HOME=str(root/'cache/hf'),XDG_CACHE_HOME=str(root/'cache/xdg'),TORCH_HOME=str(root/'cache/torch'),
        CUDA_CACHE_PATH=str(root/'cache/cuda'),TORCHINDUCTOR_CACHE_DIR=str(root/'cache/inductor'),
        TRITON_CACHE_DIR=str(root/'cache/triton'))
    guards=[];logs=[];seen={};error=None;cleanups=[];signals=[];stop=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda s,_:stop.append(s))
    r.save(root/'LAUNCH_BEFORE.json',dict(unix=time.time(),gpu=initial,arm=args.arm,deadline_unix=deadline),True)
    def observe():
        for guard in guards:
            for pid in tree(guard.proc.pid):
                value=own.inspect(pid)
                if value and value['state'] not in ('Z','X'):
                    identity=own.stable(value)
                    if pid in seen:assert seen[pid]['start']==identity['start'],'OWNED_TREE_PID_REUSE'
                    seen[pid]=identity
    try:
        for rank in range(3):
            argv=[str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B','-u',str(HERE/'train.py'),'--arm',args.arm]
            log=(out/f'RANK{rank}.log').open('x');logs.append(log)
            child=subprocess.Popen(argv,cwd=ROOT,env=dict(env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='3'),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            guards.append(own.OwnedChild(child,argv,ROOT))
        r.save(root/'PROCESSES.json',dict(ranks=[g.identity for g in guards],unix=time.time()),True)
        while True:
            observe();codes=[g.proc.poll() for g in guards]
            if all(c is not None for c in codes):
                assert codes==[0,0,0],'RANK_EXIT:'+repr(codes);break
            assert not stop,'SUPERVISOR_SIGNAL'
            assert not any(c not in (None,0) for c in codes),'RANK_FAILURE:'+repr(codes)
            assert time.monotonic()-started<p['wall_seconds_per_arm']+180,'HARD_ARM_DEADLINE'
            for g in guards:g.check()
            gpu=gpu_info();allowed=set(seen)
            assert all(set(x['pids'])<=allowed for x in gpu),'FOREIGN_TRAINING_CONTEXT'
            total_rss=sum(rss(pid) for pid in allowed)
            assert total_rss<64*1024**3,'TOTAL_CPU_RSS_BUDGET'
            size,transient_misses=scan.tree_size(root)
            assert size<2*1024**3,'OUTPUT_BUDGET'
            assert all(x['memory_mib']<28*1024 for x in gpu),'NVML_MEMORY_BUDGET'
            r.save(root/'RESOURCE.json',dict(unix=time.time(),arm=args.arm,gpu=gpu,total_rss_bytes=total_rss,
                output_bytes=size,tree_scan_transient_misses=transient_misses,wall_seconds=time.monotonic()-started,owned_pids=sorted(allowed)))
            time.sleep(5)
        result=json.loads((out/'RESULT.json').read_text())
        assert result['status']=='COMPLETE' and result['updates']==4000 and result['global_decisions']==384000
    except BaseException as exc:
        error=repr(exc);r.save(root/'FAILURE.json',dict(unix=time.time(),error=error,traceback=traceback.format_exc()),True)
    finally:
        try:observe()
        except BaseException as exc:error=error or repr(exc)
        for guard in guards:cleanups.append(guard.cleanup(timeout=15))
        for sig in (signal.SIGTERM,signal.SIGKILL):
            for pid,identity in seen.items():
                if pid in {g.proc.pid for g in guards}:continue
                try:
                    if signal_known(identity,sig):signals.append(dict(pid=pid,signal=int(sig)))
                except BaseException as exc:error=error or repr(exc)
            if sig==signal.SIGTERM:time.sleep(2)
        for log in logs:log.close()
        try:after=gpu_info()
        except BaseException as exc:
            error=error or repr(exc);after=[]
        clean=len(cleanups)==3 and all(x['exited'] for x in cleanups) and len(after)==3 and all(not x['pids'] for x in after)
        r.save(root/'LAUNCH_RESULT.json',dict(status='COMPLETE' if error is None and clean else 'FAILED',
            unix=time.time(),error=error,rank_cleanup=cleanups,descendant_signals=signals,
            gpu_after=after,wall_seconds=time.monotonic()-started,other_processes_signaled=[]),True)
    if error or not clean:raise RuntimeError(error or 'UNCLEAN_TRAINING_EXIT')
    print(json.dumps(dict(status='COMPLETE',arm=args.arm)),flush=True)
if __name__=='__main__':main()
