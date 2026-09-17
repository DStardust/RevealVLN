"""One empty-GPU1 diagnostic; race-aware owned child cleanup always leaves a receipt."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
ROOT = LINE.parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
c=load('history8_common',HERE/'common.py')
owned=load('history8_owned',LINE/'sft_acceptance/ordinary_prefix_history8_v1/owned_process_r1.py')
def gpu():
    xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=15))
    device=next(x for x in xml.findall('gpu') if x.findtext('uuid')=='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8')
    return dict(processes=[int(x.findtext('pid')) for x in device.findall('processes/process_info')],
                memory_mib=float(device.findtext('fb_memory_usage/used').split()[0]))
def rss(pid):
    try:
        rows=(Path('/proc')/str(pid)/'status').read_text().splitlines()
        return next((int(x.split()[1])*1024 for x in rows if x.startswith('VmRSS:')),0)
    except FileNotFoundError:
        return None
def output_size():
    total=0
    for path in HERE.rglob('*'):
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:pass
    return total
def main():
    lock=(HERE/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (HERE/'PREFLIGHT.json').exists(), 'NODE_ALREADY_ATTEMPTED'
    c.verify_lock();assert c.read(HERE/'CPU_TEST_RESULT.json')['status']=='PASS'
    initial=gpu();assert initial['processes']==[] and initial['memory_mib']<1024, 'GPU1_NOT_EMPTY'
    temp=LINE/'.th8';temp.mkdir(exist_ok=True)
    assert temp.resolve().is_relative_to(LINE) and temp.is_dir()
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX'):env.pop(key,None)
    env.update(HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='1',TMPDIR=str(temp),
        TORCHINDUCTOR_CACHE_DIR=str(HERE/'inductor_cache'),TRITON_CACHE_DIR=str(HERE/'triton_cache'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    c.write(HERE/'PREFLIGHT.json',dict(unix=time.time(),gpu=initial,budget_seconds=600,source_lock_sha256=c.sha(HERE/'SOURCE_LOCK.json')),True)
    proc=None;guard=None;error=None;cleanup=None;started=time.monotonic()
    try:
        argv=[str(LINE/'.envs/q35n_qwen_g2_v1/bin/python3'),'-I','-B',str(HERE/'worker.py')]
        with (HERE/'worker.log').open('x') as log:
            proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            guard=owned.OwnedChild(proc,argv,ROOT)
            c.write(HERE/'PROCESS.json',dict(**guard.identity,unix=time.time()),True)
            while guard.check()!='exited':
                info=gpu();assert set(info['processes'])<={proc.pid},'FOREIGN_GPU1_CONTEXT'
                used=rss(proc.pid);size=output_size();wall=time.monotonic()-started
                c.write(HERE/'RESOURCE.json',dict(unix=time.time(),wall_seconds=wall,gpu=info,rss_bytes=used,output_bytes=size),False)
                assert wall<=600 and info['memory_mib']<=25*1024 and (used is None or used<=16*1024**3) and size<=1024**3,'RESOURCE_BUDGET'
                time.sleep(2)
            assert proc.returncode==0, 'WORKER_EXIT_'+str(proc.returncode)
            assert c.read(HERE/'RESULT.json')['status']=='PASS_INTERFACE_ONLY'
    except BaseException as exc:
        error=repr(exc)
        c.write(HERE/'LAUNCH_FAILURE.json',dict(unix=time.time(),error=error,traceback=traceback.format_exc()),True)
    finally:
        if guard is not None:
            cleanup=guard.cleanup()
        elif proc is not None:
            # Registration failed: do not signal an unverified live identity.
            cleanup=dict(pid=proc.pid,exit_code=proc.poll(),exited=proc.poll() is not None,
                         signals=[],cleanup_error='NO_REGISTERED_IDENTITY')
        try:after=gpu()
        except BaseException as exc:after=dict(error=repr(exc))
        clean=(proc is None or (cleanup and cleanup['exited'])) and after.get('processes')==[]
        c.write(HERE/'LAUNCH_RESULT.json',dict(status='COMPLETE' if error is None and clean else 'FAILED',
            unix=time.time(),error=error,cleanup=cleanup,gpu_after=after,wall_seconds=time.monotonic()-started,
            other_processes_signaled=[],holders_released=[]),True)
    if error or not clean:
        raise RuntimeError(error or 'CLEANUP_INCOMPLETE')
    print(json.dumps(dict(status='COMPLETE',wall_seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':
    main()


