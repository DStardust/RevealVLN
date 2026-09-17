"""Bounded GPU1 supervisor; includes graphics processes and conservative memory."""
import fcntl
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
import xml.etree.ElementTree as ET

HERE = Path('/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/data_pipeline/mechanism_runtime_v1/witness_first_v1/new_hub_scout_v1')
OUT = HERE/'run_v1'
LINE = Path('/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav')
ENV = LINE/'.envs/q35n_habitat_v017_g0r'
UUID = 'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'

def parse_gpu(raw):
    cards = ET.fromstring(raw).findall('gpu')
    assert len(cards) == 1 and cards[0].findtext('uuid') == UUID, 'GPU_IDENTITY'
    node = cards[0]
    def number(value):
        tokens = value.split(); assert len(tokens) == 2 and tokens[1] == 'MiB'
        value = int(tokens[0]); assert value >= 0; return value
    processes = {}
    for process in node.findall('processes/process_info'):
        pid = int(process.findtext('pid')); assert pid not in processes
        processes[pid] = {'mib':number(process.findtext('used_memory')), 'type':process.findtext('type')}
    return dict(uuid=UUID, memory_mib=number(node.findtext('fb_memory_usage/used')),
                utilization=int(node.findtext('utilization/gpu_util').split()[0]), processes=processes)

def gpu():
    return parse_gpu(subprocess.check_output(['nvidia-smi','-i','2','-q','-x'], text=True, timeout=15))

def check_gpu(snapshot, worker=None):
    external = [v['mib'] for p,v in snapshot['processes'].items() if p != worker]
    assert all(m <= 768 for m in external) and sum(external) <= 2048, 'EXTERNAL_RESOURCE_LOAD'
    upper = snapshot['memory_mib']-sum(external)
    assert 0 <= upper < 4096, 'OWN_GPU_MEMORY_UPPER_BOUND'
    assert sum(v['mib'] for v in snapshot['processes'].values()) <= snapshot['memory_mib'], 'MEMORY_ACCOUNTING'
    return upper

def disk_size():
    total = 0
    for parent, dirs, files in os.walk(OUT, followlinks=False):
        for name in dirs+files:
            path = Path(parent)/name
            try: info = path.lstat()
            except FileNotFoundError:
                if name in ('.HEAD.next','PROGRESS.pending') or (path.parent == OUT/'content' and name.endswith('.partial')):
                    total += 1024**2; continue
                raise
            assert not stat.S_ISLNK(info.st_mode), 'OUTPUT_SYMLINK'
            total += info.st_size
    return total

def save(name, value):
    with (OUT/name).open('x') as stream: json.dump(value, stream, indent=2)

def main():
    lock = (OUT/'SUPERVISOR.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before = gpu(); check_gpu(before); assert before['utilization'] == 0, 'GPU_NOT_IDLE'
    save('GPU_BEFORE.json', before)
    cache=OUT/'cache'; cache.mkdir()
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES'): env.pop(key,None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        XDG_CACHE_HOME=str(cache),CUDA_CACHE_PATH=str(cache/'cuda'),NUMBA_CACHE_DIR=str(cache/'numba'),
        MPLCONFIGDIR=str(cache/'matplotlib'),TMPDIR=str(cache),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def interrupt(signum, frame): raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,interrupt); signal.signal(signal.SIGINT,interrupt)
    started=time.monotonic(); proc=None; error=None; count=0
    try:
        with (OUT/'worker.log').open('x') as log:
            proc=subprocess.Popen([str(ENV/'bin/python3'),'-I','-B',str(HERE/'worker.py')],
                cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save('PROCESS.json',dict(pid=proc.pid,supervisor_pid=os.getpid(),started_unix=time.time()))
            while proc.poll() is None:
                time.sleep(5); count+=1
                snapshot=gpu(); upper=check_gpu(snapshot,proc.pid)
                sample=dict(snapshot, own_memory_upper_mib=upper, elapsed=time.monotonic()-started)
                assert sample['elapsed'] < 3000, 'WALL_BUDGET'
                rss=subprocess.run(['ps','-p',str(proc.pid),'-o','rss='],text=True,capture_output=True)
                assert rss.returncode==0 or proc.poll() is not None, 'RSS_READ_ERROR'
                assert int(rss.stdout.strip() or 0) < 8*1024**2, 'RSS_BUDGET'
                if count%12==0:
                    sample['disk_bytes']=disk_size(); assert sample['disk_bytes'] < 7*1024**3, 'DISK_BUDGET'
                with (OUT/'RESOURCE_SAMPLES.jsonl').open('a') as stream: stream.write(json.dumps(sample)+'\n')
    except BaseException as exc: error=repr(exc)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        after=None; clean=False
        try:
            after=gpu(); clean=not proc or proc.pid not in after['processes']
        except BaseException as exc: error=(error or '')+'; FINAL_GPU_QUERY: '+repr(exc)
        result=dict(returncode=proc.returncode if proc else None,error=error,wall_seconds=time.monotonic()-started,
            cleanup_complete=clean,external_processes_stopped=0,holders_touched=False,
            holder_restoration_required=False,gpu_after=after,scientific_pass=False)
        save('SUPERVISOR_RESULT.json',result); print(json.dumps(result),flush=True)
    if error or not proc or proc.returncode or not clean: raise SystemExit(1)

if __name__ == '__main__': main()
