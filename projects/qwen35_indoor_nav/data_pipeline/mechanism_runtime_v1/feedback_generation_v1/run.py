"""Bounded nonexclusive GPU1 rendering; external processes are never signalled."""
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
HERE=Path(__file__).resolve().parent
OUT=HERE/'run_v1'
LINE=HERE.parents[2]
ENV=LINE/'.envs/q35n_habitat_v017_g0r'
def call(*args):return subprocess.check_output(args,text=True).strip()
def gpu():
    row=call('nvidia-smi','-i','1','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits').split(',')
    assert row[0].strip()=='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8'
    text=call('nvidia-smi','-i','1','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits')
    procs={int(r.split(',')[0]):int(r.split(',')[1]) for r in text.splitlines() if r.strip()}
    return dict(uuid=row[0].strip(),memory_mib=int(row[1]),utilization=int(row[2]),processes=procs)
def size():
    total=0
    for parent,dirs,files in os.walk(OUT,followlinks=False):
        for name in dirs+files:
            p=Path(parent)/name
            try:s=p.lstat()
            except FileNotFoundError:
                if name=='.HEAD.next' or name=='PROGRESS.pending' or (p.parent==OUT/'content' and name.endswith('.partial')):
                    total+=1024**2;continue
                raise
            assert not stat.S_ISLNK(s.st_mode)
            total+=s.st_size
    return total
def save(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2)
def main():
    import fcntl
    lock=(OUT/'SUPERVISOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=gpu();save('GPU_BEFORE.json',before)
    assert before['utilization']==0 and sum(before['processes'].values())<=2048
    assert all(n<=768 for n in before['processes'].values())
    cache=OUT/'cache';cache.mkdir()
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin',PYTHONNOUSERSITE='1',XDG_CACHE_HOME=str(cache),
        NUMBA_CACHE_DIR=str(cache/'numba'),MPLCONFIGDIR=str(cache/'matplotlib'),TMPDIR=str(cache),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def interrupted(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    started=time.monotonic();proc=None;error=None;counter=0
    try:
        with (OUT/'worker.log').open('x') as log:
            proc=subprocess.Popen([str(ENV/'bin/python3'),'-I','-B',str(HERE/'worker.py')],
                cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save('PROCESS.json',dict(pid=proc.pid,supervisor_pid=os.getpid(),started_unix=time.time()))
            while proc.poll() is None:
                time.sleep(5);g=gpu();counter+=1
                other=[n for p,n in g['processes'].items() if p!=proc.pid]
                assert all(n<=768 for n in other) and sum(other)<=2048,'EXTERNAL_RESOURCE_LOAD'
                assert g['processes'].get(proc.pid,0)<4096
                rss=subprocess.run(['ps','-p',str(proc.pid),'-o','rss='],capture_output=True,text=True)
                if rss.returncode and proc.poll() is None:raise RuntimeError('RSS_READ_FAILED')
                assert int(rss.stdout.strip() or 0)<8*1024**2
                assert time.monotonic()-started<3000
                sample=dict(g,elapsed=time.monotonic()-started)
                if counter%12==0:
                    sample['disk_bytes']=size();assert sample['disk_bytes']<7*1024**3
                with (OUT/'RESOURCE_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
    except BaseException as exc:error=repr(exc)
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        after=gpu()
        result=dict(returncode=proc.returncode if proc else None,error=error,wall_seconds=time.monotonic()-started,
            cleanup_complete=not proc or proc.pid not in after['processes'],external_processes_stopped=0,
            holders_touched=False,holder_restoration_required=False,gpu_after=after)
        save('SUPERVISOR_RESULT.json',result);print(json.dumps(result),flush=True)
    if error or not proc or proc.returncode:raise SystemExit(1)
if __name__=='__main__':main()

