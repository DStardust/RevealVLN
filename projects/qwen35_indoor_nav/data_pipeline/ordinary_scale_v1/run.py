"""Supervisor: one idle GPU, no external process signals or holder changes."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ENV=LINE/'.envs/q35n_habitat_v017_g0r'
UUID='GPU-be1b30d0-517b-b079-871b-de195d35a1a2'

def call(*args):
    return subprocess.check_output(args,text=True).strip()

def gpu():
    fields=call('nvidia-smi','-i','2','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits').split(',')
    rows=call('nvidia-smi','-i','2','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits')
    processes={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.splitlines() if r.strip()}
    assert fields[0].strip()==UUID
    return dict(uuid=UUID,memory_mib=int(fields[1]),utilization=int(fields[2]),processes=processes)

def save(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2)

def main():
    run_id='run_'+time.strftime('%Y%m%d_%H%M%S')
    out=OUT/'supervision'/run_id;out.mkdir(parents=True,exist_ok=False)
    # Exclusive flock prevents concurrent producers in this immutable batch.
    import fcntl
    lock=(OUT/'PRODUCER.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=gpu();save(out/'GPU_BEFORE.json',before)
    assert before['utilization']==0 and before['memory_mib']<1024
    existing=set(before['processes']);cache=out/'cache';cache.mkdir()
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin',PYTHONNOUSERSITE='1',XDG_CACHE_HOME=str(cache),
        NUMBA_CACHE_DIR=str(cache/'numba'),MPLCONFIGDIR=str(cache/'matplotlib'),TMPDIR=str(cache),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def interrupted(signum,frame):raise InterruptedError('SUPERVISOR_SIGNAL:'+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    started=time.time();error=None;proc=None;samples=0
    try:
        with (out/'worker.log').open('x') as log:
            proc=subprocess.Popen([str(ENV/'bin/python3'),'-I','-B',str(OUT/'worker.py')],
                cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(out/'PROCESS.json',dict(pid=proc.pid,supervisor_pid=os.getpid(),started_unix=started))
            while proc.poll() is None:
                time.sleep(5);g=gpu();samples+=1
                assert set(g['processes'])<=existing|{proc.pid},'NEW_EXTERNAL_GPU_PROCESS'
                assert g['processes'].get(proc.pid,0)<4096,'GPU_CAP'
                rss=subprocess.run(['ps','-p',str(proc.pid),'-o','rss='],capture_output=True,text=True).stdout.strip()
                assert int(rss or 0)<12*1024**2,'RAM_CAP'
                assert time.time()-started<14400,'WALL_CAP'
                if samples%12==0:
                    size=int(call('du','-sb',str(OUT)).split()[0])
                    assert size<200*1024**3,'DISK_CAP'
                with (out/'RESOURCE_SAMPLES.jsonl').open('a') as f:
                    f.write(json.dumps(dict(g,elapsed=time.time()-started))+'\n')
    except BaseException as ex:error=repr(ex)
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        after=gpu()
        result=dict(returncode=proc.returncode if proc else None,error=error,
            wall_seconds=time.time()-started,cleanup_complete=not proc or proc.pid not in after['processes'],
            holders_touched=False,holder_restoration_required=False,external_processes_stopped=0,
            gpu_after=after,scientific_pass=False)
        save(out/'RESULT.json',result)
        print(json.dumps(result),flush=True)
    if error or not proc or proc.returncode:raise SystemExit(1)

if __name__=='__main__':main()
