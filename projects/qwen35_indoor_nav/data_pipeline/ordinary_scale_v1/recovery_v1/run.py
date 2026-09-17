import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
LINE=BASE.parents[1]
ENV=LINE/'.envs/q35n_habitat_v017_g0r'
s=importlib.util.spec_from_file_location('supervisor_helpers',BASE/'run.py')
old=importlib.util.module_from_spec(s);s.loader.exec_module(old)

def main():
    lock=(BASE/'PRODUCER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=HERE/'run';out.mkdir(exist_ok=False)
    before=old.gpu();old.save(out/'GPU_BEFORE.json',before)
    assert before['utilization']==0 and before['memory_mib']<1024
    existing=set(before['processes']);cache=out/'cache';cache.mkdir()
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin',PYTHONNOUSERSITE='1',XDG_CACHE_HOME=str(cache),
        NUMBA_CACHE_DIR=str(cache/'numba'),MPLCONFIGDIR=str(cache/'matplotlib'),TMPDIR=str(cache),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def interrupted(signum,frame):raise InterruptedError(str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    started=time.time();proc=None;error=None;samples=0
    try:
        with (out/'worker.log').open('x') as log:
            proc=subprocess.Popen([str(ENV/'bin/python3'),'-I','-B',str(HERE/'worker.py')],
                cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            old.save(out/'PROCESS.json',dict(pid=proc.pid,supervisor_pid=os.getpid(),started_unix=started))
            while proc.poll() is None:
                time.sleep(5);g=old.gpu();samples+=1
                assert set(g['processes'])<=existing|{proc.pid}
                assert g['processes'].get(proc.pid,0)<4096
                rss=subprocess.run(['ps','-p',str(proc.pid),'-o','rss='],capture_output=True,text=True).stdout.strip()
                assert int(rss or 0)<12*1024**2
                assert time.time()-started<14250
                if samples%12==0:assert int(old.call('du','-sb',str(BASE)).split()[0])<200*1024**3
                with (out/'RESOURCE_SAMPLES.jsonl').open('a') as f:
                    f.write(json.dumps(dict(g,elapsed=time.time()-started))+'\n')
    except BaseException as ex:error=repr(ex)
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
        after=old.gpu()
        result=dict(returncode=proc.returncode if proc else None,error=error,
            wall_seconds=time.time()-started,cleanup_complete=not proc or proc.pid not in after['processes'],
            holders_touched=False,holder_restoration_required=False,external_processes_stopped=0,
            gpu_after=after,scientific_pass=False)
        old.save(out/'RESULT.json',result);print(json.dumps(result),flush=True)
    if error or not proc or proc.returncode:raise SystemExit(1)

if __name__=='__main__':main()

