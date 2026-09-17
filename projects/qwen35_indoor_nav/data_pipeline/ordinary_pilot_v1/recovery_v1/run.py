import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

OUT=Path(__file__).resolve().parent
PARENT=OUT.parent
LINE=PARENT.parents[1]
ENV=LINE/'.envs/q35n_habitat_v017_g0r'


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


lease=module('lease',PARENT/'lease.py');lease.OUT=OUT
prior=module('previous_launcher',PARENT/'launch.py')


def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2)


def disk_size():
    size=0
    for p in PARENT.rglob('*'):
        try:
            if p.is_file():size+=p.stat().st_size
        except FileNotFoundError:pass  # Atomic PNG promotion may remove the temporary name.
    return size


def main():
    lock=json.loads((PARENT/'GPU3_CODE_AND_INPUT_LOCK.json').read_text())
    for name,h in lock.items():assert hashlib.sha256((PARENT/name).read_bytes()).hexdigest()==h,name
    assert json.loads((PARENT/'LEASE_RESTORED.json').read_text())['restored']
    remaining=3600-json.loads((PARENT/'EXECUTION_RESULT.json').read_text())['wall_seconds']
    save('INPUT_LOCK.json',{str(p.relative_to(PARENT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
         [PARENT/'worker.py',PARENT/'loader.py',PARENT/'JOBS.json',PARENT/'ASSET_LOCK.json',PARENT/'SPLIT_FREEZE.json',PARENT/'LEDGER.jsonl',OUT/'worker.py',OUT/'run.py',OUT/'AMENDMENT_ZH.md']})
    lease.acquire()
    try:
        before=prior.gpu();assert before['utilization']==0 and before['memory_mib']<1024
        existing={int(p['pid']) for p in before['processes']};save('GPU_BEFORE.json',before)
        cache=OUT/'cache';cache.mkdir()
        env=os.environ.copy()
        for k in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
        env.update(PATH=f'{ENV}/bin:/usr/bin:/bin',PYTHONNOUSERSITE='1',XDG_CACHE_HOME=str(cache),
          NUMBA_CACHE_DIR=str(cache/'numba'),MPLCONFIGDIR=str(cache/'matplotlib'),TMPDIR=str(cache),
          OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        started=time.time();samples=[];error=None
        with (OUT/'worker.log').open('x') as log:
            proc=subprocess.Popen([str(ENV/'bin/python3'),'-I','-B',str(OUT/'worker.py')],cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT)
            save('WORKER_PROCESS.json',{'pid':proc.pid,'started_unix':started})
            try:
                while proc.poll() is None:
                    g=prior.gpu();g['elapsed_seconds']=time.time()-started;samples.append(g)
                    assert g['memory_mib']<8192 and {int(p['pid']) for p in g['processes']}<=existing|{proc.pid}
                    ps=subprocess.run(['ps','-p',str(proc.pid),'-o','rss='],capture_output=True,text=True)
                    g['child_rss_kib']=int(ps.stdout.strip() or '0');assert g['child_rss_kib']<16*1024**2
                    assert time.time()-started<remaining
                    if len(samples)%10==0:assert disk_size()<20*1024**3
                    time.sleep(2)
            except BaseException as e:error=repr(e)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
                after=prior.gpu();save('GPU_AFTER.json',after);save('RESOURCE_SAMPLES.json',samples)
                save('EXECUTION_RESULT.json',{'returncode':proc.returncode,'runner_error':error,
                    'wall_seconds':time.time()-started,'cleanup_complete':not any(int(p['pid'])==proc.pid for p in after['processes']),
                    'real_tasks_stopped':0,'placeholder_restoration_required':True})
        print(json.dumps({'worker_returncode':proc.returncode,'runner_error':error}))
        if error or proc.returncode:raise SystemExit(1)
    finally:lease.restore()


if __name__=='__main__':main()
