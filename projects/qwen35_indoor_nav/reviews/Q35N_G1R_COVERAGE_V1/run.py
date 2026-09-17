import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ENV=LINE/'.envs/q35n_habitat_v017_g0r'


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def save(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2)


def main():
    assert not (OUT/'GPU_BEFORE.json').exists()
    for rel in ['reviews/Q35N_G1F_MINIMAL_FAMILY_REPLAY_ACCEPTANCE_V2','reviews/Q35N_G0R_DEPENDENCY_RECOVERY_V1']:
        r=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=LINE/rel,capture_output=True,text=True);assert r.returncode==0
    gpu=module('gpu_reader',LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/run_probe.py').gpu
    lease=module('lease',LINE/'data_pipeline/ordinary_pilot_v1/lease.py');lease.OUT=OUT
    save('CODE_LOCK.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.py','.md')})
    lease.acquire()
    try:
        before=gpu();save('GPU_BEFORE.json',before);assert before['memory_mib']<1024
        existing={int(p['pid']) for p in before['processes']}
        cache=OUT/'cache';cache.mkdir()
        env=os.environ.copy()
        for k in ['PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES']:env.pop(k,None)
        env.update(PATH=f'{ENV}/bin:/usr/bin:/bin',PYTHONNOUSERSITE='1',XDG_CACHE_HOME=str(cache),
            TMPDIR=str(cache),NUMBA_CACHE_DIR=str(cache/'numba'),MPLCONFIGDIR=str(cache/'mpl'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        started=time.time();samples=[];error=None
        with (OUT/'worker.log').open('x') as log:
            proc=subprocess.Popen([str(ENV/'bin/python3'),'-I','-B',str(OUT/'worker.py')],cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT)
            save('PROCESS.json',{'pid':proc.pid,'started_unix':started})
            try:
                while proc.poll() is None:
                    g=gpu();g['elapsed_seconds']=time.time()-started;samples.append(g)
                    assert g['memory_mib']<8192 and {int(p['pid']) for p in g['processes']}<=existing|{proc.pid}
                    p=subprocess.run(['ps','-p',str(proc.pid),'-o','rss='],capture_output=True,text=True)
                    g['rss_kib']=int(p.stdout.strip() or '0');assert g['rss_kib']<16*1024**2
                    assert time.time()-started<3600
                    if len(samples)%10==0:
                        total=0
                        for f in OUT.rglob('*'):
                            try:
                                if f.is_file():total+=f.stat().st_size
                            except FileNotFoundError:pass
                        assert total<8*1024**3
                    time.sleep(2)
            except BaseException as e:error=repr(e)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
                after=gpu();save('GPU_AFTER.json',after);save('RESOURCE_SAMPLES.json',samples)
                save('EXECUTION.json',{'returncode':proc.returncode,'runner_error':error,'wall_seconds':time.time()-started,
                    'cleanup_complete':not any(int(p['pid'])==proc.pid for p in after['processes'])})
        if error or proc.returncode:raise RuntimeError('Worker failed; preserve recorded results')
    finally:lease.restore()


if __name__=='__main__':main()
