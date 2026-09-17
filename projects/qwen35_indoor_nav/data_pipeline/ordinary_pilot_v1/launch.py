"""Bounded GPU lease; signals only its own worker, always records cleanup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ROOT=LINE.parents[1]
ENV=LINE/'.envs/q35n_habitat_v017_g0r'


def save(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2)


def gpu():
    g=ET.fromstring(subprocess.check_output(['nvidia-smi','-i','3','-q','-x'],text=True,timeout=15)).find('gpu')
    assert g.findtext('uuid')=='GPU-a62dba8b-285b-57e6-b6d2-cf6e5788864a'
    return {'uuid':g.findtext('uuid'),'memory_mib':float(g.findtext('fb_memory_usage/used').split()[0]),
            'utilization':float(g.findtext('utilization/gpu_util').split()[0]),
            'processes':[{x.tag:x.text for x in p} for p in g.findall('processes/process_info')]}


def main():
    assert not (OUT/'GPU_BEFORE.json').exists(),'New execution requires a new version'
    for name in ['Q35N_G0R_DEPENDENCY_RECOVERY_V1','Q35N_G1F_MINIMAL_FAMILY_REPLAY_ACCEPTANCE_V2']:
        r=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=LINE/'reviews'/name,capture_output=True,text=True)
        assert r.returncode==0,r.stdout+r.stderr
    fp=json.loads((LINE/'reviews/Q35N_G0R_DEPENDENCY_RECOVERY_V1/ENVIRONMENT_FINGERPRINT.json').read_text())
    assert hashlib.sha256((ENV/'lib/python3.10/_sysconfigdata__linux_x86_64-linux-gnu.py').read_bytes()).hexdigest()==fp['sysconfig_sha256']
    # Prepared files are immutable launch inputs; syntax/loader tests must have passed.
    assert json.loads((OUT/'TEST_RESULT.json').read_text())['pass']
    locked={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file()}
    save('GPU3_CODE_AND_INPUT_LOCK.json',locked)
    before=gpu();assert before['utilization']==0 and before['memory_mib']<1024
    existing={int(p['pid']) for p in before['processes']}
    # Existing ancillary contexts stay untouched; no PID is ever signaled by this check.
    for pid in existing:
        args=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
        assert '--device cuda:0' in args and 'eval.scripts.evaluate_pointgoal' in args
    save('GPU_BEFORE.json',before)
    cache=OUT/'cache';cache.mkdir(exist_ok=True)
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES'):env.pop(k,None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin',PYTHONNOUSERSITE='1',XDG_CACHE_HOME=str(cache),
        NUMBA_CACHE_DIR=str(cache/'numba'),MPLCONFIGDIR=str(cache/'matplotlib'),TMPDIR=str(cache),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    samples=[];started=time.time();error=None
    with (OUT/'worker.log').open('x') as log:
        proc=subprocess.Popen([str(ENV/'bin/python3'),'-I','-B',str(OUT/'worker.py')],cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT)
        save('WORKER_PROCESS.json',{'pid':proc.pid,'started_unix':started})
        try:
            while proc.poll() is None:
                s=gpu();s['elapsed_seconds']=time.time()-started;samples.append(s)
                assert s['memory_mib']<8192
                assert {int(p['pid']) for p in s['processes']}<=existing|{proc.pid}
                if proc.poll() is None:
                    ps=subprocess.run(['ps','-p',str(proc.pid),'-o','rss='],capture_output=True,text=True)
                    rss=int(ps.stdout.strip() or '0');s['child_rss_kib']=rss;assert rss<16*1024**2
                assert time.time()-started<3600
                if len(samples)%10==0:
                    size=int(subprocess.check_output(['du','-sb',str(OUT)],text=True).split()[0]);assert size<20*1024**3
                time.sleep(2)
        except BaseException as e:
            error=repr(e)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            after=gpu();save('GPU_AFTER.json',after);save('RESOURCE_SAMPLES.json',samples)
            save('EXECUTION_RESULT.json',{'returncode':proc.returncode,'runner_error':error,
                'wall_seconds':time.time()-started,'cleanup_complete':not any(int(p['pid'])==proc.pid for p in after['processes']),
                'other_processes_stopped':0,'placeholder_restoration_required':False})
    print(json.dumps({'worker_returncode':proc.returncode,'runner_error':error}))
    if error or proc.returncode:raise SystemExit(1)


if __name__=='__main__':
    # Lease module also restores occupancy if main fails before worker creation.
    sys.path.insert(0,str(OUT))
    import lease
    lease.acquire()
    try:main()
    finally:lease.restore()
