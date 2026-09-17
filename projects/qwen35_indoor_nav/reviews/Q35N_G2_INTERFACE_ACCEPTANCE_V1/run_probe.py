"""One guarded model process, leased GPU3, finally restore only verified occupancy."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ENV=LINE/'.envs/q35n_qwen_g2_v1'
DATA=LINE/'data_pipeline/ordinary_pilot_v1'
MODEL=LINE/'runtime/models/Qwen3.5-2B_15852e8'


def save(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,indent=2)


def gpu():
    g=ET.fromstring(subprocess.check_output(['nvidia-smi','-i','3','-q','-x'],text=True,timeout=15)).find('gpu')
    assert g.findtext('uuid')=='GPU-a62dba8b-285b-57e6-b6d2-cf6e5788864a'
    return {'uuid':g.findtext('uuid'),'memory_mib':float(g.findtext('fb_memory_usage/used').split()[0]),
        'processes':[{x.tag:x.text for x in p} for p in g.findall('processes/process_info')]}


def main():
    assert not (OUT/'GPU_BEFORE.json').exists()
    assert json.loads((OUT/'setup_1_result.json').read_text())['returncode']==0
    files=json.loads((OUT/'MODEL_FILE_LOCK.json').read_text())+json.loads((OUT/'MODEL_SUPPLEMENT_LOCK.json').read_text())
    for r in files:
        p=MODEL/r['file'];h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        assert h.hexdigest()==r['sha256']
    for directory in [DATA,LINE/'reviews/Q35N_G0R_DEPENDENCY_RECOVERY_V1',LINE/'reviews/Q35N_P2R1_SPEC_CORRECTIONS_V1']:
        r=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=directory,capture_output=True,text=True)
        assert r.returncode==0
    save('PROBE_CODE_LOCK.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')})
    s=importlib.util.spec_from_file_location('verified_lease',DATA/'lease.py');lease=importlib.util.module_from_spec(s);s.loader.exec_module(lease);lease.OUT=OUT
    lease.acquire()
    try:
        before=gpu();assert before['memory_mib']<1024
        existing={int(p['pid']) for p in before['processes']};save('GPU_BEFORE.json',before)
        cache=LINE/'.cache/q35n_qwen_g2_v1';env=os.environ.copy()
        for k in ('PYTHONPATH','PYTHONHOME','CONDA_PREFIX','LD_LIBRARY_PATH'):env.pop(k,None)
        env.update(PATH=f'{ENV}/bin:/usr/bin:/bin',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='3',
            HF_HOME=str(cache/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
            XDG_CACHE_HOME=str(cache),TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),
            TMPDIR=str(cache),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        samples=[];started=time.time();error=None
        with (OUT/'probe.log').open('x') as log:
            proc=subprocess.Popen([str(ENV/'bin/python3'),'-I','-B',str(OUT/'probe.py')],cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT)
            save('PROBE_PROCESS.json',{'pid':proc.pid,'physical_gpu':3,'logical_gpu':0,'started_unix':started})
            try:
                while proc.poll() is None:
                    state=gpu();state['elapsed_seconds']=time.time()-started
                    assert state['memory_mib']<28*1024 and {int(p['pid']) for p in state['processes']}<=existing|{proc.pid}
                    ps=subprocess.run(['ps','-p',str(proc.pid),'-o','rss='],capture_output=True,text=True)
                    state['rss_kib']=int(ps.stdout.strip() or '0');assert state['rss_kib']<48*1024**2
                    samples.append(state);assert time.time()-started<1800
                    time.sleep(2)
            except BaseException as e:error=repr(e)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
                after=gpu();save('GPU_AFTER.json',after);save('RESOURCE_SAMPLES.json',samples)
                save('EXECUTION_RESULT.json',{'returncode':proc.returncode,'runner_error':error,
                    'wall_seconds':time.time()-started,'worker_cleanup':not any(int(p['pid'])==proc.pid for p in after['processes']),
                    'real_tasks_stopped':0,'occupancy_restore_required':True})
        if proc.returncode or error:raise RuntimeError('Probe failed; original log and partial stages preserved')
    finally:lease.restore()


if __name__=='__main__':main()
