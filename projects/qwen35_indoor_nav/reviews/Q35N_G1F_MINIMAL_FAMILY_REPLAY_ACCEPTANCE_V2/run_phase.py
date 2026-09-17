"""GPU-guarded phase runner. No install, network, model or external process signals."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
LINE=OUT.parents[1]
ROOT=LINE.parents[1]
ENV=LINE/'.envs/q35n_habitat_v017_g0r'

def save(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def gpu():
    r=ET.fromstring(subprocess.check_output(['nvidia-smi','-i','2','-q','-x'],text=True,timeout=15))
    g=r.find('gpu'); assert g.findtext('uuid')=='GPU-be1b30d0-517b-b079-871b-de195d35a1a2'
    return {'uuid':g.findtext('uuid'),'memory_mib':float(g.findtext('fb_memory_usage/used').split()[0]),
            'utilization':float(g.findtext('utilization/gpu_util').split()[0]),
            'processes':[{x.tag:x.text for x in p} for p in g.findall('processes/process_info')]}

def child(phase):
    sys.path.insert(0,str(OUT))
    from engine import Engine,Reject,save
    eng=None
    try:
        eng=Engine(phase)
        if phase=='preview':
            eng.preview(); eng.u_pool()
            decision='WITNESS_PREVIEW_PASS'
        elif phase in ('discovery','discovery_pruned'):
            eng.views=json.loads((OUT/'WITNESS_POOL.json').read_text())
            assert all(eng.views.values())
            candidate=eng.discovery(); decision='FROZEN_CANDIDATE_READY'
        else:raise ValueError(phase)
        save(f'{phase}_RESULT.json',{'decision':decision,'phase_pass':True,'scientific_pass':False})
    except Reject as e:
        save(f'{phase}_RESULT.json',{'decision':e.code,'detail':e.detail,'phase_pass':False,'scientific_pass':False})
        print(f'{phase}: {e.code}',flush=True)
    except KeyboardInterrupt:
        save(f'{phase}_RESULT.json',{'decision':'INTERRUPTED_FOR_EQUIVALENT_PRUNING','phase_pass':False,'scientific_pass':False})
        raise
    except Exception as e:
        save(f'{phase}_RESULT.json',{'decision':'IMPLEMENTATION_ERROR','exception':repr(e),'phase_pass':False,'scientific_pass':False})
        raise
    finally:
        if eng:eng.close()

def main(phase):
    assert phase in ('preview','discovery','discovery_pruned')
    assert not (OUT/f'{phase}_RESULT.json').exists(), 'Preserve completed phase'
    runtime=OUT.parent/'Q35N_G0R_DEPENDENCY_RECOVERY_V1'
    for d in (runtime,OUT.parent/'Q35N_P2R1_SPEC_CORRECTIONS_V1',OUT.parent/'Q35N_P2_DATA_AND_IMPLEMENTATION_PLAN_V1'):
        c=subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=d,capture_output=True,text=True)
        assert c.returncode==0,c.stdout+c.stderr
    fp=json.loads((runtime/'ENVIRONMENT_FINGERPRINT.json').read_text())
    path=ENV/'lib/python3.10/_sysconfigdata__linux_x86_64-linux-gnu.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==fp['sysconfig_sha256']
    scene=json.loads((OUT.parent/'Q35N_G0R_RUNTIME_SETUP_ACCEPTANCE_V1/SCENE_ASSET_FINGERPRINT.json').read_text())
    for r in scene['files']:
        p=Path(scene['root'])/r['name']; assert p.resolve().is_relative_to(ROOT)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    before=gpu(); assert before['utilization']==0 and before['memory_mib']<1024
    existing={int(p['pid']) for p in before['processes']}
    assert existing<={2781015,2900192}
    for pid in existing:
        args=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
        assert '--device cuda:0' in args and 'eval.scripts.evaluate_pointgoal' in args
    save(f'{phase}_GPU_BEFORE.json',before)
    save(f'{phase}_CODE_LOCK.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(OUT.glob('*.py'))})
    cache=OUT/'cache'; cache.mkdir(exist_ok=True)
    env=os.environ.copy()
    for k in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES'):
        env.pop(k,None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin',PYTHONNOUSERSITE='1',XDG_CACHE_HOME=str(cache),
               NUMBA_CACHE_DIR=str(cache/'numba'),MPLCONFIGDIR=str(cache/'matplotlib'),
               TMPDIR=str(cache),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    started=time.time(); samples=[]
    with (OUT/f'{phase}.log').open('x') as log:
        proc=subprocess.Popen([str(ENV/'bin/python3'),'-I','-B',str(OUT/'run_phase.py'),'--child',phase],
                              cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT)
        save(f'{phase}_PROCESS.json',{'pid':proc.pid,'started_unix':started})
        try:
            while proc.poll() is None:
                s=gpu(); s['elapsed_seconds']=time.time()-started; samples.append(s)
                assert s['memory_mib']<8192
                assert {int(p['pid']) for p in s['processes']}<=existing|{proc.pid}
                rss=int(subprocess.check_output(['ps','-p',str(proc.pid),'-o','rss='],text=True).strip() or '0') if proc.poll() is None else 0
                s['child_rss_kib']=rss; assert rss<16*1024**2
                assert time.time()-started<8*3600
                if len(samples)%15==0:
                    total=int(subprocess.check_output(['du','-sb',str(OUT)],text=True).split()[0]); assert total<5*1024**3
                time.sleep(2)
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
            after=gpu();save(f'{phase}_GPU_AFTER.json',after);save(f'{phase}_RESOURCE_SAMPLES.json',samples)
            cleanup=not any(int(p['pid'])==proc.pid for p in after['processes'])
            save(f'{phase}_EXECUTION.json',{'returncode':proc.returncode,'elapsed_seconds':time.time()-started,
                 'max_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                 'cleanup_complete':cleanup,'other_processes_stopped':0,'placeholder_restoration_required':False})
    print((OUT/f'{phase}_RESULT.json').read_text(),flush=True)
    raise SystemExit(proc.returncode)

if __name__=='__main__':
    if sys.argv[1]=='--child':child(sys.argv[2])
    else:main(sys.argv[1])
