"""Lease an inactive rendering device without stopping any other process."""
import datetime
import json
import os
import resource
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

CODE = Path(__file__).resolve().parent
OUT = CODE / sys.argv[1] if len(sys.argv) > 1 else CODE
assert OUT.resolve().is_relative_to(CODE)
OUT.mkdir(exist_ok=True)
LINE = CODE.parents[1]
ENV = LINE / '.envs/q35n_habitat_v017_g0r'
CACHE = LINE / '.cache/q35n_habitat_v017_g0r'
GPU = 2
UUID = 'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj,indent=2) + '\n')

def gpu_state():
    raw = subprocess.check_output(['nvidia-smi','-i',str(GPU),'-q','-x'],text=True,timeout=15)
    root = ET.fromstring(raw)
    gpu = root.find('gpu')
    assert gpu.findtext('uuid') == UUID
    return {'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'index':GPU,'uuid':UUID,'name':gpu.findtext('product_name'),
            'driver':root.findtext('driver_version'),'memory_used':gpu.findtext('fb_memory_usage/used'),
            'utilization':gpu.findtext('utilization/gpu_util'),
            'processes':[{x.tag:x.text for x in p} for p in gpu.findall('processes/process_info')]}

def main():
    assert not (OUT / 'smoke.result.json').exists()
    before = gpu_state()
    assert float(before['utilization'].split()[0]) == 0, 'GPU gained active work'
    assert float(before['memory_used'].split()[0]) < 1024
    # The two observed small CUDA contexts belong to cuda:0 jobs, not GPU 2 jobs.
    # Recheck their commands rather than declaring arbitrary occupied cards idle.
    existing = {int(p['pid']) for p in before['processes']}
    assert existing <= {2781015,2900192}, 'Unreviewed process; do not displace'
    for pid in sorted(existing):
        args = subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
        assert 'eval.scripts.evaluate_pointgoal' in args and '--device cuda:0' in args
    before['selection_basis'] = '0% active use; <1 GiB resident ancillary contexts of separately verified cuda:0 jobs; no process termination or displacement'
    before['stopped_placeholders'] = []
    save('GPU_LEASE_BEFORE.json',before)
    env = os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES'):
        env.pop(key,None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin', PYTHONNOUSERSITE='1',
               XDG_CACHE_HOME=str(CACHE), NUMBA_CACHE_DIR=str(CACHE/'numba'),
               MPLCONFIGDIR=str(CACHE/'matplotlib'), TMPDIR=str(CACHE/'tmp_recovery'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    cmd = [str(ENV/'bin/python3'),'-I','-B',str(CODE/'renderer_smoke.py'),str(GPU),str(OUT)]
    save('smoke.command.json',{'command':cmd,'gpu_ordinal_is_physical':True,'cuda_visible_devices_unset':True})
    samples = []
    started = time.time()
    rc = None
    with (OUT/'smoke.log').open('x') as log:
        proc = subprocess.Popen(cmd,cwd=LINE,env=env,stdout=log,stderr=subprocess.STDOUT)
        save('SMOKE_PROCESS.json',{'pid':proc.pid,'started_at':before['timestamp']})
        try:
            while proc.poll() is None:
                sample = gpu_state()
                sample['elapsed_seconds'] = time.time()-started
                samples.append(sample)
                assert float(sample['memory_used'].split()[0]) < 8192, 'Conservative whole-device VRAM cap'
                unexpected = {int(p['pid']) for p in sample['processes']} - existing - {proc.pid}
                assert not unexpected, 'Device gained an unrelated process'
                assert time.time()-started < 300, 'Smoke timeout'
                time.sleep(0.2)
            rc = proc.returncode
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            after = gpu_state()
            save('GPU_SAMPLES.json',samples)
            save('GPU_LEASE_AFTER.json',after)
            own_remaining = any(int(p['pid']) == proc.pid for p in after['processes'])
            save('GPU_RESTORE_RESULT.json',{'stopped_placeholders':[], 'restoration_required':False,
                 'new_placeholder_created':False,'own_process_remaining_on_gpu':own_remaining,
                 'cleanup_complete':not own_remaining,'no_external_process_signals_sent':True})
            save('smoke.result.json',{'returncode':proc.returncode,'elapsed_seconds':time.time()-started,
                 'max_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                 'sampled_peak_total_device_mib':max((float(s['memory_used'].split()[0]) for s in samples),default=None),
                 'peak_measurement':'sampled whole-device upper bound, not exact continuous process peak',
                 'renderer_pass':rc == 0 and not own_remaining,'scientific_pass':False})
    print(f'Renderer smoke returncode={rc}; no placeholders were stopped.',flush=True)
    raise SystemExit(rc)

if __name__ == '__main__':
    main()
