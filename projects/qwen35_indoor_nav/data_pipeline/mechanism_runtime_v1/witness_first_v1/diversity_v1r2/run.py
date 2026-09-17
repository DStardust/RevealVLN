"""Independent GPU1 supervisor adapter; no GPU/locks/launch side effects on import."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fcntl
import hashlib
import json
import os
import types

from common import HERE, LINE, OUT, RUNTIME, WF, load, save, verify_inputs


def build_supervisor():
    path=RUNTIME/'compact_loop_v2/run.py'
    source=path.read_text()
    changes={"sample['elapsed'] < 3000":"sample['elapsed'] < 3900",
             'snapshot=gpu(); upper=check_gpu(snapshot,proc.pid)':'snapshot,upper=_active_sample(proc.pid,started)',
             'sample=dict(snapshot, own_memory_upper_mib=upper, elapsed=time.monotonic()-started)':
             'sample=dict(snapshot, own_memory_upper_mib=upper, elapsed=time.monotonic()-started, active_accounting_sample_index=_active_count)'}
    for old,new in changes.items():
        assert source.count(old)==1
        source=source.replace(old,new)
    module=types.ModuleType('diversity_supervisor');module.__file__=str(path)
    exec(compile(source,str(HERE/'run.py')+'::sealed_supervisor_adapter','exec'),module.__dict__)
    module.HERE=HERE;module.OUT=OUT;module.LINE=LINE
    module.ENV=LINE/'.envs/q35n_habitat_v017_g0r';module._active_count=0
    telemetry=load('diversity_existing_active_telemetry',WF/'special_scale_transport_v1/telemetry.py')
    def sample(worker,started):
        module._active_count+=1;index=module._active_count
        def record(kind,value):
            event=dict(sample_index=index,kind=kind,monotonic=module.time.monotonic(),
                       supervisor_started_monotonic=started,payload=value)
            with (OUT/'ACTIVE_ACCOUNTING.jsonl').open('a') as stream:
                stream.write(json.dumps(event)+'\n');stream.flush();os.fsync(stream.fileno())
        raw=module.subprocess.check_output(['nvidia-smi','-i','1','-q','-x'],text=True,timeout=15)
        record('raw_xml',dict(xml=raw,sha256=hashlib.sha256(raw.encode()).hexdigest()))
        snapshot=module.parse_gpu(raw);record('parsed_snapshot',snapshot)
        result=telemetry.assess(snapshot,worker,module.check_gpu,lambda value:record('decision',value))
        assert module.time.monotonic()-started<3900
        return snapshot,result['guard_upper_mib']
    module._active_sample=sample
    return module


def main():
    verify_inputs()
    mutex=LINE/'data_pipeline/auto_production_v1/gpu_locks/gpu_1.lock'
    with mutex.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        module=build_supervisor()
        # No unknown process or holder is borrowed for this bounded batch.
        before=module.gpu()
        assert not before['processes'] and before['utilization']==0,'GPU1_HAS_EXISTING_USER'
        save(OUT/'LAUNCH_RESULT.json',dict(accounting_acceptance_amended=True,
             idle_final_guard_unchanged=True,external_processes_stopped=0,holders_touched=False,
             status='LAUNCH_ADMITTED_NOT_COMPLETED',scientific_pass=False))
        module.main()


if __name__=='__main__': main()

