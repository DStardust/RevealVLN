"""Fresh-batch launcher: bounded genuine first-query readiness, unchanged guards.

Importing this module does not query GPUs or launch any process. Only run_main
does so after validating a fresh direct child of this batch-execution directory.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import traceback
import types

HERE=Path(__file__).resolve().parent
WF=HERE.parent
RUNTIME=WF.parent
WAIT_SECONDS=60.0


class FirstIdleQuery:
    """Return the actual first idle sample; subsequent calls use original gpu()."""
    def __init__(self,normal_query,initial_query,check_gpu,record,*,clock=time.monotonic,
                 sleeper=time.sleep,poll_seconds=1.0,wait_seconds=WAIT_SECONDS):
        if not 0<poll_seconds<=5 or wait_seconds!=60.0:raise ValueError('READINESS_BOUNDS')
        self.normal_query=normal_query;self.initial_query=initial_query
        self.check_gpu=check_gpu;self.record=record;self.clock=clock;self.sleeper=sleeper
        self.poll_seconds=poll_seconds;self.wait_seconds=wait_seconds
        self.completed=False;self.attempted=False;self.samples=0;self.elapsed=None

    def __call__(self):
        if self.completed:return self.normal_query()
        if self.attempted:raise RuntimeError('READINESS_FAILED_NO_AUTOMATIC_RETRY')
        self.attempted=True;started=self.clock();deadline=started+self.wait_seconds
        while True:
            remaining=deadline-self.clock()
            if remaining<=0:
                self.elapsed=self.clock()-started
                self.record({'event':'readiness_timeout','elapsed_seconds':self.elapsed,'samples':self.samples})
                raise TimeoutError('FIRST_REAL_IDLE_NOT_OBSERVED_WITHIN_60_SECONDS')
            snapshot=None
            try:
                # The first-query subprocess timeout is shortened to remaining
                # readiness time; never issue a fresh 15s query at t=59s.
                snapshot=self.initial_query(min(15.0,remaining))
                self.samples+=1
                upper=self.check_gpu(snapshot)
            except BaseException as error:
                self.elapsed=self.clock()-started
                self.record({'event':'readiness_query_or_guard_error','elapsed_seconds':self.elapsed,
                             'samples':self.samples,'error_type':type(error).__name__,'error':str(error),
                             'snapshot':locals().get('snapshot')})
                raise
            self.elapsed=self.clock()-started
            self.record({'event':'readiness_sample','elapsed_seconds':self.elapsed,
                         'sample_number':self.samples,'snapshot':snapshot,
                         'original_check_gpu_upper_mib':upper})
            # Include query/check/receipt time; never accept a late idle sample.
            if self.clock()>=deadline:
                self.elapsed=self.clock()-started
                self.record({'event':'readiness_timeout','elapsed_seconds':self.elapsed,'samples':self.samples})
                raise TimeoutError('FIRST_REAL_IDLE_NOT_OBSERVED_WITHIN_60_SECONDS')
            if snapshot['utilization']==0:
                self.elapsed=self.clock()-started
                self.completed=True
                return snapshot
            self.sleeper(min(self.poll_seconds,max(0.0,deadline-self.clock())))


def _load_shared():
    spec=importlib.util.spec_from_file_location('readiness_frozen_shared',HERE/'shared.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def build_supervisor(batch,cfg):
    shared=_load_shared()
    assert cfg['supervision_wall_seconds']==3900 and cfg['budget']['total_seconds']==3600
    gpu=cfg['gpu_device'];assert cfg['gpu_uuid']==shared.GPUS[gpu]
    source=RUNTIME/'compact_loop_v2/run.py'
    module=types.ModuleType('bounded_first_idle_original_supervisor');module.__file__=str(source)
    # Exact same source adapter as shared.run_main. No guard/source replacement.
    exec(compile(shared.supervisor_source(source.read_text(),gpu),str(source),'exec'),module.__dict__)
    module.HERE=batch;module.OUT=batch/'run_v1';module.LINE=RUNTIME.parents[1]
    module.ENV=module.LINE/'.envs/q35n_habitat_v017_g0r';module.UUID=shared.GPUS[gpu]
    return module


def validate_target(batch):
    given=Path(batch)
    batch=given.resolve(strict=True)
    if batch.parent!=HERE.resolve() or not re.fullmatch(r'batch_[0-9]+(?:r[0-9]+)?',batch.name):
        raise ValueError('BATCH_MUST_BE_NAMED_DIRECT_CHILD')
    if given.is_symlink() or not batch.is_dir():raise ValueError('BATCH_DIRECTORY_REQUIRED')
    out=batch/'run_v1'
    if out.is_symlink() or not out.is_dir() or out.resolve()!=out:raise ValueError('RUN_DIRECTORY_REQUIRED')
    protected=('SUPERVISOR.lock','GPU_BEFORE.json','PROCESS.json','worker.log','SUPERVISOR_RESULT.json',
               'LAUNCH_RESULT.json','LAUNCH_RESERVATION.json','READINESS_SAMPLES.jsonl')
    if any((out/name).exists() for name in protected):raise ValueError('FRESH_BATCH_ONLY_OLD_ATTEMPT_PRESERVED')
    return batch,out


def _save_new(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())


def run_main(batch,*,module_builder=build_supervisor,clock=time.monotonic,sleeper=time.sleep):
    # Out-of-scope or already-attempted targets are rejected without writing into
    # them. Every new in-scope admitted launch gets an independent outer receipt.
    batch,out=validate_target(batch)
    _save_new(out/'LAUNCH_RESERVATION.json',{'launcher':'readiness_v1','pid':os.getpid(),
        'readiness_limit_seconds':60,'original_supervision_limit_seconds':3900})
    started=clock();readiness=None;error=None;success=False;module=None
    try:
        cfg=json.loads((out/'EXECUTION_CONFIG.json').read_text())
        assert cfg['runtime_allowed'] and cfg['executable'] and not cfg['training_allowed'],'RUNTIME_CONFIG_NOT_APPROVED'
        module=module_builder(batch,cfg)
        original_gpu=module.gpu
        with (out/'READINESS_SAMPLES.jsonl').open('x') as receipt:
            def record(value):
                receipt.write(json.dumps(value,allow_nan=False)+'\n');receipt.flush();os.fsync(receipt.fileno())
            def actual_query(timeout):
                raw=module.subprocess.check_output(['nvidia-smi','-i',str(cfg['gpu_device']),'-q','-x'],
                    text=True,timeout=timeout)
                return module.parse_gpu(raw)
            readiness=FirstIdleQuery(original_gpu,actual_query,module.check_gpu,record,clock=clock,sleeper=sleeper)
            module.gpu=readiness
            module.main()
        success=True
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        raise
    finally:
        receipt={'launcher':'readiness_v1','status':'ORIGINAL_SUPERVISOR_RETURNED' if success else 'LAUNCH_FAILED',
            'exception':error,'wall_seconds':clock()-started,
            'readiness_limit_seconds':60,'original_supervisor_limit_seconds':3900,
            'nominal_readiness_plus_supervision_seconds':3960,
            'original_bounded_cleanup_grace_unchanged':True,
            'first_real_idle_observed':bool(readiness and readiness.completed),
            'readiness_samples':readiness.samples if readiness else 0,
            'readiness_elapsed_seconds':readiness.elapsed if readiness else None,
            'process_record_present':(out/'PROCESS.json').is_file(),
            'supervisor_result_present':(out/'SUPERVISOR_RESULT.json').is_file(),
            'cleanup_authority':'original_supervisor_only_no_outer_process_termination',
            'external_processes_stopped_by_launcher':0,'guards_relaxed':False,
            'raw_gpu_samples_not_modified':True,'scientific_pass':False}
        _save_new(out/'LAUNCH_RESULT.json',receipt)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('batch',type=Path)
    run_main(parser.parse_args().batch)
