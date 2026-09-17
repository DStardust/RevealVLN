"""GPU5 exact-holder lease for unchanged winding batches. Import is CPU only.

Only main-agent-approved run_main issues GPU/tmux queries or signals. The
stdlib lacks pidfd: consecutive identity checks reduce, not eliminate, PID races.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import types

HERE=Path(__file__).resolve().parent
BE=HERE.parent
WF=BE.parent
RUNTIME=WF.parent
LINE=RUNTIME.parents[1]
ROOT=LINE.parents[1]
UUID='GPU-2b3b2c3a-7398-a845-293a-d2844dd7524b'
HOLDER_PID=3133144
PANE='vla_idle_occupancy_20260904:2.0'
PANE_ID='%150'
COMMAND='.envs/etpr1/bin/python -u scripts/occupy_idle_gpu.py --gpu 5 --reserve-mib 2048 --max-initial-used-mib 1024 --tag vla_idle_occupancy_20260904'
SOURCE_LOCK={
    BE/'shared.py':'3b717bdf07a53ce06191a2445a20e86982073c17b43412b30b1332c8e01a53b0',
    BE/'readiness_v1.py':'6ad5b369845917a57963ce2fb4718bf2580b8d87a408c8f467cb70ddc61b7600',
    BE/'prepare.py':'14b0288ccdc027df9556ef3ff9cb1b478c3fd0c9158e6587a1a6432d8278d718',
    RUNTIME/'compact_loop_v2/run.py':'9aabce69787bc717f6283b7679fef3c2f811018f0456741dc51516ebe6e986ca',
}

def require(value,reason):
    if not value:raise ValueError(reason)
def sha(path):
    path=Path(path);require(path.resolve()==path and path.is_relative_to(ROOT),'SOURCE_SCOPE')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
    with Path(path).open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def verify_sources():
    for p,h in SOURCE_LOCK.items():require(sha(p)==h,'FROZEN_TRANSPORT_SOURCE_CHANGED:'+str(p))

def supervisor_source(source):
    """Apply the unchanged approved adapter, then exactly one GPU-index rewrite."""
    shared=load('gpu5_frozen_shared',BE/'shared.py')
    result=shared.supervisor_source(source,1)
    old="'nvidia-smi','-i','1'"
    require(result.count(old)==1,'GPU5_SUPERVISOR_SUBSTITUTION_COUNT')
    return result.replace(old,"'nvidia-smi','-i','5'")

def build_supervisor(batch,cfg):
    verify_sources()
    require(cfg['gpu_device']==5 and cfg['gpu_uuid']==UUID,'GPU5_CONFIG_IDENTITY')
    require(cfg['factory_variant']=='winding_v1','UNCHANGED_WINDING_ONLY')
    require(cfg['supervision_wall_seconds']==3900 and cfg['budget']['total_seconds']==3600,'BUDGET_CHANGED')
    path=RUNTIME/'compact_loop_v2/run.py'
    module=types.ModuleType('gpu5_original_supervisor');module.__file__=str(path)
    exec(compile(supervisor_source(path.read_text()),str(path),'exec'),module.__dict__)
    module.HERE=Path(batch);module.OUT=module.HERE/'run_v1'
    module.LINE=LINE;module.ENV=LINE/'.envs/q35n_habitat_v017_g0r';module.UUID=UUID
    module._gpu5_raw_gpu=module.gpu
    module._gpu5_child=None
    original_subprocess=module.subprocess
    proxy=types.SimpleNamespace(**{name:getattr(original_subprocess,name) for name in dir(original_subprocess)})
    def own_popen(*args,**kwargs):
        require(module._gpu5_child is None and kwargs.get('start_new_session') is True,'ONE_OWN_SESSION_WORKER')
        proc=original_subprocess.Popen(*args,**kwargs);module._gpu5_child=proc;return proc
    proxy.Popen=own_popen;module.subprocess=proxy
    return module

def normalize_identity(document):
    require(document.get('project_root')==str(ROOT) and document.get('gpu_device')==5
            and document.get('gpu_uuid')==UUID,'FROZEN_GPU_SCOPE')
    require(document.get('cmdline')==COMMAND.split() and document.get('proc_uid')==0,'FROZEN_ARGV_UID')
    require(document.get('pane_pid')==HOLDER_PID and document.get('pane_dead') is False,'FROZEN_PANE_STATE')
    require(document.get('restore_same_cwd_and_cmdline') is True and
            document.get('revalidate_all_fields_before_signal') is True and
            document.get('unknown_or_nonholder_process_signal_allowed') is False,'FROZEN_SAFETY_SCOPE')
    identity={'pid':document['pid'],'starttime_ticks':document['starttime_ticks'],'command':' '.join(document['cmdline']),
              'cwd':document['cwd'],'pane':document['pane_target'],'pane_id':document['pane_id'],'proc_uid':document['proc_uid']}
    validate_identity(identity);return identity

def validate_identity(identity):
    expected={'pid':HOLDER_PID,'command':COMMAND,'cwd':str(ROOT),'pane':PANE,'pane_id':PANE_ID,'proc_uid':0}
    require(all(identity.get(k)==v for k,v in expected.items()),'FROZEN_HOLDER_IDENTITY')
    require(type(identity.get('starttime_ticks')) is int and identity['starttime_ticks']>0,'STARTTIME_REQUIRED')

def validate_authorization(document,batch,snapshot,indices,identity_path):
    expected={'approved':True,'gpu_device':5,'gpu_uuid':UUID,'supervision_wall_seconds':3900,
              'factory_wall_seconds':3600,'max_actions':60000,'per_candidate_discovery_seconds':1000,
              'per_candidate_discovery_actions':15000,'per_candidate_certification_seconds':1500,
              'per_candidate_certification_actions':20000,'first_idle_wait_seconds':60,
              'training_allowed':False,'replace_failed_candidate_within_batch':False,
              'restore_holder_on_success_failure_or_interruption':True,
              'original_semantic_physical_query_count_and_resource_thresholds_unchanged':True}
    require(all(document.get(k)==v for k,v in expected.items()),'AUTHORIZATION_LIMITS_OR_APPROVAL')
    for key,path in [('batch',batch),('snapshot',snapshot),('holder_identity',identity_path)]:
        require(document.get(key)==str(Path(path).relative_to(LINE)),'AUTHORIZATION_PATH:'+key)
    require(document.get('snapshot_indices')==indices,'AUTHORIZATION_SELECTION')

class RealOps:
    """No operations happen at construction; injectable for CPU-only tests."""
    clock=staticmethod(time.monotonic)
    sleep=staticmethod(time.sleep)
    def call(self,*args):return subprocess.check_output(args,text=True,timeout=15).strip()
    def pane(self,fmt):return self.call('tmux','display-message','-p','-t',PANE,fmt)
    def proc(self,pid):
        root=Path('/proc')/str(pid)
        # stat field 22: tokens following the final ')' begin with field 3.
        stat=(root/'stat').read_text();start=int(stat.rsplit(')',1)[1].split()[19])
        cmd=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        uid=int(next(x for x in (root/'status').read_text().splitlines() if x.startswith('Uid:')).split()[1])
        return {'pid':pid,'starttime_ticks':start,'command':cmd,'cwd':str((root/'cwd').resolve(strict=True)),'proc_uid':uid}
    def exists(self,pid):return (Path('/proc')/str(pid)).exists()
    def identity(self):
        pid=int(self.pane('#{pane_pid}'));result=self.proc(pid)
        result.update(pane=PANE,pane_id=self.pane('#{pane_id}'))
        require(self.pane('#{pane_current_path}')==str(ROOT),'PANE_CWD')
        require(self.pane('#{pane_dead}')=='0','HOLDER_PANE_DEAD')
        return result
    def remain(self):return self.call('tmux','show-options','-w','-v','-t',PANE,'remain-on-exit')
    def set_remain(self,value):self.call('tmux','set-option','-w','-t',PANE,'remain-on-exit',value)
    def terminate(self,pid):os.kill(pid,signal.SIGTERM)
    def restore_command(self):
        # No -k: even an identity race cannot kill an unexpected live pane.
        self.call('tmux','respawn-pane','-t',PANE,'-c',str(ROOT),COMMAND)
    def gpu(self,module):return module._gpu5_raw_gpu()
    def cleanup_own(self,module):
        proc=module._gpu5_child
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)

def identity_equal(actual,frozen):
    return all(actual.get(k)==frozen.get(k) for k in ('pid','starttime_ticks','command','cwd','pane','pane_id','proc_uid'))

def run_lease(out,identity,module,execute,*,ops=None,signals=signal):
    """All post-admission paths run restoration; execute cleans its own worker."""
    ops=ops or RealOps();out=Path(out);validate_identity(identity)
    original_handlers={s:signals.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    attempted=False;remain=None;remain_changed=False;error=None;restoration=None;success=False
    def interrupted(signum,frame):raise InterruptedError('GPU5_LEASE_SIGNAL:'+str(signum))
    for s in original_handlers:signals.signal(s,interrupted)
    try:
        require(identity_equal(ops.identity(),identity),'HOLDER_IDENTITY_CHANGED')
        before=ops.gpu(module)
        require(before['uuid']==UUID and HOLDER_PID in before['processes'],'HOLDER_GPU_IDENTITY')
        require(before['processes'][HOLDER_PID]['mib']>20000,'HOLDER_NOT_OCCUPYING_EXPECTED_GPU')
        external_values=[v['mib'] for p,v in before['processes'].items() if p!=HOLDER_PID]
        require(all(0<=m<=768 for m in external_values) and sum(external_values)<=2048,'EXTERNAL_RESOURCE_LOAD')
        require(sum(v['mib'] for v in before['processes'].values())<=before['memory_mib'],'MEMORY_ACCOUNTING')
        external=sum(external_values)
        require(external<1024,'RESTORE_EXTERNAL_MEMORY_BUDGET')
        remain=ops.remain();require(remain in ('on','off','failed'),'TMUX_REMAIN_VALUE')
        save(out/'LEASE_BEFORE.json',{'identity':identity,'gpu':before,'remain_on_exit':remain,
            'restore_argv':['tmux','respawn-pane','-t',PANE,'-c',str(ROOT),COMMAND],
            'restore_no_force_kill':True,'pidfd_available':False,
            'residual_race':'double /proc identity checks then exact PID SIGTERM; no atomic pidfd'})
        remain_changed=True;ops.set_remain('on')
        require(identity_equal(ops.identity(),identity),'HOLDER_IDENTITY_CHANGED_BEFORE_SIGNAL')
        require(identity_equal(ops.identity(),identity),'HOLDER_IDENTITY_CHANGED_FINAL_CHECK')
        attempted=True;ops.terminate(HOLDER_PID)
        for _ in range(80):
            if not ops.exists(HOLDER_PID):break
            # Never signal a replacement PID, including during timeout cleanup.
            require(ops.proc(HOLDER_PID)['starttime_ticks']==identity['starttime_ticks'],'HOLDER_PID_REUSED')
            ops.sleep(.25)
        require(not ops.exists(HOLDER_PID),'HOLDER_EXIT_FAILED_NO_ESCALATION')
        save(out/'LEASE_ACTIVE.json',{'holder_exited':True,'pane':PANE,'pane_id':PANE_ID})
        execute();success=True
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
    finally:
        for s in original_handlers:signals.signal(s,signals.SIG_IGN)
        try:
            # The original supervisor normally already cleaned up. This fallback
            # only knows the Popen object created by this exact supervisor.
            ops.cleanup_own(module)
            if not attempted:
                restoration={'restored':True,'not_borrowed':True}
            elif ops.exists(HOLDER_PID) and identity_equal(ops.identity(),identity):
                restoration={'restored':True,'original_holder_retained':True,'pid':HOLDER_PID}
            else:
                # The original supervisor must have completed own-worker cleanup.
                process=read(out/'PROCESS.json') if (out/'PROCESS.json').exists() else None
                if process:require(not ops.exists(process['pid']),'OWN_WORKER_STILL_ALIVE_NO_RESTORE')
                require(ops.pane('#{pane_id}')==PANE_ID and ops.pane('#{pane_dead}')=='1','PANE_NOT_OWN_DEAD_HOLDER')
                require(int(ops.pane('#{pane_pid}'))==HOLDER_PID,'DEAD_PANE_IDENTITY_CHANGED')
                snapshot=ops.gpu(module);module.check_gpu(snapshot)
                require(snapshot['memory_mib']<1024,'RESTORE_INITIAL_GPU_MEMORY_TOO_HIGH')
                save(out/'GPU_PRE_RESTORE.json',snapshot)
                ops.restore_command()
                restored_pid=int(ops.pane('#{pane_pid}'));observed=None;passed=False
                for _ in range(60):
                    ops.sleep(.5)
                    if not ops.exists(restored_pid):break
                    observed=ops.proc(restored_pid)
                    require(observed['command']==COMMAND and observed['cwd']==str(ROOT)
                            and observed['proc_uid']==identity['proc_uid'],'RESTORED_PROCESS_IDENTITY')
                    snapshot=ops.gpu(module)
                    if snapshot['processes'].get(restored_pid,{}).get('mib',0)>20000:
                        passed=True;break
                restoration={'restored':passed,'pid':restored_pid,'process_identity':observed,'gpu':snapshot}
        except BaseException as exc:
            restoration={'restored':False,'error':type(exc).__name__+': '+str(exc),
                         'manual_recovery_required':True,'unknown_processes_stopped':0}
        finally:
            if remain_changed:
                try:
                    require(ops.pane('#{pane_id}')==PANE_ID,'PANE_CHANGED_NO_OPTION_RESTORE')
                    ops.set_remain(remain)
                except BaseException as exc:
                    restoration['restored']=False;restoration['option_restore_error']=repr(exc)
            try:
                save(out/'RESTORATION.json',restoration)
                save(out/'LEASE_RESULT.json',{'execute_returned':success,'error':error,'holder_signal_attempted':attempted,
                    'holder_restored':restoration['restored'],'external_processes_stopped':0,
                    'scientific_pass':False,'original_supervisor_worker_cleanup_unchanged':True})
            finally:
                for s,handler in original_handlers.items():signals.signal(s,handler)
    if error or not success or not restoration['restored']:raise RuntimeError('GPU5_LEASE_OR_EXECUTION_FAILED:'+str(error))

def run_main(batch):
    verify_sources();readiness=load('gpu5_frozen_readiness',BE/'readiness_v1.py')
    batch,out=readiness.validate_target(batch)
    cfg=read(out/'EXECUTION_CONFIG.json');identity=normalize_identity(read(cfg['gpu5_holder_identity_path']))
    validate_authorization(read(cfg['gpu5_authorization_path']),batch,cfg['source_snapshot'],
                           cfg['source_selection_indices'],cfg['gpu5_holder_identity_path'])
    lock=read(out/'INPUT_LOCK.json')
    require(len(lock)<=1024,'INPUT_LOCK_COUNT_CAP')
    for p,h in lock.items():require(sha(Path(p))==h,'INPUT_LOCK_CHANGED:'+p)
    approval=read(batch/'MAIN_AGENT_GPU5_APPROVAL.json')
    require(approval=={'approved':True,'gpu':5,'input_lock_sha256':sha(out/'INPUT_LOCK.json'),
                      'holder_identity_sha256':sha(Path(cfg['gpu5_holder_identity_path']))},'MAIN_AGENT_APPROVAL')
    require(str(HERE/'transport.py') in lock and str(HERE/'prepare.py') in lock,'TRANSPORT_NOT_LOCKED')
    require(cfg['gpu5_authorization_path'] in lock and cfg['gpu5_holder_identity_path'] in lock,'AUTHORIZATION_NOT_LOCKED')
    lease_lock=(HERE/'GPU5_LEASE.lock').open('a')
    fcntl.flock(lease_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(out/'LEASE_RESERVATION.json',{'supervisor_pid':os.getpid(),'gpu':5,'identity':identity})
    module=build_supervisor(batch,cfg)
    def same_supervisor(path,configuration):
        require(Path(path)==batch and configuration==cfg,'SUPERVISOR_CONFIG_CHANGED')
        return module
    run_lease(out,identity,module,lambda:readiness.run_main(batch,module_builder=same_supervisor))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('batch',type=Path)
    run_main(parser.parse_args().batch)
