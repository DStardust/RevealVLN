"""Exact holder chain over reviewed active telemetry and original V2 lease.

Imports are CPU only. run_main alone can borrow, after exact main approval and
immediate predecessor restoration binding; no name-based process discovery.
"""
import fcntl
import importlib.util
import os
from pathlib import Path
import shlex
import types
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('special_holder_private_core',HERE/'core.py')
k=importlib.util.module_from_spec(s);s.loader.exec_module(k)
core=k.private;c=core.c
telemetry=core.telemetry
BASE=k.BASE
LEASE_SOURCE=c.BE/'gpu5_transport_v2/transport.py'
_base_dependencies=core.dependency_paths

def dependency_paths():
    return [*_base_dependencies(),*sorted(HERE.glob('*.py')),HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',
            LEASE_SOURCE,c.BE/'gpu5_transport_v2/test_transport.py']
core.dependency_paths=dependency_paths

def check_identity(document,gpu):
    c.require(gpu in k.GPUS and document.get('gpu_device')==gpu and document.get('gpu_uuid')==k.GPUS[gpu],'HOLDER_GPU_IDENTITY')
    c.require(document.get('project_root')==str(c.ROOT) and document.get('cwd')==str(c.ROOT) and document.get('proc_uid')==0,'HOLDER_CWD_UID')
    c.require(type(document.get('pid')) is int and document['pid']>0 and document.get('pane_pid')==document['pid'],'EXACT_HOLDER_PID')
    c.require(type(document.get('starttime_ticks')) is int and document['starttime_ticks']>0 and document.get('pane_dead') is False,'HOLDER_START_PANE')
    c.require(isinstance(document.get('pane_target'),str) and document['pane_target'].startswith('vla_idle_occupancy_20260904:'),'HOLDER_PANE_SCOPE')
    c.require(isinstance(document.get('pane_id'),str) and document['pane_id'].startswith('%'),'EXACT_PANE_ID')
    c.require(document.get('remain_on_exit') in ('on','off','failed'),'FROZEN_REMAIN_STATE')
    argv=document.get('cmdline')
    c.require(isinstance(argv,list) and len(argv)==11 and argv[:3]==['.envs/etpr1/bin/python','-u','scripts/occupy_idle_gpu.py'],'OCCUPANCY_ARGV_ONLY')
    c.require(argv[3:7]==['--gpu',str(gpu),'--reserve-mib','2048'] and argv[7]=='--max-initial-used-mib' and argv[8] in ('1024','1536') and argv[9:]==['--tag','vla_idle_occupancy_20260904'],'EXACT_REGISTERED_OCCUPANCY_ARGV')
    c.require(document.get('restore_same_cwd_and_cmdline') is True and document.get('revalidate_all_fields_before_signal') is True and document.get('unknown_or_nonholder_process_signal_allowed') is False,'HOLDER_SIGNAL_SCOPE')
    c.require(document.get('holder_source_sha256')==c.sha(c.ROOT/'scripts/occupy_idle_gpu.py'),'HOLDER_SOURCE_CHANGED')
    env=document.get('project_cache_environment',{})
    allowed={'PYTHONDONTWRITEBYTECODE','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR','NUMBA_CACHE_DIR','MPLCONFIGDIR'}
    c.require(isinstance(env,dict) and set(env)<=allowed,'HOLDER_ENV_ALLOWLIST')
    for key,value in env.items():
        c.require(isinstance(value,str) and '\0' not in value,'HOLDER_ENV_VALUE')
        if key=='PYTHONDONTWRITEBYTECODE':c.require(value=='1','HOLDER_NO_BYTECODE')
        else:c.require(Path(value).is_absolute() and Path(value).resolve().is_relative_to(c.ROOT),'HOLDER_CACHE_SCOPE')
    return document

def approval_value(batch):
    cfg=c.read(c.scoped(batch)/'run_v1/EXECUTION_CONFIG.json');auth=c.read(cfg['scale_authorization_path'])
    return dict(approved=True,gpu=cfg['gpu_device'],input_lock_sha256=c.sha(Path(batch)/'run_v1/INPUT_LOCK.json'),
        authorization_sha256=c.sha(cfg['scale_authorization_path']),transport_version='special_scale_holder_v1',
        initial_holder_identity_sha256=auth['initial_holder_identity_sha256'])

def check_inputs(batch,worker=False):
    cfg=core.check_inputs(batch,worker=worker)
    c.require(cfg.get('holder_transport_version')=='special_scale_holder_v1','HOLDER_TRANSPORT_VERSION')
    auth=c.read(cfg['scale_authorization_path']);items=c.validate_authorization(auth,cfg['queue_path'])
    c.require(auth['mode']=='holder_chain','EXPLICIT_HOLDER_AUTHORIZATION')
    index=next(i for i,r in enumerate(items) if r['id']==Path(batch).name)
    expected=str(c.BE/items[index-1]['id']) if index else None
    c.require(cfg.get('previous_holder_batch')==expected,'IMMEDIATE_PREDECESSOR_REQUIRED')
    initial=c.read(auth['initial_holder_identity_path']);check_identity(initial,cfg['gpu_device'])
    c.require(initial.get('collection_path')==cfg['holder_identity_collection_path'],'IDENTITY_COLLECTION_BINDING')
    collection=c.read(cfg['holder_identity_collection_path'])
    broad=c.read(auth['main_scope_authorization_path'])
    c.require(cfg['holder_identity_collection_path']==str(c.scoped(c.LINE/broad['holder_identities'])),'MAIN_COLLECTION_PATH_REQUIRED')
    c.require(c.sha(cfg['holder_identity_collection_path'])==initial['collection_sha256'],'HOLDER_COLLECTION_HASH')
    rows=[r for r in collection['holders'] if r['gpu_device']==cfg['gpu_device']]
    c.require(len(rows)==1 and {key:value for key,value in initial.items() if key not in ('collection_path','collection_sha256')}==rows[0],'EXACT_MAIN_HOLDER_COLLECTION')
    lock=c.read(Path(batch)/'run_v1/INPUT_LOCK.json')
    for p in [auth['initial_holder_identity_path'],cfg['holder_identity_collection_path'],str(c.ROOT/'scripts/occupy_idle_gpu.py')]:
        c.require(lock.get(p)==c.sha(p),'HOLDER_IDENTITY_SOURCE_NOT_LOCKED')
    if expected:
        path=Path(expected)/'run_v1/INPUT_LOCK.json'
        c.require(c.sha(path)==cfg['previous_holder_input_lock_sha256'] and lock.get(str(path))==c.sha(path),'PREDECESSOR_INPUT_LOCK_BINDING')
    return cfg

def resolve_identity(batch,cfg):
    auth=c.read(cfg['scale_authorization_path']);initial=c.read(auth['initial_holder_identity_path'])
    check_identity(initial,cfg['gpu_device']);document=dict(initial)
    proof=dict(initial_holder_identity_path=auth['initial_holder_identity_path'],initial_holder_identity_sha256=auth['initial_holder_identity_sha256'],
               predecessor_batch=cfg['previous_holder_batch'],source_files={},name_based_discovery=False)
    prior=cfg['previous_holder_batch']
    if prior:
        prior=c.scoped(prior);root=prior/'run_v1';previous_cfg=c.read(root/'EXECUTION_CONFIG.json')
        c.require(c.sha(root/'INPUT_LOCK.json')==cfg['previous_holder_input_lock_sha256'],'PREDECESSOR_INPUT_CHANGED')
        previous_lock=c.read(root/'INPUT_LOCK.json')
        c.require(0<len(previous_lock)<=2048,'PREDECESSOR_SOURCE_COUNT')
        c.require(previous_lock.get(str(root/'EXECUTION_CONFIG.json'))==c.sha(root/'EXECUTION_CONFIG.json'),'PREDECESSOR_CONFIG_NOT_LOCKED')
        for path,digest in previous_lock.items():c.require(c.sha(path)==digest,'PREDECESSOR_SOURCE_CHANGED:'+path)
        c.require(previous_cfg['gpu_device']==cfg['gpu_device'] and previous_cfg['scale_authorization_path']==cfg['scale_authorization_path'],'PREDECESSOR_OTHER_LANE')
        c.require(c.read(prior/'MAIN_AGENT_SCALE_APPROVAL.json')==approval_value(prior),'PREDECESSOR_NOT_MAIN_APPROVED')
        final=c.read(root/'SUPERVISOR_RESULT.json');lease=c.read(root/'LEASE_RESULT.json');restored=c.read(root/'RESTORATION.json')
        launch=c.read(root/'LAUNCH_RESULT.json')
        c.require(final.get('returncode')==0 and final.get('error') is None and final.get('cleanup_complete') is True,'PREDECESSOR_NOT_NORMALLY_CLOSED')
        c.require(lease.get('execute_returned') is True and lease.get('error') is None and lease.get('holder_restored') is True,'PREDECESSOR_LEASE_FAILED')
        c.require(restored.get('restored') is True and restored.get('remain_on_exit_restored') is True,'PREDECESSOR_RESTORE_FAILED')
        c.require(launch.get('status')=='ORIGINAL_SUPERVISOR_RETURNED','PREDECESSOR_LAUNCH_FAILED')
        identity=restored['process_identity']
        c.require(identity['command']==' '.join(initial['cmdline']) and identity['cwd']==initial['cwd'] and identity['proc_uid']==initial['proc_uid'],'RESTORED_HOLDER_OTHER_COMMAND')
        c.require(identity['pid']==restored['pid'] and restored['gpu']['uuid']==cfg['gpu_uuid'],'RESTORED_PID_GPU_BINDING')
        entry=restored['gpu']['processes'].get(str(identity['pid']))
        c.require(entry and entry['mib']>20000,'RESTORED_OCCUPANCY_NOT_PROVEN')
        document.update(pid=identity['pid'],pane_pid=identity['pid'],starttime_ticks=identity['starttime_ticks'])
        check_identity(document,cfg['gpu_device'])
        for p in [prior/'MAIN_AGENT_SCALE_APPROVAL.json',root/'INPUT_LOCK.json',root/'EXECUTION_CONFIG.json',root/'SUPERVISOR_RESULT.json',root/'LEASE_RESULT.json',root/'RESTORATION.json',root/'LAUNCH_RESULT.json']:
            proof['source_files'][str(p)]=c.sha(p)
    proof['resolved_identity']=document
    return document,proof

def lease_module(document):
    """All original lease/restore guard functions unchanged; exact globals only."""
    check_identity(document,document['gpu_device'])
    module=c.load('special_private_frozen_holder_lease',LEASE_SOURCE)
    module.UUID=document['gpu_uuid'];module.HOLDER_PID=document['pid']
    module.PANE=document['pane_target'];module.PANE_ID=document['pane_id'];module.COMMAND=' '.join(document['cmdline'])
    identity=dict(pid=document['pid'],starttime_ticks=document['starttime_ticks'],command=module.COMMAND,
        cwd=document['cwd'],pane=module.PANE,pane_id=module.PANE_ID,proc_uid=document['proc_uid'])
    module.validate_identity(identity)
    return module,identity

def build_supervisor(batch,cfg):
    module=core.build_supervisor(batch,cfg)
    module._gpu5_raw_gpu=module.gpu;module._gpu5_child=None
    original=module.subprocess
    proxy=types.SimpleNamespace(**{name:getattr(original,name) for name in dir(original)})
    def popen(*args,**kwargs):
        c.require(module._gpu5_child is None and kwargs.get('start_new_session') is True,'ONE_OWN_WORKER_SESSION')
        proc=original.Popen(*args,**kwargs);module._gpu5_child=proc;return proc
    proxy.Popen=popen;module.subprocess=proxy
    return module

def make_ops(document,lease):
    """Reuse original process/guard/signal operations; restore frozen cache env."""
    check_identity(document,document['gpu_device'])
    class Ops(lease.RealOps):
        def remain(self):
            value=super().remain()
            c.require(value==document['remain_on_exit'],'FROZEN_REMAIN_OPTION_CHANGED')
            return value
        def restore_command(self):
            env=document.get('project_cache_environment',{})
            allowed={'PYTHONDONTWRITEBYTECODE','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR','NUMBA_CACHE_DIR','MPLCONFIGDIR'}
            argv=['env']
            for key in sorted(allowed-set(env)):argv.extend(['-u',key])
            argv += [key+'='+env[key] for key in sorted(env)]+document['cmdline']
            self.call('tmux','respawn-pane','-t',lease.PANE,'-c',str(c.ROOT),shlex.join(argv))
    return Ops()

def worker_main(batch):
    cfg=check_inputs(batch,worker=True)
    original=c.load('holder_scale_frozen_clock_worker',c.BE/'budget_batched_transport_v1/transport.py')
    original.build_worker(c.scoped(batch),cfg).main()

def run_main(batch):
    cfg=check_inputs(batch);batch=c.scoped(batch)
    c.require(c.read(batch/'MAIN_AGENT_SCALE_APPROVAL.json')==approval_value(batch),'MAIN_HOLDER_LAUNCH_APPROVAL')
    readiness=c.load('holder_original_readiness',c.BE/'readiness_v1.py');batch,out=readiness.validate_target(batch)
    document,proof=resolve_identity(batch,cfg);lease,identity=lease_module(document)
    lock_dir=HERE/'locks';lock_dir.mkdir(exist_ok=True)
    handle=(lock_dir/('gpu_'+str(cfg['gpu_device'])+'.lock')).open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c.save(out/'CHAIN_IDENTITY_BINDING.json',proof)
    c.save(out/'LEASE_RESERVATION.json',dict(supervisor_pid=os.getpid(),gpu=cfg['gpu_device'],identity=identity))
    module=build_supervisor(batch,cfg)
    original_save=readiness._save_new
    def save_launch(path,value):
        if Path(path).name=='LAUNCH_RESULT.json':
            value=dict(value);value['original_launcher_guards_relaxed_field']=value.pop('guards_relaxed')
            value.update(accounting_acceptance_amended=True,quantitative_thresholds_relaxed=False,
                idle_final_guard_unchanged=True,resource_protocol='special_active_conservative_accounting_v1',holder_lease_version='special_scale_holder_v1')
        return original_save(path,value)
    readiness._save_new=save_launch
    def builder(path,value):
        c.require(Path(path)==batch and value==cfg,'LEASE_SUPERVISOR_CONFIG_CHANGED');return module
    try:lease.run_lease(out,identity,module,lambda:readiness.run_main(batch,module_builder=builder),ops=make_ops(document,lease))
    finally:handle.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);a=p.parse_args();run_main(a.batch)
