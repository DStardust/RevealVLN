"""Owner-approved GPU7 scout; no GPU access on import or CPU preparation."""
import fcntl
import importlib.util
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('new_hub_scale_common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)

def approval_value(job):
    job=c.job_root(job);launch=c.read(job/'LAUNCH_CONFIG.json')
    return dict(approved=True,node='Q35N_NEW_HUB_SCALE_RUNTIME_V1',gpu=7,
        source_lock_sha256=c.sha(job/'SOURCE_LOCK.json'),authorization_sha256=c.sha(launch['authorization_path']))

def check(job,approved=True):
    job=c.job_root(job);item=c.item_for(job);lock=c.verify_lock(job/'SOURCE_LOCK.json')
    launch=c.read(job/'LAUNCH_CONFIG.json');auth=c.authorization(launch['authorization_path'])
    c.require(job.name in auth['job_ids'],'UNAUTHORIZED_JOB')
    index=auth['job_ids'].index(job.name)
    expected=str(HERE/'jobs'/auth['job_ids'][index-1]) if index else None
    c.require(launch['previous_job']==expected and launch['initial_holder_identity_path']==auth['initial_holder_identity_path'],'EXACT_SCOUT_PREDECESSOR')
    for p in [HERE/'common.py',HERE/'runtime.py',HERE/'adapters.py',HERE/'bank.py',HERE/'prepare.py',job/'LAUNCH_CONFIG.json',launch['authorization_path'],launch['initial_holder_identity_path']]:
        c.require(lock.get(str(p))==c.sha(p),'REQUIRED_LOCK_DEPENDENCY')
    if expected:c.require(launch['previous_source_lock_sha256']==c.sha(Path(expected)/'SOURCE_LOCK.json'),'PREDECESSOR_SOURCE_BINDING')
    cfg=c.read(job/'PREPARED_CONFIG.json');c.require(cfg==c.prepared_config(item),'EXACT_PREPARED_CONFIG')
    if approved:
        c.require(c.read(job/'MAIN_AGENT_APPROVAL.json')==approval_value(job),'MAIN_AGENT_APPROVAL')
        cfg.update(runtime_allowed=True,executable=True,runtime_adapter_ready=True,
            source_lock_sha256=c.sha(job/'SOURCE_LOCK.json'),main_agent_approval_sha256=c.sha(job/'MAIN_AGENT_APPROVAL.json'))
    return cfg,launch,auth

def resolve_identity(job,launch,holder):
    document=c.read(launch['initial_holder_identity_path']);holder.check_identity(document,7)
    proof={'initial_identity':launch['initial_holder_identity_path'],'previous_job':launch['previous_job'],'source_files':{}}
    prior=launch['previous_job']
    if prior:
        prior=c.job_root(prior);check(prior)
        out=prior/'run_v1';sup=c.read(out/'SUPERVISOR_RESULT.json');lease=c.read(out/'LEASE_RESULT.json');restored=c.read(out/'RESTORATION.json')
        c.require(sup.get('returncode')==0 and sup.get('error') is None and sup.get('cleanup_complete') is True,'PREDECESSOR_SUPERVISOR_FAILED')
        c.require(lease.get('execute_returned') is True and lease.get('error') is None and lease.get('holder_restored') is True,'PREDECESSOR_LEASE_FAILED')
        c.require(restored.get('restored') is True and restored.get('remain_on_exit_restored') is True,'PREDECESSOR_RESTORATION_FAILED')
        c.require(c.read(out/'LAUNCH_RESULT.json').get('supervisor_returned') is True,'PREDECESSOR_LAUNCH_FAILED')
        state=restored['process_identity'];c.require(state['command']==' '.join(document['cmdline']) and state['cwd']==document['cwd'] and state['proc_uid']==document['proc_uid'],'RESTORED_HOLDER_COMMAND')
        c.require(state['pid']==restored['pid'] and restored['gpu']['uuid']==c.UUID and restored['gpu']['processes'].get(str(state['pid']),{}).get('mib',0)>20000,'RESTORED_HOLDER_GPU')
        document.update(pid=state['pid'],pane_pid=state['pid'],starttime_ticks=state['starttime_ticks']);holder.check_identity(document,7)
        for name in ['SUPERVISOR_RESULT.json','LEASE_RESULT.json','RESTORATION.json','LAUNCH_RESULT.json']:
            proof['source_files'][str(out/name)]=c.sha(out/name)
    proof['resolved_identity']=document;return document,proof

def worker_main(job):
    job=c.job_root(job);cfg,_,_=check(job)
    c.require(c.read(job/'run_v1/EXECUTION_CONFIG.json')==cfg,'WORKER_CONFIG_BINDING')
    c.load('new_hub_scale_worker_adapter',HERE/'adapters.py').build_worker(job,cfg).main()

def run_main(job):
    job=c.job_root(job);cfg,launch,auth=check(job)
    holder=c.load('new_hub_scale_sealed_holder',c.WF/'special_scale_holder_v1/transport.py')
    mutex=c.scoped(auth['shared_gpu_lock']);mutex.parent.mkdir(exist_ok=True)
    with mutex.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        document,proof=resolve_identity(job,launch,holder)
        lease,identity=holder.lease_module(document)
        out=job/'run_v1';out.mkdir(exist_ok=False)
        c.save(out/'EXECUTION_CONFIG.json',cfg);c.save(out/'INPUT_LOCK.json',c.read(job/'SOURCE_LOCK.json'))
        c.save(out/'CHAIN_IDENTITY_BINDING.json',proof)
        c.save(out/'LEASE_RESERVATION.json',dict(supervisor_pid=os.getpid(),gpu=7,identity=identity))
        returned=False;error=None
        def execute():
            nonlocal returned
            module.main();returned=True
        try:
            module=c.load('new_hub_scale_active_adapter',HERE/'adapters.py').build_supervisor(job)
            lease.run_lease(out,identity,module,execute,ops=holder.make_ops(document,lease))
        except BaseException as exc:error=repr(exc);raise
        finally:c.save(out/'LAUNCH_RESULT.json',dict(supervisor_returned=returned,error=error,gpu=7,
            external_processes_stopped=0,scientific_pass=False,holder_results='LEASE_RESULT.json'))
def audit_main(job):
    # Separate bounded CPU command: the queue calls this only after transport.
    job=c.job_root(job);out=job/'run_v1'
    bank=c.load('new_hub_scale_closed_bank',HERE/'bank.py')
    try:bank.main(job)
    except BaseException as exc:
        c.save(out/'BANK_REJECTED.json',dict(error=repr(exc),scientific_pass=False,new_physical_families=0));raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    (worker_main if a.worker else run_main)(a.job)
