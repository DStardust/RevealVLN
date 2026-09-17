"""CPU strong family/resource audit with mandatory exact holder chain closure."""
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('holder_audit_transport',HERE/'transport.py')
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
c=t.c
base=c.load('holder_private_scale_audit',t.BASE/'audit.py');base.t=t;base.c=c
AUDIT_ROOT=base.GATE.parent/'special_scale_holder_v1'

def verify_holder_terminal(run,cfg):
    document,expected=t.resolve_identity(run.parent,cfg)
    actual=c.read(run/'CHAIN_IDENTITY_BINDING.json')
    c.require(c.canonical(actual)==c.canonical(expected),'LEASE_CHAIN_PROOF_CHANGED')
    lease=c.read(run/'LEASE_RESULT.json');restored=c.read(run/'RESTORATION.json')
    c.require(lease.get('execute_returned') is True and lease.get('error') is None and lease.get('holder_restored') is True and lease.get('external_processes_stopped')==0,'LEASE_NOT_SUCCESSFUL')
    c.require(restored.get('restored') is True and restored.get('remain_on_exit_restored') is True,'HOLDER_NOT_RESTORED')
    identity=restored['process_identity'];c.require(identity['pid']==restored['pid'],'RESTORED_PID')
    c.require(identity['command']==' '.join(document['cmdline']) and identity['cwd']==document['cwd'] and identity['proc_uid']==document['proc_uid'],'RESTORED_COMMAND_CWD_UID')
    c.require(type(identity['starttime_ticks']) is int and identity['starttime_ticks']>0,'RESTORED_STARTTIME')
    gpu=restored['gpu'];c.require(gpu['uuid']==cfg['gpu_uuid'],'RESTORED_GPU_UUID')
    own=gpu['processes'].get(str(identity['pid']));c.require(own and own['mib']>20000,'RESTORED_GPU_OCCUPANCY')
    return dict(holder_chain_verified=True,restore_verified=True,previous_holder_batch=cfg['previous_holder_batch'],
        resolved_holder_pid=document['pid'],restored_holder_pid=identity['pid'],external_processes_signalled=0)

_active=base.verify_active_log
def active_with_holder(run,cfg,proc):
    verify_holder_terminal(run,cfg)
    return _active(run,cfg,proc)
base.verify_active_log=active_with_holder
_adapted=base.adapted_acceptance
def adapted(source):
    return c.exact(_adapted(source),"'ACTIVE_ACCOUNTING.jsonl','LAUNCH_RESULT.json'):",
        "'ACTIVE_ACCOUNTING.jsonl','LAUNCH_RESULT.json','CHAIN_IDENTITY_BINDING.json','LEASE_RESERVATION.json','LEASE_RESULT.json','RESTORATION.json'):")
base.adapted_acceptance=adapted

def audit_main(batch):
    batch=c.scoped(batch);cfg=t.check_inputs(batch);run=batch/'run_v1'
    c.require(c.read(run/'SUPERVISOR_RESULT.json').get('cleanup_complete') is True,'WAIT_FOR_CLEANUP')
    proof=verify_holder_terminal(run,cfg)
    out=AUDIT_ROOT/batch.name;out.mkdir(parents=True,exist_ok=False)
    observation=base.stack().observe_run(run,out)
    result=dict(status='HOLDER_CHAIN_RESOURCE_AMENDED_STRONG_BATCH_AUDIT',observation=observation,holder_audit=proof,
        strongly_accepted_families=sum(x.get('quality_pass') is True and x.get('source_and_phase_binding_verified') is True for x in observation['attempts']),
        registered_candidates=len(cfg['candidates']),accounting_guard_amended=True,
        original_semantic_and_physical_thresholds_unchanged=True,old_cohort_replacement=False,
        scientific_pass=False,model_gain_pass=False,training_admission=False,gpu_operations=0)
    c.save(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='observation'},indent=2));return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);a=p.parse_args();audit_main(a.batch)
