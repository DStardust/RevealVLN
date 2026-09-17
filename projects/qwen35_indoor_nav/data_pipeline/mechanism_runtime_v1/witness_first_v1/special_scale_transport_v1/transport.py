"""Fresh GPU1/2 production. Only active accounting is versionedly amended.

No holder support in this frozen first entry: scaleout borrowing is a separately
reviewed future module, so adding it cannot mutate active GPU1/2 input closures.
"""
import importlib.util
import json
import os
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('special_scale_common',HERE/'common.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
telemetry=c.load('special_scale_telemetry',HERE/'telemetry.py')
GPUS={1:'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',2:'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'}

def check_prior_unattempted(item,cfg):
    prior=item.get('previous_run_root')
    if item.get('runtime_recheck_unstarted_required') is True:
        c.require(bool(prior),'PRIOR_RECHECK_TARGET_REQUIRED')
    if prior:
        root=c.scoped(prior)
        c.require(root.parent.parent==c.BE and root.name=='run_v1','PRIOR_RUN_SCOPE')
        c.require({p.name for p in root.iterdir()}=={'EXECUTION_CONFIG.json','INPUT_LOCK.json'},'OLD_RESERVATION_NOW_ATTEMPTED')
        old=c.read(root/'EXECUTION_CONFIG.json')
        c.require(old['candidates']==cfg['candidates'],'MIGRATED_CANDIDATES_CHANGED')

def check_inputs(batch,worker=False):
    batch=c.scoped(batch);c.require(batch.parent==c.BE and batch.name.startswith('batch_'),'EXACT_NEW_BATCH_SCOPE')
    out=batch/'run_v1';cfg=c.read(out/'EXECUTION_CONFIG.json');c.check_config(cfg)
    c.require(cfg['gpu_device'] in GPUS and cfg['gpu_uuid']==GPUS[cfg['gpu_device']],'NONEXCLUSIVE_GPU12_ONLY')
    c.require(cfg['lease_mode']=='nonexclusive','NO_HOLDER_IN_FROZEN_NONEXCLUSIVE_ENTRY')
    queue=c.scoped(cfg['queue_path']);auth_path=c.scoped(cfg['scale_authorization_path']);auth=c.read(auth_path)
    items=c.validate_authorization(auth,queue)
    matches=[r for r in items if r['id']==batch.name];c.require(len(matches)==1,'BATCH_NOT_AUTHORIZED')
    item=matches[0]
    c.require(item['prepared_snapshot']==cfg['source_snapshot'] and cfg['source_selection_indices']==[0,1,2],'QUEUE_SOURCE_BINDING')
    c.require(item['candidate_ids']==[r['candidate_id'] for r in cfg['candidates']],'QUEUE_CANDIDATE_BINDING')
    check_prior_unattempted(item,cfg)
    c.require(auth['gpu_device']==cfg['gpu_device'] and auth['gpu_uuid']==cfg['gpu_uuid'],'AUTH_GPU_CONFIG')
    lock=c.read(out/'INPUT_LOCK.json');c.require(1<=len(lock)<=2048,'INPUT_COUNT_CAP')
    c.require(sum(Path(p).stat().st_size for p in lock)<=32*1024**3,'INPUT_BYTE_CAP')
    for p,h in lock.items():c.require(c.sha(p)==h,'LOCKED_SOURCE_CHANGED:'+p)
    for p in [*dependency_paths(),queue,queue.parent/'SOURCE_LOCK.json',auth_path,
              c.scoped(auth['main_scope_authorization_path']),batch/'worker.py',batch/'run.py',out/'EXECUTION_CONFIG.json']:
        c.require(lock.get(str(p))==c.sha(p),'REQUIRED_INPUT_NOT_LOCKED:'+str(p))
    if worker:
        c.require(not any((out/p).exists() for p in ('bundles','content','journal','PROGRESS.json','result.json','BUDGET_FINAL.json')),'FRESH_WORKER_REQUIRED')
    return cfg

def dependency_paths():
    return [*sorted(HERE.glob('*.py')),HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',*c.SOURCES,
        c.WF/'budget_clock_batching_cpu_v1/ledger.py',c.WF/'budget_clock_batching_cpu_v1/SHA256SUMS',
        c.WF/'quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/SHA256SUMS']

def supervisor_source(source,gpu):
    shared=c.load('special_scale_shared_supervisor',c.BE/'shared.py')
    result=shared.supervisor_source(source,gpu)
    result=c.exact(result,'snapshot=gpu(); upper=check_gpu(snapshot,proc.pid)',
                   'snapshot,upper=_active_sample(proc.pid,started)')
    return c.exact(result,'sample=dict(snapshot, own_memory_upper_mib=upper, elapsed=time.monotonic()-started)',
        'sample=dict(snapshot, own_memory_upper_mib=upper, elapsed=time.monotonic()-started, active_accounting_sample_index=_active_count)')

def build_supervisor(batch,cfg):
    c.check_sources();c.check_config(cfg)
    c.require(cfg['gpu_uuid']==GPUS[cfg['gpu_device']],'SUPERVISOR_GPU_IDENTITY')
    path=c.RUNTIME/'compact_loop_v2/run.py'
    module=types.ModuleType('special_scale_original_supervisor');module.__file__=str(path)
    exec(compile(supervisor_source(path.read_text(),cfg['gpu_device']),str(path)+'::active_accounting_v1','exec'),module.__dict__)
    module.HERE=c.scoped(batch);module.OUT=module.HERE/'run_v1';module.LINE=c.LINE
    module.ENV=c.LINE/'.envs/q35n_habitat_v017_g0r';module.UUID=cfg['gpu_uuid'];module._active_count=0
    def sample(worker,started):
        module._active_count+=1;index=module._active_count
        def record(kind,value):
            c.append(module.OUT/'ACTIVE_ACCOUNTING.jsonl',dict(sample_index=index,kind=kind,
                monotonic=module.time.monotonic(),supervisor_started_monotonic=started,payload=value))
        # Do not call or replace gpu(): original idle/final/restore queries stay raw.
        raw=module.subprocess.check_output(['nvidia-smi','-i',str(cfg['gpu_device']),'-q','-x'],text=True,timeout=15)
        import hashlib
        record('raw_xml',dict(xml=raw,sha256=hashlib.sha256(raw.encode()).hexdigest()))
        snapshot=module.parse_gpu(raw)
        record('parsed_snapshot',snapshot)
        def decision(value):record('decision',value)
        result=telemetry.assess(snapshot,worker,module.check_gpu,decision)
        c.require(module.time.monotonic()-started<3900,'WALL_BUDGET_AFTER_DURABLE_ACCOUNTING')
        return snapshot,result['guard_upper_mib']
    module._active_sample=sample
    return module

def worker_main(batch):
    cfg=check_inputs(batch,worker=True)
    original=c.load('special_scale_frozen_clock_worker',c.BE/'budget_batched_transport_v1/transport.py')
    original.build_worker(c.scoped(batch),cfg).main()

def approval_value(batch):
    cfg=c.read(c.scoped(batch)/'run_v1/EXECUTION_CONFIG.json')
    return dict(approved=True,gpu=cfg['gpu_device'],input_lock_sha256=c.sha(Path(batch)/'run_v1/INPUT_LOCK.json'),
        authorization_sha256=c.sha(cfg['scale_authorization_path']),transport_version='special_scale_transport_v1')

def run_main(batch):
    cfg=check_inputs(batch)
    c.require(c.read(c.scoped(batch)/'MAIN_AGENT_SCALE_APPROVAL.json')==approval_value(batch),'MAIN_LAUNCH_APPROVAL_REQUIRED')
    readiness=c.load('special_scale_original_readiness',c.BE/'readiness_v1.py')
    original_save=readiness._save_new
    def save_launch(path,value):
        if Path(path).name=='LAUNCH_RESULT.json':
            value=dict(value)
            value['original_launcher_guards_relaxed_field']=value.pop('guards_relaxed')
            value.update(accounting_acceptance_amended=True,quantitative_thresholds_relaxed=False,
                         idle_final_guard_unchanged=True,resource_protocol='special_active_conservative_accounting_v1')
        return original_save(path,value)
    readiness._save_new=save_launch
    readiness.run_main(c.scoped(batch),module_builder=build_supervisor)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);a=p.parse_args();run_main(a.batch)
