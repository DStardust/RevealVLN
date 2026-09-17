"""Original family semantics plus explicitly amended active-resource audit.

Never label a newly accepted inconsistent sample as passing the old guard.
Original final/idle checks and 18 cells /27 replays /54 evaluations remain.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import types
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('scale_audit_transport',HERE/'transport.py')
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
c=t.c
GATE=c.WF/'quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/gate.py'
AUDIT_ROOT=GATE.parent/'special_scale_transport_v1'

def verify_events(events,module,worker):
    c.require(bool(events) and len(events)<=3200,'ACTIVE_EVENT_COUNT')
    c.require(len(events)%4==0,'INCOMPLETE_ACTIVE_ACCOUNTING_SEQUENCE')
    result={};previous_time=None;started=None
    for offset in range(0,len(events),4):
        group=events[offset:offset+4];index=offset//4+1
        c.require([r['kind'] for r in group]==['raw_xml','parsed_snapshot','decision','decision'],'RAW_PARSE_DECISION_ORDER')
        c.require(all(type(r['sample_index']) is int and r['sample_index']==index for r in group),'ACTIVE_INDEX_ORDER')
        for r in group:
            now=r['monotonic'];origin=r['supervisor_started_monotonic']
            c.require(type(now) in (int,float) and type(origin) in (int,float) and math.isfinite(now) and math.isfinite(origin),'INVALID_ACCOUNTING_TIME')
            if started is None:started=origin
            c.require(origin==started and 0<=now-started<3900,'ACCOUNTING_WALL_LIMIT')
            c.require(previous_time is None or now>=previous_time,'ACCOUNTING_CLOCK_REVERSED')
            previous_time=now
        raw=group[0]['payload'];c.require(set(raw)=={'xml','sha256'},'RAW_SCHEMA')
        c.require(hashlib.sha256(raw['xml'].encode()).hexdigest()==raw['sha256'],'RAW_XML_HASH')
        snapshot=module.parse_gpu(raw['xml'])
        c.require(c.canonical(json.loads(c.canonical(snapshot)))==c.canonical(group[1]['payload']),'RAW_PARSE_MISMATCH')
        c.require(all(v['type'] in ('C','G','C+G') for v in snapshot['processes'].values()),'GRAPHICS_INVENTORY_SCHEMA')
        observed=[]
        decision=t.telemetry.assess(snapshot,worker,module.check_gpu,observed.append)
        c.require(c.canonical(observed)==c.canonical([r['payload'] for r in group[2:]]),'RECOMPUTED_DECISION_MISMATCH')
        c.require([r['status'] for r in observed]==['PENDING','ACCEPTED'],'ACTIVE_REJECT_OR_PENDING_NOT_ACCEPTED')
        result[index]=dict(snapshot=snapshot,upper=decision['guard_upper_mib'],decision=decision,
                           last_relative_time=group[-1]['monotonic']-started)
    return dict(samples=result,seen=set(),amended=sum(r['decision']['amended_acceptance'] for r in result.values()))

def verify_active_log(run,cfg,proc):
    c.check_config(cfg)
    path=run/'ACTIVE_ACCOUNTING.jsonl'
    c.require(path.stat().st_size<=64*1024**2,'ACTIVE_LOG_BYTE_CAP')
    raw=path.read_bytes();c.require(raw.endswith(b'\n'),'PARTIAL_ACTIVE_LOG')
    events=[json.loads(line) for line in raw.splitlines()]
    module=t.build_supervisor(run.parent,cfg)  # constructs CPU functions only
    result=verify_events(events,module,proc['pid'])
    before=c.read(run/'GPU_BEFORE.json');module.check_gpu(dict(before,processes={int(p):v for p,v in before['processes'].items()}))
    launch=c.read(run/'LAUNCH_RESULT.json')
    c.require(launch.get('accounting_acceptance_amended') is True and launch.get('idle_final_guard_unchanged') is True,'LAUNCH_AMENDMENT_NOT_DECLARED')
    return result

def verify_resource(context,sample):
    index=sample.get('active_accounting_sample_index')
    c.require(type(index) is int and index==len(context['seen'])+1 and index in context['samples'],'RESOURCE_INDEX_BINDING')
    row=context['samples'][index]
    original={k:sample[k] for k in ('uuid','memory_mib','utilization','processes')}
    c.require(c.canonical(json.loads(c.canonical(original)))==c.canonical(json.loads(c.canonical(row['snapshot']))),'RESOURCE_RAW_VALUE_CHANGED')
    c.require(type(sample['own_memory_upper_mib']) is type(row['upper']) and sample['own_memory_upper_mib']==row['upper'],'RESOURCE_UPPER_CHANGED')
    c.require(type(sample['elapsed']) in (int,float) and math.isfinite(sample['elapsed']) and sample['elapsed']>=row['last_relative_time'],'RESOURCE_BEFORE_DURABLE_DECISION')
    context['seen'].add(index)

def adapted_acceptance(source):
    source=c.exact(source,"    require(bool(samples),'RESOURCE_SAMPLES_MISSING')",
        "    require(bool(samples),'RESOURCE_SAMPLES_MISSING')\n    active_context=verify_active_log(run,cfg,proc)")
    source=c.exact(source,"        gpu_snapshot(sample,proc['pid'],cfg['gpu_uuid'])",
        '        verify_resource(active_context,sample)')
    source=c.exact(source,"    final_gpu=gpu_snapshot(final['gpu_after'],proc['pid'],cfg['gpu_uuid'])",
        "    require(len(active_context['seen'])==len(active_context['samples']),'UNBOUND_ACTIVE_DECISIONS')\n    final_gpu=gpu_snapshot(final['gpu_after'],proc['pid'],cfg['gpu_uuid'])")
    source=c.exact(source,"    evidence['source_files']=dict(lock)",
        "    evidence['active_accounting']={'guard_amended':True,'amended_samples':active_context['amended'],'sample_count':len(active_context['samples']),'quality_thresholds_unchanged':True,'old_guard_pass_not_claimed_for_amended_samples':True}\n    evidence['source_files']=dict(lock)")
    source=c.exact(source,"'journal/events.jsonl','journal/HEAD.json'):",
        "'journal/events.jsonl','journal/HEAD.json','ACTIVE_ACCOUNTING.jsonl','LAUNCH_RESULT.json'):")
    source=c.exact(source,"        response.update(source_and_phase_binding_verified=True,preaction_configuration_verified=True,",
        "        response['original_semantic_quality_grade']=response['grade']\n        if response.get('quality_pass') is True:response['grade']='QUALITY_VERIFIED_FIT_AMENDED_ACTIVE_ACCOUNTING_NOT_MODEL_GAIN'\n        response['resource_guard_amended']=True\n        response['active_accounting_audit']=source['active_accounting']\n        response.update(source_and_phase_binding_verified=True,preaction_configuration_verified=True,")
    return source

def stack():
    base=c.load('special_scale_original_gate3',GATE)
    source=adapted_acceptance(base.ADAPTED_SOURCE)
    private=types.ModuleType('special_scale_private_acceptance');private.__file__=str(base.ACCEPTANCE)
    private.verify_active_log=verify_active_log;private.verify_resource=verify_resource
    exec(compile(source,str(base.ACCEPTANCE)+'::active_resource_amendment_v1','exec'),private.__dict__)
    base.gate.old=private;base.gate.require=private.require
    return base.gate

def audit_main(batch):
    batch=c.scoped(batch);cfg=t.check_inputs(batch);run=batch/'run_v1'
    c.require(c.read(run/'SUPERVISOR_RESULT.json').get('cleanup_complete') is True,'WAIT_FOR_CLEANUP')
    out=AUDIT_ROOT/batch.name;out.mkdir(parents=True,exist_ok=False)
    gate=stack();observation=gate.observe_run(run,out)
    result=dict(status='RESOURCE_AMENDED_STRONG_BATCH_AUDIT',observation=observation,
        strongly_accepted_families=sum(x.get('quality_pass') is True and x.get('source_and_phase_binding_verified') is True for x in observation['attempts']),
        registered_candidates=len(cfg['candidates']),accounting_guard_amended=True,
        original_semantic_and_physical_thresholds_unchanged=True,old_cohort_replacement=False,
        scientific_pass=False,model_gain_pass=False,training_admission=False,gpu_operations=0)
    c.save(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='observation'},indent=2));return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);a=p.parse_args();audit_main(a.batch)
