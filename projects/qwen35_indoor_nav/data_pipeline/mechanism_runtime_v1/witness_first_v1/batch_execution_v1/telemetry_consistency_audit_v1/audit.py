"""Additional closed-run audit, not a replacement for family/lease validation."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
BE=HERE.parent
WF=BE.parent

def require(value,why):
    if not value:raise ValueError(why)
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def normalized(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def unique_pairs(items):
    out={}
    for key,value in items:
        require(key not in out,'DUPLICATE_JSON_KEY');out[key]=value
    return out
def lines(raw):
    require(len(raw)<=2*1024**3,'LOG_BYTE_CAP')
    require(raw.endswith(b'\n') and raw.strip(),'EMPTY_OR_PARTIAL_LOG')
    return [json.loads(line,object_pairs_hook=unique_pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('NONFINITE_JSON'))) for line in raw.splitlines()]

def audit_samples(events,resources,parse,check,retry_eligible,worker,wall_limit=3900):
    require(type(worker) is int and worker>0 and wall_limit==3900,'EXACT_WORKER_AND_DEADLINE')
    require(events and resources,'NO_COMPLETE_SAMPLES')
    require(len(events)<=10000 and len(resources)<=780,'ORIGINAL_BOUNDED_SAMPLE_COUNT')
    previous=-float('inf')
    for event in events:
        now=event.get('monotonic');require(finite(now) and now>=previous,'EVENT_CLOCK_ORDER');previous=now
    cursor=0;passes=[];retried=0;failed_raw=0;first_error_repr="AssertionError('MEMORY_ACCOUNTING')"
    def take(kind,attempt):
        nonlocal cursor
        require(cursor<len(events),'TRUNCATED_PROTOCOL')
        event=events[cursor];cursor+=1
        require(event.get('event')==kind and type(event.get('attempt')) is int and event['attempt']==attempt,'PROTOCOL_ORDER:'+kind)
        return event
    while cursor<len(events):
        deadline=None;first=None;attempt=0
        while True:
            raw=take('raw_xml',attempt)
            require(type(raw.get('raw_xml')) is str,'RAW_XML_TEXT_REQUIRED')
            require(hashlib.sha256(raw['raw_xml'].encode()).hexdigest()==raw['utf8_sha256'],'RAW_XML_HASH')
            before=raw['query_started'];received=raw['monotonic'];timeout=raw['timeout']
            require(finite(before) and before<=received and finite(timeout) and 0<timeout<=15,'QUERY_CLOCK_TIMEOUT')
            if cursor>1:require(before>=events[cursor-2]['monotonic'],'QUERY_STARTED_BEFORE_PRIOR_EVENT')
            if deadline is not None:
                require(before<deadline and received<deadline and timeout<=deadline-before,'RETRY_DEADLINE_OR_TIMEOUT')
            # Always rerun from XML; serialized PID keys must not enter check().
            parsed=parse(raw['raw_xml'])
            recorded=take('parsed_sample',attempt)
            require(normalized(parsed)==recorded['snapshot'],'PARSED_VALUE_CHANGED')
            try:upper=check(parsed,worker);error=None
            except BaseException as exc:error=exc
            if error is None:
                passed=take('guard_pass',attempt)
                require(passed['own_memory_upper_mib']==upper and passed['retries']==attempt,'PASS_VALUE_OR_RETRY_COUNT')
                require(passed['first_error']==('None' if first is None else first_error_repr),'PASS_FIRST_ERROR')
                if deadline is not None:require(passed['monotonic']<deadline,'LATE_FINAL_PASS')
                passes.append({'snapshot':parsed,'upper':upper,'monotonic':passed['monotonic'],
                               'first_query_started':raw['query_started'] if attempt==0 else cycle_start,'retries':attempt})
                retried+=int(attempt>0);break
            failed_raw+=1
            require(type(error) is AssertionError and error.args==('MEMORY_ACCOUNTING',),'NONRETRYABLE_OR_UNSAFE_PRIOR_SAMPLE:'+repr(error))
            failure=take('guard_error',attempt)
            require(failure['error']==first_error_repr and failure['first_error']==first_error_repr,'ERROR_CHANGED')
            if first is None:
                first=failure['monotonic'];deadline=first+1.;cycle_start=raw['query_started']
            require(failure['retry_deadline']==deadline,'RETRY_WINDOW_RESET')
            gross=retry_eligible(parsed,worker)  # Reject before any later low sample.
            eligible=take('bounded_resample_eligible',attempt)
            require(all(eligible[k]==v for k,v in gross.items()),'ELIGIBILITY_VALUE_CHANGED')
            require(eligible['monotonic']<deadline and attempt<2,'RETRY_LIMIT_OR_LATE_ELIGIBILITY')
            attempt+=1
    require(len(passes)==len(resources),'RESOURCE_PASS_COUNT_MISMATCH')
    previous_elapsed=-1.;lower=-float('inf');upper_start=float('inf')
    for i,(passed,resource) in enumerate(zip(passes,resources)):
        snapshot={key:resource[key] for key in ('uuid','memory_mib','utilization','processes')}
        require(normalized(passed['snapshot'])==snapshot,'RESOURCE_RAW_VALUE_MISMATCH')
        require(resource['own_memory_upper_mib']==passed['upper'],'RESOURCE_UPPER_CHANGED')
        elapsed=resource['elapsed'];require(finite(elapsed) and previous_elapsed<elapsed<wall_limit,'ORIGINAL_RESOURCE_WALL_LIMIT')
        previous_elapsed=elapsed;lower=max(lower,passed['monotonic']-elapsed)
        if i+1<len(passes):upper_start=min(upper_start,passes[i+1]['first_query_started']-elapsed)
        if 'disk_bytes' in resource:require(resource['disk_bytes']<7*1024**3,'ORIGINAL_DISK_LIMIT')
    require(lower<=upper_start,'NO_COMMON_ORIGINAL_STARTED_INTERVAL')
    return {'raw_protocol_pass':True,'resource_passes_bound':len(passes),'raw_queries_verified':len(passes)+failed_raw,
        'retained_failed_accounting_samples':failed_raw,'sampling_calls_with_retries':retried,
        'max_extra_queries_per_call':max(p['retries'] for p in passes),
        'quantitative_thresholds_unchanged':True,'unsafe_prior_samples_not_discarded':True,
        'lease_restore_sampling_not_audited_here':True,
        'original_started_interval_lower':lower,'original_started_interval_upper':None if math.isinf(upper_start) else upper_start,
        'durable_ack_timestamps_directly_recorded':False,
        'ack_deadline_evidence':'Recorded pre-write times plus locked return-after-I/O deadline checks and matching subsequent RESOURCE samples; not an independently measured fsync completion time.',
        'scientific_pass':False,'model_gain_pass':False}

def audit_closed(run,output):
    run=Path(run).absolute();output=Path(output).absolute()
    require(run==BE/'batch_06r2/run_v1' and run==run.resolve(),'EXACT_RUN_SCOPE')
    require(output==output.resolve() and output.is_relative_to(HERE) and output!=HERE,'OUTPUT_SCOPE')
    v5=BE/'gpu5_transport_v5'
    for line in (v5/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split(None,1);require(hashlib.sha256((v5/name).read_bytes()).hexdigest()==expected,'V5_SOURCE_CHANGED')
    t=load('raw_audit_v5',v5/'transport.py')
    a=load('raw_audit_capacity',WF/'quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/gate.py')
    old=a.gate.old
    final=old.read(run/'SUPERVISOR_RESULT.json');lease=old.read(run/'LEASE_RESULT.json');launch=old.read(run/'LAUNCH_RESULT.json')
    require(final['returncode']==0 and final['error'] is None and final['cleanup_complete'] is True,'ACTIVE_OR_FAILED_NOT_COMPLETE')
    require(lease['execute_returned'] is True and lease['error'] is None and lease['holder_restored'] is True,'LEASE_FAILED')
    require(launch['status']=='ORIGINAL_SUPERVISOR_RETURNED','LAUNCH_FAILED')
    _,cfg,records,source=old.inspect_run(run)
    t.check_amendment(cfg);t.verify_sources()
    t.validate_authorization(old.read(cfg['gpu5_authorization_path']),run.parent,Path(cfg['source_snapshot']),cfg['source_selection_indices'],Path(cfg['gpu5_holder_identity_path']))
    module=t.original_build_supervisor(run.parent,cfg)  # Constructor only: no queries.
    for name in ('GPU_RAW_SAMPLES.jsonl','RESOURCE_SAMPLES.jsonl'):require((run/name).stat().st_size<=2*1024**3,'LOG_BYTE_CAP')
    raw=(run/'GPU_RAW_SAMPLES.jsonl').read_bytes();resources=(run/'RESOURCE_SAMPLES.jsonl').read_bytes()
    result=audit_samples(lines(raw),lines(resources),module.parse_gpu,module.check_gpu,t.telemetry.retry_eligible,old.read(run/'PROCESS.json')['pid'])
    result.update(closed_source_binding_pass=True,run_root=str(run),input_lock_sha256=old.sha(run/'INPUT_LOCK.json'),
        raw_log_sha256=hashlib.sha256(raw).hexdigest(),resource_log_sha256=hashlib.sha256(resources).hexdigest(),
        supervisor_result_sha256=old.sha(run/'SUPERVISOR_RESULT.json'),source_journal_sha256=source['journal_sha256'],
        not_a_replacement_for_family_quality=True,training_admission=False)
    output.mkdir(exist_ok=False);old.save(output/'REPORT.json',result)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(audit_closed(a.run,a.output),indent=2))
