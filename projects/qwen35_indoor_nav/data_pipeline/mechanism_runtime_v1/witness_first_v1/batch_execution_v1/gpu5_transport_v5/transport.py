"""Private V4 lease stack; only worker-loop GPU telemetry sampling is amended."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
V4=HERE.parent/'gpu5_transport_v4'
V4_HASHES={'transport.py':'bff767876c6440dd130069dd40f2004c376369e1dbc6fec220a0cb3ebf27acfb',
           'prepare.py':'3195407f2c6b4e4993b585b67c72930efa0c88e71178c6ceded1c2518d1f3904'}
TELEMETRY=HERE.parent/'telemetry_consistency_v1/wrapper.py'
TELEMETRY_SHA='8b3035e5371b9fc16a94f25b438a04692bc8e9252eed9d8124c86b4367616544'
AMENDMENT={'version':'bounded_worker_telemetry_consistency_v1','retry_only':'MEMORY_ACCOUNTING',
    'max_queries_per_sample':3,'retry_window_seconds':1.0,'raw_xml_saved_before_guard':True,
    'original_thresholds_unchanged':True,'lease_restore_queries_wrapped':False,
    'supervision_deadline_seconds':3900}
AUTH_AMENDMENT='raw_before_check__only_safe_accounting_mismatch__max_two_requeries_within_one_second__final_original_guard_required'

def checked_v4(name):
    path=V4/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==V4_HASHES[name],'SEALED_V4_CHANGED'
    return path

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

v4=load('v5_private_v4',checked_v4('transport.py'))
assert hashlib.sha256(TELEMETRY.read_bytes()).hexdigest()==TELEMETRY_SHA,'TELEMETRY_SOURCE_CHANGED'
telemetry=load('v5_sealed_telemetry',TELEMETRY)
base=v4.v3.base
original_supervisor_source=base.supervisor_source
original_build_supervisor=base.build_supervisor

def __getattr__(name):return getattr(v4,name)

def supervisor_source(source):
    result=original_supervisor_source(source)
    old='snapshot=gpu(); upper=check_gpu(snapshot,proc.pid)'
    new='snapshot,upper=_telemetry_sample(proc.pid,started+3900)'
    assert result.count(old)==1 and result.count(new)==0,'EXACT_LOOP_SAMPLING_SUBSTITUTION'
    adapted=result.replace(old,new)
    assert adapted.replace(new,old)==result
    return adapted

def check_amendment(cfg):
    assert cfg['sampling_amendment']==AMENDMENT
    assert cfg['thresholds_unchanged'] is True and cfg['supervision_sampling_policy_changed'] is True
    assert cfg['runtime_transport_version']=='gpu5_transport_v5'
    assert cfg['cohort_member'] is False and cfg['cohort_replacement'] is False
    assert cfg['engineering_retry_count']==1
    assert cfg['engineering_retry_of']==str(base.BE/'batch_06r1/run_v1')
    assert cfg['source_selection_indices']==[9,10,11]

def build_supervisor(batch,cfg):
    check_amendment(cfg)
    module=original_build_supervisor(batch,cfg)
    record=None
    def sampling(worker,deadline):
        nonlocal record
        if record is None:record=telemetry.DurableLog(module.OUT/'GPU_RAW_SAMPLES.jsonl',module.OUT)
        def query(timeout):
            return module.subprocess.check_output(['nvidia-smi','-i','5','-q','-x'],text=True,timeout=timeout)
        return telemetry.sample(query,module.parse_gpu,module.check_gpu,record,worker,
            wall_deadline=deadline,clock=module.time.monotonic)
    module._telemetry_sample=sampling
    original_main=module.main
    def close_receipt():
        if record is not None:record.close()
    def main_with_receipt_close():
        try:return original_main()
        finally:close_receipt()
    module.main=main_with_receipt_close
    module._telemetry_close=close_receipt
    module._v5_original_main=original_main
    return module

# Isolated imported V4/V3/V2 instance only. Lease/restore functions and raw GPU
# query object remain the sealed implementation, with no global module patch.
base.supervisor_source=supervisor_source
base.build_supervisor=build_supervisor

def dependency_paths():
    return [*map(checked_v4,V4_HASHES),*map(v4.checked_v3,v4.V3_HASHES),
        *map(v4.v3.checked,v4.v3.EXPECTED),TELEMETRY,TELEMETRY.parent/'test_wrapper.py',*previous_evidence_paths()]

def previous_evidence_paths():
    prior=base.BE/'batch_06r1/run_v1'
    candidate=v4.read(prior/'EXECUTION_CONFIG.json')['candidates'][0]['candidate_id']
    return [prior/name for name in ('SUPERVISOR_RESULT.json','LEASE_RESULT.json','RESTORATION.json',
        'PROCESS.json','EXECUTION_CONFIG.json','INPUT_LOCK.json','RESOURCE_SAMPLES.jsonl',
        'READINESS_SAMPLES.jsonl','LAUNCH_RESULT.json','journal/events.jsonl','journal/HEAD.json',
        'PROGRESS.pending')]+[prior/'bundles'/candidate/name for name in ('SEMANTIC_INVENTORY.json','traces/000000.json')]+sorted((prior/'content').glob('*.npy'))

def validate_prior_attempt(document,snapshot,indices,identity_path):
    prior=base.BE/'batch_06r1/run_v1'
    cfg=v4.read(prior/'EXECUTION_CONFIG.json');final=v4.read(prior/'SUPERVISOR_RESULT.json')
    lease=v4.read(prior/'LEASE_RESULT.json');restoration=v4.read(prior/'RESTORATION.json')
    assert final['error']=="AssertionError('MEMORY_ACCOUNTING')" and final['returncode']==-15
    assert final['cleanup_complete'] is True and final['external_processes_stopped']==0
    assert lease['holder_restored'] is True and lease['external_processes_stopped']==0 and lease['execute_returned'] is False
    assert restoration['restored'] is True and restoration['remain_on_exit_restored'] is True
    assert document['previous_supervisor_wall_seconds']==final['wall_seconds']
    assert cfg['source_selection_indices']==indices and Path(cfg['source_snapshot'])==Path(snapshot)
    draft=v4.read(Path(snapshot)/'CONFIG_DRAFT.json')
    for index,old in zip(indices,cfg['candidates']):
        row=dict(draft['candidates'][index]);row.update(runtime_allowed=True,executable=True)
        assert row==old,'PRIOR_SOURCE_CANDIDATE_CHANGED'
    assert len(cfg['candidates'])==len(indices)==3
    identity=v4.read(identity_path)
    assert identity['pid']==restoration['pid'] and identity['starttime_ticks']==restoration['process_identity']['starttime_ticks']
    raw=(prior/'journal/events.jsonl').read_bytes();head=v4.read(prior/'journal/HEAD.json')
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    previous='0'*64;records=[]
    for i,line in enumerate(raw.splitlines()):
        record=json.loads(line);body={k:v for k,v in record.items() if k!='hash'}
        assert canonical(record)==line and record['seq']==i and record['prev']==previous
        assert hashlib.sha256(canonical(body)).hexdigest()==record['hash']
        previous=record['hash'];records.append(record)
    assert raw.endswith(b'\n') and head['count']==len(records) and head['byte_length']==len(raw) and head['last_hash']==previous
    assert records[0]['kind']=='__config__' and records[0]['payload']==cfg
    assert head['config_hash']==hashlib.sha256(canonical(cfg)).hexdigest()
    traces=[r['payload'] for r in records if r['kind']=='trace_saved']
    assert len(traces)==1 and not any(r['kind']=='action_completed' for r in records)
    path=prior/'bundles'/cfg['candidates'][0]['candidate_id']/'traces/000000.json'
    trace=v4.read(path)
    assert traces[0]['sha256']==v4.sha(path) and trace['complete'] is True
    assert trace['actions']==[] and len(trace['observations'])==1
    assert not (prior/'result.json').exists(),'PRIOR_FAILURE_EVIDENCE_CHANGED'
    assert len(list((prior/'content').glob('*.npy')))==2
    return {'prior_wall_seconds':final['wall_seconds'],'physical_observation_traces':1,
            'confirmed_actions':0,'cohort_failure_preserved':True}

def verify_sources():
    for name in V4_HASHES:checked_v4(name)
    assert hashlib.sha256(TELEMETRY.read_bytes()).hexdigest()==TELEMETRY_SHA,'TELEMETRY_SOURCE_CHANGED'
    v4.verify_sources()

def validate_authorization(document,batch,snapshot,indices,identity_path):
    assert Path(batch)==base.BE/'batch_06r2' and indices==[9,10,11],'FIXED_ENGINEERING_RETRY_ONLY'
    assert Path(snapshot)==base.WF/'multi_program_bank_v1/language_ready_v2','SAME_SOURCE_REQUIRED'
    assert document.get('supervisor_sampling_protocol_changed') is True
    assert document.get('sampling_amendment')==AUTH_AMENDMENT
    assert document.get('lease_and_restore_guards_unchanged') is True
    assert document.get('old_cohort_replacement_allowed') is False and document.get('previous_failure_preserved') is True
    assert document.get('previous_attempt')==str((base.BE/'batch_06r1/run_v1').relative_to(base.LINE))
    v4.validate_authorization(document,batch,snapshot,indices,identity_path)
    validate_prior_attempt(document,snapshot,indices,identity_path)

def run_main(batch):
    verify_sources();batch=Path(batch).resolve(strict=True)
    assert batch==base.BE/'batch_06r2','FIXED_ENGINEERING_RETRY_ONLY'
    cfg=v4.read(batch/'run_v1/EXECUTION_CONFIG.json');check_amendment(cfg)
    validate_authorization(v4.read(cfg['gpu5_authorization_path']),batch,Path(cfg['source_snapshot']),
                           cfg['source_selection_indices'],Path(cfg['gpu5_holder_identity_path']))
    lock=v4.read(batch/'run_v1/INPUT_LOCK.json')
    for path in [HERE/'transport.py',HERE/'prepare.py',HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',*dependency_paths()]:
        assert lock.get(str(path))==v4.sha(path),'V5_SOURCE_NOT_LOCKED'
    v4.run_main(batch)
