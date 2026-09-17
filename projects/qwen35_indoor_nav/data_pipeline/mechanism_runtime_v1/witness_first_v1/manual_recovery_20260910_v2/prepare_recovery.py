"""One explicit user-approved retry wave. All old code/locks/results stay read-only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
WF=HERE.parent
BE=WF/'batch_execution_v1'
LINE=WF.parents[2]
ROOT=LINE.parents[1]
OLD=WF/'special_scale_transport_v1'
AUTO=LINE/'data_pipeline/auto_production_v1'
PAIRS=[(225,500),(244,501),(226,502),(245,503),(227,504),(246,505),(228,506),(247,507)]
AUTH=HERE/'MAIN_AUTHORIZATION.json'

def require(ok,reason):
    if not ok:raise ValueError(reason)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    p=Path(path).resolve();require(p.is_relative_to(ROOT),'PROJECT_HASH_SCOPE')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    path=Path(path);require(path.resolve().is_relative_to(LINE),'LINE_OUTPUT_SCOPE')
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def classify(old_id,state,receipt):
    if old_id in (225,244):
        require(state=='FAILED_NO_AUTOMATIC_RETRY','FAILED_SOURCE_REQUIRED')
        if old_id==225:
            require(receipt.get('error')=="AssertionError('EXTERNAL_RESOURCE_LOAD')" and receipt.get('cleanup_complete') is True,'RESOURCE_INTERRUPTION_ONLY')
            require(receipt.get('external_processes_stopped')==0,'NO_FOREIGN_SIGNALS')
        else:
            require(receipt.get('status')=='LAUNCH_FAILED' and receipt.get('exception',{}).get('type')=='AssertionError' and receipt.get('exception',{}).get('message')=='EXTERNAL_RESOURCE_LOAD','PREWORKER_RESOURCE_FAILURE_ONLY')
            require(receipt.get('external_processes_stopped_by_launcher')==0,'NO_244_FOREIGN_SIGNALS')
            require(receipt.get('process_record_present') is False and receipt.get('supervisor_result_present') is False,'NO_244_WORKER_OR_SUPERVISOR')
        return 'EXPLICIT_ONCE_MANUAL_RETRY_RESOURCE_FAILURE'
    require(state=='PENDING','ONLY_UNATTEMPTED_PENDING_MIGRATION')
    return 'UNATTEMPTED_RESERVATION_MIGRATION'

def precheck():
    for name,digest in read(HERE/'CODE_SEAL.json').items():require(sha(HERE/name)==digest,'RECOVERY_CODE_CHANGED')
    auth=read(AUTH)
    require(auth['approved'] is True and auth['explicit_manual_retry_batches']==[225,244],'EXACT_MANUAL_AUTHORITY')
    require(auth['special_gpus']==[1] and auth['max_special_jobs']==8,'SINGLE_IDLE_GPU_BOUNDED_WAVE')
    lanes={gpu:AUTO/f'special_gpu{gpu}_scale_v2' for gpu in (1,2)}
    outputs={gpu:read(path/f'lane_gpu_{gpu}/RESULT.json') for gpu,path in lanes.items()}
    for gpu,path in lanes.items():
        ident=read(path/f'lane_gpu_{gpu}/IDENTITY.json')
        require(not Path('/proc',str(ident['pid'])).exists(),'OLD_QUEUE_STILL_PRESENT')
    transport=load('sealed_recovery_checks',OLD/'transport.py')
    entries=[];sources={str(AUTH):sha(AUTH)}
    for name in ('prepare_recovery.py','audit.py','test_recovery.py','CODE_SEAL.json','SPEC_ZH.md'):
        sources[str(HERE/name)]=sha(HERE/name)
    for old_id,new_id in PAIRS:
        old_gpu=1 if old_id<240 else 2
        batch=BE/f'batch_{old_id}';run=batch/'run_v1'
        cfg=transport.check_inputs(batch,worker=old_id not in (225,244))
        receipt_path=run/('SUPERVISOR_RESULT.json' if old_id==225 else 'LAUNCH_RESULT.json')
        receipt=read(receipt_path) if old_id in (225,244) else {}
        mode=classify(old_id,outputs[old_gpu]['states'][batch.name],receipt)
        if mode=='UNATTEMPTED_RESERVATION_MIGRATION':
            require({p.name for p in run.iterdir()}=={'EXECUTION_CONFIG.json','INPUT_LOCK.json'},'PENDING_RUN_NOT_PRISTINE')
        else:
            audit=WF/'quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/special_scale_transport_v1'/batch.name/'result.json'
            require(not audit.exists(),'DO_NOT_REPEAT_ALREADY_AUDITED_FAILED_BATCH')
            sources[str(receipt_path)]=sha(receipt_path)
        require(not (BE/f'batch_{new_id}').exists(),'NEW_BATCH_COLLISION')
        ids=[r['candidate_id'] for r in cfg['candidates']]
        require(len(ids)==3 and len(set(ids))==3,'THREE_EXACT_CANDIDATES')
        entry=dict(id=f'batch_{new_id}',gpu=1,prepared_snapshot=cfg['source_snapshot'],candidate_ids=ids,
            recovery_mode=mode,recovery_of_batch=batch.name,old_gpu=old_gpu,independent_new_candidate_count=0)
        if mode=='UNATTEMPTED_RESERVATION_MIGRATION':
            entry.update(previous_run_root=str(run),runtime_recheck_unstarted_required=True)
        else:
            entry.update(manual_retry_of_run=str(run),failure_receipt=str(receipt_path),failure_receipt_sha256=sha(receipt_path))
        entries.append(entry)
        for path in (run/'EXECUTION_CONFIG.json',run/'INPUT_LOCK.json',Path(cfg['source_snapshot'])/'CONFIG_DRAFT.json',Path(cfg['source_snapshot'])/'SOURCE_LOCK.json',lanes[old_gpu]/f'lane_gpu_{old_gpu}/RESULT.json',lanes[old_gpu]/'PLAN.json'):
            sources[str(path)]=sha(path)
    ids=[candidate for item in entries for candidate in item['candidate_ids']]
    require(len(ids)==len(set(ids))==24,'DUPLICATE_RECOVERY_CANDIDATES')
    return entries,sources

def prepare():
    require(not (HERE/'QUEUE.json').exists(),'FRESH_RECOVERY_PREPARATION_ONLY')
    entries,sources=precheck()
    save(HERE/'QUEUE.json',dict(node='EXPLICIT_MANUAL_FAILURE_RECOVERY_20260910_V1',batches=entries,
        user_basis='有用的信息做成图表，然后把失败的任务重新挂上',automatic_retry=False,
        manual_retry_count=2,unattempted_migrations=6,old_results_replaced=False))
    save(HERE/'SOURCE_LOCK.json',sources)
    old=load('sealed_prepare_for_explicit_manual_recovery',OLD/'prepare.py')
    # Capture original runtime wrappers before redirecting preparation outputs.
    wrappers={entry:old.wrapper(entry) for entry in ('run_main','worker_main')}
    old.HERE=HERE
    old.wrapper=lambda entry:wrappers[entry]
    jobs=[];checks=[]
    for index,entry in enumerate(entries):
        job=old.prepare_one(HERE/'QUEUE.json',index,AUTH)
        batch=BE/entry['id'];approval=batch/'MAIN_AGENT_SCALE_APPROVAL.json'
        save(approval,job['main_agent_approval_required'])
        old.t.check_inputs(batch,worker=True)
        row={k:job[k] for k in ('id','gpu','command','max_seconds','transport_upper_seconds','audit_seconds','completion','audit_command','candidate_ids','input_hashes')}
        for path in (approval,HERE/'audit.py',AUTO/'queue.py',HERE/'QUEUE.json',HERE/'SOURCE_LOCK.json',AUTH):row['input_hashes'][str(path)]=sha(path)
        jobs.append(row)
        checks.append(dict(batch=entry['id'],retry_of=entry['recovery_of_batch'],mode=entry['recovery_mode'],
                           input_count=len(read(batch/'run_v1/INPUT_LOCK.json')),worker_inputs_pass=True))
    plan_dir=AUTO/'manual_failure_recovery_gpu1_20260910_v1';plan_dir.mkdir(exist_ok=False)
    plan=dict(node='EXPLICIT_FAILED_LANES_RECOVERED_SEQUENTIALLY_GPU1_V1',training_allowed=False,automatic_retry=False,
              wall_seconds=43200,jobs=jobs,user_authorization=str(AUTH),independent_new_candidates=0)
    queue=load('sealed_queue_validation',AUTO/'queue.py');queue.validate(plan)
    for job in jobs:queue.verify_job(job)
    save(plan_dir/'PLAN.json',plan)
    save(plan_dir/'MAIN_AGENT_APPROVAL.json',dict(approved=True,plan_sha256=sha(plan_dir/'PLAN.json'),queue_source_sha256=sha(AUTO/'queue.py')))
    save(HERE/'PREPARED_RESULT.json',dict(status='CPU_PREPARED_EXPLICIT_ONCE_RECOVERY',unix=time.time(),checks=checks,
        plan_path=str(plan_dir/'PLAN.json'),plan_sha256=sha(plan_dir/'PLAN.json'),candidate_attempts=24,
        new_independent_candidates=0,gpu_started=False,old_results_replaced=False))
    print(json.dumps(dict(plan=str(plan_dir/'PLAN.json'),checks=checks),indent=2))

if __name__=='__main__':prepare()
