"""CPU-only chained-holder preparation. Never borrow a GPU here."""
import importlib.util
import json
from pathlib import Path
import types
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('holder_preparation_transport',HERE/'transport.py')
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
c=t.c

def authorization(queue_path,gpu,main_path):
    queue_path=c.scoped(queue_path);main_path=c.scoped(main_path)
    queue=c.read(queue_path);broad=c.read(main_path)
    collection_path=c.scoped(c.LINE/broad['holder_identities']);collection=c.read(collection_path)
    matches=[r for r in collection['holders'] if r['gpu_device']==gpu]
    c.require(len(matches)==1,'UNIQUE_MAIN_HOLDER')
    document=dict(matches[0]);t.check_identity(document,gpu)
    document.update(collection_path=str(collection_path),collection_sha256=c.sha(collection_path))
    out=HERE/'authorizations';out.mkdir(exist_ok=True)
    initial=out/('gpu_'+str(gpu)+'_initial_'+c.sha(collection_path)[:16]+'.json')
    if initial.exists():c.require(c.read(initial)==document,'FROZEN_INITIAL_IDENTITY_CHANGED')
    else:c.save(initial,document)
    value=dict(version='special_scale_transport_v1',approved=True,training_allowed=False,
        main_scope_authorization_path=str(main_path),main_scope_authorization_sha256=c.sha(main_path),
        gpu_device=gpu,gpu_uuid=t.k.GPUS[gpu],mode='holder_chain',budget=c.BUDGET,
        supervision_wall_seconds=3900,accounting_amendment=c.AMENDMENT,automatic_retry=False,
        replace_failed_candidate=False,lane_wall_seconds=43200,queue_path=str(queue_path),queue_sha256=c.sha(queue_path),
        batch_ids=[r['id'] for r in queue['batches'] if r['gpu']==gpu],
        initial_holder_identity_path=str(initial),initial_holder_identity_sha256=c.sha(initial),
        restore_holder_on_success_failure_or_interruption=True,
        chain_identity_from_immediate_previous_restoration_only=True)
    c.validate_authorization(value,queue_path)
    path=out/('gpu_'+str(gpu)+'_queue_'+c.sha(queue_path)[:16]+'.json')
    if path.exists():c.require(c.read(path)==value,'FROZEN_LANE_AUTHORIZATION_CHANGED')
    else:c.save(path,value)
    return path,initial,collection_path

def wrapper(entry):
    c.require(entry in ('run_main','worker_main'),'ENTRY')
    return "import importlib.util\nfrom pathlib import Path\nHERE=Path(__file__).resolve().parent\ns=importlib.util.spec_from_file_location('frozen_holder_scale_entry',"+repr(str(HERE/'transport.py'))+")\nt=importlib.util.module_from_spec(s);s.loader.exec_module(t)\nif __name__=='__main__':t."+entry+"(HERE)\n"

def prepare_source(source):
    c.require(c.sha(c.BE/'prepare.py')==c.SOURCES[c.BE/'prepare.py'],'ORIGINAL_PREPARE_SOURCE')
    value=c.exact(source,"assert gpu in (1,2) and variant in ('lr_v2','winding_v1')", "assert gpu in HOLDER_GPUS and variant == 'winding_v1'")
    value=c.exact(value,"gpu_uuid={1:'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',2:'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'}[gpu]",'gpu_uuid=HOLDER_GPUS[gpu]')
    value=c.exact(value,"    cfg['split_sha256']=sha(split)","    cfg['split_sha256']=sha(split)\n    cfg.update(HOLDER_METADATA)")
    return c.exact(value,'    for p in code:lock[str(p.resolve())]=sha(p)',
                   '    code += HOLDER_DEPENDENCIES\n    for p in code:lock[str(p.resolve())]=sha(p)')

def prepare_one(queue_path,index,main_authorization):
    queue_path=c.scoped(queue_path);queue=c.read(queue_path)
    c.require(type(index) is int and 0<=index<len(queue['batches']),'QUEUE_INDEX')
    item=queue['batches'][index];c.require(item['gpu'] in t.k.GPUS,'HOLDER_GPU_SCOPE')
    auth,initial,collection=authorization(queue_path,item['gpu'],main_authorization)
    lane=c.read(auth)['batch_ids'];position=lane.index(item['id'])
    previous=c.BE/lane[position-1] if position else None
    prior_lock=previous/'run_v1/INPUT_LOCK.json' if previous else None
    if prior_lock:c.require(prior_lock.is_file(),'PREPARE_PREDECESSOR_FIRST')
    batch=c.BE/item['id'];c.require(not batch.exists(),'FRESH_OUTPUT_NO_REPREPARE')
    snapshot=c.scoped(item['prepared_snapshot']);draft=c.read(snapshot/'CONFIG_DRAFT.json')
    c.require(len(draft['candidates'])==3 and [r['candidate_id'] for r in draft['candidates']]==item['candidate_ids'],'EXACT_THREE_SOURCE_ROWS')
    original=c.BE/'prepare.py';m=types.ModuleType('holder_private_original_prepare');m.__file__=str(original)
    exec(compile(prepare_source(original.read_text()),str(original)+'::holder_metadata','exec'),m.__dict__)
    m.HOLDER_GPUS=t.k.GPUS
    m.HOLDER_METADATA=dict(runtime_transport_version='special_scale_transport_v1',holder_transport_version='special_scale_holder_v1',
        accounting_amendment=c.AMENDMENT,budget_transport='clock_batching_fresh_only_v1',fresh_only=True,resume_allowed=False,
        cohort_membership=None,auto_retry=False,lease_mode='holder_chain',queue_path=str(queue_path),
        scale_authorization_path=str(auth),resource_guard_amended=True,holder_identity_collection_path=str(collection),
        previous_holder_batch=str(previous) if previous else None,previous_holder_input_lock_sha256=c.sha(prior_lock) if prior_lock else None,
        comparison='unpaired_actual_yield_and_wall_only',candidate_is_accepted_family=False)
    m.HOLDER_DEPENDENCIES=[*t.dependency_paths(),queue_path,queue_path.parent/'SOURCE_LOCK.json',auth,initial,collection,
        c.scoped(main_authorization),c.ROOT/'scripts/occupy_idle_gpu.py']+([prior_lock] if prior_lock else [])
    for path,h in c.read(queue_path.parent/'SOURCE_LOCK.json').items():
        c.require(c.sha(path)==h,'QUEUE_SOURCE_CHANGED');m.HOLDER_DEPENDENCIES.append(Path(path))
    batch.mkdir()
    for name,entry in [('run.py','run_main'),('worker.py','worker_main')]:
        with (batch/name).open('x') as f:f.write(wrapper(entry))
    m.prepare(snapshot,item['id'],[0,1,2],item['gpu'],'winding_v1')
    cfg=t.check_inputs(batch,worker=True)
    paths=[batch/'run.py',batch/'run_v1/INPUT_LOCK.json',*t.dependency_paths()]
    python=str(c.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3')
    job=dict(id=item['id'],gpu=item['gpu'],command=[python,'-I','-B',str(batch/'run.py')],
        max_seconds=5400,transport_upper_seconds=4200,audit_seconds=1200,
        completion=[dict(path=str(batch/'run_v1/SUPERVISOR_RESULT.json'),equals={'returncode':0,'error':None,'cleanup_complete':True}),
            dict(path=str(batch/'run_v1/LEASE_RESULT.json'),equals={'execute_returned':True,'error':None,'holder_restored':True}),
            dict(path=str(batch/'run_v1/RESTORATION.json'),equals={'restored':True,'remain_on_exit_restored':True})],
        audit_command=[python,'-I','-B',str(HERE/'audit.py'),'--batch',str(batch)],
        candidate_ids=item['candidate_ids'],input_lock=str(batch/'run_v1/INPUT_LOCK.json'),
        input_lock_sha256=c.sha(batch/'run_v1/INPUT_LOCK.json'),main_agent_approval_required=t.approval_value(batch),
        input_hashes={str(p):c.sha(p) for p in paths},scientific_pass=False,training_allowed=False,
        previous_holder_batch=cfg['previous_holder_batch'])
    out=HERE/'jobs';out.mkdir(exist_ok=True);c.save(out/(batch.name+'.json'),job)
    return job

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--indices',nargs='+',required=True,type=int);p.add_argument('--main-authorization',required=True)
    a=p.parse_args()
    for index in a.indices:
        job=prepare_one(a.queue,index,a.main_authorization)
        print(json.dumps(dict(id=job['id'],gpu=job['gpu'],input_lock_sha256=job['input_lock_sha256'])),flush=True)
