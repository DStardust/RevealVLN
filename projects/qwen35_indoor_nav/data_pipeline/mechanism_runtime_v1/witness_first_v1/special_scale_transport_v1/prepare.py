"""CPU-only immutable preparation of prospective GPU1/2 continuation jobs."""
import importlib.util
import json
from pathlib import Path
import types
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('scale_preparation_transport',HERE/'transport.py')
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
c=t.c

def authorization(queue_path,gpu,main_path):
    queue_path=c.scoped(queue_path);main_path=c.scoped(main_path);queue=c.read(queue_path)
    c.require(gpu in t.GPUS,'NONEXCLUSIVE_FIRST_VERSION')
    value=dict(version='special_scale_transport_v1',approved=True,training_allowed=False,
        main_scope_authorization_path=str(main_path),main_scope_authorization_sha256=c.sha(main_path),
        gpu_device=gpu,gpu_uuid=t.GPUS[gpu],mode='nonexclusive',budget=c.BUDGET,
        supervision_wall_seconds=3900,accounting_amendment=c.AMENDMENT,automatic_retry=False,
        replace_failed_candidate=False,lane_wall_seconds=43200,queue_path=str(queue_path),
        queue_sha256=c.sha(queue_path),batch_ids=[r['id'] for r in queue['batches'] if r['gpu']==gpu])
    c.validate_authorization(value,queue_path)
    out=HERE/'authorizations';out.mkdir(exist_ok=True)
    path=out/('gpu_'+str(gpu)+'_'+c.sha(queue_path)[:16]+'.json')
    if path.exists():c.require(c.read(path)==value,'AUTHORIZATION_ALREADY_DIFFERENT')
    else:c.save(path,value)
    return path

def wrapper(entry):
    c.require(entry in ('run_main','worker_main'),'ENTRY')
    return "import importlib.util\nfrom pathlib import Path\nHERE=Path(__file__).resolve().parent\ns=importlib.util.spec_from_file_location('frozen_special_scale_entry',"+repr(str(HERE/'transport.py'))+")\nt=importlib.util.module_from_spec(s);s.loader.exec_module(t)\nif __name__=='__main__':t."+entry+"(HERE)\n"

def prepare_source(source):
    c.require(c.sha(c.BE/'prepare.py')==c.SOURCES[c.BE/'prepare.py'],'ORIGINAL_PREPARE_SOURCE')
    result=c.exact(source,"    cfg['split_sha256']=sha(split)","    cfg['split_sha256']=sha(split)\n    cfg.update(SCALE_METADATA)")
    return c.exact(result,'    for p in code:lock[str(p.resolve())]=sha(p)',
                   '    code += SCALE_DEPENDENCIES\n    for p in code:lock[str(p.resolve())]=sha(p)')

def check_original_unattempted(item):
    # Prospective source may register an exact old reservation migration. No
    # inference from missing success: source namespace must still be untouched.
    prior=item.get('previous_run_root') or item.get('migrated_from_run')
    if prior:
        root=c.scoped(prior)
        c.require(root.parent.parent==c.BE and root.name=='run_v1','PRIOR_RUN_SCOPE')
        c.require({p.name for p in root.iterdir()}=={'EXECUTION_CONFIG.json','INPUT_LOCK.json'},'PRIOR_RESERVATION_ALREADY_ATTEMPTED')
        cfg=c.read(root/'EXECUTION_CONFIG.json')
        c.require(item['candidate_ids']==[r['candidate_id'] for r in cfg['candidates']],'PRIOR_CANDIDATES_CHANGED')

def prepare_one(queue_path,index,main_authorization):
    queue_path=c.scoped(queue_path);queue=c.read(queue_path)
    c.require(type(index) is int and 0<=index<len(queue['batches']),'QUEUE_INDEX')
    item=queue['batches'][index];check_original_unattempted(item)
    auth=authorization(queue_path,item['gpu'],main_authorization)
    batch=c.BE/item['id'];c.require(not batch.exists(),'FRESH_OUTPUT_NO_REPREPARE')
    snapshot=c.scoped(item['prepared_snapshot']);draft=c.read(snapshot/'CONFIG_DRAFT.json')
    c.require(len(draft['candidates'])==3 and [r['candidate_id'] for r in draft['candidates']]==item['candidate_ids'],'EXACT_THREE_SOURCE_ROWS')
    source=c.BE/'prepare.py'
    m=types.ModuleType('scale_private_original_prepare');m.__file__=str(source)
    exec(compile(prepare_source(source.read_text()),str(source)+'::scale_metadata','exec'),m.__dict__)
    m.SCALE_METADATA=dict(runtime_transport_version='special_scale_transport_v1',accounting_amendment=c.AMENDMENT,
        budget_transport='clock_batching_fresh_only_v1',fresh_only=True,resume_allowed=False,
        cohort_membership=None,auto_retry=False,lease_mode='nonexclusive',queue_path=str(queue_path),
        scale_authorization_path=str(auth),resource_guard_amended=True,
        comparison='unpaired_actual_yield_and_wall_only',candidate_is_accepted_family=False)
    m.SCALE_DEPENDENCIES=[*t.dependency_paths(),queue_path,queue_path.parent/'SOURCE_LOCK.json',auth,c.scoped(main_authorization)]
    for p,h in c.read(queue_path.parent/'SOURCE_LOCK.json').items():
        c.require(c.sha(p)==h,'QUEUE_SOURCE_CHANGED')
        m.SCALE_DEPENDENCIES.append(Path(p))
    batch.mkdir()
    for name,entry in [('run.py','run_main'),('worker.py','worker_main')]:
        with (batch/name).open('x') as f:f.write(wrapper(entry))
    m.prepare(snapshot,item['id'],[0,1,2],item['gpu'],'winding_v1')
    cfg=t.check_inputs(batch,worker=True)
    runtime_lock=c.read(batch/'run_v1/INPUT_LOCK.json')
    job=dict(id=batch.name,gpu=item['gpu'],command=[str(c.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-B',str(batch/'run.py')],
        max_seconds=5400,transport_upper_seconds=4200,audit_seconds=1200,
        completion=[dict(path=str(batch/'run_v1/SUPERVISOR_RESULT.json'),equals={'returncode':0,'error':None,'cleanup_complete':True})],
        audit_command=[str(c.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-B',str(HERE/'audit.py'),'--batch',str(batch)],
        candidate_ids=item['candidate_ids'],input_lock=str(batch/'run_v1/INPUT_LOCK.json'),
        input_lock_sha256=c.sha(batch/'run_v1/INPUT_LOCK.json'),main_agent_approval_required=t.approval_value(batch),
        input_hashes={str(p):c.sha(p) for p in [batch/'run.py',batch/'run_v1/INPUT_LOCK.json',*t.dependency_paths()]},
        source_candidate_count=3,scientific_pass=False,training_allowed=False)
    out=HERE/'jobs';out.mkdir(exist_ok=True);c.save(out/(batch.name+'.json'),job)
    return job

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--indices',nargs='+',required=True,type=int);p.add_argument('--main-authorization',required=True)
    a=p.parse_args();print(json.dumps([prepare_one(a.queue,i,a.main_authorization) for i in a.indices],indent=2))
