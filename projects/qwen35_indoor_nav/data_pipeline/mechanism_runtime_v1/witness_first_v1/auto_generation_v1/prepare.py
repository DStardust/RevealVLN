"""CPU-only prepare_many API; no launch, resume or holder operations."""
import argparse
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('auto_production_transport',HERE/'transport.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
PYTHON=t.WF.parents[4]/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
GATE=t.WF/'quality_cpu/batch_acceptance_v1/gate_v3_manifest_capacity/gate.py'
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def adapted_prepare_source(source):
    assert t.sha(t.BE/'prepare.py')==t.private.HASHES['prepare.py']
    old="cfg['split_sha256']=sha(split)"
    added="\n    cfg.update(budget_transport='clock_batching_fresh_only_v1',fresh_only=True,resume_allowed=False,comparison='unpaired_actual_yield_and_wall_only',cohort_membership=None,auto_retry=False)"
    old2='for p in code:lock[str(p.resolve())]=sha(p)';new2='code += extra_code\n    '+old2
    assert source.count(old)==source.count(old2)==1
    value=source.replace(old,old+added).replace(old2,new2)
    assert value.replace(new2,old2).replace(old+added,old)==source
    return value
def wrapper(entry):
    assert entry in ('worker_main','run_main','audit_main')
    path=HERE/('audit.py' if entry=='audit_main' else 'transport.py')
    return "import importlib.util\nfrom pathlib import Path\nHERE=Path(__file__).resolve().parent\ns=importlib.util.spec_from_file_location('auto_generation_frozen_entry',"+repr(str(path))+")\nt=importlib.util.module_from_spec(s);s.loader.exec_module(t)\nif __name__=='__main__':t."+entry+"(HERE)\n"
def dependencies(snapshot):
    original=t.BE/'budget_batched_transport_v1'
    prior=t.BE/'budget_batched_transport_v2'
    return [*HERE.glob('*.py'),HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',t.QUEUE,HERE/'queue_v1/SOURCE_LOCK.json',
        snapshot/'SOURCE_LOCK.json',original/'transport.py',original/'prepare.py',original/'SHA256SUMS',
        prior/'transport.py',prior/'prepare.py',prior/'SHA256SUMS',t.CLOCK,t.CLOCK.parent/'SHA256SUMS',
        t.CLOCK.parent/'SPEC_ZH.md',GATE,GATE.parent/'SHA256SUMS']
def prepare_one(index):
    queue=json.loads(t.QUEUE.read_text());assert type(index) is int and 0<=index<len(queue['batches'])
    item=queue['batches'][index];assert item['id']=='batch_'+str(100+index) and item['gpu']==2
    snapshot=Path(item['prepared_snapshot']);batch=t.BE/item['id']
    assert not batch.exists(),'NO_REPREPARE_OR_RETRY'
    module=types.ModuleType('auto_private_frozen_prepare');module.__file__=str(t.BE/'prepare.py')
    exec(compile(adapted_prepare_source((t.BE/'prepare.py').read_text()),str(HERE/'prepare.py'),'exec'),module.__dict__)
    module.extra_code=dependencies(snapshot)
    batch.mkdir()
    for name,entry in [('worker.py','worker_main'),('run.py','run_main')]:
        with (batch/name).open('x') as f:f.write(wrapper(entry))
    # Original admission requires exactly these two wrappers before prepare.
    module.prepare(snapshot,item['id'],[0,1,2],2,'winding_v1')
    cfg=t.check_inputs(batch,worker=True)
    # Audit wrapper lives in an independent immutable job directory, never
    # appended into the already frozen batch closure after preparation.
    command=[str(PYTHON),'-I','-S','-B',str(batch/'run.py')]
    audit_command=[str(PYTHON),'-I','-S','-B',str(HERE/'audit.py'),'--batch',str(batch)]
    result={'id':item['id'],'gpu':2,'command':command,'max_seconds':5400,'transport_upper_seconds':4200,'audit_seconds':1200,
        'completion':[{'path':str(batch/'run_v1/SUPERVISOR_RESULT.json'),'equals':{'returncode':0,'error':None,'cleanup_complete':True}}],
        'audit_command':audit_command,'input_lock':str(batch/'run_v1/INPUT_LOCK.json'),
        'input_lock_sha256':t.sha(batch/'run_v1/INPUT_LOCK.json'),
        'candidate_ids':item['candidate_ids'],'runtime_approval_by_cpu_node':False,
        'input_hashes':{str(p):t.sha(p) for p in [batch/'run.py',batch/'run_v1/INPUT_LOCK.json',*HERE.glob('*.py'),t.QUEUE]}}
    save(HERE/'queue_v1'/(item['id']+'_JOB.json'),result)
    return result
def prepare_many(indices=None):
    if indices is None:indices=range(12)
    jobs=[prepare_one(i) for i in indices]
    save(HERE/'queue_v1/JOBS.json',{'jobs':jobs,'gpu_devices':[2],'total_wall_seconds':43200,
        'sequential_per_lane':True,'stop_lane_on_transport_failure':True,'auto_retry':False,
        'hold_processes_touched':False,'gpu_operations':0,'scientific_pass':False})
    print(json.dumps({'prepared_jobs':len(jobs),'job_manifest':str(HERE/'queue_v1/JOBS.json'),
        'max_input_files':max(len(json.loads(Path(j['input_lock']).read_text())) for j in jobs),'gpu_operations':0},indent=2))
    return jobs
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--indices',nargs='+',type=int);a=p.parse_args();prepare_many(a.indices)
