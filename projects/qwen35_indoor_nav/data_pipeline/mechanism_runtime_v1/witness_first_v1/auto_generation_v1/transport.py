"""Fresh GPU2 transport: exact frozen clock transport, queue-bound new paths."""
import hashlib
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent;WF=HERE.parent;BE=WF/'batch_execution_v1'
ORIGINAL=BE/'budget_batched_transport_v1/transport.py'
ORIGINAL_SHA='386e05e4ae16a1ed062846120fd5352ed4819594a8b4da4462a8645a2e9b1a2c'
QUEUE=HERE/'queue_v1/QUEUE.json'

def source_changes():
    return [("BE=HERE.parent\nWF=BE.parent", "WF=HERE.parent\nBE=WF/'batch_execution_v1'"),
            ("assert batch==given and batch==BE/'batch_07','EXACT_FRESH_BATCH07_ONLY'",
             "assert batch==given and batch in allowed_batches(),'EXACT_FROZEN_AUTO_QUEUE_BATCH_ONLY'"),
            ('assert 1<=len(lock)<=1024','assert 1<=len(lock)<=2048')]
def adapted_source(source):
    assert hashlib.sha256(source.encode()).hexdigest()==ORIGINAL_SHA,'FROZEN_CLOCK_TRANSPORT_CHANGED'
    value=source
    for old,new in source_changes():assert value.count(old)==1;value=value.replace(old,new)
    reverse=value
    for old,new in reversed(source_changes()):assert reverse.count(new)==1;reverse=reverse.replace(new,old)
    assert reverse==source
    return value
def allowed_batches():
    queue=json.loads(QUEUE.read_text())
    assert queue['gpu_devices']==[2] and queue['batch_count']==12 and queue['candidate_count']==36
    result=[]
    for index,row in enumerate(queue['batches']):
        assert row['id']=='batch_'+str(100+index) and row['gpu']==2
        result.append(BE/row['id'])
    return result
private=types.ModuleType('auto_frozen_clock_transport');private.__file__=str(HERE/'transport.py')
private.allowed_batches=allowed_batches
exec(compile(adapted_source(ORIGINAL.read_text()),str(ORIGINAL)+'::auto_queue_paths_capacity_only','exec'),private.__dict__)
sha=private.sha;load=private.load;base=private.base;CLOCK=private.CLOCK;CLOCK_SHA=private.CLOCK_SHA
def check_inputs(batch,worker=False):
    cfg=private.check_inputs(batch,worker=worker)
    lock=json.loads((Path(batch)/'run_v1/INPUT_LOCK.json').read_text())
    for path in (QUEUE,HERE/'queue_v1/SOURCE_LOCK.json',HERE/'queue.py',HERE/'audit.py'):
        assert lock.get(str(path))==sha(path),'QUEUE_AND_AUDIT_CLOSURE_REQUIRED'
    queue=json.loads(QUEUE.read_text());rows=[r for r in queue['batches'] if BE/r['id']==Path(batch)]
    assert len(rows)==1
    item=rows[0]
    assert cfg['source_snapshot']==item['prepared_snapshot'] and cfg['source_selection_indices']==[0,1,2]
    assert [r['candidate_id'] for r in cfg['candidates']]==item['candidate_ids']
    assert cfg['cohort_membership'] is None and cfg['auto_retry'] is False
    return cfg
def worker_main(batch):
    cfg=check_inputs(batch,worker=True);private.build_worker(batch,cfg).main()
def run_main(batch):
    check_inputs(batch);base('readiness_v1.py').run_main(batch)
