"""Exact, hash-bound GPU1 adapters; imports never launch/query a GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
WF=HERE.parent
OLD=WF/'auto_generation_v1'
HASHES={
    'queue.py':'529e8e77c7e37a3f1f2b408d89a192752197cf93ec163e1b1781689a2053c12c',
    'transport.py':'deb355db04568effc38b30daecb2f3809c767fdd463ba6de48c9ca8b00605bbd',
    'prepare.py':'be3f791b1ba1870113d98d664ee5458e9f6cb297910546e8af27794581f87c11',
    'audit.py':'6eef05c22c041abcd7040752897b1773ace0fac0a26a1af347b7d12191d546fc',
    'audit_adapter_v2/audit.py':'75a217158477b3744b2459ddbaa726f1401dc09eb64826f7cf9965826af63264'}
GPU2_ASSERT="assert cfg['gpu_device']==2 and cfg['gpu_uuid']=='GPU-be1b30d0-517b-b079-871b-de195d35a1a2'"
GPU1_ASSERT="assert cfg['gpu_device']==1 and cfg['gpu_uuid']=='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8'"

def exact(source,changes,expected):
    assert hashlib.sha256(source.encode()).hexdigest()==expected,'FROZEN_GPU2_SOURCE_CHANGED'
    value=source
    for old,new in changes:
        assert old!=new and value.count(old)==1,'EXACT_UNIQUE_REPLACEMENT_REQUIRED'
        value=value.replace(old,new)
    reversed_value=value
    for old,new in reversed(changes):
        assert reversed_value.count(new)==1,'EXACT_UNIQUE_REVERSE_REQUIRED'
        reversed_value=reversed_value.replace(new,old)
    assert reversed_value==source,'SOURCE_DIFF_OUTSIDE_REGISTERED_TRANSPORT'
    return value
def changes(name):
    if name=='queue.py':
        return [("'batch_'+str(100+len(batches)),'gpu':2", "'batch_'+str(200+len(batches)),'gpu':1"),
            ("'schema_version':'q35n.auto_generation_queue.v1','gpu_devices':[2]", "'schema_version':'q35n.auto_generation_queue.gpu1.v1','gpu_devices':[1]"),
            ("'Q35N_AUTO_GENERATION_V1_'", "'Q35N_AUTO_GENERATION_GPU1_V1_'"),
            ("if (run/'journal/HEAD.json').exists():", "if run.parent.name not in GPU2_RESERVED_BATCHES and (run/'journal/HEAD.json').exists():"),
            ('    pool_entries=[]\n    for pool in POOLS:', '    verify_gpu2_reservations(prior,lock)\n    pool_entries=[]\n    for pool in POOLS:'),
            ("HERE/'SHA256SUMS',CORE,VERBALIZER,", "HERE/'SHA256SUMS',CORE,VERBALIZER,*original_dependencies(),")]
    if name=='transport.py':
        old="('assert 1<=len(lock)<=1024','assert 1<=len(lock)<=2048')]"
        new="('assert 1<=len(lock)<=1024','assert 1<=len(lock)<=2048'),\n            ("+repr(GPU2_ASSERT)+","+repr(GPU1_ASSERT)+")]"
        return [(old,new),("queue['gpu_devices']==[2]", "queue['gpu_devices']==[1]"),
            ("str(100+index) and row['gpu']==2", "str(200+index) and row['gpu']==1")]
    if name=='prepare.py':
        return [("str(100+index) and item['gpu']==2", "str(200+index) and item['gpu']==1"),
            ("module.prepare(snapshot,item['id'],[0,1,2],2,'winding_v1')", "module.prepare(snapshot,item['id'],[0,1,2],1,'winding_v1')"),
            ("result={'id':item['id'],'gpu':2", "result={'id':item['id'],'gpu':1"),
            ("{'jobs':jobs,'gpu_devices':[2]", "{'jobs':jobs,'gpu_devices':[1]"),
            ("command=[str(PYTHON),'-I','-S','-B',str(batch/'run.py')]", "command=[str(PYTHON),'-I','-B',str(batch/'run.py')]"),
            ("audit_command=[str(PYTHON),'-I','-S','-B',str(HERE/'audit.py'),'--batch',str(batch)]", "audit_command=[str(PYTHON),'-I','-B',str(HERE/'audit.py'),'--batch',str(batch)]"),
            ("t.CLOCK.parent/'SPEC_ZH.md',GATE,GATE.parent/'SHA256SUMS']", "t.CLOCK.parent/'SPEC_ZH.md',GATE,GATE.parent/'SHA256SUMS',*original_dependencies()]")]
    if name=='audit.py':
        return [("out=HERE/'audits'/batch.name", "out=GATE.parent/'auto_generation_gpu1_v1'/batch.name")]
    raise ValueError('UNREGISTERED_ADAPTER')
def original_dependencies():
    paths=[OLD/name for name in HASHES]
    for path in paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==HASHES[str(path.relative_to(OLD))]
    return paths+[OLD/'SHA256SUMS',OLD/'audit_adapter_v2/SHA256SUMS']
def verify_gpu2_reservations(prior,lock):
    path=OLD/'queue_v1/QUEUE.json';source_lock=OLD/'queue_v1/SOURCE_LOCK.json'
    raw=path.read_bytes();lock_value=json.loads(source_lock.read_text())
    assert hashlib.sha256(raw).hexdigest()==lock_value[str(path)],'GPU2_QUEUE_HASH_CHANGED'
    queue=json.loads(raw)
    assert queue['gpu_devices']==[2] and queue['batch_count']==12 and queue['candidate_count']==36
    ids=[]
    for index,batch in enumerate(queue['batches']):
        assert batch['id']=='batch_'+str(100+index) and batch['gpu']==2
        assert len(batch['candidate_ids'])==3
        expected=WF/'batch_execution_v1'/batch['id']/'run_v1'
        actual={p['candidate_id'] for p in prior if p['run_root']==str(expected)}
        assert set(batch['candidate_ids'])==actual,'GPU2_FROZEN_CANDIDATE_NOT_RESERVED'
        ids.extend(batch['candidate_ids'])
    assert len(set(ids))==36
    assert_no_gpu2_mutable(lock)
    for file in (path,source_lock):lock[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    return set(ids)
def assert_no_gpu2_mutable(lock):
    roots=[WF/'batch_execution_v1'/('batch_'+str(i))/'run_v1' for i in range(100,112)]
    for key in lock:
        path=Path(key)
        for root in roots:
            if path.is_relative_to(root):
                assert path==root/'EXECUTION_CONFIG.json','GPU2_ROLLING_STATE_MUST_NOT_BE_SOURCE'
def module(name):
    path=OLD/name;source=path.read_text()
    adapted=exact(source,changes(name),HASHES[name])
    m=types.ModuleType('gpu1_exact_'+name.replace('.','_'));m.__file__=str(HERE/name)
    m.original_dependencies=original_dependencies;m.verify_gpu2_reservations=verify_gpu2_reservations
    m.GPU2_RESERVED_BATCHES={'batch_'+str(i) for i in range(100,112)}
    exec(compile(adapted,str(path)+'::gpu1_prospective_only','exec'),m.__dict__)
    return m
