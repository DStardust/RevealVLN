"""One fixed GPU1 scout shard; import CPU only, no holder process operations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
a=load('bulk_scoped_adapters',HERE/'adapters.py')
UUID='GPU-734a5268-31fe-6452-105b-36cd08c3d9c8'
def sha(path):
    path=Path(path);assert path.resolve()==path and path.is_relative_to(a.ROOT)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def approval_value(shard):
    assert type(shard) is int and shard==0
    return {'approved':True,'shard':0,'gpu':1,'source_lock_sha256':sha(HERE/'SOURCE_LOCK.json')}
def runtime_config(shard):
    assert type(shard) is int and shard==0
    lock=json.loads((HERE/'SOURCE_LOCK.json').read_text())
    for path,h in lock.items():assert sha(Path(path))==h,path
    approval=HERE/'MAIN_AGENT_APPROVAL_SHARD_00.json'
    assert json.loads(approval.read_text())==approval_value(shard),'EXACT_SHARD_MAIN_AGENT_APPROVAL'
    cfg=json.loads((a.shard_root(shard)/'PREPARED_CONFIG.json').read_text())
    assert cfg['runtime_allowed'] is False and cfg['executable'] is False and cfg['training_allowed'] is False
    assert cfg['gpu_device']==1 and cfg['gpu_uuid']==UUID and cfg['shard_id']==0
    assert cfg['budget']['total_seconds']==4200 and cfg['budget']['total_actions']==60000
    assert a.ROOT/cfg['intended_output_root']==a.shard_root(shard)/'run_v1'
    cfg.update(runtime_allowed=True,executable=True,runtime_adapter_ready=True,
        main_agent_approval_sha256=sha(approval),source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'))
    return cfg
def module(name,source,path):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(source,str(path),'exec'),m.__dict__);return m
def run(shard=0):
    cfg=runtime_config(shard);out=a.shard_root(shard)/'run_v1';out.mkdir(exist_ok=False)
    save(out/'EXECUTION_CONFIG.json',cfg);save(out/'TRANSPORT_APPLIED.json',a.transport_record(shard))
    error=None;returned=False
    try:
        m=module('bulk_scout_supervisor',a.supervisor_source(shard),a.RUNTIME/'compact_loop_v2/run.py')
        assert m.OUT==out and m.UUID==UUID
        m.main();returned=True
    except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
        save(out/'LAUNCH_RESULT.json',{'supervisor_returned':returned,'error':error,
            'gpu':1,'holders_touched':False,'external_processes_stopped':0,'scientific_pass':False})
def worker(shard=0):
    import sys
    cfg=runtime_config(shard)
    name=f'bulk_scout_scoped_common_{shard:02d}'
    common=module(name,a.common_source(shard),a.SCOUT/'common.py')
    common.runtime_config=lambda:cfg
    assert common.HERE==a.shard_root(shard) and common.ROOT==a.ROOT
    sys.modules[name]=common
    work=module('bulk_scout_worker',a.worker_source(shard),a.SCOUT/'worker.py')
    assert work.HERE==a.shard_root(shard)
    work.main()
