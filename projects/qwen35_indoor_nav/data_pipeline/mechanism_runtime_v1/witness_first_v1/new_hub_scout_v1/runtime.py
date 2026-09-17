"""Prospective GPU2 transport. CPU import only; main owner must approve and run."""
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=load('newhub_scoped_transport',HERE/'adapters.py')
def sha(path):
    import hashlib
    p=Path(path);assert p.resolve()==p and p.is_relative_to(a.a.ROOT)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):
    assert p.is_relative_to(HERE)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def approval_value():
    return {'approved':True,'gpu':2,'source_lock_sha256':sha(HERE/'SOURCE_LOCK.json'),
        'node':'Q35N_NEW_HUB_SCOUT_FOUR_V1','new_hubs':4,'training_allowed':False}
def config():
    lock=read(HERE/'SOURCE_LOCK.json');assert len(lock)<=2048
    assert sum(Path(p).stat().st_size for p in lock)<=32*1024**3
    for p,h in lock.items():assert sha(Path(p))==h,p
    approval=HERE/'MAIN_AGENT_APPROVAL.json';assert read(approval)==approval_value()
    cfg=read(HERE/'PREPARED_CONFIG.json')
    assert cfg['runtime_allowed'] is False and cfg['executable'] is False and cfg['training_allowed'] is False
    assert cfg['gpu_device']==2 and cfg['gpu_uuid']==a.UUID
    assert cfg['budget']=={'total_actions':40000,'total_seconds':2700,'discovery_actions':40000,
        'discovery_seconds':2400,'certification_actions':1,'certification_seconds':1}
    assert cfg['new_hub_plan']['limit']==4 and len(cfg['candidates'])==1
    cfg.update(runtime_allowed=True,executable=True,runtime_adapter_ready=True,
        source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),main_agent_approval_sha256=sha(approval))
    return cfg
def module(name,source,path):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(source,str(path),'exec'),m.__dict__);return m
def run():
    cfg=config();out=HERE/'run_v1';out.mkdir(exist_ok=False)
    save(out/'EXECUTION_CONFIG.json',cfg);save(out/'INPUT_LOCK.json',read(HERE/'SOURCE_LOCK.json'))
    error=None;returned=False
    try:
        supervisor=module('newhub_bounded_supervisor',a.supervisor_source(),a.RUNTIME/'compact_loop_v2/run.py')
        assert supervisor.OUT==out and supervisor.UUID==a.UUID;supervisor.main();returned=True
    except BaseException as exc:error=repr(exc);raise
    finally:save(out/'LAUNCH_RESULT.json',{'supervisor_returned':returned,'error':error,'gpu':2,
        'holders_touched':False,'external_processes_stopped':0,'scientific_pass':False})
def worker():
    cfg=config();name='new_hub_scoped_common_v1'
    common=module(name,a.common_source(),a.a.SCOUT/'common.py');common.runtime_config=lambda:cfg
    assert common.HERE==HERE;sys.modules[name]=common
    work=module('newhub_worker',a.worker_source(),a.a.SCOUT/'worker.py');assert work.HERE==HERE;work.main()
if __name__=='__main__':run()
