"""Strict CPU utilities and frozen scout/telemetry dependencies."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;WF=HERE.parent;RUNTIME=WF.parent
LINE=RUNTIME.parents[1];ROOT=LINE.parents[1]
GPU=7;UUID='GPU-3f8830f6-45f2-6c4d-776b-2b348159ccb3'
QUEUE=WF/'new_hub_scale_v1/snapshot_v1/SCOUT_QUEUE.json'
OLD=WF/'new_hub_scout_v1'
BUDGET=dict(total_actions=40000,total_seconds=2700,discovery_actions=40000,discovery_seconds=2400,
    certification_actions=1,certification_seconds=1)
PINNED={OLD/'SOURCE_LOCK.json':'4b79d384471801926ee23c60d7106184c614be4d8c5906b34608169fae5ce257',
    OLD/'adapters.py':'b4861cc1b8efc8247aa21d26df1dce19cced5410ac8bc028b35ba30c3bfdc1da',
    OLD/'selector.py':'838602cfa40bbddb53d7cc54cf71ecee9eea4b30755b9434adb485e05327134a',
    WF/'special_scale_transport_v1/telemetry.py':'114f1958c5932fb5e7de10b634edf5150e102cee7764ff28fca5ec9d0bd7c50c',
    WF/'special_scale_transport_v1/common.py':'54c7d79f4ea4095131d1ef28fa74a97286925ffa684bebe7d437530d2647ec54',
    WF/'special_scale_holder_v1/transport.py':'811fe211c373a670b0dfd2c2bf8e2c784adf57d88eea14357261652b93ac016f',
    WF/'special_scale_holder_v1/SHA256SUMS':'ea91ec3493ed6cbc2a74e5a61a301642797b779e6831a0f850ddde6d7ed50b0a'}
def require(ok,reason):
    if not ok:raise ValueError(reason)
def scoped(path):
    p=Path(path).absolute();require(p==p.resolve() and p.is_relative_to(LINE),'EXACT_LINE_SCOPE');return p
def sha(path):
    p=Path(path).absolute();require(p==p.resolve() and p.is_relative_to(ROOT),'HASH_SCOPE')
    before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    after=p.stat();require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'SOURCE_CHANGED')
    return h.hexdigest()
def read(path):return json.loads(scoped(path).read_text())
def save(path,value):
    with scoped(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def append(path,value):
    with scoped(path).open('a') as f:f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def load(name,path):
    p=scoped(path)
    if p in PINNED:require(sha(p)==PINNED[p],'FROZEN_SOURCE_CHANGED:'+str(p))
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def exact(source,old,new):
    require(source.count(old)==1 and old!=new,'EXACT_UNIQUE_SOURCE_CHANGE')
    changed=source.replace(old,new);require(changed.count(new)==1 and changed.replace(new,old)==source,'EXACT_SOURCE_REVERSE');return changed
def dependencies():
    for p,h in PINNED.items():require(sha(p)==h,'PINNED_SOURCE_CHANGED')
    old=read(OLD/'SOURCE_LOCK.json');result=dict(PINNED)
    for p,h in old.items():
        if p.endswith('.py') or p.endswith('/SHA256SUMS'):
            require(sha(p)==h,'OLD_SCOUT_CODE_CLOSURE_CHANGED');result[Path(p)]=h
    ordinary=LINE/'data_pipeline/ordinary_fullscale_source_v1/auto_generation_v4/telemetry.py'
    require(sha(ordinary)=='c0488a18a30477ec5323b409f0b4a31d38b668677be8fe923195fb682d6f3976','FROZEN_ACCOUNTING_MATH')
    result[ordinary]=sha(ordinary)
    return result
def job_root(job):
    job=scoped(job);require(job.parent==HERE/'jobs' and re.fullmatch(r'scout_[0-9]{3}',job.name),'FROZEN_SCOUT_JOB_ROOT');return job
def item_for(job):
    job=job_root(job);rows=[r for r in read(QUEUE)['jobs'] if r['id']==job.name]
    require(len(rows)==1,'REGISTERED_SCOUT_ONLY');return rows[0]

def verify_lock(path):
    lock=read(path);require(isinstance(lock,dict) and 0<len(lock)<=2048,'SOURCE_READ_CAP')
    require(all(Path(p).absolute()==Path(p).resolve() and Path(p).is_relative_to(ROOT) for p in lock),'LOCK_PROJECT_SCOPE')
    require(sum(Path(p).stat().st_size for p in lock)<=32*1024**3,'SOURCE_BYTE_CAP')
    for p,h in lock.items():require(sha(p)==h,'SOURCE_HASH_CHANGED:'+p)
    return lock

def prepared_config(item):
    import copy
    value=copy.deepcopy(read(item['prepared_config']))
    require(value['budget']==BUDGET and value['runtime_allowed'] is False and value['executable'] is False,'PROSPECTIVE_ONLY')
    require(value['new_hub_plan']['limit']==4 and len(value['candidates'])==1,'FIXED_FOUR_ONE_HOUSE')
    require(value['candidates'][0]['house_id']==item['house_id'],'HOUSE_BINDING')
    value.update(gpu_device=GPU,gpu_uuid=UUID,supervision_wall_seconds=3000,
        total_disk_budget_bytes=7*1024**3,max_hubs_per_house=4,
        intended_output_root=str((HERE/'jobs'/item['id']/'run_v1').relative_to(ROOT)),
        runtime_transport_version='new_hub_scale_runtime_v1')
    return value

def authorization(path):
    value=read(path)
    require(value.get('approved') is True and value.get('node')=='Q35N_NEW_HUB_SCALE_RUNTIME_V1','EXPLICIT_SCOUT_AUTHORIZATION')
    require(value.get('gpu')==GPU and value.get('gpu_uuid')==UUID,'EXACT_GPU7')
    require(value.get('queue_path')==str(QUEUE) and value.get('queue_sha256')==sha(QUEUE),'QUEUE_AUTH_BINDING')
    require(value.get('budget')==BUDGET and value.get('supervision_wall_seconds')==3000,'ORIGINAL_SCOUT_BUDGET')
    require(value.get('training_allowed') is False and value.get('no_unregistered_retry') is True,'DATA_ONLY_NO_RETRY')
    rows=read(QUEUE)['jobs'];ids=value.get('job_ids')
    require(isinstance(ids,list) and 1<=len(ids)<=43 and len(set(ids))==len(ids),'BOUNDED_JOB_IDS')
    require(ids==[r['id'] for r in rows if r['id'] in ids],'QUEUE_SOURCE_ORDER')
    require(value.get('lane_wall_seconds')==43200 and value.get('shared_gpu_lock')==str(WF/'special_scale_holder_v1/locks/gpu_7.lock'),'SCOUT_LANE_MUTEX_AND_WALL')
    identity=scoped(value['initial_holder_identity_path'])
    require(sha(identity)==value['initial_holder_identity_sha256'],'INITIAL_IDENTITY_BINDING')
    return value
