"""CPU source transport for future independently approved scout shards only."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
WF=HERE.parent
RUNTIME=WF.parent
ROOT=next(p for p in HERE.parents if p.name=='vla')
SCOUT=WF/'scout_v1'
RECIPE=WF/'winding_balance_v1/next_source_v1/prepare.py'
EXPECTED={SCOUT/'common.py':'415b798f25ef134a6027db4a4206553506addc00d6a6685011b229df6d825782',
          SCOUT/'worker.py':'e18c7dd27996f15d00a1b814eab6a31b0ebac3be3f602e82c7cda50ffa501113',
          RECIPE:'1aa5eb685bf8a23f5c9bbe825442c76ba4cc84570e59993945566ed6e56edf67',
          RUNTIME/'compact_loop_v2/run.py':'9aabce69787bc717f6283b7679fef3c2f811018f0456741dc51516ebe6e986ca'}

def exact(source,old,new):
    if source.count(old)!=1:raise ValueError('EXACT_SOURCE_TRANSPORT_COUNT:'+old)
    return source.replace(old,new)
def checked(path):
    source=path.read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=EXPECTED[path]:raise ValueError('FROZEN_SOURCE_CHANGED')
    return source
def shard_root(shard):
    if type(shard) is not int or not 0<=shard<6:raise ValueError('FIRST_WAVE_SIX_SHARDS_ONLY')
    return HERE/f'shard_{shard:02d}'
def common_source(shard):
    source=checked(SCOUT/'common.py')
    for old,new in [('HERE=Path(__file__).resolve().parent',f'HERE=Path({str(shard_root(shard))!r})'),
        ('RUNTIME=HERE.parents[1]',f'RUNTIME=Path({str(RUNTIME)!r})'),
        ("str(HERE.parent/'budget_and_trace')",f'str(Path({str(WF/"budget_and_trace")!r}))'),
        ("HERE.parent/'bank_cpu/bank.py'",f'Path({str(WF/"bank_cpu/bank.py")!r})')]:source=exact(source,old,new)
    # Caller must supply separately approved config. Do not retain original
    # single-GPU approval reader accidentally reachable in the transported code.
    marker='def runtime_config():\n'
    if source.count(marker)!=1:raise ValueError('RUNTIME_CONFIG_DEFINITION_COUNT')
    source=source[:source.index(marker)]+"def runtime_config():\n    raise RuntimeError('BULK_SCOUT_REQUIRES_SEPARATE_RUNTIME_ADMISSION')\n"
    return source
def worker_source(shard,gpu=1):
    if gpu!=1:raise ValueError('ONLY_GPU1_SCOUT_TRANSPORT_DEFINED')
    source=checked(SCOUT/'worker.py')
    for old,new in [('HERE=Path(__file__).resolve().parent',f'HERE=Path({str(shard_root(shard))!r})'),
        ('from common import (',f'from bulk_scout_scoped_common_{shard:02d} import ('),
        ("HabitatBackend(candidate['scene_glb'],1,candidate['roles']",f"HabitatBackend(candidate['scene_glb'],{gpu},candidate['roles']")]:source=exact(source,old,new)
    return source
def supervisor_source(shard):
    if shard!=0:raise ValueError('ONLY_FIRST_SHARD_RUNTIME_TRANSPORTED')
    source=checked(RUNTIME/'compact_loop_v2/run.py')
    for old,new in [('HERE = Path(__file__).resolve().parent',f'HERE = Path({str(shard_root(shard))!r})'),
        ('LINE = HERE.parents[2]',f'LINE = Path({str(ROOT/"projects/qwen35_indoor_nav")!r})'),
        ("sample['elapsed'] < 3000","sample['elapsed'] < 4500"),
        ("sample['disk_bytes'] < 7*1024**3","sample['disk_bytes'] < 8*1024**3"),
        ("str(HERE/'worker.py')",f"{str(HERE/'worker.py')!r},'--shard','{shard}'")]:source=exact(source,old,new)
    return source
def recipe_source(shard):
    """Exact old bank recipe, limited to a new shard's completed-house sources."""
    root=shard_root(shard);source=checked(RECIPE)
    for old,new in [('HERE=Path(__file__).resolve().parent',f'HERE=Path({str(root/"bank_snapshots")!r})'),
        ('WF=HERE.parents[1]',f'WF=Path({str(WF)!r})'),
        ('sys.path.insert(0,str(HERE.parent))',f'sys.path.insert(0,{str(WF/"winding_balance_v1")!r})'),
        ("source=WF/'scout_next_v1/shard_0/run_v1'",f'source=Path({str(root/"run_v1")!r})'),
        ("[HERE.parent/'method.py',HERE/'prepare.py',WF/'bank_cpu/bank.py'",
         f"[Path({str(WF/'winding_balance_v1/method.py')!r}),Path({str(HERE/'adapters.py')!r}),Path({str(RECIPE)!r}),WF/'bank_cpu/bank.py'")]:source=exact(source,old,new)
    return source
def transport_record(shard):
    versions={name:hashlib.sha256(fn(shard).encode()).hexdigest() for name,fn in
        [('common',common_source),('worker_gpu1',worker_source),('bank_recipe',recipe_source)]}
    return {'shard_id':shard,'output_root':str(shard_root(shard)/'run_v1'),
        'store_scope':str(shard_root(shard)),'gpu_assignment':1,'prospective_worker_gpu':1,
        'runtime_allowed':False,'executable':False,'main_agent_runtime_approval_required':True,
        'source_hashes':versions,'algorithm_changed':False,'scientific_pass':False}
