"""Read-only, exact-hash transport of reviewed runtime; no GPU side effects."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent.parent/'ordinary_parallel_v1/runtime_v1'
ROOT=HERE.parents[4]
EXPECTED={
    'common.py':'c6fa96828a5d5d0e07be467b9dcb02180ce5b73184a789748aa8b5ffb69e4840',
    'run.py':'c674250e1dff16028e933041ed63c5242427ce6ee22f617dab604e896258e5dd',
    'worker.py':'d25df35d2d27840100e1fdf1bc525a0105dd27094a2fb420c9dd69593fed1bc2',
    'safe_size.py':'957ec7851bd448685690d9065882ce208461f733731bbaaa65838cf287b4530d',
    'merge.py':'0c16272e4430b229f199c8b5329f681129c4ad0cb948fb10fd99896db352735f',
    'test_runtime.py':'d4b184434f29432969c3dd5ba6a2f895f1b69274f79e528744abf7c9bf76c66f',
    'INPUT_LOCK.json':'1aa3343a7d176d8eb53587e02adb8e24c3bfc93b7a422e78b63ad298eba856a2'}


def original(name):
    path=(SOURCE/name).resolve(strict=True)
    assert path.is_relative_to(ROOT)
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==EXPECTED[name],('SEALED_SOURCE_CHANGED',name)
    return raw.decode()


def exact(source,old,new):
    assert source.count(old)==1,('EXACT_TRANSPORT_COUNT',old)
    return source.replace(old,new)


def selected_queues(value):
    assert isinstance(value,dict) and value
    lanes={int(k):tuple(v) for k,v in value.items()}
    assert len(lanes)==len(value) and set(lanes)<={3,4,6,7}
    flat=[]
    for gpu,shards in lanes.items():
        assert 1<=len(shards)<=2,'MAX_TWO_SHARDS_PER_7200S_LANE'
        assert all(type(s) is int and 0<=s<7 for s in shards)
        flat.extend(shards)
    assert len(flat)==len(set(flat)),'DUPLICATE_SHARD_OWNERSHIP'
    return lanes


def setup():
    path=HERE/'SETUP.json'
    if path.exists():
        value=json.loads(path.read_text())
        selected_queues(value['gpu_to_shards'])
        return value
    return dict(gpu_to_shards={'3':[0],'4':[1]},
        identities_path=str((HERE/'MAIN_IDENTITIES_REQUIRED.json').relative_to(ROOT)),
        authorization_path=str((HERE/'MAIN_AUTHORIZATION_REQUIRED.json').relative_to(ROOT)),
        executable=False)


def common_source():
    source=original('common.py');cfg=setup()
    replacements=[
        ("IDENTITIES=LINE/'authorizations/ORDINARY_PARALLEL_GPU67_IDENTITIES_V1.json'",
         f"IDENTITIES=ROOT/{cfg['identities_path']!r}"),
        ("AUTH=LINE/'authorizations/ORDINARY_PARALLEL_PRODUCTION_V1.json'",
         f"AUTH=ROOT/{cfg['authorization_path']!r}"),
        ('LANES={6:(0,2),7:(1,3)}',f'LANES={selected_queues(cfg["gpu_to_shards"])!r}\nSELECTED_SHARDS=tuple(sorted(s for v in LANES.values() for s in v))'),
        ('0<=shard<4','0<=shard<7'),
        ('def lane_for(shard):return 6 if shard in (0,2) else 7',
         'def lane_for(shard):\n    owners=[gpu for gpu,shards in LANES.items() if shard in shards]\n    assert len(owners)==1,"UNASSIGNED_SHARD"\n    return owners[0]')]
    for old,new in replacements:source=exact(source,old,new)
    return source


def run_source():
    source=exact(original('run.py'),'choices=[6,7]','choices=[3,4,6,7]')
    helpers='''def checked_holder_environment(identity):
    values=identity.get('project_cache_environment',{})
    allowed={'PYTHONDONTWRITEBYTECODE','XDG_CACHE_HOME','CUDA_CACHE_PATH','TMPDIR','NUMBA_CACHE_DIR','MPLCONFIGDIR'}
    assert isinstance(values,dict) and set(values)<=allowed,'UNREGISTERED_HOLDER_ENV_KEY'
    for key,value in values.items():
        assert isinstance(value,str) and '\\0' not in value
        if key=='PYTHONDONTWRITEBYTECODE':assert value=='1'
        else:assert Path(value).is_absolute() and Path(value).resolve().is_relative_to(c.ROOT),'EXTERNAL_HOLDER_CACHE_PATH'
    return values


def holder_environment_matches(identity,pid):
    expected=checked_holder_environment(identity)
    if not expected:return True
    try:raw=Path('/proc',str(pid),'environ').read_bytes()
    except FileNotFoundError:return False
    actual=dict(item.decode().split('=',1) for item in raw.split(b'\\0') if b'=' in item)
    return all(actual.get(key)==value for key,value in expected.items())


def holder_command(identity):
    values=checked_holder_environment(identity)
    prefix=['env']+[key+'='+value for key,value in sorted(values.items())] if values else []
    return shlex.join(prefix+identity['cmdline'])


'''
    source=exact(source,'def pane(identity,fmt):',helpers+'def pane(identity,fmt):')
    source=exact(source,"    return actual\n\n\ndef same_process_running", 
        "    assert holder_environment_matches(identity,identity['pid']),'HOLDER_ENVIRONMENT_CHANGED'\n    return actual\n\n\ndef same_process_running")
    source=exact(source,"shlex.join(identity['cmdline'])",'holder_command(identity)')
    source=exact(source,"and new_identity['proc_uid']==identity['proc_uid'] and snapshot['processes'].get(pid,0)>20000)",
        "and new_identity['proc_uid']==identity['proc_uid'] and holder_environment_matches(identity,pid)\n                and snapshot['processes'].get(pid,0)>20000)")
    return source


def merge_source():
    source=original('merge.py')
    replacements=[
        ("out=c.PARALLEL/'merge'", "out=HERE/'merge'"),
        ("expected_all={j['physical_source_route_sha256'] for j in c.read(c.PARALLEL/'JOBS.json')}",
         "expected_all={j['physical_source_route_sha256'] for s in c.SELECTED_SHARDS for j in c.read(c.shard_root(s)/'JOBS.json')}"),
        ("['unique_physical_route_keys_excluded']", "['physical_route_keys']"),
        ("assert len(expected_all)==1000 and expected_all.isdisjoint(excluded)",
         "assert len(expected_all)==sum(len(c.read(c.shard_root(s)/'JOBS.json')) for s in c.SELECTED_SHARDS) and expected_all.isdisjoint(excluded)"),
        ('for shard in range(4):','for shard in c.SELECTED_SHARDS:'),
        ('<199*1024**3', '<(49*len(c.SELECTED_SHARDS)+3)*1024**3'),
        ("status='STRICT_CPU_READBACK_MERGED_NOT_TRAINED',shards=summary,",
         "status='STRICT_CPU_READBACK_MERGED_NOT_TRAINED',scope='SELECTED_WAVE_NOT_FULL_POOL',selected_shards=list(c.SELECTED_SHARDS),all_plan_shards=7,shards=summary,")]
    for old,new in replacements:source=exact(source,old,new)
    return source
