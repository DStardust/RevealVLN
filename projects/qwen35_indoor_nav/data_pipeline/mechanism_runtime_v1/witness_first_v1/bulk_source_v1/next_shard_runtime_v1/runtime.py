"""Independent second scout shard on idle GPU2; unchanged physical algorithm."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
BULK = HERE.parent
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
p = load('next_scout_source_prepare', BULK/'prepare.py')
a = load('next_scout_source_adapter', BULK/'adapters.py')
SHARD = 1
GPU = 2
UUID = 'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'
OUT = HERE/'run_v1'
AUTH = p.LINE/'authorizations/SPECIAL_BULK_SCOUT_NEXT_GPU2_V1.json'
OLD_ROOT = a.shard_root(SHARD)

def common_source():
    return a.common_source(SHARD).replace(repr(str(OLD_ROOT)), repr(str(HERE)))

def worker_source():
    source = a.worker_source(SHARD)
    source = a.exact(source, repr(str(OLD_ROOT)), repr(str(HERE)))
    return a.exact(source, "HabitatBackend(candidate['scene_glb'],1,candidate['roles']",
                   "HabitatBackend(candidate['scene_glb'],2,candidate['roles']")

def supervisor_source():
    source = a.supervisor_source(0)
    source = a.exact(source, repr(str(a.shard_root(0))), repr(str(HERE)))
    source = a.exact(source, "'nvidia-smi','-i','1'", "'nvidia-smi','-i','2'")
    source = a.exact(source, "UUID = 'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8'", 'UUID = '+repr(UUID))
    return a.exact(source, repr(str(BULK/'worker.py'))+",'--shard','0'", repr(str(HERE/'worker.py')))

def save(path, value):
    path = Path(path)
    assert path.is_relative_to(HERE)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)

def prepare():
    assert not (HERE/'SOURCE_LOCK.json').exists() and not OUT.exists()
    auth = p.read(AUTH)
    assert auth['approved'] and not auth['training_allowed']
    lock = p.read(BULK/'SOURCE_LOCK.json')
    for path, digest in lock.items():
        assert p.sha(path) == digest, path
    inventory = p.read(BULK/'INVENTORY_ALL_37.json')['rows'][3:6]
    houses = [r['house_id'] for r in inventory]
    assert len(set(houses)) == 3
    first = p.read(BULK/'shard_00/PREPARED_CONFIG.json')
    assert not set(houses) & {r['house_id'] for r in first['candidates']}
    rows = p.read(p.MANIFEST)
    positions = p.read(BULK/'SOURCE_POSITIONS.json')
    fit = set(p.read(p.SPLIT)['FIT'])
    candidates = []
    for inv in inventory:
        house = inv['house_id']
        row = copy.deepcopy(next(r for r in rows if r['house_id'] == house and r['house_candidate_rank'] == 0))
        assert house in fit and row['split'] == 'FIT'
        assets = {x['path']:p.sha(x['path']) for x in inv['assets']}
        assert all(x['sha256'] is None or x['sha256'] == assets[x['path']] for x in inv['assets'])
        lock.update(assets)
        row.update(assets=assets, scene_glb=next(x for x in assets if x.endswith('.glb')),
                   source_positions=positions[house]['positions'], scene_group_id='mp3d:'+house,
                   source_house_order=inv['source_order'], metadata_permission_roles_are_not_actual_witness_roles=True)
        candidates.append(row)
    cfg = copy.deepcopy(first)
    cfg.update(node='Q35N_BULK_SCOUT_NEXT_GPU2_V1', candidates=candidates, gpu_device=GPU,
               gpu_uuid=UUID, shard_id=SHARD, intended_output_root=str(OUT.relative_to(p.ROOT)))
    save(HERE/'PREPARED_CONFIG.json', cfg)
    for name, fn in [('common',common_source), ('worker',worker_source), ('supervisor',supervisor_source)]:
        compile(fn(), '<checked_'+name+'>', 'exec')
    for path in [*HERE.glob('*.py'), HERE/'PREPARED_CONFIG.json', AUTH, BULK/'SOURCE_LOCK.json']:
        lock[str(path)] = p.sha(path)
    save(HERE/'SOURCE_LOCK.json', lock)
    print(json.dumps({'houses':houses, 'locked_files':len(lock), 'source_lock_sha256':p.sha(HERE/'SOURCE_LOCK.json'),
                      'runtime_allowed':False, 'gpu_operations':0}))

def approval_value():
    return dict(approved=True, shard=SHARD, gpu=GPU, source_lock_sha256=p.sha(HERE/'SOURCE_LOCK.json'))

def config():
    lock = p.read(HERE/'SOURCE_LOCK.json')
    for path, digest in lock.items():
        assert p.sha(path) == digest, path
    assert p.read(HERE/'MAIN_AGENT_APPROVAL.json') == approval_value()
    cfg = p.read(HERE/'PREPARED_CONFIG.json')
    assert cfg['gpu_device'] == GPU and cfg['gpu_uuid'] == UUID and cfg['shard_id'] == SHARD
    assert not cfg['training_allowed'] and not cfg['runtime_allowed'] and not cfg['executable']
    assert cfg['budget']['total_seconds'] == 4200 and cfg['budget']['total_actions'] == 60000
    assert p.ROOT/cfg['intended_output_root'] == OUT
    cfg.update(runtime_allowed=True, executable=True, runtime_adapter_ready=True,
               source_lock_sha256=p.sha(HERE/'SOURCE_LOCK.json'),
               main_agent_approval_sha256=p.sha(HERE/'MAIN_AGENT_APPROVAL.json'))
    return cfg

def module(name, source, path):
    result = types.ModuleType(name)
    result.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), result.__dict__)
    return result

def run():
    cfg = config()
    OUT.mkdir(exist_ok=False)
    save(OUT/'EXECUTION_CONFIG.json', cfg)
    save(OUT/'INPUT_LOCK.json', p.read(HERE/'SOURCE_LOCK.json'))
    error = None
    returned = False
    try:
        supervisor = module('next_scout_supervisor', supervisor_source(), a.RUNTIME/'compact_loop_v2/run.py')
        assert supervisor.OUT == OUT and supervisor.UUID == UUID
        supervisor.main()
        returned = True
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        save(OUT/'LAUNCH_RESULT.json', dict(supervisor_returned=returned, error=error,
             gpu=GPU, holders_touched=False, external_processes_stopped=0, scientific_pass=False))

def worker():
    cfg = config()
    name = 'bulk_scout_scoped_common_01'
    common = module(name, common_source(), a.SCOUT/'common.py')
    common.runtime_config = lambda:cfg
    assert common.HERE == HERE and common.ROOT == p.ROOT
    sys.modules[name] = common
    work = module('next_scout_worker', worker_source(), a.SCOUT/'worker.py')
    assert work.HERE == HERE
    work.main()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare','run'])
    globals()[parser.parse_args().operation]()
