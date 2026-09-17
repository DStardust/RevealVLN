"""V2 exact, prospective transport of sealed EnvDrop production, no quality changes."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'runtime_v1'
ROOT = HERE.parents[4]
DATA = HERE.parent / 'envdrop_production_v2'
MANIFEST = HERE.parent / 'envdrop_source_v1'
REC = HERE.parent / 'recovery_inventory_20260911_v1/run_001'
LANES = {g: tuple(range(g - 3, 42, 3)) for g in (3, 4, 5)}
SELECTED_SHARDS = tuple(range(42))
SEALED = {'transport.py': 'af9ddd2548175a9d6c5d1fce8e29c3a556fb01595182a6d445757eba55c0ab6a',
          'sentinel_gate.py': '9f3fdcb1915d541cfb6667000d082a7dbd9b07cefe60e5b8639cc99aa5ddbe45',
          'telemetry.py': '13d023919d8647085497d2c01bdb6be899bed3c228a069f32e522d16e2a67e8b'}

def sealed(name):
    raw = (OLD / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SEALED[name], ('SEALED_CHANGED', name)
    return raw.decode()

spec = importlib.util.spec_from_file_location('sealed_envdrop_transport_v1', OLD / 'transport.py')
prior = importlib.util.module_from_spec(spec)
exec(compile(sealed('transport.py'), str(OLD / 'transport.py'), 'exec'), prior.__dict__)
V4 = prior.V4
SOURCE = prior.SOURCE
EXPECTED = prior.EXPECTED
original = prior.original
exact = prior.exact

def common_source():
    s = prior.common_source()
    s = exact(s, "IDENTITIES=LINE/'authorizations/ORDINARY_ENVDROP_GPU6_IDENTITY_V1.json'", "IDENTITIES=HERE/'IDENTITIES.json'")
    s = exact(s, "AUTH=LINE/'authorizations/ORDINARY_ENVDROP_GPU6_PRODUCTION_V1.json'", "AUTH=HERE/'AUTHORIZATION.json'")
    s = exact(s, 'LANES={6:tuple(range(21))}\nSELECTED_SHARDS=tuple(range(21))', f'LANES={LANES!r}\nSELECTED_SHARDS=tuple(range(42))')
    s = exact(s, '0<=shard<21', '0<=shard<42')
    s = exact(s, "PARALLEL/'envdrop_production_v1'", "PARALLEL/'envdrop_production_v2'")
    s = exact(s, '    return 6', '    return 3 + shard % 3')
    return s

def run_source():
    s = prior.run_source()
    s = exact(s, 'lane_limit==82800', 'lane_limit==43200')
    s = exact(s, '<23*1024**3', '<11*1024**3')
    s = exact(s, 'allocated 24GiB cap.', 'allocated 12GiB cap.')
    s = exact(s, 'if shard!=0:require_sentinel_receipt(out)', 'if shard!=c.LANES[gpu][0]:require_sentinel_receipt(out,gpu)')
    s = exact(s, 'if shard==0:run_sentinel_gate(out,env)', 'if shard==c.LANES[gpu][0]:run_sentinel_gate(out,env,gpu)')
    s = exact(s, 'def require_sentinel_receipt(out):', 'def require_sentinel_receipt(out,gpu):')
    s = exact(s, "c.shard_root(0)/'JOBS.json'", "c.shard_root(c.LANES[gpu][0])/'JOBS.json'")
    s = exact(s, 'def run_sentinel_gate(out,env):', 'def run_sentinel_gate(out,env,gpu):')
    s = exact(s, "'--output',str(out/'SENTINEL_GATE.json')", "'--output',str(out/'SENTINEL_GATE.json'),'--gpu',str(gpu)")
    s = exact(s, '    require_sentinel_receipt(out)', '    require_sentinel_receipt(out,gpu)')
    s = exact(s, 'choices=[6]', 'choices=[3,4,5]')
    s = exact(s, "    return actual\n", "    assert project_environment(identity['pid'])==identity['project_cache_environment'],'HOLDER_CACHE_ENV_CHANGED'\n    return actual\n")
    s = exact(s, "shlex.join(identity['cmdline']))", "shlex.join(restore_argv(identity)))")
    s = exact(s, "new_identity['proc_uid']==identity['proc_uid'] and snapshot['processes'].get(pid,0)>20000", "new_identity['proc_uid']==identity['proc_uid'] and project_environment(pid)==identity['project_cache_environment'] and snapshot['processes'].get(pid,0)>20000")
    s = exact(s, 'def call(*args):', ENV_HELPERS + '\n\ndef call(*args):')
    return s

ENV_HELPERS = '''CACHE_KEYS=('XDG_CACHE_HOME','CUDA_CACHE_PATH','NUMBA_CACHE_DIR','MPLCONFIGDIR','TMPDIR','TORCH_HOME','HF_HOME')
def project_environment(pid):
    raw=Path('/proc',str(pid),'environ').read_bytes().split(b'\\0')
    env=dict(x.decode().split('=',1) for x in raw if b'=' in x)
    result={k:env[k] for k in CACHE_KEYS if k in env}
    for value in result.values():assert Path(value).resolve().is_relative_to(c.ROOT),'CACHE_OUTSIDE_PROJECT'
    return result

def restore_argv(identity):
    env=identity['project_cache_environment']
    return (['env']+[k+'='+v for k,v in sorted(env.items())] if env else [])+identity['cmdline']
'''

def sentinel_source():
    s = sealed('sentinel_gate.py')
    s = exact(s, 'def execute(output):', 'def execute(output,gpu):')
    s = exact(s, 'c.approved(6);root=c.shard_root(0)', 'c.approved(gpu);shard=c.LANES[gpu][0];root=c.shard_root(shard)')
    s = exact(s, "c.state(0)['complete']", "c.state(shard)['complete']")
    s = exact(s, "p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);execute(p.parse_args().output)", "p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,choices=[3,4,5],required=True);a=p.parse_args();execute(a.output,a.gpu)")
    return s

def merge_source():
    s = prior.merge_source()
    s = exact(s, "c.read(c.PARALLEL/'envdrop_source_v1/JOBS.json')", "c.read(HERE/'JOBS.json')")
    s = exact(s, 'len(expected_all)==20000', 'len(expected_all)==9661')
    return s
