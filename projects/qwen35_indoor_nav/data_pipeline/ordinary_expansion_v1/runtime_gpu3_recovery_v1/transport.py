"""Exact adaptation of frozen V2: GPU3 only, 50-route phases, same physical audit."""
import hashlib
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'runtime_v2'
ROOT=HERE.parents[4]
DATA=HERE.parent/'envdrop_production_gpu3_recovery_v1'
LANES={3:tuple(range(64))}
SELECTED_SHARDS=tuple(range(64))
raw=(OLD/'transport.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='2552c3773921fbbafb97f212a26df123b1330053771fec2574296fa0995cc163'
spec=importlib.util.spec_from_file_location('sealed_v2_transport_gpu3',OLD/'transport.py')
prior=importlib.util.module_from_spec(spec)
exec(compile(raw,str(OLD/'transport.py'),'exec'),prior.__dict__)
exact=prior.exact
original=prior.original
SOURCE=prior.SOURCE
V4=prior.V4
MANIFEST=prior.MANIFEST

def common_source():
    s=prior.common_source()
    s=exact(s,f'LANES={prior.LANES!r}\nSELECTED_SHARDS=tuple(range(42))',f'LANES={LANES!r}\nSELECTED_SHARDS=tuple(range(64))')
    s=exact(s,'0<=shard<42','0<=shard<64')
    s=exact(s,"PARALLEL/'envdrop_production_v2'","PARALLEL/'envdrop_production_gpu3_recovery_v1'")
    return exact(s,'return 3 + shard % 3','return 3')

def run_source():
    s=prior.run_source()
    s=exact(s,'<11*1024**3','<3*1024**3')
    s=exact(s,'allocated 12GiB cap.','allocated 4GiB cap.')
    return exact(s,'choices=[3,4,5]','choices=[3]')

def sentinel_source():
    return exact(prior.sentinel_source(),'choices=[3,4,5]','choices=[3]')

def merge_source():
    s=exact(prior.merge_source(),'len(expected_all)==9661','len(expected_all)==3112')
    return exact(s,'<511*1024**3','<263*1024**3')

def telemetry_source():
    return prior.sealed('telemetry.py')
