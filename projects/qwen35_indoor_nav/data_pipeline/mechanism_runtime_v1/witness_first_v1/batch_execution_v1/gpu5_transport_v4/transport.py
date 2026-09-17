"""V3 exact lease binding with a 2048-file input-manifest parsing cap only."""
import hashlib
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
V3=HERE.parent/'gpu5_transport_v3'
V3_HASHES={'transport.py':'03e08161eaf28fb6a75a0974e2676468e9aacfd293b6af8b13e1f2460e07cbae',
           'prepare.py':'48ef96f0fe3a6695b4599087d2a2087ffd559fd34288cc94531d09402c011239'}
def checked_v3(name):
    path=V3/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==V3_HASHES[name],'SEALED_V3_CHANGED'
    return path
spec=importlib.util.spec_from_file_location('gpu5_v4_v3_binding',checked_v3('transport.py'))
v3=importlib.util.module_from_spec(spec);spec.loader.exec_module(v3)

def adapted_run_source():
    source=v3.checked('transport.py').read_text()
    source=source[source.index('def run_main(batch):'):source.index("\nif __name__=='__main__':")]
    old="require(len(lock)<=1024,'INPUT_LOCK_COUNT_CAP')"
    assert source.count(old)==1
    return source.replace(old,"require(len(lock)<=2048,'INPUT_LOCK_COUNT_CAP')")

# Only this entry function is replaced; every safety, quality, signal, budget,
# cleanup and restoration function is still the exact sealed V2 implementation.
exec(compile(adapted_run_source(),str(HERE/'transport.py'),'exec'),v3.base.__dict__)

def __getattr__(name):return getattr(v3,name)
def verify_sources():
    for name in V3_HASHES:checked_v3(name)
    v3.verify_sources()
def run_main(batch):
    verify_sources()
    batch=Path(batch).resolve(strict=True)
    cfg=v3.read(batch/'run_v1/EXECUTION_CONFIG.json')
    assert cfg['input_manifest_capacity_version']=='gpu5_transport_v4_2048'
    lock=v3.read(batch/'run_v1/INPUT_LOCK.json')
    for path in [HERE/'transport.py',HERE/'prepare.py',*map(checked_v3,V3_HASHES)]:
        assert lock.get(str(path))==v3.sha(path),'V4_DEPENDENCY_NOT_LOCKED'
    v3.run_main(batch)
