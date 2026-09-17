"""Exact frozen identity binding over the proven V2 lease; no import-time GPU work."""
import hashlib
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'gpu5_transport_v2'
EXPECTED = {'transport.py':'a48d8f1c3454b84804531a051b209db6ce419995493d103078a4a2cb92749a7f',
            'prepare.py':'7374eb675549c9e622e6204aadacb861de96c889ba08bea287e567ae154a3ae9'}
def checked(name):
    path = OLD/name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED[name], 'SEALED_V2_SOURCE_CHANGED'
    return path
spec = importlib.util.spec_from_file_location('gpu5_v3_exact_v2_core', checked('transport.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

def __getattr__(name):
    return getattr(base,name)

def bind_identity(document):
    # Only an explicitly frozen exact holder identity may replace the historical
    # PID. No process discovery, no accepting the current pane occupant by name.
    pid = document.get('pid')
    base.require(type(pid) is int and pid > 0 and document.get('pane_pid') == pid,
                 'FROZEN_POSITIVE_PID_REQUIRED')
    base.HOLDER_PID = pid
    return base.normalize_identity(document)

def normalize_identity(document):
    return bind_identity(document)

def verify_sources():
    for name in EXPECTED:checked(name)
    base.verify_sources()

def run_main(batch):
    verify_sources()
    batch = Path(batch).resolve(strict=True)
    cfg = base.read(batch/'run_v1/EXECUTION_CONFIG.json')
    lock = base.read(batch/'run_v1/INPUT_LOCK.json')
    for path in [HERE/'transport.py',HERE/'prepare.py',*map(checked,EXPECTED)]:
        base.require(lock.get(str(path)) == base.sha(path),'IDENTITY_BINDING_SOURCE_NOT_LOCKED')
    base.require(cfg['identity_binding_version'] == 'gpu5_transport_v3', 'BINDING_VERSION')
    bind_identity(base.read(cfg['gpu5_holder_identity_path']))
    base.run_main(batch)
