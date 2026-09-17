import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[3]

def verify():
    lock=json.loads((HERE/'INPUT_LOCK.json').read_text())
    for path,digest in lock['immutable'].items():
        file=ROOT/path
        assert file.resolve().is_relative_to(ROOT)
        assert hashlib.sha256(file.read_bytes()).hexdigest()==digest,path
    s=importlib.util.spec_from_file_location('v3_preflight',BASE/'recovery_v3/preflight.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.verify(prepared=True)
    return lock

if __name__=='__main__':print(json.dumps({'preflight_pass':bool(verify()),'gpu_operations':0}))
