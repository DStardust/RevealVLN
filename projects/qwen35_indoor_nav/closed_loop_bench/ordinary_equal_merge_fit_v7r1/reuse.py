"""Identical pure policy/geometry/metrics; only diagnostic label differs."""
import hashlib,importlib.util
from pathlib import Path
PARENT=Path(__file__).resolve().parent.parent/'ordinary_stop_calibration_fit_v2/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='974af0eb4fad27c75420a1f08bb109909b6a555b6a16c4536253264f9dc5d820'
s=importlib.util.spec_from_file_location('equal_merge_same_runtime',PARENT);parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
def source(name):
    t=parent.source(name)
    if name=='aggregate.py':
        old='R2R-CE train FIT threshold calibration; not validation'
        assert t.count(old)==1;t=t.replace(old,'R2R-CE train FIT equal-merge compatibility; not validation')
    return t
def execute(name,namespace):exec(compile(source(name),str(Path(namespace['__file__']))+':same-policy-and-metrics','exec'),namespace)
