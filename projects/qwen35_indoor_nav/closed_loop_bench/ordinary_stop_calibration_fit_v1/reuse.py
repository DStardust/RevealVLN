"""Unmodified batch-one base policy; FIT report label corrected explicitly."""
import hashlib
import importlib.util
from pathlib import Path

PARENT=Path(__file__).resolve().parent.parent/'ordinary_expanded_dev_after_single_v1/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='e51c014048a6c189cbda790ff6e3c3608fc00ce3e437b2b269a325fc5a04038a'
s=importlib.util.spec_from_file_location('stop_fit_parent',PARENT)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)


def source(name):
    text=parent.source(name)
    if name=='aggregate.py':
        old='R2R-CE train held-out INTERNAL_DEV diagnostic'
        assert text.count(old)==1
        text=text.replace(old,'R2R-CE train FIT threshold calibration; not validation')
    return text


def execute(name,namespace):
    exec(compile(source(name),str(Path(namespace['__file__']))+':hash-bound-parent','exec'),namespace)
