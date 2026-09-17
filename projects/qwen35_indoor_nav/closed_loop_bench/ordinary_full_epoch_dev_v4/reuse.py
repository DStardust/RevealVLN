"""Unmodified pure model and metrics, batch one, with the proven scan repair."""
import hashlib
import importlib.util
from pathlib import Path
PARENT=Path(__file__).resolve().parent.parent/'ordinary_stop_calibration_fit_v2/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='974af0eb4fad27c75420a1f08bb109909b6a555b6a16c4536253264f9dc5d820'
s=importlib.util.spec_from_file_location('full_epoch_eval_parent',PARENT)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)


def source(name):
    text=parent.source(name)
    if name=='aggregate.py':
        old='R2R-CE train FIT threshold calibration; not validation';assert text.count(old)==1
        text=text.replace(old,'R2R-CE train INTERNAL_DEV; full-epoch ordinary base')
    return text


def execute(name,namespace):
    exec(compile(source(name),str(Path(namespace['__file__']))+':hash-bound-parent','exec'),namespace)
