"""Byte-identical fixed-batch-one policy and simulator for checkpoint comparison."""
import hashlib
import importlib.util
from pathlib import Path

PARENT=Path(__file__).resolve().parent.parent/'ordinary_expanded_dev_after_single_v1/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='e51c014048a6c189cbda790ff6e3c3608fc00ce3e437b2b269a325fc5a04038a'
s=importlib.util.spec_from_file_location('continued_eval_parent',PARENT)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
source=parent.source


def execute(name,namespace):exec(compile(source(name),str(Path(namespace['__file__']))+':parent','exec'),namespace)
