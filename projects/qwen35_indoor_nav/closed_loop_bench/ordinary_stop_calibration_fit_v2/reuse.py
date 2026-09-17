"""Transport-only correction after the retained pre-episode V1 failure."""
import hashlib
import importlib.util
from pathlib import Path

PARENT=Path(__file__).resolve().parent.parent/'ordinary_stop_calibration_fit_v1/reuse.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='8caaae5a00dfdc65f5e6837fa2e687acaf4907bd52018633c93ae75e2ef775b2'
s=importlib.util.spec_from_file_location('stop_fit_v2_parent',PARENT)
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)


def source(name):
    text=parent.source(name)
    if name=='launch.py':
        changes={
          "OUT=HERE/'run_001'":"""OUT=HERE/'run_001'
_scan_spec=importlib.util.spec_from_file_location('race_safe_tree_size',HERE/'tree_size.py')
_scan=importlib.util.module_from_spec(_scan_spec);_scan_spec.loader.exec_module(_scan)""",
          "size=sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file())":"size,transient_misses=_scan.tree_size(OUT)",
          "own_pids=sorted(pids),misplaced=misplaced,foreign=foreign,stop_reason=reason)":"own_pids=sorted(pids),misplaced=misplaced,foreign=foreign,stop_reason=reason,tree_scan_transient_misses=transient_misses)",
        }
        for old,new in changes.items():
            assert text.count(old)==1,old;text=text.replace(old,new)
    return text


def execute(name,namespace):
    exec(compile(source(name),str(Path(namespace['__file__']))+':hash-bound-parent','exec'),namespace)
