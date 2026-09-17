"""Select one fixed history arm; all shared source is locked before evaluation."""
import importlib.util
from pathlib import Path
PARENT=Path(__file__).resolve().parent.parent/'ordinary_history_pair_eval_r1/reuse.py'
s=importlib.util.spec_from_file_location('history_pair_shared',PARENT)
shared=importlib.util.module_from_spec(s);s.loader.exec_module(shared)
ARM='control_recent2'
def source(name):return shared.source(name,ARM)
def execute(name,namespace):exec(compile(source(name),str(Path(namespace['__file__']))+':history-pair','exec'),namespace)

