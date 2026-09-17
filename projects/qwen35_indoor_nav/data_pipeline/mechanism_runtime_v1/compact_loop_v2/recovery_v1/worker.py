"""Explicit output-scope adapter; sealed V2 launch failure remains unchanged."""
import importlib.util
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
RUNTIME=BASE.parent
OLD=RUNTIME/'feedback_generation_v1'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def scoped_store():
    module=load('compact_recovery_scoped_store',OLD/'store.py')
    module.FEEDBACK_ROOT=HERE
    return module.TrackedContentStore

def main():
    adapter=load('compact_v2_worker_adapter',BASE/'worker.py')
    module=types.ModuleType('compact_v2_recovery_worker')
    module.__file__=str(OLD/'worker.py')
    exec(compile(adapter.adapted_source(),str(OLD/'worker.py'),'exec'),module.__dict__)
    module.HERE=HERE
    module.FeedbackFactory=adapter.CompactLoopFactory
    module.rank_reachable_positions=adapter.rank_reachable_positions
    module.TrackedContentStore=scoped_store()
    module.main()

if __name__=='__main__':main()
