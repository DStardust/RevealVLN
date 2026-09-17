"""Resume only unattempted jobs; old terminal results and failed run preserved."""
import importlib.util
import json
from pathlib import Path
import hashlib

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    lock=json.loads((HERE/'RECOVERY_INPUT_LOCK.json').read_text())
    root=BASE.parents[3]
    for name,h in lock.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    w=load('production_worker_v1',BASE/'worker.py')
    w.audit=load('strict_quarantine_audit_v2',HERE/'audit.py')
    # Append new route data/immutable shard indices in original batch. Only
    # recovery's live summaries are updated; original PROGRESS remains evidence.
    def atomic(name,obj):
        p=HERE/name;tmp=p.with_suffix(p.suffix+'.pending')
        with tmp.open('w') as f:json.dump(obj,f,indent=2,allow_nan=False)
        tmp.replace(p)
    w.atomic=atomic
    w.main()

if __name__=='__main__':main()

