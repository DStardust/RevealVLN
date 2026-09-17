"""Execute, rather than mock, exact approval checks in each requested interpreter."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-import',action='store_true');args=p.parse_args()
    lock=c.approved(6)
    roots=[]
    if args.worker_import:
        for shard in (0,1,20):
            m=c.worker_module(shard);assert m.OUT==c.shard_root(shard);roots.append(str(m.OUT))
    print(json.dumps(dict(approved=True,actual_interpreter=sys.executable,input_lock_sha256=c.sha(HERE/'INPUT_LOCK.json'),
        immutable_inputs=len(lock),worker_import_roots=roots,simulator_constructed=False)))
