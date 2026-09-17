"""Actual approval check in each process/environment, no mocking or GPU calls."""
from pathlib import Path
import json
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
if __name__=='__main__':
    lock=c.approved(6)
    print(json.dumps(dict(approved=True,interpreter=sys.executable,common=str(c.__file__),immutable=len(lock),
                         roots={str(s):str(c.shard_root(s)) for s in c.SELECTED_SHARDS},gpu_operations=0)))
