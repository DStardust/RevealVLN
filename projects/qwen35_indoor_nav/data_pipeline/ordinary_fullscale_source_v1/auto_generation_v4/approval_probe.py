"""Actual exact approval verification in either runtime interpreter; no GPU calls."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
if __name__=='__main__':
    lock=c.approved(6)
    print(json.dumps(dict(approved=True,interpreter=sys.executable,common=str(c.__file__),immutable=len(lock),
                         roots={str(s):str(c.shard_root(s)) for s in c.SELECTED_SHARDS},gpu_operations=0)))
