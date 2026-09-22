"""Private physical service: full history replay, no oracle fields returned to actor."""
import base64
import json
from pathlib import Path
import socket
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
collector=load('ident_pilot_store',V16/'collect.py')
class ContentStore(collector.ContentStore):
    """Keep the frozen array writer; constrain writes to this runtime's own runs."""
    def __init__(self, root, max_bytes):
        self.root=Path(root)
        if self.root.is_symlink() or not self.root.resolve().is_relative_to(HERE/'runs'):
            raise ValueError('CONTENT_PATH')
        self.root.mkdir(exist_ok=True,parents=True)
        self.bytes=sum(p.stat().st_size for p in self.root.iterdir())
        self.max_bytes=max_bytes
        self.verified=set()

from evaluator_v16 import legacy,evaluate
backend_module=load('v16_live_habitat',LINE/'data_pipeline/mechanism_runtime_v1/habitat_backend.py')

class NoInteriorJoin(backend_module.HabitatBackend):
    def reconstruct(self,pose):raise ValueError('INTERIOR_STATE_ASSIGNMENT_FORBIDDEN')

def exact_pose(a,b):
    return a['position']==b['position'] and (a['rotation']==b['rotation'] or a['rotation']==[-x for x in b['rotation']]) and a.get('sensors',{}).keys()==b.get('sensors',{}).keys() and all(exact_pose(a['sensors'][k],b['sensors'][k]) for k in a.get('sensors',{}))

