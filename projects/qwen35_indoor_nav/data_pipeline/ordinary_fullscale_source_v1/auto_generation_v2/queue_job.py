"""Print frozen new-runtime job; no launch and no approval bridge."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
def descriptor():
    c.immutable_verify();cfg=c.read(HERE/'PREPARED_CONFIG.json');previous=c.read(HERE/'PREVIOUS_FAILURES.json')
    remaining=cfg['lane_seconds']['6']-previous['prior_wall_seconds']
    return dict(id='ordinary_gpu6_unattempted_1424_fresh_runtime_v2',gpu=6,
        command=[str(c.ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-B',str(HERE/'run.py'),'--gpu','6'],
        max_seconds=int(remaining),completion=[dict(path=str(HERE/'lanes/gpu_6/attempt_000/RESULT.json'),equals={'error':None,'restoration.restored':True})],
        audit_command=[str(c.ENV/'bin/python3'),'-I','-B',str(HERE/'merge.py')],
        input_lock_sha256=c.sha(HERE/'INPUT_LOCK.json'),main_agent_approval_required=c.approval_value(6),
        prior_preworker_wall_seconds=previous['prior_wall_seconds'],remaining_chain_wall_seconds=remaining,
        source_routes=1424,source_aliases=4272,production_retry_allowed=False,guard_amended=False,approval_bridge_used=False)
if __name__=='__main__':print(json.dumps(descriptor(),indent=2))
