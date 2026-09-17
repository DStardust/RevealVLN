"""Independent actual-window audit; no simulator truth reaches policy."""
import hashlib
from pathlib import Path
import runpy
LINE=Path(__file__).resolve().parents[2]
BLACK_SHA=hashlib.sha256(bytes(224*224*3)).hexdigest()
old=runpy.run_path(str(LINE/'closed_loop_bench/r2r_ce_tiny_v1/common.py'))
class RecentWindow(old['Window']):
    def __init__(self):
        super().__init__()
        self.decision_t=None
    def receive(self,payload,executed=None):
        alive=super().receive(payload,executed)
        if alive:self.decision_t=0 if executed is None else self.decision_t+1
        return alive
    def input_audit(self):
        t=self.decision_t
        return dict(input_frame_indices=[0] if t==0 else [t-1,t],
            input_rgb_sha256=[hashlib.sha256(x).hexdigest() for x in self.images],
            history_rule='recent_two',padding_mask=[False]*len(self.images))
def expected_indices(arm,t):
    assert type(t) is int and 0<=t<500
    if arm=='control_recent2':return [0] if t==0 else [t-1,t]
    assert arm=='treatment_prefix8'
    # Independently derive each sampled position using rational integer division.
    visible=t+1
    padded=max(8,visible)
    leading=padded-visible
    positions=[divmod(j*(padded-1),7)[0]-leading for j in range(7)]+[t]
    return [None if x<0 else x for x in positions]
def validate(arm,policy,step,observed):
    t=step['step']-1
    assert len(observed)==t+1,'OBSERVATION_SEQUENCE'
    expected=expected_indices(arm,t)
    assert policy['input_frame_indices']==expected,'FRAME_SELECTION'
    assert policy['images']==len(expected),'FRAME_COUNT'
    assert policy['executed_history']==min(8,t),'EXECUTED_COUNT'
    assert policy['padding_mask']==[x is None for x in expected],'PADDING_MASK'
    assert policy['history_rule']==('recent_two' if arm=='control_recent2' else 'navila_prefix_quantiles_8'),'HISTORY_RULE'
    assert policy['input_rgb_sha256']==[BLACK_SHA if j is None else observed[j] for j in expected],'RAW_RGB_SELECTION'

