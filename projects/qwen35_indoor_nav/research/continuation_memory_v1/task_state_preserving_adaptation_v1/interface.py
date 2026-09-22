"""CPU-tested interface for a proposed selective adapter; not a trained new method."""
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn

class InterventionRouter(nn.Module):
    """Only causal current features, actual recurrent memory/state and native logits."""
    def __init__(self, feature_width=2048, memory_width=512):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(feature_width+memory_width+8,64),nn.Tanh(),nn.Linear(64,1))
    def forward(self, feature, memory, predicted_state, native_logits):
        x=torch.cat((feature,memory.flatten(1),predicted_state,native_logits),-1)
        if not torch.isfinite(x).all():raise ValueError('NONFINITE_CAUSAL_INPUT')
        return self.net(x).squeeze(-1)

def route_logits(native,proposal,enabled):
    """Exact inactive-row copy, with no native STOP override. No safety theorem."""
    if native.ndim!=2 or native.shape[-1]!=4 or not torch.isfinite(native).all():raise ValueError('INVALID_NATIVE')
    if enabled.shape!=(native.shape[0],) or enabled.dtype!=torch.bool:raise ValueError('INVALID_GATE')
    if proposal.shape!=native.shape or proposal.dtype!=native.dtype or proposal.device!=native.device:raise ValueError('PROPOSAL_CONTRACT')
    if not torch.isfinite(proposal).all():raise ValueError('INVALID_PROPOSAL')
    result=native.clone();result[enabled]=proposal[enabled]
    return result

@dataclass(frozen=True)
class Branch:
    prefix_sha256: str
    base_sha256: str
    continuation_policy_sha256: str
    environment_sha256: str
    seed: int
    remaining_decisions: int
    executed_first_action: int
    complete: bool
    legal: bool
    label: str
    trace_sha256: str
    decisions: int

@dataclass(frozen=True)
class InterventionTarget:
    known: bool
    benefit: Optional[int]
    reason: str

def branch_target(native:Branch,alternative:Branch):
    """Certificate-level contract only. Physical traces must be independently verified."""
    for field in ('prefix_sha256','base_sha256','continuation_policy_sha256','environment_sha256','seed','remaining_decisions'):
        if getattr(native,field)!=getattr(alternative,field):raise ValueError('UNPAIRED_'+field)
    for b in (native,alternative):
        if b.label not in ('PASS','FAIL','UNKNOWN'):raise ValueError('LABEL_DOMAIN')
        if not 1<=b.remaining_decisions<=500 or not 1<=b.decisions<=b.remaining_decisions:raise ValueError('BUDGET')
        if b.executed_first_action not in range(4):raise ValueError('ACTION')
        if not b.trace_sha256:raise ValueError('MISSING_TRACE')
    if native.executed_first_action==alternative.executed_first_action:raise ValueError('NO_FIRST_ACTION_INTERVENTION')
    if any(not b.complete or not b.legal or b.label=='UNKNOWN' for b in (native,alternative)):
        return InterventionTarget(False,None,'Incomplete/illegal/unknown is masked, never a synthetic negative.')
    gain=int(alternative.label=='PASS')-int(native.label=='PASS')
    return InterventionTarget(True,gain,'Finite registered branch comparison; not global action optimality.')
