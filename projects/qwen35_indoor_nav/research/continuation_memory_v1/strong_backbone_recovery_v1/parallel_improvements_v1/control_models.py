"""Simple controls for future comparisons, not the proposed paper contribution."""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent.parent/'recovery_confirmation_v2')]
import torch
import torch.nn.functional as F
from confirmation_model import ConfirmationMemory


class UnitReadoutMemory(ConfirmationMemory):
    """Same stored parameters and actor; remove per-query state norm as a cue.

    LOCAL retains only current information in the added branch. EMA retains a
    simple running average. RECURRENT uses the existing learned recurrence.
    The frozen backbone still has its original history in every condition.
    """
    def __init__(self, width, mode):
        if mode not in ('LOCAL','EMA','RECURRENT'):
            raise ValueError('UNKNOWN_MEMORY_CONTROL')
        super().__init__(width,'CONCAT')
        self.mode = mode

    def update(self, feature, memory):
        if self.mode=='RECURRENT':
            return super().update(feature,memory)
        candidate = torch.tanh(self.writer(self.norm(feature))).reshape(-1,8,64)
        if self.mode=='LOCAL':
            return candidate
        return .99*memory+.01*candidate

    def read_state(self, memory):
        # Fixed unit-L2 convention for all controls, including training.
        # Unlike the read-only v4 diagnostic, no full-history norm is supplied
        # to LOCAL or EMA. Zero reset remains zero; no learned gate is added.
        return F.normalize(memory.flatten(1),p=2,dim=-1,eps=1e-6)

    def action_delta(self, feature, memory):
        return self.actor(torch.cat((self.norm(feature),self.read_state(memory)),-1))
