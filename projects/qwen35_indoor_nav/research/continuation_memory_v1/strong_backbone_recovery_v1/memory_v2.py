"""A zero-initialized action residual on causal StreamVLN features.

The recurrent architecture is inherited from Q35N (8x64, retention .99), not a
novel architecture. Future continuations and evaluator state are not inputs.
The proposed learning difference is implemented in interaction_loss below.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ExecutionMemory(nn.Module):
    def __init__(self, feature_width):
        super().__init__()
        self.norm = nn.LayerNorm(feature_width, elementwise_affine=False)
        self.writer = nn.Linear(feature_width, 512)
        self.recurrent = nn.Linear(512, 512, bias=False)
        self.actor = nn.Sequential(nn.Linear(feature_width + 512, 128), nn.Tanh(), nn.Linear(128, 4))
        self.state_reader = nn.Linear(512, 4)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def reset(self, batch_size=1):
        return self.writer.weight.new_zeros(batch_size, 8, 64)

    def update(self, feature, memory):
        current = memory.flatten(1)
        return (.99 * current + .01 * torch.tanh(self.writer(self.norm(feature)) + self.recurrent(current))).reshape(-1,8,64)

    def action_delta(self, feature, memory):
        return self.actor(torch.cat([self.norm(feature), memory.flatten(1)], -1))

    def forward(self, features, lengths, actor_features=None):
        memory = self.reset(features.shape[0]); states=[]; deltas=[]
        for step in range(features.shape[1]):
            new = self.update(features[:,step], memory)
            memory = torch.where((step < lengths)[:,None,None], new, memory)
            states.append(memory)
            deltas.append(self.action_delta(features[:,step] if actor_features is None else actor_features[:,step], memory))
        return dict(memory=torch.stack(states,1), delta=torch.stack(deltas,1))


def interaction_loss(action_logits, action_pairs, observed_returns, valid):
    """Train actual policy preferences on registered lawful history contrasts.

    Shapes: logits [family,task,history=2,4], action_pairs [family,task,2],
    returns/mask [family,task,history=2,continuation=2]. Continuation first
    actions must differ at the registered decision point; no invented conflict.
    All four real cells are required. Equal finite outcome rows only constrain
    these tested action preferences; they never collapse full memories.
    """
    if action_logits.shape[-2:] != (2,4) or observed_returns.shape != valid.shape:
        raise ValueError('INVALID_INTERACTION_SHAPE')
    if bool((action_pairs[...,0] == action_pairs[...,1]).any()):
        raise ValueError('NO_IDENTIFIABLE_ACTION_FORK')
    if bool((~torch.isfinite(observed_returns[valid])).any()):
        raise ValueError('NONFINITE_OBSERVED_RETURN')
    keep = valid.all(-1).all(-1)
    if not bool(keep.any()):
        return action_logits.sum() * 0
    selected = action_logits.gather(-1, action_pairs.unsqueeze(-2).expand(*action_logits.shape[:-1],2))
    # Bounded preference has the same [-1,1] range as binary return contrast.
    preferences = torch.tanh((selected[...,0] - selected[...,1]) / 2)
    returns = torch.where(valid, observed_returns, torch.zeros_like(observed_returns))
    target = returns[...,0] - returns[...,1]
    return F.mse_loss((preferences[...,0] - preferences[...,1])[keep], (target[...,0] - target[...,1])[keep])


def masked_state_loss(logits, target, known):
    if not bool(known.any()):
        return logits.sum() * 0
    if not bool(torch.isfinite(target[known]).all()):
        raise ValueError('NONFINITE_KNOWN_STATE')
    return F.binary_cross_entropy_with_logits(logits[known], target[known])
