"""Causal event/state prediction and shared state-conditioned action readout.

DIRECT is the strong simple state-prediction baseline. MONOTONIC accumulates
uncertain events. REVISE learns to retract a mistaken historical belief.
None of these generic ideas is claimed as newly invented here.
"""
from typing import NamedTuple
import torch
from torch import nn
from common import old

MODES = ('DIRECT', 'MONOTONIC', 'REVISE')


class State(NamedTuple):
    memory: torch.Tensor
    belief: torch.Tensor
    first: torch.Tensor


class EvidencePolicy(nn.Module):
    def __init__(self, seed=1209, mode='REVISE'):
        super().__init__()
        if mode not in MODES:
            raise ValueError('UNKNOWN_STATE_MODE')
        self.mode = mode
        self.core = old.initialize(seed)
        # There is no future-query reader in this policy or its optimizer.
        del self.core.result_head
        torch.manual_seed(seed + 10000)
        self.events = nn.Sequential(nn.Linear(2048, 128), nn.Tanh(), nn.Linear(128, 2))
        self.initial_belief = nn.Linear(2048, 1)
        self.revision = nn.Sequential(nn.Linear(515, 64), nn.Tanh(), nn.Linear(64, 1))
        self.state_action = nn.Linear(4, 4, bias=False)
        nn.init.zeros_(self.state_action.weight)

    def reset(self, batch_size, device):
        return State(self.core.reset(batch_size, device),
                     torch.zeros(batch_size, device=device),
                     torch.ones(batch_size, dtype=torch.bool, device=device))

    def step(self, feature, native_logits, state):
        if feature.ndim != 2 or feature.shape[-1] != 2048 or native_logits.shape != (feature.shape[0], 4):
            raise ValueError('POLICY_INPUT_SHAPE')
        if not torch.isfinite(feature).all() or not torch.isfinite(native_logits).all():
            raise ValueError('NONFINITE_POLICY_INPUT')
        memory, _ = self.core.update(feature, state.memory)
        normalized = self.core.feature_norm(feature)
        event_logits = self.events(normalized)
        event = event_logits.sigmoid()
        revision = self.revision(torch.cat([memory.flatten(1), state.belief[:, None], event], -1)).sigmoid().squeeze(-1)
        if self.mode == 'DIRECT':
            probability = self.core.state_head(memory.flatten(1)).sigmoid()
        else:
            # task_T's initial prior is learned from the causal instruction feature;
            # no task ID, parser, exact program state or private compiler enters here.
            prior = state.belief * (1 - revision) if self.mode == 'REVISE' else state.belief
            prior = torch.where(state.first, self.initial_belief(normalized).sigmoid().squeeze(-1), prior)
            after = prior + (1 - prior) * event[:, 0]
            probability = torch.stack([prior, after, event[:, 1], prior * event[:, 1]], -1)
        logits = self.core.action_logits(memory, native_logits, feature) + self.state_action(probability)
        next_state = State(memory, probability[:, 1], torch.zeros_like(state.first))
        return logits, next_state, dict(state=probability, event_logits=event_logits, revision=revision)

    def forward(self, features, native_logits, alive):
        if features.shape[:2] != alive.shape or native_logits.shape != (*alive.shape, 4):
            raise ValueError('SEQUENCE_INPUT_SHAPE')
        state = self.reset(features.shape[0], features.device)
        logits, probabilities, events, revisions = [], [], [], []
        for t in range(features.shape[1]):
            action, updated, detail = self.step(features[:, t], native_logits[:, t], state)
            valid = alive[:, t]
            state = State(torch.where(valid[:, None, None], updated.memory, state.memory),
                          torch.where(valid, updated.belief, state.belief),
                          torch.where(valid, updated.first, state.first))
            logits.append(action); probabilities.append(detail['state'])
            events.append(detail['event_logits']); revisions.append(detail['revision'])
        return dict(logits=torch.stack(logits, 1), state=torch.stack(probabilities, 1),
                    event_logits=torch.stack(events, 1), revision=torch.stack(revisions, 1), final_state=state)


def select_action(logits):
    if logits.shape[-1] != 4 or not torch.isfinite(logits).all():
        raise ValueError('INVALID_ACTION_LOGITS')
    return logits.argmax(-1)
