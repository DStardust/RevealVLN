"""CPU prototype: causal feature change and actual previous action write memory.

x is the existing mixed visual/instruction/previous-action feature, not a pure
visual embedding. Its difference also includes changes in the encoded previous
action; the explicit action embedding may duplicate information already in x.
This is a trainable adaptation prototype, not novelty or visual-effect evidence.
It adds capacity; a fair future comparison needs an equally large ordinary
writer. This module is not wired into any current GPU run.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'recovery_confirmation_v2'))
import torch
from torch import nn
import torch.nn.functional as F
from confirmation_model import ConfirmationMemory


STOP, FORWARD, LEFT, RIGHT, START = range(5)


class ActionOutcomeMemory(ConfirmationMemory):
    """Existing 8x64 recurrence plus two explicit causal write terms."""

    def __init__(self, width):
        super().__init__(width, 'CONCAT')
        self.change_writer = nn.Linear(width, 512, bias=False)
        self.executed_action = nn.Embedding(5, 512)

    def update(self, feature, memory, previous_feature, previous_action):
        """Update once per arriving observation; never update after STOP.

        Previous feature/action and memory are supplied by the owning episode.
        START is a separate index and requires a zero reset state. Batched resets
        are supported by using START for reset rows; their previous feature is
        ignored. No internal episode cache, future query or evaluator input.
        """
        if previous_action.dtype != torch.long or previous_action.shape != (len(feature),):
            raise ValueError('PREVIOUS_ACTION_MUST_BE_LONG_BATCH_VECTOR')
        if bool(((previous_action < 0) | (previous_action > START)).any()):
            raise ValueError('INVALID_PREVIOUS_ACTION')
        if bool((previous_action == STOP).any()):
            raise ValueError('STOP_HAS_NO_NEXT_OBSERVATION')
        start = previous_action == START
        if bool((memory[start] != 0).any()):
            raise ValueError('START_REQUIRES_RESET_MEMORY')
        if previous_feature is None:
            if not bool(start.all()):
                raise ValueError('MOTION_REQUIRES_PREVIOUS_OBSERVATION')
            change = torch.zeros_like(feature)
        else:
            if previous_feature.shape != feature.shape:
                raise ValueError('PREVIOUS_FEATURE_SHAPE_MISMATCH')
            change = torch.where(start[:, None], torch.zeros_like(feature), feature - previous_feature)
        current = memory.flatten(1)
        candidate = torch.tanh(self.writer(self.norm(feature))
                               + self.recurrent(current)
                               + self.change_writer(change)
                               + self.executed_action(previous_action))
        return (.99 * current + .01 * candidate).reshape(-1, 8, 64)

    def action_delta(self, feature, memory):
        state = F.normalize(memory.flatten(1), p=2, dim=-1, eps=1e-6)
        return self.actor(torch.cat((self.norm(feature), state), -1))

    def forward(self, features, executed_actions, query_steps, actor_features, native_logits):
        """One real trajectory; each feature precedes actions[t].

        query_steps only locates recorded action-head calls. Dense memory updates
        include intervening real observations, with no detach or cutoff repeat.
        A final STOP has its pre-action observation and generates no extra one.
        """
        if features.ndim != 2 or executed_actions.shape != (len(features),):
            raise ValueError('ONE_ACTION_PER_PRE_ACTION_OBSERVATION_REQUIRED')
        if executed_actions.dtype != torch.long or bool(((executed_actions < STOP) | (executed_actions > RIGHT)).any()):
            raise ValueError('INVALID_EXECUTED_ACTION')
        if len(features) == 0 or bool((executed_actions[:-1] == STOP).any()):
            raise ValueError('EMPTY_TRACE_OR_OBSERVATION_AFTER_STOP')
        steps = query_steps.tolist()
        if not steps or steps != sorted(set(steps)) or steps[0] < 0 or steps[-1] >= len(features):
            raise ValueError('INVALID_QUERY_STEPS')
        if actor_features.shape != (len(steps), features.shape[-1]) or native_logits.shape != (len(steps), 4):
            raise ValueError('QUERY_FEATURE_OR_LOGIT_ALIGNMENT')
        if not bool(torch.isfinite(native_logits).all()):
            raise ValueError('NONFINITE_NATIVE_LOGITS')
        lookup = {step: index for index, step in enumerate(steps)}
        memory = self.reset()
        states, deltas = [], []
        for step, feature in enumerate(features):
            previous_feature = None if step == 0 else features[step - 1:step]
            previous_action = (executed_actions.new_tensor([START]) if step == 0
                               else executed_actions[step - 1:step])
            memory = self.update(feature[None], memory, previous_feature, previous_action)
            states.append(memory[0])
            if step in lookup:
                q = lookup[step]
                deltas.append(self.action_delta(actor_features[q:q + 1], memory)[0])
        return {'memory': torch.stack(states),
                'logits': native_logits.float() + torch.stack(deltas)}
