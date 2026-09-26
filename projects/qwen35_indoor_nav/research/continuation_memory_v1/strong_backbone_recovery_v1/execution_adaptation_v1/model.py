"""Matched-capacity causal feature-change versus current-feature writers.

The existing 3584-wide feature already mixes visual, instruction and previous
action information. DELTA is not a pure visual-effect representation. Both arms
use the same actual previous-action embedding and unit-norm state readout.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'parallel_improvements_v1'))
import torch
from execution_memory import ActionOutcomeMemory, START, STOP


class ExecutionAdaptation(ActionOutcomeMemory):
    """Only the input of the extra writer differs; parameter keys are identical."""

    def __init__(self, width, mode):
        if mode not in ('DELTA', 'CURRENT'):
            raise ValueError('UNKNOWN_EXECUTION_ADAPTATION')
        super().__init__(width)
        self.mode = mode

    def update(self, feature, memory, previous_feature, previous_action):
        if self.mode == 'DELTA':
            return super().update(feature, memory, previous_feature, previous_action)

        # Preserve the source prototype's input contract. CURRENT does not use
        # previous_feature numerically, but requires the same causal interface.
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
        elif previous_feature.shape != feature.shape:
            raise ValueError('PREVIOUS_FEATURE_SHAPE_MISMATCH')

        current = memory.flatten(1)
        candidate = torch.tanh(self.writer(self.norm(feature))
                               + self.recurrent(current)
                               + self.change_writer(feature)
                               + self.executed_action(previous_action))
        return (.99 * current + .01 * candidate).reshape(-1, 8, 64)
