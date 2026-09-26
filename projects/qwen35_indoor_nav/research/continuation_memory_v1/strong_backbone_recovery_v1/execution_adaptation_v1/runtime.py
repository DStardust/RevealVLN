"""Causal live state and all-token residuals; no backbone forward is simulated."""
import hashlib
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from action_boundary_v2 import ActionBoundary

STOP, START = 0, 4


def tensor_hash(value):
    raw = value.detach().contiguous().cpu()
    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def selected_action(logits):
    """Shared four-class argmax; no native-STOP override or score threshold."""
    if logits.shape[-1:] != (4,) or not bool(torch.isfinite(logits).all()):
        raise ValueError('NONFINITE_OR_INVALID_ACTION_LOGITS')
    return logits.argmax(dim=-1)


class FeatureMemoryState:
    """One episode; only actual observations and executed actions write state."""

    def __init__(self, head):
        self.head = head
        self.reset()

    def reset(self):
        self.memory = self.head.reset()
        self.previous_feature = None
        self.previous_action = None
        self.last_feature = None
        self.writes = 0
        self.stopped = False
        self.query_memory = None

    def validate_observation(self, previous_action):
        if self.stopped or previous_action == STOP:
            raise ValueError('STOP_MUST_NOT_CREATE_OBSERVATION')
        if self.query_memory is not None:
            raise ValueError('PHYSICAL_OBSERVATION_DURING_GENERATION')
        if self.writes == 0 and previous_action is not None:
            raise ValueError('INITIAL_OBSERVATION_REQUIRES_START')
        if self.writes > 0 and previous_action not in (1, 2, 3):
            raise ValueError('OBSERVATION_REQUIRES_EXECUTED_MOTION')

    @torch.inference_mode()
    def observe(self, feature, previous_action):
        self.validate_observation(previous_action)
        if feature.ndim == 1:
            feature = feature[None]
        if feature.shape != (1, self.head.writer.in_features) or not bool(torch.isfinite(feature).all()):
            raise ValueError('BAD_CAUSAL_FEATURE')
        x = feature.to(device=self.memory.device, dtype=self.memory.dtype)
        action = torch.tensor([START if previous_action is None else previous_action],
                              dtype=torch.long, device=self.memory.device)
        updated = self.head.update(x, self.memory, self.previous_feature, action)
        if not bool(torch.isfinite(updated).all()):
            raise ValueError('NONFINITE_MEMORY')
        self.memory = updated
        self.previous_feature = x.detach().clone()
        self.previous_action = previous_action
        self.last_feature = feature[0].detach().cpu().clone()
        self.writes += 1
        return dict(memory_input_sha256=tensor_hash(x), memory_sha256=tensor_hash(self.memory),
                    executed_previous_action=previous_action, writes=self.writes)

    def mark_stopped(self):
        if self.query_memory is not None:
            raise ValueError('EXECUTION_DURING_GENERATION')
        self.previous_action = STOP
        self.stopped = True

    @torch.inference_mode()
    def begin_query(self):
        if self.query_memory is not None or not self.writes or self.stopped:
            raise ValueError('INVALID_QUERY_START')
        self.query_memory = self.memory.detach().clone()
        self.query_writes = self.writes
        self.query_memory_sha256 = tensor_hash(self.query_memory)

    def end_query(self):
        if self.query_memory is None:
            raise ValueError('NO_ACTIVE_QUERY')
        self.query_memory = None

    @torch.inference_mode()
    def delta(self, actor_feature):
        if self.query_memory is None or self.writes != self.query_writes:
            raise ValueError('NO_FIXED_QUERY_MEMORY')
        if actor_feature.shape != (1, self.head.writer.in_features) or not bool(torch.isfinite(actor_feature).all()):
            raise ValueError('BAD_LIVE_ACTOR_FEATURE')
        x = actor_feature.to(device=self.query_memory.device, dtype=self.query_memory.dtype)
        delta = self.head.action_delta(x, self.query_memory)
        if delta.shape != (1, 4) or not bool(torch.isfinite(delta).all()):
            raise ValueError('NONFINITE_RESIDUAL')
        return delta


class LiveRuntime(FeatureMemoryState):
    def __init__(self, base, tokenizer, processor, head, token_ids, instruction):
        super().__init__(head)
        # Read-only reuse of the frozen feature encoder, including its dtype,
        # visual/instruction/action mixture and original processor behavior.
        from capture_runtime_v3 import DenseRuntime
        self.encoder = DenseRuntime(base, tokenizer, processor, head, token_ids, instruction)
        self.feature = self.encoder.feature

    @torch.inference_mode()
    def observe(self, rgb, previous_action):
        self.validate_observation(previous_action)
        return super().observe(self.feature(rgb, previous_action), previous_action)


class ChunkActionProcessor:
    """Transformers LogitsProcessor interface, with one frozen state per query.

    actor_feature_provider must return the latest REAL backbone forward-hook
    feature for this token. Generated tokens never write execution memory.
    Full-vocabulary EOS behavior is preserved; four-class and generated-token
    selections are logged separately. Offsets beyond 3 receive no residual.
    """

    def __init__(self, state, header, token_ids, actor_feature_provider, apply_residual=True):
        if len(token_ids) != 4 or len(set(token_ids)) != 4:
            raise ValueError('FOUR_DISTINCT_ACTION_TOKENS_REQUIRED')
        if state.query_memory is None:
            raise ValueError('BEGIN_QUERY_REQUIRED')
        self.state = state
        self.boundary = ActionBoundary(header)
        self.token_ids = list(token_ids)
        self.actor_feature_provider = actor_feature_provider
        self.apply_residual = apply_residual
        self.records = []

    @torch.inference_mode()
    def __call__(self, input_ids, scores):
        offset = self.boundary.offset(input_ids)
        if offset is None:
            return scores
        if scores.ndim != 2 or scores.shape[0] != 1 or scores.dtype != torch.float32:
            raise ValueError('ACTION_SCORE_INTERFACE_CHANGED')
        if offset != len(self.records):
            raise ValueError('SKIPPED_OR_DUPLICATED_TOKEN_OFFSET')
        native = scores[:, self.token_ids]
        selected_action(native)
        actor = self.actor_feature_provider()
        if actor is None:
            raise ValueError('MISSING_REAL_FORWARD_HOOK_FEATURE')
        bias = self.state.delta(actor) if self.apply_residual and offset < 4 else torch.zeros_like(native)
        revised = scores.clone()
        revised[:, self.token_ids] += bias.to(scores.device)
        method = revised[:, self.token_ids]
        action = selected_action(method)
        native_token, method_token = int(scores.argmax(-1)[0]), int(revised.argmax(-1)[0])
        if offset >= 4 and method_token in self.token_ids:
            raise ValueError('UNREGISTERED_FIFTH_ACTION_TOKEN')
        self.records.append(dict(offset=offset, native_logits=native[0].detach().cpu().tolist(),
            method_logits=method[0].detach().cpu().tolist(), residual=bias[0].detach().cpu().tolist(),
            native_action=int(selected_action(native)[0]), method_action=int(action[0]),
            native_token=native_token, method_token=method_token,
            actor_feature_sha256=tensor_hash(actor), query_writes=self.state.query_writes,
            query_memory_sha256=self.state.query_memory_sha256, residual_applied=self.apply_residual and offset < 4))
        return revised


def make_logits_processors(processor):
    # Lazy import keeps CPU memory-state use independent of Transformers.
    from transformers import LogitsProcessorList
    return LogitsProcessorList([processor])
