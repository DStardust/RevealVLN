"""Fixed Stage-1 reveal/expiry encoder for MF3ZN-TUAD v1.

The module deliberately has no intervention-utility input or head.  Its only
job is to encode a causal prefix and supervise the three factors from which
U/A/D is deterministically derived, together with reveal/expiry hazards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TEMPORAL_HIDDEN_DIM = 64


def _frozen_projection(input_dim: int, output_dim: int) -> Tensor:
    """Create a deterministic, non-learned dense projection matrix."""

    row = torch.arange(1, input_dim + 1, dtype=torch.float32)[:, None]
    column = torch.arange(1, output_dim + 1, dtype=torch.float32)[None, :]
    value = torch.sin(row * column * 0.017) + torch.cos(
        row * (column + 1.0) * 0.013
    )
    return value / value.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


@dataclass(frozen=True)
class RevealExpiryOutput:
    """Per-prefix factor/hazard logits plus the causal GRU state."""

    target_in_set_logit: Tensor
    separation_logit: Tensor
    evidence_logit: Tensor
    reveal_hazard_logit: Tensor
    expiry_hazard_logit: Tensor
    state_embedding: Tensor


def _validate_sequence_inputs(
    sequence_features: Tensor,
    sequence_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    if sequence_features.ndim != 3:
        raise ValueError("sequence_features must have shape [batch,time,feature]")
    if sequence_mask.ndim != 2 or sequence_mask.shape != sequence_features.shape[:2]:
        raise ValueError("sequence_mask must have shape [batch,time]")
    if sequence_mask.dtype is not torch.bool:
        raise TypeError("sequence_mask must be boolean")
    if sequence_features.shape[0] == 0 or sequence_features.shape[1] == 0:
        raise ValueError("empty temporal batch")
    if not bool(sequence_mask.any(dim=1).all()):
        raise ValueError("every sequence must contain at least one causal prefix")
    # Only right padding is permitted.  This makes the final valid state and
    # every prefix prediction unambiguous.
    if bool((sequence_mask[:, 1:] & ~sequence_mask[:, :-1]).any()):
        raise ValueError("sequence_mask must be right padded")
    if not bool(torch.isfinite(sequence_features).all()):
        raise ValueError("sequence_features contain non-finite values")
    return sequence_features, sequence_mask


class TemporalRevealExpiryEncoder(nn.Module):
    """The single pre-registered causal GRU architecture.

    ``hidden_dim`` is accepted only so a drifted caller fails with a useful
    error instead of silently creating an architecture variant.
    """

    def __init__(self, input_dim: int, hidden_dim: int = TEMPORAL_HIDDEN_DIM):
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim must be positive")
        if int(hidden_dim) != TEMPORAL_HIDDEN_DIM:
            raise ValueError("MF3ZN-TUAD v1 fixes the temporal hidden size at 64")
        self.input_dim = int(input_dim)
        self.hidden_dim = TEMPORAL_HIDDEN_DIM
        self.register_buffer(
            "fixed_input_projection",
            _frozen_projection(self.input_dim, self.hidden_dim),
            persistent=True,
        )
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.target_in_set_head = nn.Linear(self.hidden_dim, 1)
        self.separation_head = nn.Linear(self.hidden_dim, 1)
        self.evidence_head = nn.Linear(self.hidden_dim, 1)
        self.reveal_hazard_head = nn.Linear(self.hidden_dim, 1)
        self.expiry_hazard_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        sequence_features: Tensor,
        sequence_mask: Tensor,
    ) -> RevealExpiryOutput:
        sequence_features, sequence_mask = _validate_sequence_inputs(
            sequence_features, sequence_mask
        )
        masked = sequence_features.masked_fill(~sequence_mask.unsqueeze(-1), 0.0)
        normalized = F.layer_norm(masked, (self.input_dim,))
        projected = torch.tanh(normalized @ self.fixed_input_projection)
        state, _ = self.gru(projected)
        state = state.masked_fill(~sequence_mask.unsqueeze(-1), 0.0)

        def head(module: nn.Module) -> Tensor:
            value = module(state).squeeze(-1)
            return value.masked_fill(~sequence_mask, 0.0)

        return RevealExpiryOutput(
            target_in_set_logit=head(self.target_in_set_head),
            separation_logit=head(self.separation_head),
            evidence_logit=head(self.evidence_head),
            reveal_hazard_logit=head(self.reveal_hazard_head),
            expiry_hazard_logit=head(self.expiry_hazard_head),
            state_embedding=state,
        )


def last_causal_state(output: RevealExpiryOutput, sequence_mask: Tensor) -> Tensor:
    """Return the state at each sequence's decision prefix."""

    if sequence_mask.ndim != 2 or sequence_mask.dtype is not torch.bool:
        raise ValueError("invalid sequence mask")
    if output.state_embedding.shape[:2] != sequence_mask.shape:
        raise ValueError("state/mask shape mismatch")
    lengths = sequence_mask.sum(dim=1)
    if bool((lengths < 1).any()):
        raise ValueError("empty temporal sequence")
    rows = torch.arange(len(lengths), device=sequence_mask.device)
    return output.state_embedding[rows, lengths - 1]


def _masked_binary_loss(logit: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if logit.shape != target.shape or mask.shape != target.shape:
        raise ValueError("temporal supervision shapes do not match")
    valid = mask & torch.isfinite(target)
    if not bool(valid.any()):
        return logit.sum() * 0.0
    values = target[valid]
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("binary temporal targets must lie in [0,1]")
    return F.binary_cross_entropy_with_logits(logit[valid], values)


@dataclass(frozen=True)
class RevealExpiryTargets:
    """Oracle-only Stage-1 labels; this type never contains utility."""

    target_in_set: Tensor
    separation: Tensor
    evidence: Tensor
    reveal_event: Tensor
    expiry_event: Tensor
    factor_mask: Tensor
    reveal_at_risk: Tensor
    expiry_at_risk: Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.target_in_set,
            self.separation,
            self.evidence,
            self.reveal_event,
            self.expiry_event,
            self.factor_mask,
            self.reveal_at_risk,
            self.expiry_at_risk,
        )
        shape = tensors[0].shape
        if len(shape) != 2 or any(value.shape != shape for value in tensors):
            raise ValueError("all reveal/expiry targets must have shape [batch,time]")
        if any(value.dtype is not torch.bool for value in tensors[5:]):
            raise TypeError("target masks must be boolean")


class TemporalRevealExpiryLoss(nn.Module):
    """Fixed equal-weight factor and discrete-hazard objective."""

    def forward(
        self,
        output: RevealExpiryOutput,
        targets: RevealExpiryTargets,
        sequence_mask: Tensor,
    ) -> Tensor:
        if sequence_mask.dtype is not torch.bool:
            raise TypeError("sequence_mask must be boolean")
        factor_mask = sequence_mask & targets.factor_mask
        losses = (
            _masked_binary_loss(
                output.target_in_set_logit, targets.target_in_set, factor_mask
            ),
            _masked_binary_loss(
                output.separation_logit, targets.separation, factor_mask
            ),
            _masked_binary_loss(
                output.evidence_logit, targets.evidence, factor_mask
            ),
            _masked_binary_loss(
                output.reveal_hazard_logit,
                targets.reveal_event,
                sequence_mask & targets.reveal_at_risk,
            ),
            _masked_binary_loss(
                output.expiry_hazard_logit,
                targets.expiry_event,
                sequence_mask & targets.expiry_at_risk,
            ),
        )
        return torch.stack(losses).mean()


def freeze_temporal_encoder(model: TemporalRevealExpiryEncoder) -> None:
    """Freeze Stage 1 before any exact intervention utility is consumed."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


__all__ = [
    "TEMPORAL_HIDDEN_DIM",
    "RevealExpiryOutput",
    "RevealExpiryTargets",
    "TemporalRevealExpiryEncoder",
    "TemporalRevealExpiryLoss",
    "freeze_temporal_encoder",
    "last_causal_state",
]
