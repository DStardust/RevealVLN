"""UAD-only readiness heads and a bounded ETP candidate-logit adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


MF3B_SCOPE = {
    "instruction_guided_vln_mainline": True,
    "method_scope": "uad_readiness_residual_adapter",
    "uses_secondary_topology": False,
    "uses_checkpoint_memory": False,
    "uses_physical_backtracking": False,
    "uses_branch_exploration": False,
    "public_unseen_authorized": False,
}


class StructuredUADOutput(NamedTuple):
    target_logits: Tensor
    target_in_set_logit: Tensor
    separation_logit: Tensor
    evidence_logit: Tensor
    reveal_hazard_logit: Tensor
    expiry_hazard_logit: Tensor
    uad_probabilities: Tensor


class ResidualFusionOutput(NamedTuple):
    logits: Tensor
    authorized: Tensor


class NativeConditionedUADOutput(NamedTuple):
    native_error_logit: Tensor
    alternative_logits: Tensor


class PairwiseSwitchUtilityOutput(NamedTuple):
    outcome_logits: Tensor


class CausalReturnSafetyOutput(NamedTuple):
    expected_utility: Tensor
    beneficial_logit: Tensor


class CausalReturnSafetyCritic(nn.Module):
    """Predict final task gain from pre-decision causal state only.

    This critic deliberately excludes the post-excursion observation and all
    future trajectory features.  It consumes only quantities available at the
    instant the frozen policy proposes its native action and runner-up.
    """

    def __init__(self, feature_dim: int = 768, projection_dim: int = 32) -> None:
        super().__init__()
        if min(feature_dim, projection_dim) < 1:
            raise ValueError("critic dimensions must be positive")
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.normalizer = nn.LayerNorm(feature_dim)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, projection_dim), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(projection_dim * 5),
            nn.Linear(projection_dim * 5, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.expected_utility = nn.Linear(64, 1)
        self.beneficial = nn.Linear(64, 1)

    def forward(
        self, instruction: Tensor, checkpoint: Tensor,
        native: Tensor, alternative: Tensor,
    ) -> CausalReturnSafetyOutput:
        values = (instruction, checkpoint, native, alternative)
        if any(value.shape[-1] != self.feature_dim for value in values):
            raise ValueError("critic feature dimension drift")
        encoded = [self.project(self.normalizer(value)) for value in values]
        contrast = self.project(self.normalizer(alternative - native))
        fused = self.fusion(torch.cat([*encoded, contrast], dim=-1))
        return CausalReturnSafetyOutput(
            self.expected_utility(fused).squeeze(-1),
            self.beneficial(fused).squeeze(-1),
        )


class PolicyAnchoredTop2Output(NamedTuple):
    target_logits: Tensor
    residual: Tensor


class PolicyAnchoredTop2UAD(nn.Module):
    """Estimate target posterior while preserving the frozen policy prior.

    The head learns a bounded residual over the frozen current-local logits.
    Deployment may inspect only the posterior advantage of the frozen
    policy's runner-up over its native action; this module never proposes an
    unconstrained third action.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        candidate_feature_dim: int = 1536,
        hidden_dim: int = 64,
        correction_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if min(feature_dim, candidate_feature_dim, hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        if not isinstance(correction_bound, (int, float)) or correction_bound <= 0:
            raise ValueError("correction_bound must be positive")
        self.feature_dim = feature_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.hidden_dim = hidden_dim
        self.correction_bound = float(correction_bound)
        self.history_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.instruction_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(candidate_feature_dim),
            nn.Linear(candidate_feature_dim, hidden_dim), nn.GELU(),
        )
        self.temporal = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 3, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1),
        )
        # At initialization the adapter delegates exactly to the frozen policy.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        history_embeddings: Tensor,
        candidate_embeddings: Tensor,
        candidate_mask: Tensor,
        instruction_embedding: Tensor,
        native_scores: Tensor,
        native_index: Tensor,
    ) -> PolicyAnchoredTop2Output:
        if history_embeddings.ndim != 3 or candidate_embeddings.ndim != 4:
            raise ValueError("history must be 3-D and candidates must be 4-D")
        batch, steps, feature_dim = history_embeddings.shape
        if feature_dim != self.feature_dim:
            raise ValueError("history feature dimension drift")
        if candidate_embeddings.shape[:2] != (batch, steps):
            raise ValueError("history and candidate axes do not match")
        if candidate_embeddings.shape[-1] != self.candidate_feature_dim:
            raise ValueError("candidate feature dimension drift")
        if candidate_mask.shape != candidate_embeddings.shape[:-1]:
            raise ValueError("candidate mask shape mismatch")
        if native_scores.shape != candidate_mask.shape:
            raise ValueError("native score shape mismatch")
        if native_index.shape != (batch, steps):
            raise ValueError("native index shape mismatch")
        if instruction_embedding.shape != (batch, feature_dim):
            raise ValueError("instruction embedding shape mismatch")
        if candidate_mask.dtype is not torch.bool:
            raise TypeError("candidate_mask must be boolean")
        if native_index.dtype is not torch.long:
            raise TypeError("native_index must be torch.long")
        if not (
            torch.isfinite(history_embeddings).all()
            and torch.isfinite(candidate_embeddings).all()
            and torch.isfinite(instruction_embedding).all()
            and torch.isfinite(native_scores[candidate_mask]).all()
        ):
            raise ValueError("policy-anchored inputs must be finite")

        candidate_count = candidate_mask.sum(-1)
        safe_native = native_index.clamp(0, candidate_mask.shape[-1] - 1)
        valid_native = (native_index >= 0) & (
            native_index < candidate_mask.shape[-1]
        )
        valid_native &= candidate_mask.gather(
            -1, safe_native.unsqueeze(-1)
        ).squeeze(-1)
        history = self.history_projection(history_embeddings)
        instruction = self.instruction_projection(instruction_embedding)
        instruction_steps = instruction.unsqueeze(1).expand(batch, steps, -1)
        temporal, _ = self.temporal(torch.cat((history, instruction_steps), -1))
        candidates = self.candidate_projection(candidate_embeddings)
        native = candidates.gather(
            2, safe_native[..., None, None].expand(
                batch, steps, 1, candidates.shape[-1]
            ),
        ).squeeze(2)
        native = native * valid_native.unsqueeze(-1).to(native.dtype)
        native_score = native_scores.gather(
            -1, safe_native.unsqueeze(-1)
        ).squeeze(-1)
        native_score = torch.where(
            valid_native, native_score, torch.zeros_like(native_score)
        )
        relative_score = (native_scores - native_score.unsqueeze(-1)).clamp(
            min=-20.0, max=20.0
        ).masked_fill(~candidate_mask, 0.0)
        safe_scores = native_scores.masked_fill(~candidate_mask, -torch.inf)
        top_two = torch.topk(safe_scores, k=2, dim=-1).values
        margin = torch.where(
            candidate_count >= 2,
            top_two[..., 0] - top_two[..., 1],
            torch.zeros_like(top_two[..., 0]),
        )
        count_feature = candidate_count.to(candidates.dtype) / (
            candidate_count.to(candidates.dtype) + 1.0
        )
        expanded = lambda value: value.unsqueeze(2).expand(
            batch, steps, candidates.shape[2], value.shape[-1]
        )
        raw = self.residual_head(torch.cat((
            expanded(temporal), expanded(instruction_steps), expanded(native),
            candidates, candidates - expanded(native),
            relative_score.unsqueeze(-1),
            margin.unsqueeze(-1).unsqueeze(-1).expand(
                batch, steps, candidates.shape[2], 1
            ),
            count_feature.unsqueeze(-1).unsqueeze(-1).expand(
                batch, steps, candidates.shape[2], 1
            ),
        ), -1)).squeeze(-1)
        residual = self.correction_bound * torch.tanh(raw)
        residual = residual.masked_fill(~candidate_mask, 0.0)
        center = residual.sum(-1, keepdim=True) / candidate_count.clamp_min(
            1
        ).unsqueeze(-1)
        residual = (residual - center).masked_fill(~candidate_mask, 0.0)
        target_logits = (native_scores + residual).masked_fill(
            ~candidate_mask, -torch.inf
        )
        return PolicyAnchoredTop2Output(target_logits, residual)


def policy_anchored_target_loss(
    output: PolicyAnchoredTop2Output,
    batch: dict[str, Tensor],
) -> Tensor:
    """Proper candidate-level target-posterior loss."""

    target = batch["target_index"]
    candidate_mask = batch["candidate_mask"]
    valid = batch["step_mask"] & (target >= 0)
    safe_target = target.clamp(0, candidate_mask.shape[-1] - 1)
    valid &= candidate_mask.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    if not valid.any():
        return output.residual.sum() * 0.0
    return F.cross_entropy(output.target_logits[valid], target[valid])


def top2_switch_indices(
    native_scores: Tensor,
    candidate_mask: Tensor,
    native_index: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the frozen policy runner-up and rows with a valid top-2 pair."""

    if native_scores.shape != candidate_mask.shape:
        raise ValueError("native scores and mask shape mismatch")
    if native_index.shape != candidate_mask.shape[:-1]:
        raise ValueError("native index shape mismatch")
    safe_native = native_index.clamp(0, candidate_mask.shape[-1] - 1)
    valid_native = (native_index >= 0) & (
        native_index < candidate_mask.shape[-1]
    )
    valid_native &= candidate_mask.gather(
        -1, safe_native.unsqueeze(-1)
    ).squeeze(-1)
    alternatives = candidate_mask.clone().scatter(
        -1, safe_native.unsqueeze(-1), False
    )
    runner = native_scores.masked_fill(~alternatives, -torch.inf).argmax(-1)
    valid = valid_native & (candidate_mask.sum(-1) >= 2)
    return runner, valid


def top2_switch_targets(
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return NEITHER/RESCUE/HARM labels for the fixed top-2 switch."""

    runner, valid = top2_switch_indices(
        batch["native_scores"], batch["candidate_mask"], batch["native_index"]
    )
    native = batch["native_index"]
    target = batch["target_index"]
    valid &= batch["step_mask"] & (target >= 0)
    labels = torch.zeros_like(native)
    labels = torch.where(native == target, torch.full_like(labels, 2), labels)
    labels = torch.where(
        (native != target) & (runner == target), torch.ones_like(labels), labels
    )
    return labels, runner, valid


def top2_horizon_switch_targets(
    batch: dict[str, Tensor], *, horizon: int = 3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Label a top-2 switch by short-horizon teacher-path consistency.

    The future target indices are training-only supervision.  At deployment
    the controller receives only the current history and candidate features.
    A rescue means the runner is supported by the teacher path within the
    next ``horizon`` observations while the native action is not; harm is the
    converse.  Ambiguous cases are neutral rather than guessed.
    """

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    labels, runner, valid = top2_switch_targets(batch)
    target = batch["target_index"]
    steps = target.shape[1]
    for batch_index in range(target.shape[0]):
        for time_index in range(steps):
            if not bool(valid[batch_index, time_index]):
                continue
            native = int(batch["native_index"][batch_index, time_index])
            alternative = int(runner[batch_index, time_index])
            end = min(steps, time_index + horizon)
            native_supported = False
            alternative_supported = False
            for future in range(time_index, end):
                if not bool(batch["step_mask"][batch_index, future]):
                    continue
                future_target = int(target[batch_index, future])
                native_supported |= future_target == native
                alternative_supported |= future_target == alternative
            labels[batch_index, time_index] = (
                1 if alternative_supported and not native_supported else
                2 if native_supported and not alternative_supported else 0
            )
    return labels, runner, valid


def top2_posterior_advantage(
    output: PolicyAnchoredTop2Output,
    native_scores: Tensor,
    candidate_mask: Tensor,
    native_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return P(runner-up target) - P(native target)."""

    runner, valid = top2_switch_indices(
        native_scores, candidate_mask, native_index
    )
    safe_native = native_index.clamp(0, candidate_mask.shape[-1] - 1)
    probabilities = torch.softmax(output.target_logits, dim=-1)
    runner_probability = probabilities.gather(
        -1, runner.unsqueeze(-1)
    ).squeeze(-1)
    native_probability = probabilities.gather(
        -1, safe_native.unsqueeze(-1)
    ).squeeze(-1)
    advantage = runner_probability - native_probability
    return advantage, runner, valid


def top2_conditional_advantage(
    output: PolicyAnchoredTop2Output,
    native_scores: Tensor,
    candidate_mask: Tensor,
    native_index: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the candidate-set-invariant conditional top-2 advantage.

    For target logits ``z``, this is
    ``tanh((z_runner - z_native) / 2)``, equivalently
    ``(P_runner - P_native) / (P_runner + P_native)``.  Unlike the absolute
    posterior difference, its scale cannot shrink merely because unrelated
    candidates enter the current set.
    """

    runner, valid = top2_switch_indices(
        native_scores, candidate_mask, native_index
    )
    safe_native = native_index.clamp(0, candidate_mask.shape[-1] - 1)
    runner_logit = output.target_logits.gather(
        -1, runner.unsqueeze(-1)
    ).squeeze(-1)
    native_logit = output.target_logits.gather(
        -1, safe_native.unsqueeze(-1)
    ).squeeze(-1)
    return torch.tanh((runner_logit - native_logit) / 2.0), runner, valid


def policy_anchored_conditional_top2_loss(
    output: PolicyAnchoredTop2Output,
    batch: dict[str, Tensor],
    *,
    top2_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Joint proper candidate and conditional native-vs-runner loss."""

    if not isinstance(top2_weight, (int, float)) or top2_weight < 0:
        raise ValueError("top2_weight must be non-negative")
    target_loss = policy_anchored_target_loss(output, batch)
    _, runner, valid = top2_switch_targets(batch)
    target = batch["target_index"]
    native = batch["native_index"]
    comparable = valid & ((target == native) | (target == runner))
    safe_native = native.clamp(0, batch["candidate_mask"].shape[-1] - 1)
    delta = output.target_logits.gather(
        -1, runner.unsqueeze(-1)
    ).squeeze(-1) - output.target_logits.gather(
        -1, safe_native.unsqueeze(-1)
    ).squeeze(-1)
    top2 = (
        F.binary_cross_entropy_with_logits(
            delta[comparable], (target[comparable] == runner[comparable]).to(
                delta.dtype
            ),
        )
        if comparable.any()
        else output.residual.sum() * 0.0
    )
    return {
        "total": target_loss + float(top2_weight) * top2,
        "target": target_loss,
        "top2": top2,
    }


def median_mad_lower_confidence(
    member_advantages: Tensor,
    *,
    mad_weight: float,
) -> Tensor:
    """Robust ensemble score: member median minus a MAD disagreement cost."""

    if member_advantages.ndim < 1 or member_advantages.shape[0] < 1:
        raise ValueError("member_advantages must have a non-empty member axis")
    if not isinstance(mad_weight, (int, float)) or mad_weight < 0:
        raise ValueError("mad_weight must be non-negative")
    if not torch.isfinite(member_advantages).all():
        raise ValueError("member advantages must be finite")
    median = member_advantages.median(0).values
    mad = (member_advantages - median.unsqueeze(0)).abs().median(0).values
    return median - float(mad_weight) * mad


class PairwiseSwitchUtility(nn.Module):
    """Predict ``neither/rescue/harm`` for each native-to-alternative switch."""

    def __init__(
        self,
        feature_dim: int = 768,
        candidate_feature_dim: int = 1536,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(feature_dim, candidate_feature_dim, hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        self.feature_dim = feature_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.history_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.instruction_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(candidate_feature_dim),
            nn.Linear(candidate_feature_dim, hidden_dim), nn.GELU(),
        )
        self.temporal = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 3, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        history_embeddings: Tensor,
        candidate_embeddings: Tensor,
        candidate_mask: Tensor,
        instruction_embedding: Tensor,
        native_scores: Tensor,
        native_index: Tensor,
    ) -> PairwiseSwitchUtilityOutput:
        if history_embeddings.ndim != 3 or candidate_embeddings.ndim != 4:
            raise ValueError("history must be 3-D and candidates must be 4-D")
        batch, steps, feature_dim = history_embeddings.shape
        if feature_dim != self.feature_dim:
            raise ValueError("history feature dimension drift")
        if candidate_embeddings.shape[:2] != (batch, steps):
            raise ValueError("history and candidate axes do not match")
        if candidate_embeddings.shape[-1] != self.candidate_feature_dim:
            raise ValueError("candidate feature dimension drift")
        if candidate_mask.shape != candidate_embeddings.shape[:-1]:
            raise ValueError("candidate mask shape mismatch")
        if native_scores.shape != candidate_mask.shape:
            raise ValueError("native score shape mismatch")
        if native_index.shape != (batch, steps):
            raise ValueError("native index shape mismatch")
        if instruction_embedding.shape != (batch, feature_dim):
            raise ValueError("instruction embedding shape mismatch")
        if candidate_mask.dtype is not torch.bool:
            raise TypeError("candidate_mask must be boolean")
        if native_index.dtype is not torch.long:
            raise TypeError("native_index must be torch.long")
        if not (
            torch.isfinite(history_embeddings).all()
            and torch.isfinite(candidate_embeddings).all()
            and torch.isfinite(instruction_embedding).all()
            and torch.isfinite(native_scores[candidate_mask]).all()
        ):
            raise ValueError("switch-utility inputs must be finite")

        candidate_count = candidate_mask.sum(-1)
        safe_native = native_index.clamp(0, candidate_mask.shape[-1] - 1)
        valid_native = (native_index >= 0) & (
            native_index < candidate_mask.shape[-1]
        )
        valid_native &= candidate_mask.gather(
            -1, safe_native.unsqueeze(-1)
        ).squeeze(-1)
        history = self.history_projection(history_embeddings)
        instruction = self.instruction_projection(instruction_embedding)
        instruction_steps = instruction.unsqueeze(1).expand(batch, steps, -1)
        temporal, _ = self.temporal(torch.cat((history, instruction_steps), -1))
        candidates = self.candidate_projection(candidate_embeddings)
        native = candidates.gather(
            2, safe_native[..., None, None].expand(
                batch, steps, 1, candidates.shape[-1]
            ),
        ).squeeze(2)
        native = native * valid_native.unsqueeze(-1).to(native.dtype)
        native_score = native_scores.gather(
            -1, safe_native.unsqueeze(-1)
        ).squeeze(-1)
        native_score = torch.where(
            valid_native, native_score, torch.zeros_like(native_score)
        )
        relative_score = (native_scores - native_score.unsqueeze(-1)).clamp(
            min=-20.0, max=20.0
        ).masked_fill(~candidate_mask, 0.0)
        safe_scores = native_scores.masked_fill(~candidate_mask, -torch.inf)
        top_two = torch.topk(safe_scores, k=2, dim=-1).values
        margin = torch.where(
            candidate_count >= 2,
            top_two[..., 0] - top_two[..., 1],
            torch.zeros_like(top_two[..., 0]),
        )
        count_feature = candidate_count.to(candidates.dtype) / (
            candidate_count.to(candidates.dtype) + 1.0
        )
        expanded = lambda value: value.unsqueeze(2).expand(
            batch, steps, candidates.shape[2], value.shape[-1]
        )
        outcome_logits = self.outcome_head(torch.cat((
            expanded(temporal), expanded(instruction_steps),
            expanded(native), candidates, candidates - expanded(native),
            relative_score.unsqueeze(-1),
            margin.unsqueeze(-1).unsqueeze(-1).expand(
                batch, steps, candidates.shape[2], 1
            ),
            count_feature.unsqueeze(-1).unsqueeze(-1).expand(
                batch, steps, candidates.shape[2], 1
            ),
        ), -1))
        native_slots = torch.zeros_like(candidate_mask).scatter(
            -1, safe_native.unsqueeze(-1), True
        )
        alternative_mask = (
            candidate_mask & ~native_slots & valid_native.unsqueeze(-1)
        )
        outcome_logits = outcome_logits.masked_fill(
            ~alternative_mask.unsqueeze(-1), 0.0
        )
        return PairwiseSwitchUtilityOutput(outcome_logits=outcome_logits)


def pairwise_switch_targets(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Return outcome labels and the valid native-to-alternative pair mask."""

    native = batch["native_index"]
    target = batch["target_index"]
    candidate_mask = batch["candidate_mask"]
    safe_native = native.clamp(0, candidate_mask.shape[-1] - 1)
    native_slots = torch.zeros_like(candidate_mask).scatter(
        -1, safe_native.unsqueeze(-1), True
    )
    valid_steps = (
        batch["step_mask"] & (native >= 0) & (target >= 0)
        & (candidate_mask.sum(-1) >= 2)
    )
    pair_mask = candidate_mask & ~native_slots & valid_steps.unsqueeze(-1)
    target_slots = torch.zeros_like(candidate_mask).scatter(
        -1, target.clamp(0, candidate_mask.shape[-1] - 1).unsqueeze(-1), True
    )
    native_correct = native == target
    labels = torch.zeros_like(candidate_mask, dtype=torch.long)
    labels = torch.where(
        native_correct.unsqueeze(-1) & pair_mask,
        torch.full_like(labels, 2),
        labels,
    )
    labels = torch.where(
        ~native_correct.unsqueeze(-1) & target_slots & pair_mask,
        torch.ones_like(labels),
        labels,
    )
    return labels, pair_mask


def pairwise_switch_utility_loss(
    output: PairwiseSwitchUtilityOutput,
    batch: dict[str, Tensor],
) -> Tensor:
    labels, pair_mask = pairwise_switch_targets(batch)
    if not pair_mask.any():
        return output.outcome_logits.sum() * 0.0
    return F.cross_entropy(output.outcome_logits[pair_mask], labels[pair_mask])


def pairwise_expected_utility(
    output: PairwiseSwitchUtilityOutput,
) -> Tensor:
    probabilities = torch.softmax(output.outcome_logits, dim=-1)
    return probabilities[..., 1] - probabilities[..., 2]


def top2_switch_utility_loss(
    output: PairwiseSwitchUtilityOutput,
    batch: dict[str, Tensor],
) -> Tensor:
    """Proper three-class loss for the frozen policy's runner-up only."""

    labels, runner, valid = top2_switch_targets(batch)
    runner_logits = output.outcome_logits.gather(
        2,
        runner[..., None, None].expand(
            *runner.shape, 1, output.outcome_logits.shape[-1]
        ),
    ).squeeze(2)
    if not valid.any():
        return output.outcome_logits.sum() * 0.0
    per_step = F.cross_entropy(
        runner_logits.transpose(1, 2), labels, reduction="none"
    )
    counts = valid.sum(-1)
    per_episode = (per_step * valid).sum(-1) / counts.clamp_min(1)
    return per_episode[counts > 0].mean()


def top2_cost_sensitive_utility_loss(
    output: PairwiseSwitchUtilityOutput,
    batch: dict[str, Tensor],
    *,
    class_weights: tuple[float, float, float] = (1.0, 2.0, 0.5),
) -> Tensor:
    """Episode-balanced CE with an explicit rescue/harm utility cost.

    The weights are part of the sealed protocol; they deliberately penalize
    missed rescues more than neutral decisions and do not use future labels.
    The output is a cost-sensitive score, so threshold selection remains
    validation-only rather than pretending the softmax is an unweighted
    probability estimate.
    """

    if len(class_weights) != 3 or any(
        not isinstance(value, (int, float)) or value <= 0
        for value in class_weights
    ):
        raise ValueError("class_weights must contain three positive values")
    labels, runner, valid = top2_switch_targets(batch)
    runner_logits = output.outcome_logits.gather(
        2,
        runner[..., None, None].expand(
            *runner.shape, 1, output.outcome_logits.shape[-1]
        ),
    ).squeeze(2)
    if not valid.any():
        return output.outcome_logits.sum() * 0.0
    weights = runner_logits.new_tensor(class_weights)
    per_step = F.cross_entropy(
        runner_logits.transpose(1, 2), labels, weight=weights,
        reduction="none",
    )
    counts = valid.sum(-1)
    per_episode = (per_step * valid).sum(-1) / counts.clamp_min(1)
    return per_episode[counts > 0].mean()


def top2_expected_switch_utility(
    output: PairwiseSwitchUtilityOutput,
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return P(rescue)-P(harm) for the frozen runner-up decision."""

    _, runner, valid = top2_switch_targets(batch)
    utilities = pairwise_expected_utility(output)
    score = utilities.gather(-1, runner.unsqueeze(-1)).squeeze(-1)
    return score, runner, valid


def top2_rescue_harm_logit(
    output: PairwiseSwitchUtilityOutput,
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the direct log odds of rescue versus harm for the runner-up."""

    _, runner, valid = top2_switch_targets(batch)
    logits = output.outcome_logits.gather(
        2,
        runner[..., None, None].expand(
            *runner.shape, 1, output.outcome_logits.shape[-1]
        ),
    ).squeeze(2)
    return logits[..., 1] - logits[..., 2], runner, valid


def top2_rescue_harm_loss(
    output: PairwiseSwitchUtilityOutput,
    batch: dict[str, Tensor],
    *,
    rescue_positive_weight: float,
) -> Tensor:
    """Episode-balanced binary risk over consequential Top-2 switches."""

    if not isinstance(rescue_positive_weight, (int, float)) or (
        rescue_positive_weight <= 0
    ):
        raise ValueError("rescue_positive_weight must be positive")
    labels, _, valid = top2_switch_targets(batch)
    consequential = valid & ((labels == 1) | (labels == 2))
    logit, _, _ = top2_rescue_harm_logit(output, batch)
    if not consequential.any():
        return output.outcome_logits.sum() * 0.0
    target = (labels == 1).to(logit.dtype)
    per_step = F.binary_cross_entropy_with_logits(
        logit, target,
        pos_weight=logit.new_tensor(float(rescue_positive_weight)),
        reduction="none",
    )
    counts = consequential.sum(-1)
    per_episode = (
        (per_step * consequential).sum(-1) / counts.clamp_min(1)
    )
    return per_episode[counts > 0].mean()


def top2_rescue_harm_ranked_loss(
    output: PairwiseSwitchUtilityOutput,
    batch: dict[str, Tensor],
    *,
    rescue_positive_weight: float,
    ranking_weight: float = 0.25,
) -> dict[str, Tensor]:
    """Add episode-balanced rescue-over-harm ranking to the binary risk loss."""

    if not isinstance(ranking_weight, (int, float)) or ranking_weight < 0:
        raise ValueError("ranking_weight must be non-negative")
    binary = top2_rescue_harm_loss(
        output, batch, rescue_positive_weight=rescue_positive_weight,
    )
    labels, _, valid = top2_switch_targets(batch)
    consequential = valid & ((labels == 1) | (labels == 2))
    logits, _, _ = top2_rescue_harm_logit(output, batch)
    counts = consequential.sum(-1).clamp_min(1).to(logits.dtype)
    weights = consequential.to(logits.dtype) / counts.unsqueeze(-1)
    rescue = labels == 1
    harm = labels == 2
    episode_rank_losses = []
    for batch_index in range(logits.shape[0]):
        rescue_index = rescue[batch_index]
        harm_index = harm[batch_index]
        if not rescue_index.any() or not harm_index.any():
            continue
        differences = (
            logits[batch_index, rescue_index].unsqueeze(1)
            - logits[batch_index, harm_index].unsqueeze(0)
        )
        pair_weights = (
            weights[batch_index, rescue_index].unsqueeze(1)
            * weights[batch_index, harm_index].unsqueeze(0)
        )
        episode_rank_losses.append(
            (F.softplus(-differences) * pair_weights).sum()
            / pair_weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
        )
    if episode_rank_losses:
        ranking = torch.stack(episode_rank_losses).mean()
    else:
        ranking = output.outcome_logits.sum() * 0.0
    return {
        "total": binary + float(ranking_weight) * ranking,
        "binary": binary,
        "ranking": ranking,
    }


def top2_horizon_rescue_harm_ranked_loss(
    output: PairwiseSwitchUtilityOutput,
    batch: dict[str, Tensor], *, rescue_positive_weight: float,
    ranking_weight: float = 0.25, horizon: int = 3,
) -> dict[str, Tensor]:
    """Episode-balanced rescue/harm loss using short-horizon targets."""

    if not isinstance(ranking_weight, (int, float)) or ranking_weight < 0:
        raise ValueError("ranking_weight must be non-negative")
    labels, runner, valid = top2_horizon_switch_targets(batch, horizon=horizon)
    logits = output.outcome_logits.gather(
        2, runner[..., None, None].expand(
            *runner.shape, 1, output.outcome_logits.shape[-1]
        ),
    ).squeeze(2)
    consequential = valid & ((labels == 1) | (labels == 2))
    if not consequential.any():
        zero = output.outcome_logits.sum() * 0.0
        return {"total": zero, "binary": zero, "ranking": zero}
    target = (labels == 1).to(logits.dtype)
    counts = consequential.sum(-1).clamp_min(1).to(logits.dtype)
    weights = consequential.to(logits.dtype) / counts.unsqueeze(-1)
    binary_steps = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[..., 1] - logits[..., 2], target,
        pos_weight=logits.new_tensor(float(rescue_positive_weight)),
        reduction="none",
    )
    binary = (
        (binary_steps * consequential).sum(-1) / counts
    )[counts > 0].mean()
    rescue = labels == 1
    harm = labels == 2
    ranks = []
    for batch_index in range(logits.shape[0]):
        if not rescue[batch_index].any() or not harm[batch_index].any():
            continue
        difference = (
            (logits[batch_index, rescue[batch_index], 1]
             - logits[batch_index, rescue[batch_index], 2]).unsqueeze(1)
            - (logits[batch_index, harm[batch_index], 1]
               - logits[batch_index, harm[batch_index], 2]).unsqueeze(0)
        )
        pair_weights = (
            weights[batch_index, rescue[batch_index]].unsqueeze(1)
            * weights[batch_index, harm[batch_index]].unsqueeze(0)
        )
        ranks.append(
            (torch.nn.functional.softplus(-difference) * pair_weights).sum()
            / pair_weights.sum().clamp_min(torch.finfo(logits.dtype).eps)
        )
    ranking = torch.stack(ranks).mean() if ranks else logits.sum() * 0.0
    return {
        "total": binary + float(ranking_weight) * ranking,
        "binary": binary, "ranking": ranking,
    }


class NativeConditionedUAD(nn.Module):
    """Predict a high-precision correction to a frozen policy decision.

    The model does not relearn the full navigation action.  It first estimates
    whether the frozen policy's current-local argmax is wrong, then ranks only
    the alternatives to that action.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        candidate_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        candidate_feature_dim = candidate_feature_dim or feature_dim
        if feature_dim < 1 or hidden_dim < 1 or candidate_feature_dim < 1:
            raise ValueError("model dimensions must be positive")
        self.feature_dim = feature_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.history_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(candidate_feature_dim),
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.instruction_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.error_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1),
        )
        self.alternative_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        history_embeddings: Tensor,
        candidate_embeddings: Tensor,
        candidate_mask: Tensor,
        instruction_embedding: Tensor,
        native_scores: Tensor,
        native_index: Tensor,
    ) -> NativeConditionedUADOutput:
        if history_embeddings.ndim != 3 or candidate_embeddings.ndim != 4:
            raise ValueError("history must be 3-D and candidates must be 4-D")
        batch, steps, feature_dim = history_embeddings.shape
        if feature_dim != self.feature_dim:
            raise ValueError("history feature dimension drift")
        if candidate_embeddings.shape[:2] != (batch, steps):
            raise ValueError("history and candidate axes do not match")
        if candidate_embeddings.shape[-1] != self.candidate_feature_dim:
            raise ValueError("candidate feature dimension drift")
        if candidate_mask.shape != candidate_embeddings.shape[:-1]:
            raise ValueError("candidate_mask shape mismatch")
        if native_scores.shape != candidate_mask.shape:
            raise ValueError("native score shape mismatch")
        if native_index.shape != (batch, steps):
            raise ValueError("native index shape mismatch")
        if instruction_embedding.shape != (batch, feature_dim):
            raise ValueError("instruction embedding shape mismatch")
        if candidate_mask.dtype is not torch.bool:
            raise TypeError("candidate_mask must be boolean")
        if native_index.dtype is not torch.long:
            raise TypeError("native_index must be torch.long")
        if not (
            torch.isfinite(history_embeddings).all()
            and torch.isfinite(candidate_embeddings).all()
            and torch.isfinite(instruction_embedding).all()
            and torch.isfinite(native_scores[candidate_mask]).all()
        ):
            raise ValueError("native-conditioned UAD inputs must be finite")

        candidate_count = candidate_mask.sum(-1)
        valid_native = (native_index >= 0) & (native_index < candidate_mask.shape[-1])
        safe_native = native_index.clamp(0, candidate_mask.shape[-1] - 1)
        valid_native &= candidate_mask.gather(
            -1, safe_native.unsqueeze(-1)
        ).squeeze(-1)

        history = self.history_projection(history_embeddings)
        candidates = self.candidate_projection(candidate_embeddings)
        instruction = self.instruction_projection(instruction_embedding)
        instruction_steps = instruction.unsqueeze(1).expand(batch, steps, -1)
        temporal, _ = self.temporal(torch.cat((history, instruction_steps), -1))
        native = candidates.gather(
            2,
            safe_native[..., None, None].expand(batch, steps, 1, candidates.shape[-1]),
        ).squeeze(2)
        native = native * valid_native.unsqueeze(-1).to(native.dtype)
        mask_values = candidate_mask.unsqueeze(-1).to(candidates.dtype)
        mean_candidate = (candidates * mask_values).sum(2) / mask_values.sum(
            2
        ).clamp_min(1.0)

        safe_scores = native_scores.masked_fill(~candidate_mask, -torch.inf)
        top_two = torch.topk(safe_scores, k=2, dim=-1).values
        score_margin = torch.where(
            candidate_count >= 2,
            top_two[..., 0] - top_two[..., 1],
            torch.zeros_like(top_two[..., 0]),
        )
        count_feature = candidate_count.to(candidates.dtype) / (
            candidate_count.to(candidates.dtype) + 1.0
        )
        native_error_logit = self.error_head(torch.cat((
            temporal, native, instruction_steps, mean_candidate,
            score_margin.unsqueeze(-1), count_feature.unsqueeze(-1),
        ), -1)).squeeze(-1)

        native_score = native_scores.gather(
            -1, safe_native.unsqueeze(-1)
        ).squeeze(-1)
        native_score = torch.where(
            valid_native, native_score, torch.zeros_like(native_score)
        )
        relative_score = (native_scores - native_score.unsqueeze(-1)).clamp(
            min=-20.0, max=20.0
        ).masked_fill(~candidate_mask, 0.0)
        native_expanded = native.unsqueeze(2).expand_as(candidates)
        alternative_logits = self.alternative_head(torch.cat((
            candidates - native_expanded,
            candidates,
            temporal.unsqueeze(2).expand_as(candidates),
            instruction_steps.unsqueeze(2).expand_as(candidates),
            relative_score.unsqueeze(-1),
        ), -1)).squeeze(-1)
        native_slots = torch.zeros_like(candidate_mask).scatter(
            -1, safe_native.unsqueeze(-1), True
        )
        alternative_mask = candidate_mask & ~native_slots & valid_native.unsqueeze(-1)
        alternative_logits = alternative_logits.masked_fill(
            ~alternative_mask, -torch.inf
        )
        return NativeConditionedUADOutput(
            native_error_logit=native_error_logit,
            alternative_logits=alternative_logits,
        )


def median_native_conditioned_outputs(
    outputs: tuple[NativeConditionedUADOutput, ...],
) -> NativeConditionedUADOutput:
    """Aggregate fixed training seeds into one deployable median decision."""

    if len(outputs) < 1:
        raise ValueError("native-conditioned ensemble cannot be empty")
    reference = outputs[0]
    if any(
        output.native_error_logit.shape != reference.native_error_logit.shape
        or output.alternative_logits.shape != reference.alternative_logits.shape
        for output in outputs[1:]
    ):
        raise ValueError("native-conditioned ensemble output shape drift")
    return NativeConditionedUADOutput(
        native_error_logit=torch.stack([
            output.native_error_logit for output in outputs
        ]).median(0).values,
        alternative_logits=torch.stack([
            output.alternative_logits for output in outputs
        ]).median(0).values,
    )


def native_conditioned_uad_loss(
    output: NativeConditionedUADOutput,
    batch: dict[str, Tensor],
    *,
    error_positive_weight: float,
) -> dict[str, Tensor]:
    """Supervise native-error detection and the better alternative."""

    if error_positive_weight <= 0:
        raise ValueError("error_positive_weight must be positive")
    valid = (
        batch["step_mask"]
        & (batch["native_index"] >= 0)
        & (batch["target_index"] >= 0)
        & (batch["candidate_mask"].sum(-1) >= 2)
    )
    if not valid.any():
        zero = output.native_error_logit.sum() * 0.0
        return {"total": zero, "native_error": zero, "alternative": zero}
    error_label = (
        batch["native_index"] != batch["target_index"]
    ).to(output.native_error_logit.dtype)
    native_error = F.binary_cross_entropy_with_logits(
        output.native_error_logit[valid], error_label[valid],
        pos_weight=output.native_error_logit.new_tensor(error_positive_weight),
    )
    wrong = valid & (error_label > 0.5)
    alternative = (
        F.cross_entropy(
            output.alternative_logits[wrong], batch["target_index"][wrong]
        )
        if wrong.any()
        else output.native_error_logit.sum() * 0.0
    )
    return {
        "total": native_error + alternative,
        "native_error": native_error,
        "alternative": alternative,
    }


def native_residual_logits(
    output: NativeConditionedUADOutput,
    native_scores: Tensor,
    candidate_mask: Tensor,
    *,
    correction_bound: float,
) -> tuple[Tensor, Tensor]:
    """Fuse bounded learned corrections with frozen current-local logits."""

    if native_scores.shape != output.alternative_logits.shape:
        raise ValueError("native and correction score shapes differ")
    if candidate_mask.shape != native_scores.shape:
        raise ValueError("candidate mask shape mismatch")
    if candidate_mask.dtype is not torch.bool:
        raise TypeError("candidate_mask must be boolean")
    if not isinstance(correction_bound, (int, float)) or correction_bound <= 0:
        raise ValueError("correction_bound must be positive")
    correction = torch.where(
        torch.isfinite(output.alternative_logits),
        torch.tanh(output.alternative_logits) * correction_bound,
        torch.zeros_like(output.alternative_logits),
    )
    fused = (native_scores + correction).masked_fill(~candidate_mask, -torch.inf)
    return fused, correction


def native_alternative_posterior_gain(
    output: NativeConditionedUADOutput,
    adapted_index: Tensor,
) -> Tensor:
    """Estimate one-step gain of an alternative over the native decision.

    The alternative distribution is conditional on the native action being
    wrong.  Therefore ``P(alt correct) - P(native correct)`` factorizes into
    ``p_error * p_alt_given_error - (1 - p_error)``.
    """

    if adapted_index.shape != output.native_error_logit.shape:
        raise ValueError("adapted index shape mismatch")
    if adapted_index.dtype is not torch.long:
        raise TypeError("adapted index must be torch.long")
    if output.alternative_logits.shape[:-1] != adapted_index.shape:
        raise ValueError("alternative logit axes mismatch")
    candidates = output.alternative_logits.shape[-1]
    if candidates < 1 or bool(
        ((adapted_index < 0) | (adapted_index >= candidates)).any()
    ):
        raise ValueError("adapted index outside candidate axis")
    finite = torch.isfinite(output.alternative_logits)
    has_alternative = finite.any(-1)
    safe_logits = torch.where(
        has_alternative.unsqueeze(-1),
        output.alternative_logits,
        torch.zeros_like(output.alternative_logits),
    )
    probabilities = torch.softmax(safe_logits, dim=-1) * finite.to(
        safe_logits.dtype
    )
    alternative_probability = probabilities.gather(
        -1, adapted_index.unsqueeze(-1)
    ).squeeze(-1)
    error_probability = torch.sigmoid(output.native_error_logit)
    return torch.where(
        has_alternative,
        error_probability * alternative_probability
        - (1.0 - error_probability),
        torch.full_like(error_probability, -1.0),
    )


def native_residual_uad_loss(
    output: NativeConditionedUADOutput,
    batch: dict[str, Tensor],
    *,
    correction_bound: float,
    error_weight: float = 0.25,
    regularization_weight: float = 0.01,
) -> dict[str, Tensor]:
    """Optimize a bounded correction while retaining the frozen native prior."""

    if error_weight < 0 or regularization_weight < 0:
        raise ValueError("residual loss weights must be non-negative")
    valid = (
        batch["step_mask"]
        & (batch["native_index"] >= 0)
        & (batch["target_index"] >= 0)
        & (batch["candidate_mask"].sum(-1) >= 2)
    )
    if not valid.any():
        zero = output.native_error_logit.sum() * 0.0
        return {
            "total": zero, "policy": zero,
            "native_error": zero, "regularization": zero,
        }
    fused, correction = native_residual_logits(
        output, batch["native_scores"], batch["candidate_mask"],
        correction_bound=correction_bound,
    )
    policy = F.cross_entropy(fused[valid], batch["target_index"][valid])
    error_label = (
        batch["native_index"] != batch["target_index"]
    ).to(output.native_error_logit.dtype)
    native_error = F.binary_cross_entropy_with_logits(
        output.native_error_logit[valid], error_label[valid]
    )
    valid_candidates = batch["candidate_mask"] & valid.unsqueeze(-1)
    regularization = correction[valid_candidates].square().mean()
    total = (
        policy + error_weight * native_error
        + regularization_weight * regularization
    )
    return {
        "total": total, "policy": policy,
        "native_error": native_error, "regularization": regularization,
    }


def factorized_uad_probabilities(
    target_in_set_logit: Tensor,
    separation_logit: Tensor,
    evidence_logit: Tensor,
) -> Tensor:
    """Return normalized ``[U, A, D]`` probabilities from three factors."""

    if not (
        target_in_set_logit.shape
        == separation_logit.shape
        == evidence_logit.shape
    ):
        raise ValueError("UAD factor shapes must match")
    target_in_set = torch.sigmoid(target_in_set_logit)
    closure = torch.sigmoid(separation_logit) * torch.sigmoid(evidence_logit)
    undecidable = 1.0 - target_in_set
    actionable = target_in_set * (1.0 - closure)
    decisive = target_in_set * closure
    return torch.stack((undecidable, actionable, decisive), dim=-1)


class StructuredUADHeads(nn.Module):
    """Prefix-causal UAD heads over frozen ETP history and local candidates."""

    candidate_count_encoding = "count_over_count_plus_one"

    def __init__(self, feature_dim: int = 768, hidden_dim: int = 128) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("model dimensions must be positive")
        self.feature_dim = feature_dim
        self.history_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.instruction_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.relevance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relational_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim * 3, hidden_dim, batch_first=True)
        self.expiry_temporal = nn.GRU(
            hidden_dim * 3, hidden_dim, batch_first=True
        )
        self.option_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.target_head = nn.Linear(hidden_dim, 1)
        self.readiness_heads = nn.Linear(hidden_dim, 4)
        self.expiry_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        history_embeddings: Tensor,
        candidate_embeddings: Tensor,
        candidate_mask: Tensor,
        instruction_embedding: Tensor,
    ) -> StructuredUADOutput:
        """Run UAD heads.

        Inputs are history ``[B,T,H]``, candidates ``[B,T,N,H]``, a boolean
        candidate mask ``[B,T,N]``, and instruction embeddings ``[B,H]``.
        """

        if history_embeddings.ndim != 3 or candidate_embeddings.ndim != 4:
            raise ValueError("history must be 3-D and candidates must be 4-D")
        batch, steps, feature_dim = history_embeddings.shape
        if feature_dim != self.feature_dim:
            raise ValueError("history feature dimension drift")
        if candidate_embeddings.shape[:2] != (batch, steps):
            raise ValueError("history and candidate axes do not match")
        if candidate_embeddings.shape[-1] != feature_dim:
            raise ValueError("history and candidate feature dimensions differ")
        if candidate_embeddings.shape[2] < 2:
            raise ValueError("UAD relational encoding requires at least two slots")
        if candidate_mask.shape != candidate_embeddings.shape[:-1]:
            raise ValueError("candidate_mask shape mismatch")
        if candidate_mask.dtype is not torch.bool:
            raise TypeError("candidate_mask must be boolean")
        if instruction_embedding.shape != (batch, feature_dim):
            raise ValueError("instruction_embedding shape mismatch")
        if not (
            torch.isfinite(history_embeddings).all()
            and torch.isfinite(candidate_embeddings).all()
            and torch.isfinite(instruction_embedding).all()
        ):
            raise ValueError("UAD inputs must be finite")

        history = self.history_projection(history_embeddings)
        candidates = self.candidate_projection(candidate_embeddings)
        instruction = self.instruction_projection(instruction_embedding)
        instruction_steps = instruction.unsqueeze(1).expand(batch, steps, -1)

        mask_values = candidate_mask.unsqueeze(-1).to(candidates.dtype)
        candidate_count = candidate_mask.sum(-1)
        mean_candidate = (candidates * mask_values).sum(2) / mask_values.sum(
            2
        ).clamp_min(1.0)
        query = self.relevance(instruction_steps).unsqueeze(2)
        relevance = (candidates * query).sum(-1) / math.sqrt(
            candidates.shape[-1]
        )
        masked_relevance = relevance.masked_fill(~candidate_mask, -torch.inf)
        attention = torch.softmax(
            masked_relevance.masked_fill(
                ~candidate_mask, torch.finfo(relevance.dtype).min
            ),
            dim=-1,
        ) * candidate_mask.to(relevance.dtype)
        attention /= attention.sum(-1, keepdim=True).clamp_min(1.0)
        attended_candidate = (candidates * attention.unsqueeze(-1)).sum(2)
        top_two = torch.topk(masked_relevance, k=2, dim=-1).values
        margin = torch.where(
            candidate_count >= 2,
            top_two[..., 0] - top_two[..., 1],
            torch.zeros_like(top_two[..., 0]),
        )
        count_feature = candidate_count.to(candidates.dtype) / (
            candidate_count.to(candidates.dtype) + 1.0
        )
        relational = self.relational_fusion(torch.cat((
            mean_candidate,
            attended_candidate,
            margin.unsqueeze(-1),
            count_feature.unsqueeze(-1),
        ), dim=-1))
        temporal_input = torch.cat(
            (history, relational, instruction_steps), dim=-1
        )
        temporal, _ = self.temporal(temporal_input)
        expiry_temporal, _ = self.expiry_temporal(temporal_input)

        options = self.option_fusion(torch.cat((
            candidates,
            temporal.unsqueeze(2).expand_as(candidates),
            instruction_steps.unsqueeze(2).expand_as(candidates),
        ), dim=-1))
        target_logits = self.target_head(options).squeeze(-1).masked_fill(
            ~candidate_mask, -torch.inf
        )
        readiness = self.readiness_heads(temporal)
        uad = factorized_uad_probabilities(
            readiness[..., 0], readiness[..., 1], readiness[..., 2]
        )
        return StructuredUADOutput(
            target_logits=target_logits,
            target_in_set_logit=readiness[..., 0],
            separation_logit=readiness[..., 1],
            evidence_logit=readiness[..., 2],
            reveal_hazard_logit=readiness[..., 3],
            expiry_hazard_logit=self.expiry_head(expiry_temporal).squeeze(-1),
            uad_probabilities=uad,
        )


@dataclass(frozen=True)
class StructuredUADLossConfig:
    target_weight: float = 1.0
    factor_weight: float = 1.0
    uad_weight: float = 1.0
    reveal_weight: float = 1.0
    expiry_weight: float = 1.0
    class_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    positive_weights: tuple[float, float, float, float, float] = (
        1.0, 1.0, 1.0, 1.0, 1.0
    )


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    selected = value.masked_select(mask)
    return selected.mean() if selected.numel() else value.sum() * 0.0


class StructuredUADLoss(nn.Module):
    """UAD-only objective with no topology, cost, feasibility, or Q target."""

    def __init__(
        self, config: StructuredUADLossConfig = StructuredUADLossConfig()
    ) -> None:
        super().__init__()
        if len(config.class_weights) != 3 or len(config.positive_weights) != 5:
            raise ValueError("UAD class/positive weight counts must be 3/5")
        weights = (*config.class_weights, *config.positive_weights)
        if any(not isinstance(value, (int, float)) or value <= 0 for value in weights):
            raise ValueError("UAD loss weights must be positive")
        self.config = config

    def forward(
        self, output: StructuredUADOutput, batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        candidate_mask = batch["candidate_mask"]
        target_index = batch["target_index"]
        if candidate_mask.dtype is not torch.bool:
            raise TypeError("candidate_mask must be boolean")
        if target_index.dtype is not torch.long:
            raise TypeError("target_index must be torch.long")
        if candidate_mask.shape != output.target_logits.shape:
            raise ValueError("candidate mask and UAD target logits differ")
        if target_index.shape != output.target_in_set_logit.shape:
            raise ValueError("target index and UAD time axes differ")
        valid_target = target_index >= 0
        safe_target_index = target_index.clamp(
            min=0, max=candidate_mask.shape[-1] - 1
        )
        if torch.any(
            valid_target
            & (
                (target_index >= candidate_mask.shape[-1])
                | ~candidate_mask.gather(
                    -1, safe_target_index.unsqueeze(-1)
                ).squeeze(-1)
            )
        ):
            raise ValueError("target index is not a valid candidate")
        target = (
            F.cross_entropy(
                output.target_logits[valid_target], target_index[valid_target]
            )
            if valid_target.any()
            else output.target_logits.masked_fill(~candidate_mask, 0.0).sum() * 0.0
        )

        factor_losses = []
        factor_labels = []
        for prediction, key, positive_weight in (
            (output.target_in_set_logit, "target_in_set", self.config.positive_weights[0]),
            (output.separation_logit, "separation", self.config.positive_weights[1]),
            (output.evidence_logit, "evidence_complete", self.config.positive_weights[2]),
        ):
            label = batch[key]
            if label.shape != prediction.shape:
                raise ValueError(f"{key} shape drift")
            mask = label >= 0
            raw = F.binary_cross_entropy_with_logits(
                prediction,
                label.clamp_min(0),
                reduction="none",
                pos_weight=prediction.new_tensor(positive_weight),
            )
            factor_losses.append(_masked_mean(raw, mask))
            factor_labels.append(label)
        factor = torch.stack(factor_losses).mean()

        target_in_set, separation, evidence = factor_labels
        valid_uad = (
            (target_in_set >= 0) & (separation >= 0) & (evidence >= 0)
        )
        uad_label = torch.where(
            target_in_set < 0.5,
            torch.zeros_like(target_in_set, dtype=torch.long),
            torch.where(
                (separation < 0.5) | (evidence < 0.5),
                torch.ones_like(target_in_set, dtype=torch.long),
                torch.full_like(target_in_set, 2, dtype=torch.long),
            ),
        )
        uad_rows = -output.uad_probabilities.clamp_min(
            torch.finfo(output.uad_probabilities.dtype).tiny
        ).log().gather(-1, uad_label.unsqueeze(-1)).squeeze(-1)
        class_weight = output.uad_probabilities.new_tensor(
            self.config.class_weights
        )[uad_label]
        selected_weight = class_weight * valid_uad.to(class_weight.dtype)
        uad = (uad_rows * selected_weight).sum() / selected_weight.sum().clamp_min(
            1.0
        )

        hazards = []
        for prediction, key, positive_weight in (
            (output.reveal_hazard_logit, "reveal_hazard", self.config.positive_weights[3]),
            (output.expiry_hazard_logit, "expiry_hazard", self.config.positive_weights[4]),
        ):
            label = batch[key]
            if label.shape != prediction.shape:
                raise ValueError(f"{key} shape drift")
            mask = label >= 0
            raw = F.binary_cross_entropy_with_logits(
                prediction,
                label.clamp_min(0),
                reduction="none",
                pos_weight=prediction.new_tensor(positive_weight),
            )
            hazards.append(_masked_mean(raw, mask))
        reveal, expiry = hazards
        total = (
            self.config.target_weight * target
            + self.config.factor_weight * factor
            + self.config.uad_weight * uad
            + self.config.reveal_weight * reveal
            + self.config.expiry_weight * expiry
        )
        return {
            "total": total,
            "target": target,
            "factors": factor,
            "uad": uad,
            "reveal": reveal,
            "expiry": expiry,
        }


def fuse_current_candidate_logits(
    native_logits: Tensor,
    current_candidate_indices: Tensor,
    target_scores: Tensor,
    current_candidate_mask: Tensor,
    decisive_probability: Tensor,
    *,
    alpha: float,
    decisive_threshold: float,
    margin_threshold: float,
) -> ResidualFusionOutput:
    """Apply a centered UAD residual to current local ETP candidates.

    Malformed tensor schemas raise.  Invalid or insufficient evidence in an
    individual row delegates that row bit-exactly to the native policy.
    """

    if native_logits.ndim != 2 or current_candidate_indices.ndim != 2:
        raise ValueError("native logits and current candidate indices must be 2-D")
    if not (
        current_candidate_indices.shape
        == target_scores.shape
        == current_candidate_mask.shape
    ):
        raise ValueError("current candidate tensors must have matching shapes")
    if native_logits.shape[0] != current_candidate_indices.shape[0]:
        raise ValueError("native and current candidate batch axes differ")
    if decisive_probability.shape != native_logits.shape[:1]:
        raise ValueError("decisive probability shape mismatch")
    if current_candidate_indices.dtype is not torch.long:
        raise TypeError("current candidate indices must be torch.long")
    if current_candidate_mask.dtype is not torch.bool:
        raise TypeError("current candidate mask must be boolean")
    if not (
        native_logits.is_floating_point()
        and target_scores.is_floating_point()
        and decisive_probability.is_floating_point()
    ):
        raise TypeError("UAD policy logits, scores, and probability must be floating point")
    devices = {
        value.device for value in (
            native_logits, current_candidate_indices, target_scores,
            current_candidate_mask, decisive_probability,
        )
    }
    if len(devices) != 1:
        raise ValueError("UAD adapter tensors must share one device")
    if not (
        isinstance(alpha, (int, float)) and alpha >= 0
        and isinstance(decisive_threshold, (int, float))
        and 0 <= decisive_threshold <= 1
        and isinstance(margin_threshold, (int, float)) and margin_threshold >= 0
    ):
        raise ValueError("invalid UAD adapter hyperparameters")

    fused = native_logits.clone()
    authorized = torch.zeros(
        native_logits.shape[0], dtype=torch.bool, device=native_logits.device
    )
    action_count = native_logits.shape[1]
    for row in range(native_logits.shape[0]):
        mask = current_candidate_mask[row]
        if int(mask.sum()) < 2:
            continue
        indices = current_candidate_indices[row, mask]
        scores = target_scores[row, mask]
        native_action = int(torch.argmax(native_logits[row]))
        if not (
            torch.isfinite(scores).all()
            and torch.isfinite(decisive_probability[row])
            and 0.0 <= float(decisive_probability[row]) <= 1.0
            and float(decisive_probability[row]) >= decisive_threshold
            and torch.all(indices >= 1)
            and torch.all(indices < action_count)
            and torch.unique(indices).numel() == indices.numel()
            and torch.isfinite(native_logits[row, indices]).all()
            and torch.any(indices == native_action)
        ):
            continue
        top_two = torch.topk(scores, k=2).values
        if float(top_two[0] - top_two[1]) < margin_threshold:
            continue
        centered = scores - scores.mean()
        fused[row, indices] = native_logits[row, indices] + alpha * centered
        authorized[row] = True
    return ResidualFusionOutput(fused, authorized)
