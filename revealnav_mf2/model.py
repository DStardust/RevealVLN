"""Small causal heads over frozen ETP-R1 history and candidate embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RevealOptionOutput:
    target_logits: Tensor
    option_cost: Tensor
    current_feasibility_logits: Tensor
    target_in_set_logit: Tensor
    separation_logit: Tensor
    evidence_logit: Tensor
    reveal_hazard_logit: Tensor
    checkpoint_value: Tensor


def select_topk_options(
    predicted_loss: Tensor,
    candidate_mask: Tensor,
    exhausted_mask: Tensor | None = None,
    width: int = 2,
) -> tuple[Tensor, Tensor]:
    """Select the lowest-loss viable options, returning -1 for empty slots."""

    if width < 1:
        raise ValueError("width must be positive")
    if predicted_loss.shape != candidate_mask.shape:
        raise ValueError("predicted_loss and candidate_mask shapes must match")
    if candidate_mask.dtype is not torch.bool:
        raise TypeError("candidate_mask must be boolean")
    viable = candidate_mask
    if exhausted_mask is not None:
        if exhausted_mask.shape != candidate_mask.shape:
            raise ValueError("exhausted_mask shape must match candidate_mask")
        if exhausted_mask.dtype is not torch.bool:
            raise TypeError("exhausted_mask must be boolean")
        viable = viable & ~exhausted_mask
    if predicted_loss.shape[-1] < width:
        raise ValueError("candidate dimension must be at least width")
    ranked = predicted_loss.masked_fill(~viable, torch.inf)
    values, indices = torch.topk(
        ranked, k=width, dim=-1, largest=False, sorted=True
    )
    valid = torch.isfinite(values)
    return indices.masked_fill(~valid, -1), valid


class RevealOptionHeads(nn.Module):
    """Causal REE/OPP heads; the ETP-R1 feature producer remains frozen."""

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        budget_count: int = 4,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or budget_count < 1:
            raise ValueError("model dimensions must be positive")
        self.budget_count = budget_count
        self.history_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.instruction_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.temporal = nn.GRU(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.option_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.target_head = nn.Linear(hidden_dim, 1)
        self.cost_head = nn.Linear(hidden_dim, 1)
        self.feasibility_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.event_heads = nn.Linear(hidden_dim, 5)

    def forward(
        self,
        history_embeddings: Tensor,
        candidate_embeddings: Tensor,
        candidate_mask: Tensor,
        normalized_budgets: Tensor,
        instruction_embedding: Tensor,
    ) -> RevealOptionOutput:
        """Run causal heads.

        Shapes are history ``[B,T,H]``, candidates ``[B,T,N,H]``, boolean
        mask ``[B,T,N]``, budgets ``[B,T,K]``, and one frozen instruction
        embedding per episode ``[B,H]``.
        """

        if history_embeddings.ndim != 3 or candidate_embeddings.ndim != 4:
            raise ValueError("history must be 3-D and candidates must be 4-D")
        batch, steps, feature_dim = history_embeddings.shape
        if candidate_embeddings.shape[:2] != (batch, steps):
            raise ValueError("history and candidate batch/time axes must match")
        if candidate_embeddings.shape[-1] != feature_dim:
            raise ValueError("history and candidate feature dimensions must match")
        if candidate_mask.shape != candidate_embeddings.shape[:-1]:
            raise ValueError("candidate_mask shape mismatch")
        if candidate_mask.dtype is not torch.bool:
            raise TypeError("candidate_mask must be boolean")
        if normalized_budgets.shape != (batch, steps, self.budget_count):
            raise ValueError("normalized_budgets shape mismatch")
        if instruction_embedding.shape != (batch, feature_dim):
            raise ValueError("instruction_embedding shape mismatch")

        history = self.history_projection(history_embeddings)
        candidates = self.candidate_projection(candidate_embeddings)
        instruction = self.instruction_projection(instruction_embedding)
        instruction_steps = instruction.unsqueeze(1).expand(
            batch, steps, instruction.shape[-1]
        )
        weights = candidate_mask.unsqueeze(-1).to(candidates.dtype)
        pooled = (candidates * weights).sum(dim=2) / weights.sum(
            dim=2
        ).clamp_min(1.0)
        temporal, _ = self.temporal(torch.cat(
            (history, pooled, instruction_steps), dim=-1
        ))
        expanded_temporal = temporal.unsqueeze(2).expand_as(candidates)
        expanded_instruction = instruction_steps.unsqueeze(2).expand_as(candidates)
        options = self.option_fusion(torch.cat(
            (candidates, expanded_temporal, expanded_instruction), dim=-1
        ))

        target_logits = self.target_head(options).squeeze(-1)
        option_cost = F.softplus(self.cost_head(options).squeeze(-1))
        target_logits = target_logits.masked_fill(~candidate_mask, -torch.inf)
        option_cost = option_cost.masked_fill(~candidate_mask, torch.inf)

        candidate_budget = options.unsqueeze(3).expand(
            batch, steps, options.shape[2], self.budget_count, options.shape[-1]
        )
        budget_values = normalized_budgets.unsqueeze(2).unsqueeze(-1).expand(
            batch, steps, options.shape[2], self.budget_count, 1
        )
        feasibility = self.feasibility_head(torch.cat(
            (candidate_budget, budget_values), dim=-1
        )).squeeze(-1)
        feasibility = feasibility.masked_fill(
            ~candidate_mask.unsqueeze(-1), -torch.inf
        )

        event = self.event_heads(temporal)
        return RevealOptionOutput(
            target_logits=target_logits,
            option_cost=option_cost,
            current_feasibility_logits=feasibility,
            target_in_set_logit=event[..., 0],
            separation_logit=event[..., 1],
            evidence_logit=event[..., 2],
            reveal_hazard_logit=event[..., 3],
            checkpoint_value=event[..., 4],
        )
