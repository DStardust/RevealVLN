"""Instruction-conditioned relational candidate encoder for REE events."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from revealnav_mf2 import RevealOptionOutput


class RelationalRevealOptionHeads(nn.Module):
    """REE heads whose temporal state retains candidate competition evidence."""

    def __init__(
        self, feature_dim: int = 768, hidden_dim: int = 128,
        budget_count: int = 4,
        candidate_count_encoding: str = "batch_fraction",
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or budget_count < 1:
            raise ValueError("model dimensions must be positive")
        if candidate_count_encoding not in {"batch_fraction", "saturating"}:
            raise ValueError("unsupported candidate_count_encoding")
        self.budget_count = budget_count
        self.candidate_count_encoding = candidate_count_encoding
        self.history_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.instruction_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.relevance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relational_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim * 3, hidden_dim, batch_first=True)
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

    def candidate_count_feature(
        self, count: Tensor, candidate_slots: int,
    ) -> Tensor:
        value = count.to(self.event_heads.weight.dtype)
        if self.candidate_count_encoding == "batch_fraction":
            return value / candidate_slots
        return value / (value + 1.0)

    def forward(
        self,
        history_embeddings: Tensor,
        candidate_embeddings: Tensor,
        candidate_mask: Tensor,
        normalized_budgets: Tensor,
        instruction_embedding: Tensor,
    ) -> RevealOptionOutput:
        if history_embeddings.ndim != 3 or candidate_embeddings.ndim != 4:
            raise ValueError("history must be 3-D and candidates must be 4-D")
        batch, steps, feature_dim = history_embeddings.shape
        if candidate_embeddings.shape[:2] != (batch, steps):
            raise ValueError("history and candidate axes do not match")
        if candidate_embeddings.shape[-1] != feature_dim:
            raise ValueError("history and candidate feature dimensions differ")
        if candidate_mask.shape != candidate_embeddings.shape[:-1]:
            raise ValueError("candidate_mask shape mismatch")
        if candidate_mask.dtype is not torch.bool:
            raise TypeError("candidate_mask must be boolean")
        if candidate_embeddings.shape[2] < 2:
            raise ValueError("relational encoding requires at least two slots")
        if normalized_budgets.shape != (batch, steps, self.budget_count):
            raise ValueError("normalized_budgets shape mismatch")
        if instruction_embedding.shape != (batch, feature_dim):
            raise ValueError("instruction_embedding shape mismatch")

        history = self.history_projection(history_embeddings)
        candidates = self.candidate_projection(candidate_embeddings)
        instruction = self.instruction_projection(instruction_embedding)
        instruction_steps = instruction.unsqueeze(1).expand(batch, steps, -1)
        mask_values = candidate_mask.unsqueeze(-1).to(candidates.dtype)
        mean_candidate = (candidates * mask_values).sum(2) / mask_values.sum(
            2
        ).clamp_min(1.0)

        query = self.relevance(instruction_steps).unsqueeze(2)
        scores = (candidates * query).sum(-1) / math.sqrt(candidates.shape[-1])
        masked_scores = scores.masked_fill(~candidate_mask, -1e4)
        attention = torch.softmax(masked_scores, dim=-1) * candidate_mask.to(
            scores.dtype
        )
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1.0)
        attended_candidate = (candidates * attention.unsqueeze(-1)).sum(2)
        top_two = torch.topk(masked_scores, k=2, dim=-1).values
        count = candidate_mask.sum(-1)
        margin = torch.where(
            count >= 2, top_two[..., 0] - top_two[..., 1],
            torch.zeros_like(top_two[..., 0]),
        )
        normalized_count = self.candidate_count_feature(
            count, candidate_mask.shape[-1]
        ).to(candidates.dtype)
        relational = self.relational_fusion(torch.cat((
            mean_candidate,
            attended_candidate,
            margin.unsqueeze(-1),
            normalized_count.unsqueeze(-1),
        ), dim=-1))
        temporal, _ = self.temporal(torch.cat((
            history, relational, instruction_steps
        ), dim=-1))

        expanded_temporal = temporal.unsqueeze(2).expand_as(candidates)
        expanded_instruction = instruction_steps.unsqueeze(2).expand_as(candidates)
        options = self.option_fusion(torch.cat((
            candidates, expanded_temporal, expanded_instruction
        ), dim=-1))
        target_logits = self.target_head(options).squeeze(-1).masked_fill(
            ~candidate_mask, -torch.inf
        )
        option_cost = F.softplus(self.cost_head(options).squeeze(-1)).masked_fill(
            ~candidate_mask, torch.inf
        )
        candidate_budget = options.unsqueeze(3).expand(
            batch, steps, options.shape[2], self.budget_count, options.shape[-1]
        )
        budget_values = normalized_budgets.unsqueeze(2).unsqueeze(-1).expand(
            batch, steps, options.shape[2], self.budget_count, 1
        )
        feasibility = self.feasibility_head(torch.cat((
            candidate_budget, budget_values
        ), dim=-1)).squeeze(-1).masked_fill(
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
