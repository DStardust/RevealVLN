"""Relational REE with independent reveal and expiry hazards."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from revealnav_mf2 import RevealOptionOutput
from revealnav_mf2r2 import RelationalRevealOptionHeads


@dataclass(frozen=True)
class RevealExpiryOptionOutput(RevealOptionOutput):
    expiry_hazard_logit: Tensor
    option_cost_without_checkpoint: Tensor


class RelationalRevealExpiryHeads(RelationalRevealOptionHeads):
    """R2 relational model with additive Q-delta and expiry adapters.

    The inherited decision path is intentionally shape-identical to R2.  This
    lets R3.1 load and freeze a successful R2 decision model while learning an
    independent causal expiry process, avoiding multi-task overwrite by the
    much longer post-decision sequences.
    """

    def __init__(
        self, feature_dim: int = 768, hidden_dim: int = 128,
        budget_count: int = 4,
        candidate_count_encoding: str = "batch_fraction",
    ) -> None:
        super().__init__(
            feature_dim, hidden_dim, budget_count, candidate_count_encoding
        )
        self.no_checkpoint_delta_head = nn.Linear(hidden_dim, 1)
        self.expiry_temporal = nn.GRU(hidden_dim * 3, hidden_dim, batch_first=True)
        self.expiry_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        history_embeddings: Tensor,
        candidate_embeddings: Tensor,
        candidate_mask: Tensor,
        normalized_budgets: Tensor,
        instruction_embedding: Tensor,
    ) -> RevealExpiryOptionOutput:
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
            count >= 2,
            top_two[..., 0] - top_two[..., 1],
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
        temporal_input = torch.cat((history, relational, instruction_steps), dim=-1)
        temporal, _ = self.temporal(temporal_input)
        expiry_temporal, _ = self.expiry_temporal(temporal_input)
        expanded_temporal = temporal.unsqueeze(2).expand_as(candidates)
        expanded_instruction = instruction_steps.unsqueeze(2).expand_as(candidates)
        options = self.option_fusion(torch.cat((
            candidates, expanded_temporal, expanded_instruction
        ), dim=-1))
        target_logits = self.target_head(options).squeeze(-1).masked_fill(
            ~candidate_mask, -torch.inf
        )
        option_cost_raw = F.softplus(self.cost_head(options).squeeze(-1))
        option_delta = F.softplus(
            self.no_checkpoint_delta_head(options).squeeze(-1)
        )
        option_cost = option_cost_raw.masked_fill(~candidate_mask, torch.inf)
        option_cost_without_checkpoint = (
            option_cost_raw + option_delta
        ).masked_fill(~candidate_mask, torch.inf)
        masked_delta = option_delta.masked_fill(~candidate_mask, -torch.inf)
        checkpoint_value = torch.where(
            candidate_mask.any(-1), masked_delta.max(-1).values,
            torch.zeros_like(masked_delta[..., 0]),
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
        return RevealExpiryOptionOutput(
            target_logits=target_logits,
            option_cost=option_cost,
            current_feasibility_logits=feasibility,
            target_in_set_logit=event[..., 0],
            separation_logit=event[..., 1],
            evidence_logit=event[..., 2],
            reveal_hazard_logit=event[..., 3],
            checkpoint_value=checkpoint_value,
            expiry_hazard_logit=self.expiry_head(expiry_temporal).squeeze(-1),
            option_cost_without_checkpoint=option_cost_without_checkpoint,
        )
