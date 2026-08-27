"""Causal candidate-conditioned costs for commit and checkpointed excursion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BranchExcursionQOutput:
    commit_cost: Tensor
    excursion_cost: Tensor


class BranchExcursionQHead(nn.Module):
    def __init__(
        self, feature_dim: int = 768, hidden_dim: int = 96,
        age_denominator: float = 128.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or age_denominator <= 0:
            raise ValueError("hidden_dim and age_denominator must be positive")
        self.hidden_dim = hidden_dim
        self.age_denominator = float(age_denominator)
        self.history_projection = nn.Linear(feature_dim, hidden_dim)
        self.candidate_projection = nn.Linear(feature_dim, hidden_dim)
        self.instruction_projection = nn.Linear(feature_dim, hidden_dim)
        self.age_projection = nn.Linear(1, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.commit_head = nn.Linear(hidden_dim, 1)
        self.excursion_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, history_embeddings: Tensor, candidate_embeddings: Tensor,
        candidate_mask: Tensor, instruction_embedding: Tensor,
        decision_index: Tensor,
    ) -> BranchExcursionQOutput:
        if history_embeddings.ndim != 3 or candidate_embeddings.ndim != 4:
            raise ValueError("history/candidate inputs must be 3-D and 4-D")
        batch, steps, feature_dim = history_embeddings.shape
        if candidate_embeddings.shape[:2] != (batch, steps):
            raise ValueError("history/candidate axes mismatch")
        if candidate_embeddings.shape[-1] != feature_dim:
            raise ValueError("feature dimensions mismatch")
        if candidate_mask.shape != candidate_embeddings.shape[:-1]:
            raise ValueError("candidate mask shape mismatch")
        if instruction_embedding.shape != (batch, feature_dim):
            raise ValueError("instruction embedding shape mismatch")
        if decision_index.shape != (batch,):
            raise ValueError("decision index shape mismatch")
        if bool(((decision_index < 0) | (decision_index >= steps)).any()):
            raise ValueError("decision index outside sequence")
        candidates = candidate_embeddings.shape[2]
        history = F.gelu(self.history_projection(history_embeddings))
        options = F.gelu(self.candidate_projection(candidate_embeddings))
        mask_values = candidate_mask.unsqueeze(-1).to(options.dtype)
        candidate_set = (options * mask_values).sum(2) / mask_values.sum(
            2
        ).clamp_min(1.0)
        candidate_set = candidate_set.unsqueeze(2).expand_as(options)
        instruction = F.gelu(
            self.instruction_projection(instruction_embedding)
        ).view(batch, 1, 1, self.hidden_dim).expand_as(options)
        history = history.unsqueeze(2).expand_as(options)
        age = torch.arange(
            steps, device=history.device, dtype=history.dtype
        ).view(1, steps, 1, 1).expand(batch, steps, candidates, 1)
        age = self.age_projection(age / self.age_denominator)
        fused = self.fusion(torch.cat((
            history, options, candidate_set, instruction, age,
        ), -1))
        causal = fused.permute(0, 2, 1, 3).reshape(
            batch * candidates, steps, self.hidden_dim
        )
        causal, _ = self.temporal(causal)
        causal = causal.reshape(batch, candidates, steps, self.hidden_dim)
        index = decision_index.view(batch, 1, 1, 1).expand(
            batch, candidates, 1, self.hidden_dim
        )
        decision = causal.gather(2, index).squeeze(2)
        decision_mask = candidate_mask[
            torch.arange(batch, device=candidate_mask.device), decision_index
        ]
        commit = F.softplus(self.commit_head(decision).squeeze(-1)).masked_fill(
            ~decision_mask, torch.inf
        )
        excursion = F.softplus(
            self.excursion_head(decision).squeeze(-1)
        ).masked_fill(~decision_mask, torch.inf)
        return BranchExcursionQOutput(commit, excursion)
