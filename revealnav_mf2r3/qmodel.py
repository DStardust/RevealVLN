"""Candidate-conditioned causal temporal paired-Q adapter for OPP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PairedQOutput:
    q_with_checkpoint: Tensor
    q_without_checkpoint: Tensor

    @property
    def opv_per_candidate(self) -> Tensor:
        return self.q_without_checkpoint - self.q_with_checkpoint


class CausalPairedQAdapter(nn.Module):
    """Small online-safe Q head over history, option, instruction and age."""

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
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.q_with_head = nn.Linear(hidden_dim, 1)
        self.q_delta_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.q_delta_head.bias, 0.1)

    def forward(
        self, history_embeddings: Tensor, candidate_embeddings: Tensor,
        candidate_mask: Tensor, instruction_embedding: Tensor,
    ) -> PairedQOutput:
        if history_embeddings.ndim != 3 or candidate_embeddings.ndim != 4:
            raise ValueError("history and candidates must be 3-D and 4-D")
        batch, steps, feature_dim = history_embeddings.shape
        if candidate_embeddings.shape[:2] != (batch, steps):
            raise ValueError("history/candidate axes mismatch")
        if candidate_embeddings.shape[-1] != feature_dim:
            raise ValueError("feature dimensions mismatch")
        if candidate_mask.shape != candidate_embeddings.shape[:-1]:
            raise ValueError("candidate mask shape mismatch")
        if instruction_embedding.shape != (batch, feature_dim):
            raise ValueError("instruction embedding shape mismatch")
        candidates = candidate_embeddings.shape[2]
        history = F.gelu(self.history_projection(history_embeddings))
        option = F.gelu(self.candidate_projection(candidate_embeddings))
        instruction = F.gelu(
            self.instruction_projection(instruction_embedding)
        ).view(batch, 1, 1, self.hidden_dim).expand_as(option)
        history = history.unsqueeze(2).expand_as(option)
        age = torch.arange(
            steps, device=history.device, dtype=history.dtype
        ).view(1, steps, 1, 1).expand(batch, steps, candidates, 1)
        age = self.age_projection(age / self.age_denominator)
        fused = self.fusion(torch.cat((history, option, instruction, age), -1))
        causal = fused.permute(0, 2, 1, 3).reshape(
            batch * candidates, steps, self.hidden_dim
        )
        causal, _ = self.temporal(causal)
        causal = causal.reshape(
            batch, candidates, steps, self.hidden_dim
        ).permute(0, 2, 1, 3)
        q_with_raw = F.softplus(self.q_with_head(causal).squeeze(-1))
        # ReLU is required because true checkpoint value has an atom at zero.
        delta = F.relu(self.q_delta_head(causal).squeeze(-1))
        q_with = q_with_raw.masked_fill(~candidate_mask, torch.inf)
        q_without = (q_with_raw + delta).masked_fill(
            ~candidate_mask, torch.inf
        )
        return PairedQOutput(q_with, q_without)
