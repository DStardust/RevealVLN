"""State-conditioned action costs after entering a checkpointed branch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PostExcursionQOutput:
    continue_cost: Tensor
    backtrack_cost: Tensor


class PostExcursionQHead(nn.Module):
    """Small causal head over frozen ETP and ECOG state tokens."""

    def __init__(
        self, feature_dim: int = 768, hidden_dim: int = 96,
        elapsed_denominator: float = 5.0,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or elapsed_denominator <= 0:
            raise ValueError("model dimensions and denominator must be positive")
        self.elapsed_denominator = float(elapsed_denominator)
        self.history_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.instruction_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.selected_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.checkpoint_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.local_candidate_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.elapsed_projection = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.continue_head = nn.Linear(hidden_dim, 1)
        self.backtrack_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, history_embeddings: Tensor, history_lengths: Tensor,
        instruction_embedding: Tensor, selected_branch_embedding: Tensor,
        checkpoint_embedding: Tensor, post_candidate_embedding: Tensor,
        normalized_excursion_elapsed: Tensor,
    ) -> PostExcursionQOutput:
        if history_embeddings.ndim != 3:
            raise ValueError("history embeddings must be 3-D")
        batch, steps, feature_dim = history_embeddings.shape
        expected = (batch, feature_dim)
        if any(value.shape != expected for value in (
            instruction_embedding, selected_branch_embedding,
            checkpoint_embedding, post_candidate_embedding,
        )):
            raise ValueError("state token shapes must match history features")
        if history_lengths.shape != (batch,):
            raise ValueError("history lengths must have one value per example")
        if normalized_excursion_elapsed.shape != (batch,):
            raise ValueError("elapsed input must have one value per example")
        if bool(((history_lengths < 1) | (history_lengths > steps)).any()):
            raise ValueError("history length outside padded sequence")
        if not bool(torch.isfinite(normalized_excursion_elapsed).all()):
            raise ValueError("elapsed input must be finite")
        history, _ = self.temporal(self.history_projection(history_embeddings))
        indices = history_lengths.to(history.device) - 1
        final = history[torch.arange(batch, device=history.device), indices]
        elapsed = self.elapsed_projection(
            (normalized_excursion_elapsed / self.elapsed_denominator).unsqueeze(-1)
        )
        fused = self.fusion(torch.cat((
            final,
            self.instruction_projection(instruction_embedding),
            self.selected_projection(selected_branch_embedding),
            self.checkpoint_projection(checkpoint_embedding),
            self.local_candidate_projection(post_candidate_embedding),
            elapsed,
        ), -1))
        return PostExcursionQOutput(
            F.softplus(self.continue_head(fused).squeeze(-1)),
            F.softplus(self.backtrack_head(fused).squeeze(-1)),
        )


class PostExcursionQLoss(nn.Module):
    def __init__(self, ranking_weight: float = 0.25, margin: float = 0.1) -> None:
        super().__init__()
        if ranking_weight < 0 or margin < 0:
            raise ValueError("loss weights must be non-negative")
        self.ranking_weight = float(ranking_weight)
        self.margin = float(margin)

    def forward(self, output: PostExcursionQOutput, batch: dict[str, Tensor]):
        target_continue = batch["continue_cost"]
        target_backtrack = batch["backtrack_cost"]
        if not bool(torch.isfinite(target_continue).all()) or not bool(
            torch.isfinite(target_backtrack).all()
        ):
            raise ValueError("post-excursion costs must be finite")
        continue_loss = F.smooth_l1_loss(output.continue_cost, target_continue)
        backtrack_loss = F.smooth_l1_loss(output.backtrack_cost, target_backtrack)
        target_gap = target_backtrack - target_continue
        predicted_gap = output.backtrack_cost - output.continue_cost
        gap_loss = F.smooth_l1_loss(predicted_gap, target_gap)
        strict = target_gap.abs() > 1e-6
        if bool(strict.any()):
            direction = target_gap[strict].sign()
            ranking = F.relu(
                self.margin - direction * predicted_gap[strict]
            ).mean()
        else:
            ranking = predicted_gap.sum() * 0.0
        total = (
            continue_loss + backtrack_loss + gap_loss
            + self.ranking_weight * ranking
        )
        return {
            "continue": continue_loss, "backtrack": backtrack_loss,
            "gap": gap_loss, "ranking": ranking, "total": total,
        }
