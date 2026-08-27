"""Regression and within-event action ranking for branch-excursion Q."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


class BranchExcursionQLoss(torch.nn.Module):
    def __init__(self, ranking_weight: float = 0.25, margin: float = 0.1):
        super().__init__()
        self.ranking_weight = float(ranking_weight)
        self.margin = float(margin)

    def forward(self, output, batch):
        valid = torch.isfinite(batch["commit_cost"]) & torch.isfinite(
            batch["excursion_cost"]
        )
        predicted_commit = output.commit_cost.masked_fill(~valid, 0.0)
        predicted_excursion = output.excursion_cost.masked_fill(~valid, 0.0)
        target_commit = batch["commit_cost"].masked_fill(~valid, 0.0)
        target_excursion = batch["excursion_cost"].masked_fill(~valid, 0.0)
        commit = masked_mean(F.smooth_l1_loss(
            predicted_commit, target_commit, reduction="none"
        ), valid)
        excursion = masked_mean(F.smooth_l1_loss(
            predicted_excursion, target_excursion, reduction="none"
        ), valid)
        gap = masked_mean(F.smooth_l1_loss(
            predicted_commit - predicted_excursion,
            target_commit - target_excursion,
            reduction="none",
        ), valid)
        teacher = torch.cat((
            batch["commit_cost"], batch["excursion_cost"]
        ), dim=-1)
        prediction = torch.cat((output.commit_cost, output.excursion_cost), dim=-1)
        action_valid = torch.cat((valid, valid), dim=-1)
        teacher = teacher.masked_fill(~action_valid, torch.inf)
        best_teacher, best_index = teacher.min(-1)
        best_prediction = prediction.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
        worse = action_valid & (
            teacher > best_teacher.unsqueeze(-1) + 1e-6
        )
        ranking = masked_mean(F.relu(
            self.margin - (prediction - best_prediction.unsqueeze(-1))
        ).masked_fill(~action_valid, 0.0), worse)
        total = commit + excursion + gap + self.ranking_weight * ranking
        return {
            "commit": commit, "excursion": excursion, "gap": gap,
            "ranking": ranking, "total": total,
        }
