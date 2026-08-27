"""Tie-aware listwise stabilization for branch-excursion action costs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import BranchExcursionQLoss


class StableBranchExcursionQLoss(torch.nn.Module):
    """Keep V4 cost regression and directly separate optimal action sets."""

    def __init__(self, listwise_weight: float = 1.0) -> None:
        super().__init__()
        if listwise_weight <= 0.0:
            raise ValueError("listwise_weight must be positive")
        self.regression = BranchExcursionQLoss(0.25, 0.1)
        self.listwise_weight = float(listwise_weight)

    def forward(self, output, batch):
        losses = self.regression(output, batch)
        valid = torch.isfinite(batch["commit_cost"]) & torch.isfinite(
            batch["excursion_cost"]
        )
        teacher = torch.cat((
            batch["commit_cost"], batch["excursion_cost"],
        ), dim=-1)
        prediction = torch.cat((
            output.commit_cost, output.excursion_cost,
        ), dim=-1)
        action_valid = torch.cat((valid, valid), dim=-1)
        teacher = teacher.masked_fill(~action_valid, torch.inf)
        best = teacher.min(-1, keepdim=True).values
        optimal = action_valid & torch.isclose(
            teacher, best, atol=1e-6, rtol=0.0
        )
        target = optimal.to(prediction.dtype)
        target = target / target.sum(-1, keepdim=True).clamp_min(1.0)
        log_probability = F.log_softmax(
            -prediction.masked_fill(~action_valid, torch.inf), dim=-1
        )
        listwise = -torch.where(
            optimal, target * log_probability, torch.zeros_like(log_probability)
        ).sum(-1).mean()
        losses["listwise"] = listwise
        losses["total"] = losses["total"] + self.listwise_weight * listwise
        return losses
