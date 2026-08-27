"""Class-balanced structured U/A/D loss for revision 2."""

from __future__ import annotations

import torch
from torch import Tensor

from revealnav_mf2 import RevealOptionLoss, RevealOptionOutput
from revealnav_mf2r1 import factorized_uad_probabilities


class BalancedStructuredUADLoss(RevealOptionLoss):
    def __init__(self, config, class_weights: tuple[float, float, float]) -> None:
        super().__init__(config)
        if len(class_weights) != 3 or any(value <= 0 for value in class_weights):
            raise ValueError("class_weights must contain three positive values")
        self.class_weights = tuple(float(value) for value in class_weights)

    def forward(self, output: RevealOptionOutput, batch: dict[str, Tensor]):
        losses = super().forward(output, batch)
        target = batch["target_in_set"]
        separation = batch["separation"]
        evidence = batch["evidence_complete"]
        valid = (target >= 0) & (separation >= 0) & (evidence >= 0)
        labels = torch.where(
            target < 0.5,
            torch.zeros_like(target, dtype=torch.long),
            torch.where(
                (separation < 0.5) | (evidence < 0.5),
                torch.ones_like(target, dtype=torch.long),
                torch.full_like(target, 2, dtype=torch.long),
            ),
        )
        probabilities = factorized_uad_probabilities(output)
        row_loss = -probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log().gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        weights = probabilities.new_tensor(self.class_weights)[labels]
        selected = valid.to(row_loss.dtype) * weights
        uad = (row_loss * selected).sum() / selected.sum().clamp_min(1.0)
        losses["balanced_uad"] = uad
        losses["total"] = losses["total"] + uad
        return losses
