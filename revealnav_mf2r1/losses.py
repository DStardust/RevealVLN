"""Joint U/A/D likelihood over the existing interpretable REE factors."""

from __future__ import annotations

import torch
from torch import Tensor

from revealnav_mf2 import RevealOptionLoss, RevealOptionOutput
from revealnav_mf2.losses import masked_mean


def factorized_uad_probabilities(output: RevealOptionOutput) -> Tensor:
    """Return normalized U/A/D probabilities without adding a direct head."""

    target = torch.sigmoid(output.target_in_set_logit)
    separated = torch.sigmoid(output.separation_logit)
    evidence = torch.sigmoid(output.evidence_logit)
    decisive = separated * evidence
    return torch.stack(
        (1.0 - target, target * (1.0 - decisive), target * decisive),
        dim=-1,
    )


class StructuredUADLoss(RevealOptionLoss):
    """Add exact U/A/D likelihood while retaining every frozen REE target."""

    def __init__(self, config, uad_weight: float = 1.0) -> None:
        super().__init__(config)
        if uad_weight <= 0:
            raise ValueError("uad_weight must be positive")
        self.uad_weight = float(uad_weight)

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
        log_probability = probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
        row_loss = -log_probability.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        uad = masked_mean(row_loss, valid)
        losses["uad"] = uad
        losses["total"] = losses["total"] + self.uad_weight * uad
        return losses
