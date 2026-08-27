"""Fixed R3 objective with discrete-time expiry supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from revealnav_mf2.losses import masked_mean
from revealnav_mf2r2 import BalancedStructuredUADLoss


class BalancedStructuredUADExpiryLoss(BalancedStructuredUADLoss):
    def __init__(self, config, class_weights: tuple[float, float, float]) -> None:
        super().__init__(config, class_weights)

    def forward(self, output, batch):
        losses = super().forward(output, batch)
        without_label = batch["option_cost_without_checkpoint"]
        without_mask = batch["candidate_mask"] & torch.isfinite(without_label)
        without_checkpoint = masked_mean(
            F.smooth_l1_loss(
                output.option_cost_without_checkpoint.masked_fill(
                    ~batch["candidate_mask"], 0.0
                ),
                without_label.masked_fill(~without_mask, 0.0),
                reduction="none",
            ),
            without_mask,
        )
        label = batch["expiry_hazard"]
        mask = label >= 0
        expiry = masked_mean(
            F.binary_cross_entropy_with_logits(
                output.expiry_hazard_logit,
                label.clamp_min(0),
                reduction="none",
            ),
            mask,
        )
        losses["without_checkpoint_cost"] = without_checkpoint
        losses["expiry"] = expiry
        losses["total"] = losses["total"] + without_checkpoint + expiry
        return losses


class ExpiryQAdapterLoss(torch.nn.Module):
    """Loss for additive R3.1 heads with the accepted R2 path frozen."""

    def forward(self, output, batch):
        without_label = batch["option_cost_without_checkpoint"]
        without_mask = batch["candidate_mask"] & torch.isfinite(without_label)
        without_checkpoint = masked_mean(
            F.smooth_l1_loss(
                output.option_cost_without_checkpoint.masked_fill(
                    ~batch["candidate_mask"], 0.0
                ),
                without_label.masked_fill(~without_mask, 0.0),
                reduction="none",
            ),
            without_mask,
        )
        label = batch["expiry_hazard"]
        expiry_mask = label >= 0
        expiry = masked_mean(
            F.binary_cross_entropy_with_logits(
                output.expiry_hazard_logit,
                label.clamp_min(0),
                reduction="none",
            ),
            expiry_mask,
        )
        return {
            "without_checkpoint_cost": without_checkpoint,
            "expiry": expiry,
            "total": without_checkpoint + expiry,
        }


class PairedQAdapterLoss(torch.nn.Module):
    """R3.2 paired-Q objective: two regressions plus within-step ranking."""

    def __init__(self, ranking_weight: float = 0.25, margin: float = 0.1):
        super().__init__()
        self.ranking_weight = float(ranking_weight)
        self.margin = float(margin)

    def forward(self, output, batch):
        mask = batch["candidate_mask"]
        q_with_label = batch["option_cost"]
        q_without_label = batch["option_cost_without_checkpoint"]
        with_mask = mask & torch.isfinite(q_with_label)
        without_mask = mask & torch.isfinite(q_without_label)
        q_with = masked_mean(
            F.smooth_l1_loss(
                output.option_cost.masked_fill(~mask, 0.0),
                q_with_label.masked_fill(~with_mask, 0.0),
                reduction="none",
            ),
            with_mask,
        )
        q_without = masked_mean(
            F.smooth_l1_loss(
                output.option_cost_without_checkpoint.masked_fill(~mask, 0.0),
                q_without_label.masked_fill(~without_mask, 0.0),
                reduction="none",
            ),
            without_mask,
        )
        teacher = q_with_label.masked_fill(~with_mask, torch.inf)
        best_teacher, best_index = teacher.min(dim=-1)
        best_valid = torch.isfinite(best_teacher)
        best_prediction = output.option_cost.gather(
            -1, best_index.unsqueeze(-1)
        ).squeeze(-1)
        strictly_worse = with_mask & (
            q_with_label > best_teacher.unsqueeze(-1) + 1e-6
        ) & best_valid.unsqueeze(-1)
        ranking = masked_mean(
            F.relu(
                self.margin
                - (output.option_cost - best_prediction.unsqueeze(-1))
            ),
            strictly_worse,
        )
        return {
            "q_with": q_with,
            "q_without": q_without,
            "ranking": ranking,
            "total": q_with + q_without + self.ranking_weight * ranking,
        }


class CausalPairedQAdapterLoss(torch.nn.Module):
    """Identifiable paired-Q loss including the algebraic cost gap."""

    def __init__(self, ranking_weight: float = 0.25, margin: float = 0.1):
        super().__init__()
        self.ranking_weight = float(ranking_weight)
        self.margin = float(margin)

    def forward(self, output, batch):
        mask = batch["candidate_mask"]
        q_with_label = batch["option_cost"]
        q_without_label = batch["option_cost_without_checkpoint"]
        valid = mask & torch.isfinite(q_with_label) & torch.isfinite(
            q_without_label
        )
        q_with_prediction = output.q_with_checkpoint.masked_fill(~mask, 0.0)
        q_without_prediction = output.q_without_checkpoint.masked_fill(
            ~mask, 0.0
        )
        q_with = masked_mean(F.smooth_l1_loss(
            q_with_prediction, q_with_label.masked_fill(~valid, 0.0),
            reduction="none",
        ), valid)
        q_without = masked_mean(F.smooth_l1_loss(
            q_without_prediction, q_without_label.masked_fill(~valid, 0.0),
            reduction="none",
        ), valid)
        truth_delta = (q_without_label - q_with_label).masked_fill(~valid, 0.0)
        pred_delta = (q_without_prediction - q_with_prediction).masked_fill(
            ~valid, 0.0
        )
        paired_delta = masked_mean(F.smooth_l1_loss(
            pred_delta, truth_delta, reduction="none"
        ), valid)
        teacher = q_with_label.masked_fill(~valid, torch.inf)
        best_teacher, best_index = teacher.min(-1)
        best_valid = torch.isfinite(best_teacher)
        best_prediction = output.q_with_checkpoint.gather(
            -1, best_index.unsqueeze(-1)
        ).squeeze(-1)
        strictly_worse = valid & (
            q_with_label > best_teacher.unsqueeze(-1) + 1e-6
        ) & best_valid.unsqueeze(-1)
        ranking = masked_mean(F.relu(
            self.margin
            - (output.q_with_checkpoint - best_prediction.unsqueeze(-1))
        ), strictly_worse)
        return {
            "q_with": q_with, "q_without": q_without,
            "paired_delta": paired_delta, "ranking": ranking,
            "total": q_with + q_without + paired_delta
                     + self.ranking_weight * ranking,
        }
