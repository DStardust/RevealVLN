"""Frozen MF2.1 training losses for variable candidate sets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import RevealOptionOutput


@dataclass(frozen=True)
class RevealOptionLossConfig:
    target_weight: float = 1.0
    state_weight: float = 1.0
    cost_weight: float = 1.0
    feasibility_weight: float = 1.0
    ranking_weight: float = 0.25
    checkpoint_weight: float = 0.5
    ranking_margin: float = 0.1
    state_pos_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = values.masked_select(mask)
    return selected.mean() if selected.numel() else values.sum() * 0.0


class RevealOptionLoss(nn.Module):
    def __init__(
        self, config: RevealOptionLossConfig = RevealOptionLossConfig()
    ) -> None:
        super().__init__()
        if (
            len(config.state_pos_weights) != 4
            or any(
                not isinstance(value, (int, float)) or value <= 0
                for value in config.state_pos_weights
            )
        ):
            raise ValueError("state_pos_weights must contain four positive values")
        self.config = config

    def forward(self, output: RevealOptionOutput, batch: dict[str, Tensor]):
        candidate_mask = batch["candidate_mask"]
        target_index = batch["target_index"]
        valid_steps = target_index >= 0
        flat_target = target_index.flatten()
        if valid_steps.any():
            target = F.cross_entropy(
                output.target_logits.flatten(0, 1),
                flat_target,
                ignore_index=-1,
            )
        else:
            target = output.target_logits.masked_fill(
                ~candidate_mask, 0.0
            ).sum() * 0.0

        state_losses = []
        for prediction, key, pos_weight in (
            (output.target_in_set_logit, "target_in_set", self.config.state_pos_weights[0]),
            (output.separation_logit, "separation", self.config.state_pos_weights[1]),
            (output.evidence_logit, "evidence_complete", self.config.state_pos_weights[2]),
            (output.reveal_hazard_logit, "reveal_hazard", self.config.state_pos_weights[3]),
        ):
            label = batch[key]
            mask = label >= 0
            raw = F.binary_cross_entropy_with_logits(
                prediction,
                label.clamp_min(0),
                reduction="none",
                pos_weight=prediction.new_tensor(pos_weight),
            )
            state_losses.append(masked_mean(raw, mask))
        state = torch.stack(state_losses).mean()

        cost_label = batch["option_cost"]
        cost_mask = candidate_mask & torch.isfinite(cost_label)
        cost = masked_mean(
            F.smooth_l1_loss(
                output.option_cost.masked_fill(~candidate_mask, 0.0),
                cost_label.masked_fill(~cost_mask, 0.0),
                reduction="none",
            ),
            cost_mask,
        )
        feasibility_label = batch["current_feasibility"]
        feasibility_mask = candidate_mask.unsqueeze(-1) & (
            feasibility_label >= 0
        )
        feasibility = masked_mean(
            F.binary_cross_entropy_with_logits(
                output.current_feasibility_logits.masked_fill(
                    ~candidate_mask.unsqueeze(-1), 0.0
                ),
                feasibility_label.clamp_min(0),
                reduction="none",
            ),
            feasibility_mask,
        )

        teacher = cost_label.masked_fill(~cost_mask, torch.inf)
        best_teacher, best_index = teacher.min(dim=-1)
        best_valid = torch.isfinite(best_teacher) & valid_steps
        best_prediction = output.option_cost.gather(
            -1, best_index.unsqueeze(-1)
        ).squeeze(-1)
        strictly_worse = cost_mask & (
            cost_label > best_teacher.unsqueeze(-1) + 1e-6
        ) & best_valid.unsqueeze(-1)
        ranking_raw = F.relu(
            self.config.ranking_margin
            - (output.option_cost - best_prediction.unsqueeze(-1))
        )
        ranking = masked_mean(ranking_raw, strictly_worse)

        checkpoint_label = batch["checkpoint_value"]
        checkpoint_mask = torch.isfinite(checkpoint_label)
        checkpoint = masked_mean(
            F.smooth_l1_loss(
                output.checkpoint_value,
                checkpoint_label.masked_fill(~checkpoint_mask, 0.0),
                reduction="none",
            ),
            checkpoint_mask,
        )
        total = (
            self.config.target_weight * target
            + self.config.state_weight * state
            + self.config.cost_weight * cost
            + self.config.feasibility_weight * feasibility
            + self.config.ranking_weight * ranking
            + self.config.checkpoint_weight * checkpoint
        )
        return {
            "total": total,
            "target": target,
            "state": state,
            "cost": cost,
            "feasibility": feasibility,
            "ranking": ranking,
            "checkpoint": checkpoint,
        }
