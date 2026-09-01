"""Fixed option-preservation objective and legal skill action set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .revealskill_schema import Readiness, RevealSkillAction


def legal_skill_actions(
    readiness_by_option: Mapping[str, Readiness | str],
    *,
    executable_options: Sequence[str],
    returnable_options: Sequence[str] = (),
    stop_legal: bool = True,
) -> tuple[tuple[RevealSkillAction, str | None], ...]:
    actions: list[tuple[RevealSkillAction, str | None]] = [
        (RevealSkillAction.FOLLOW, None),
        (RevealSkillAction.INSPECT, None),
    ]
    for option in executable_options:
        actions.append((RevealSkillAction.EXPLORE, str(option)))
        if Readiness(readiness_by_option[str(option)]) is Readiness.D:
            actions.append((RevealSkillAction.COMMIT, str(option)))
    for option in returnable_options:
        actions.append((RevealSkillAction.BACKTRACK, str(option)))
    if stop_legal:
        actions.append((RevealSkillAction.STOP, None))
    return tuple(actions)


def option_preservation_value(
    without_option_cost: Tensor,
    with_option_cost: Tensor,
) -> Tensor:
    if without_option_cost.shape != with_option_cost.shape:
        raise ValueError("OPV tensors must have matching shapes")
    return without_option_cost - with_option_cost


def skill_q_loss(predicted_q: Tensor, target_cost: Tensor, pairwise_pairs: tuple[Tensor, Tensor] | None = None) -> Tensor:
    if predicted_q.shape != target_cost.shape:
        raise ValueError("Q and target cost shapes must match")
    loss = F.huber_loss(predicted_q, target_cost)
    if pairwise_pairs is not None:
        good, bad = pairwise_pairs
        if good.shape != bad.shape:
            raise ValueError("ranking pair shapes must match")
        loss = loss + torch.relu(0.1 + good - bad).mean()
    return loss


__all__ = ["legal_skill_actions", "option_preservation_value", "skill_q_loss"]
