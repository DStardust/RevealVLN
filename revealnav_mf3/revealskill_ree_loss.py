"""Fixed equal-weight Reveal/Expiry estimation losses."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def discrete_event_nll(hazard_logits: Tensor, event: Tensor, at_risk: Tensor) -> Tensor:
    if hazard_logits.shape != event.shape or event.shape != at_risk.shape:
        raise ValueError("hazard tensors must have the same shape")
    if event.dtype is not torch.bool or at_risk.dtype is not torch.bool:
        raise TypeError("event and at_risk masks must be boolean")
    valid = at_risk
    if not bool(valid.any()):
        return hazard_logits.sum() * 0.0
    target = event.to(dtype=hazard_logits.dtype)
    return F.binary_cross_entropy_with_logits(hazard_logits[valid], target[valid])


def interval_event_nll(hazard_logits: Tensor, lower: int, upper: int, mask: Tensor | None = None) -> Tensor:
    """Negative log likelihood for a predeclared integer reveal interval."""

    if hazard_logits.ndim != 1 or not 0 <= lower <= upper < hazard_logits.numel():
        raise ValueError("invalid reveal interval")
    if mask is not None and (mask.ndim != 1 or mask.shape != hazard_logits.shape or mask.dtype is not torch.bool):
        raise ValueError("invalid interval mask")
    log_survival = torch.logsigmoid(-hazard_logits)
    log_hazard = torch.logsigmoid(hazard_logits)
    terms = []
    for index in range(lower, upper + 1):
        terms.append(log_hazard[index] + log_survival[:index].sum())
    return -torch.logsumexp(torch.stack(terms), dim=0)


def monotonicity_penalty(probabilities: Tensor, valid_transition: Tensor) -> Tensor:
    if probabilities.ndim != 2 or valid_transition.shape != probabilities[:, :-1].shape or valid_transition.dtype is not torch.bool:
        raise ValueError("invalid monotonicity tensors")
    if probabilities.shape[1] < 2:
        return probabilities.sum() * 0.0
    return torch.relu(probabilities[:, :-1] - probabilities[:, 1:])[valid_transition].mean() if bool(valid_transition.any()) else probabilities.sum() * 0.0


def ree_loss(
    set_logits: Tensor,
    separation_logits: Tensor,
    evidence_logits: Tensor,
    reveal_logits: Tensor,
    expiry_logits: Tensor,
    targets: dict[str, Tensor],
    *,
    monotonicity: Tensor,
) -> Tensor:
    """Equal 1/6 weighting; no tunable loss coefficients."""

    factors = []
    for key, logits in (("set", set_logits), ("separation", separation_logits), ("evidence", evidence_logits)):
        target = targets[key]
        if target.shape != logits.shape:
            raise ValueError(f"{key} target shape mismatch")
        factors.append(F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype)))
    factors.extend((
        discrete_event_nll(reveal_logits, targets["reveal_event"], targets["reveal_at_risk"]),
        discrete_event_nll(expiry_logits, targets["expiry_event"], targets["expiry_at_risk"]),
        monotonicity,
    ))
    return torch.stack(factors).mean()


__all__ = ["discrete_event_nll", "interval_event_nll", "monotonicity_penalty", "ree_loss"]
