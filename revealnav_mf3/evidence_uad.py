"""Deterministic per-constraint U/A/D and option readiness."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Mapping

from .evidence_chain import decisive_chain
from .evidence_constraints import InstructionEvidenceGraph


UAD_STABILITY_PREFIXES = 3


class ConstraintState(str, Enum):
    U = "U"
    A = "A"
    D = "D"


def derive_constraint_uad(
    instantiated: Iterable[bool],
    distinguishable: Iterable[bool],
    resolved: Iterable[bool],
    *,
    stability_k: int = UAD_STABILITY_PREFIXES,
) -> tuple[ConstraintState, ...]:
    if stability_k != UAD_STABILITY_PREFIXES:
        raise ValueError("UAD stability K is fixed at 3")
    factors = tuple(tuple(bool(value) for value in values) for values in (instantiated, distinguishable, resolved))
    if not factors[0] or len({len(value) for value in factors}) != 1:
        raise ValueError("constraint factors must be aligned and non-empty")
    streak = 0
    result: list[ConstraintState] = []
    for present, distinct, closed in zip(*factors, strict=True):
        ready = present and distinct and closed
        streak = streak + 1 if ready else 0
        if not present:
            result.append(ConstraintState.U)
        elif streak >= UAD_STABILITY_PREFIXES:
            result.append(ConstraintState.D)
        else:
            result.append(ConstraintState.A)
    return tuple(result)


def option_readiness(
    graph: InstructionEvidenceGraph,
    option_id: str,
    states: Mapping[str, ConstraintState | str],
) -> ConstraintState:
    required = decisive_chain(graph, option_id)
    if not required:
        raise ValueError(f"option has no decisive constraints: {option_id}")
    values = [ConstraintState(states[cid]) for cid in required]
    if any(value is ConstraintState.U for value in values):
        return ConstraintState.U
    if all(value is ConstraintState.D for value in values):
        return ConstraintState.D
    return ConstraintState.A


def soft_option_readiness(
    graph: InstructionEvidenceGraph,
    option_id: str,
    probabilities: Mapping[str, float],
) -> float:
    required = decisive_chain(graph, option_id)
    if not required:
        raise ValueError(f"option has no decisive constraints: {option_id}")
    values = []
    for cid in required:
        value = float(probabilities[cid])
        if not 0.0 <= value <= 1.0:
            raise ValueError("constraint resolution probabilities must lie in [0,1]")
        values.append(value)
    result = 1.0
    for value in values:
        result *= value
    return result


def option_reveal_step(
    graph: InstructionEvidenceGraph,
    option_id: str,
    per_constraint_reveal: Mapping[str, int],
) -> int:
    required = decisive_chain(graph, option_id)
    if not required or any(cid not in per_constraint_reveal for cid in required):
        raise ValueError("missing constraint reveal step")
    values = [int(per_constraint_reveal[cid]) for cid in required]
    if any(value < 0 for value in values):
        raise ValueError("reveal steps must be non-negative")
    return max(values)


def reveal_expiry_slack(reveal_step: int, expiry_step: int) -> int:
    if reveal_step < 0 or expiry_step < 0:
        raise ValueError("reveal/expiry steps must be non-negative")
    return int(expiry_step) - int(reveal_step)


__all__ = [
    "ConstraintState", "UAD_STABILITY_PREFIXES", "derive_constraint_uad",
    "option_readiness", "soft_option_readiness", "option_reveal_step",
    "reveal_expiry_slack",
]
