"""Oracle-only U/A/D and Reveal/Expiry labels for MF3ZN-TUAD v1.

This module deliberately contains no feature or model code.  Route truth,
future rollout evidence, and simulator reachability may be used to construct
``TemporalOracleLabel`` values, but those values never enter the causal input
schema in :mod:`revealnav_mf3.temporal_uad_schema`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from .temporal_uad_schema import TemporalSequence


UAD_STABILITY_PREFIXES = 3


class UADState(str, Enum):
    UNOBSERVED = "U"
    AMBIGUOUS = "A"
    DECISIVE = "D"


def _boolean_tuple(value: object, name: str) -> tuple[bool, ...]:
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error
    if not raw or any(not isinstance(item, (bool, np.bool_)) for item in raw):
        raise TypeError(f"{name} must contain one or more boolean values")
    return tuple(bool(item) for item in raw)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class TemporalOracleLabel:
    """Supervision stored physically apart from a causal temporal record."""

    target_in_set: tuple[bool, ...]
    candidate_separated: tuple[bool, ...]
    evidence_closed: tuple[bool, ...]
    reveal_interval: tuple[int, int] | None
    expiry_step: int | None
    resolvable: bool

    def __post_init__(self) -> None:
        in_set = _boolean_tuple(self.target_in_set, "target_in_set")
        separated = _boolean_tuple(
            self.candidate_separated, "candidate_separated",
        )
        evidence = _boolean_tuple(self.evidence_closed, "evidence_closed")
        if not (len(in_set) == len(separated) == len(evidence)):
            raise ValueError("oracle factor sequences must have equal lengths")

        interval = self.reveal_interval
        if interval is not None:
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise TypeError("reveal_interval must be an integer pair or None")
            lower = _nonnegative_int(interval[0], "reveal_interval lower")
            upper = _nonnegative_int(interval[1], "reveal_interval upper")
            if lower > upper:
                raise ValueError("reveal_interval lower must not exceed upper")
            interval = (lower, upper)

        expiry = (
            None
            if self.expiry_step is None
            else _nonnegative_int(self.expiry_step, "expiry_step")
        )
        if not isinstance(self.resolvable, (bool, np.bool_)):
            raise TypeError("resolvable must be boolean")

        object.__setattr__(self, "target_in_set", in_set)
        object.__setattr__(self, "candidate_separated", separated)
        object.__setattr__(self, "evidence_closed", evidence)
        object.__setattr__(self, "reveal_interval", interval)
        object.__setattr__(self, "expiry_step", expiry)
        object.__setattr__(self, "resolvable", bool(self.resolvable))

    @property
    def prefix_count(self) -> int:
        return len(self.target_in_set)


def _factor_sequences(
    target_in_set: TemporalOracleLabel | Iterable[bool],
    candidate_separated: Iterable[bool] | None,
    evidence_closed: Iterable[bool] | None,
) -> tuple[tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]]:
    if isinstance(target_in_set, TemporalOracleLabel):
        if candidate_separated is not None or evidence_closed is not None:
            raise TypeError(
                "factor sequences must not accompany a TemporalOracleLabel"
            )
        return (
            target_in_set.target_in_set,
            target_in_set.candidate_separated,
            target_in_set.evidence_closed,
        )
    if candidate_separated is None or evidence_closed is None:
        raise TypeError("derive_uad requires all three factor sequences")
    in_set = _boolean_tuple(target_in_set, "target_in_set")
    separated = _boolean_tuple(candidate_separated, "candidate_separated")
    evidence = _boolean_tuple(evidence_closed, "evidence_closed")
    if not (len(in_set) == len(separated) == len(evidence)):
        raise ValueError("UAD factor sequences must have equal lengths")
    return in_set, separated, evidence


def derive_uad(
    target_in_set: TemporalOracleLabel | Iterable[bool],
    candidate_separated: Iterable[bool] | None = None,
    evidence_closed: Iterable[bool] | None = None,
) -> tuple[UADState, ...]:
    """Deterministically derive U/A/D with the frozen K-prefix rule.

    A decisive state is emitted only after all three factors have held for
    ``UAD_STABILITY_PREFIXES`` consecutive prefixes.  Any set loss,
    separation regression, or evidence-closure regression resets the streak;
    this is the protocol's explicit occlusion/reset behavior.  There is no
    learned three-class head and no configurable stability threshold.
    """

    in_set, separated, evidence = _factor_sequences(
        target_in_set, candidate_separated, evidence_closed,
    )
    stable = 0
    states: list[UADState] = []
    for present, distinct, closed in zip(
        in_set, separated, evidence, strict=True,
    ):
        complete = present and distinct and closed
        stable = stable + 1 if complete else 0
        if not present:
            states.append(UADState.UNOBSERVED)
        elif stable >= UAD_STABILITY_PREFIXES:
            states.append(UADState.DECISIVE)
        else:
            states.append(UADState.AMBIGUOUS)
    return tuple(states)


def validate_oracle_alignment(
    sequence: TemporalSequence,
    label: TemporalOracleLabel,
) -> None:
    """Fail closed when a separately stored label is not prefix-aligned."""

    if not isinstance(sequence, TemporalSequence):
        raise TypeError("sequence must be a TemporalSequence")
    if not isinstance(label, TemporalOracleLabel):
        raise TypeError("label must be a TemporalOracleLabel")
    if label.prefix_count != len(sequence.steps):
        raise ValueError("oracle label length does not match causal sequence")
    observed_steps = {step.step for step in sequence.steps}
    if label.reveal_interval is not None and any(
        step not in observed_steps for step in label.reveal_interval
    ):
        raise ValueError("reveal interval is outside the causal prefix steps")
    if label.expiry_step is not None and label.expiry_step not in observed_steps:
        raise ValueError("expiry step is outside the causal prefix steps")


def reveal_interval_membership(
    sequence: TemporalSequence,
    label: TemporalOracleLabel,
) -> tuple[bool, ...]:
    """Return the interval-censor mask without choosing an onset post hoc."""

    validate_oracle_alignment(sequence, label)
    if label.reveal_interval is None:
        return (False,) * len(sequence.steps)
    lower, upper = label.reveal_interval
    return tuple(lower <= step.step <= upper for step in sequence.steps)

