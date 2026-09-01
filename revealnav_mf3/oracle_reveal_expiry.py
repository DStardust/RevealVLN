"""Fixed Reveal/Expiry derivations and a control-backed returnability adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from .oracle_revealskill_schema import (
    K_STABILITY,
    OracleOption,
    OracleReadiness,
    derive_constraint_state,
    option_readiness,
)


def derive_prerequisite_satisfaction(
    factors_by_constraint: Mapping[str, Sequence[tuple[bool, bool, bool]]],
    prerequisite_ids: Iterable[str],
    *,
    prefix_count: int | None = None,
) -> dict[str, tuple[bool, ...]]:
    """Return historical prerequisite completion, without a K=3 requirement."""

    result: dict[str, tuple[bool, ...]] = {}
    for cid in prerequisite_ids:
        if cid not in factors_by_constraint:
            raise KeyError(f"missing prerequisite factors: {cid}")
        rows = tuple(tuple(bool(x) for x in row) for row in factors_by_constraint[cid])
        if not rows or any(len(row) != 3 for row in rows):
            raise ValueError("prerequisite factors must be non-empty triples")
        if prefix_count is not None and len(rows) != prefix_count:
            raise ValueError("prerequisite prefix count mismatch")
        # A prerequisite is historically satisfied once all three factors are
        # true at any observed prefix; it does not need current K-prefix
        # stability and remains true thereafter.
        seen = False
        values: list[bool] = []
        for row in rows:
            seen = seen or all(row)
            values.append(seen)
        result[cid] = tuple(values)
    return result


def derive_decision_states(
    factors_by_constraint: Mapping[str, Sequence[tuple[bool, bool, bool]]],
    decisive_ids: Iterable[str],
) -> dict[str, tuple[OracleReadiness, ...]]:
    """Derive each DEC constraint independently with fixed K=3."""

    result = {}
    for cid in decisive_ids:
        if cid not in factors_by_constraint:
            raise KeyError(f"missing decisive factors: {cid}")
        result[cid] = derive_constraint_state(factors_by_constraint[cid], stability_k=K_STABILITY)
    return result


def reveal_step_for_option(
    option: OracleOption,
    states_by_constraint: Mapping[str, Sequence[OracleReadiness | str]],
    prerequisite_satisfaction: Mapping[str, Sequence[bool]],
) -> int | None:
    """Return the first prefix at which the option is hard-D and contextual."""

    if not option.decisive_constraint_ids:
        return None
    lengths = [len(states_by_constraint[cid]) for cid in option.decisive_constraint_ids]
    lengths += [len(prerequisite_satisfaction[cid]) for cid in option.prerequisite_ids]
    if not lengths:
        return None
    if len(set(lengths)) != 1:
        raise ValueError("option state sequences are not aligned")
    for index in range(lengths[0]):
        if not all(OracleReadiness(states_by_constraint[cid][index]) is OracleReadiness.D for cid in option.decisive_constraint_ids):
            continue
        if not all(bool(prerequisite_satisfaction[cid][index]) for cid in option.prerequisite_ids):
            continue
        return index
    return None


def reveal_expiry_slack(reveal_step: int | None, expiry_step: int | None) -> tuple[str, int | None]:
    if reveal_step is None or expiry_step is None:
        return "UNKNOWN", None
    delta = int(expiry_step) - int(reveal_step)
    if delta > 0:
        return "POSITIVE_SLACK", delta
    if delta == 0:
        return "TIGHT", delta
    return "UNRESOLVABLE", delta


@dataclass(frozen=True)
class ReturnabilityOracle:
    """Adapter whose callback must be supplied by the frozen controller.

    The callback is intentionally not a geometry-only shortcut.  Production
    callers pass a function that probes the frozen ETP return primitive.  A
    missing callback is an explicit unsupported condition rather than an
    optimistic guess.
    """

    callback: Callable[[OracleOption, Mapping[str, object], int], bool] | None = None
    horizon: int = 8

    def __post_init__(self) -> None:
        if self.horizon != 8:
            raise ValueError("MF3ZQ return horizon is fixed at 8")

    @property
    def available(self) -> bool:
        return self.callback is not None

    def is_returnable(self, option: OracleOption, state: Mapping[str, object], *, step: int) -> bool:
        if step < 0:
            raise ValueError("step must be non-negative")
        if self.callback is None:
            raise RuntimeError("control-backed returnability oracle is unavailable")
        return bool(self.callback(option, state, self.horizon))

    def expiry_step(
        self,
        option: OracleOption,
        states: Sequence[Mapping[str, object]],
    ) -> int | None:
        if self.callback is None:
            return None
        last: int | None = None
        for step, state in enumerate(states):
            if self.is_returnable(option, state, step=step):
                last = step
        return last


def option_slack_class(reveal_step: int | None, expiry_step: int | None) -> str:
    return reveal_expiry_slack(reveal_step, expiry_step)[0]


__all__ = [
    "ReturnabilityOracle", "derive_prerequisite_satisfaction",
    "derive_decision_states", "reveal_step_for_option", "reveal_expiry_slack",
    "option_slack_class",
]
