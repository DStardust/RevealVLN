"""Schemas for the exploratory MF3ZQ oracle RevealSkill headroom run.

The objects in this module deliberately separate *cognitive state* from
control outcomes.  They are small, serialisable values used by the fixed
oracle policy and by the fail-closed population audit; they are not a learned
policy and do not contain rewards or simulator poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


K_STABILITY = 3
OPTION_MEMORY_BUDGET = 8
RETURN_HORIZON = 8


class OracleSkill(str, Enum):
    FOLLOW = "FOLLOW"
    INSPECT = "INSPECT"
    EXPLORE = "EXPLORE"
    BACKTRACK = "BACKTRACK"
    COMMIT = "COMMIT"
    STOP = "STOP"


class OracleReadiness(str, Enum):
    U = "U"
    A = "A"
    D = "D"


class OracleOptionStatus(str, Enum):
    UNTRIED = "UNTRIED"
    ACTIVE = "ACTIVE"
    PRESERVED = "PRESERVED"
    EXHAUSTED = "EXHAUSTED"
    COMMITTED = "COMMITTED"
    INVALID = "INVALID"


FORBIDDEN_ORACLE_INPUT_KEYS = frozenset({
    "reward", "success", "spl", "ndtw", "sdtw", "utility", "delta_utility",
    "outcome", "catastrophe", "target", "future", "future_frame",
    "future_candidate_set", "pose", "navmesh", "shortest_path",
    "correct_action", "best_action", "car_result", "qwen_result",
})


def reject_forbidden_oracle_mapping(value: Mapping[str, object]) -> None:
    """Reject outcome/future fields before a state reaches the policy."""

    for raw_key, child in value.items():
        key = str(raw_key).casefold()
        if key in FORBIDDEN_ORACLE_INPUT_KEYS or key.startswith((
            "future_", "outcome_", "reward_", "treatment_", "oracle_result",
        )):
            raise ValueError(f"forbidden oracle state field: {raw_key}")
        if isinstance(child, Mapping):
            reject_forbidden_oracle_mapping(child)
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, Mapping):
                    reject_forbidden_oracle_mapping(item)


def _ids(values: Iterable[object], name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (not allow_empty and not result) or any(not item for item in result):
        raise ValueError(f"{name} must contain non-empty identifiers")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique identifiers")
    return result


@dataclass(frozen=True)
class OracleOption:
    """One executable ETP candidate represented as a high-level option."""

    option_id: str
    branch_candidate_id: str
    anchor_checkpoint_id: str
    frozen_candidate_rank: int
    first_seen_step: int
    last_seen_step: int
    prerequisite_ids: tuple[str, ...]
    decisive_constraint_ids: tuple[str, ...]
    readiness: OracleReadiness = OracleReadiness.U
    reveal_step: int | None = None
    expiry_step: int | None = None
    returnable: bool = False
    status: OracleOptionStatus = OracleOptionStatus.UNTRIED

    def __post_init__(self) -> None:
        for value, name in (
            (self.option_id, "option_id"),
            (self.branch_candidate_id, "branch_candidate_id"),
            (self.anchor_checkpoint_id, "anchor_checkpoint_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for value, name in (
            (self.frozen_candidate_rank, "frozen_candidate_rank"),
            (self.first_seen_step, "first_seen_step"),
            (self.last_seen_step, "last_seen_step"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.first_seen_step > self.last_seen_step:
            raise ValueError("option step interval is invalid")
        if self.reveal_step is not None and (
            isinstance(self.reveal_step, bool)
            or not isinstance(self.reveal_step, int)
            or self.reveal_step < 0
        ):
            raise ValueError("reveal_step must be non-negative when present")
        if self.expiry_step is not None and (
            isinstance(self.expiry_step, bool)
            or not isinstance(self.expiry_step, int)
            or self.expiry_step < 0
        ):
            raise ValueError("expiry_step must be non-negative when present")
        object.__setattr__(self, "prerequisite_ids", _ids(self.prerequisite_ids, "prerequisite_ids"))
        object.__setattr__(self, "decisive_constraint_ids", _ids(self.decisive_constraint_ids, "decisive_constraint_ids"))
        object.__setattr__(self, "readiness", self.readiness if isinstance(self.readiness, OracleReadiness) else OracleReadiness(self.readiness))
        object.__setattr__(self, "status", self.status if isinstance(self.status, OracleOptionStatus) else OracleOptionStatus(self.status))
        object.__setattr__(self, "returnable", bool(self.returnable))


@dataclass(frozen=True)
class OraclePolicyState:
    """Strictly causal state presented to the deterministic oracle policy."""

    step: int
    checkpoint_id: str
    executable_option_ids: tuple[str, ...]
    options: tuple[OracleOption, ...]
    unresolved_decisive_evidence: bool
    reveal_after_expiry: bool
    native_option_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")
        ids = _ids(self.executable_option_ids, "executable_option_ids")
        options = tuple(self.options)
        if any(not isinstance(value, OracleOption) for value in options):
            raise TypeError("options must contain OracleOption values")
        option_ids = {value.option_id for value in options}
        if not set(ids).issubset(option_ids):
            raise ValueError("executable option is not present in option memory")
        if self.native_option_id is not None and self.native_option_id not in ids:
            raise ValueError("native option is not executable")
        object.__setattr__(self, "executable_option_ids", ids)
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "unresolved_decisive_evidence", bool(self.unresolved_decisive_evidence))
        object.__setattr__(self, "reveal_after_expiry", bool(self.reveal_after_expiry))


def derive_constraint_state(
    factors: Iterable[tuple[bool, bool, bool]],
    *,
    stability_k: int = K_STABILITY,
) -> tuple[OracleReadiness, ...]:
    """Derive U/A/D with an independent consecutive-ready streak.

    ``factors`` is ordered by causal prefix.  A non-instantiated prefix resets
    the streak to U; a present but unresolved prefix is A.  K is fixed by the
    protocol and cannot be tuned by callers.
    """

    if stability_k != K_STABILITY:
        raise ValueError("MF3ZQ stability K is fixed at 3")
    rows = tuple(tuple(bool(item) for item in row) for row in factors)
    if not rows:
        raise ValueError("at least one factor row is required")
    streak = 0
    result: list[OracleReadiness] = []
    for instantiated, distinguishable, resolved in rows:
        if not instantiated:
            streak = 0
            result.append(OracleReadiness.U)
        elif not (distinguishable and resolved):
            streak = 0
            result.append(OracleReadiness.A)
        else:
            streak += 1
            result.append(OracleReadiness.D if streak >= K_STABILITY else OracleReadiness.A)
    return tuple(result)


def option_readiness(
    decisive_states: Mapping[str, OracleReadiness | str],
    prerequisite_satisfied: Mapping[str, bool],
    decisive_ids: Iterable[str],
    prerequisite_ids: Iterable[str],
) -> OracleReadiness:
    """Apply DEC-vs-prerequisite semantics without treating prerequisites as K-stable DEC."""

    dec = tuple(str(value) for value in decisive_ids)
    pre = tuple(str(value) for value in prerequisite_ids)
    if not dec:
        raise ValueError("an option needs at least one decisive constraint")
    values = [decisive_states[item] for item in dec]
    if any(OracleReadiness(value) is OracleReadiness.U for value in values):
        return OracleReadiness.U
    if not all(OracleReadiness(value) is OracleReadiness.D for value in values):
        return OracleReadiness.A
    if not all(bool(prerequisite_satisfied.get(item, False)) for item in pre):
        return OracleReadiness.A
    return OracleReadiness.D


__all__ = [
    "K_STABILITY", "OPTION_MEMORY_BUDGET", "RETURN_HORIZON", "OracleSkill",
    "OracleReadiness", "OracleOptionStatus", "OracleOption", "OraclePolicyState",
    "derive_constraint_state", "option_readiness", "reject_forbidden_oracle_mapping",
]
