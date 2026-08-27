"""Typed state exchanged between a VLN model and the topology controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose
from typing import Optional


def _probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _nonnegative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


class RevealState(str, Enum):
    UNOBSERVED = "U"
    AMBIGUOUS = "A"
    DISCRIMINABLE = "D"


class BranchStatus(str, Enum):
    UNTRIED = "untried"
    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    COMMITTED = "committed"


class DecisionKind(str, Enum):
    STOP = "stop"
    COMMIT = "commit"
    EXPLORE = "explore"
    INSPECT = "inspect"
    BACKTRACK = "backtrack"
    FAIL = "fail"


@dataclass(frozen=True)
class RevealBelief:
    """Posterior over U/A/D produced by the reveal model."""

    unobserved: float
    ambiguous: float
    discriminable: float

    def __post_init__(self) -> None:
        for name, value in (
            ("unobserved", self.unobserved),
            ("ambiguous", self.ambiguous),
            ("discriminable", self.discriminable),
        ):
            _probability(name, value)
        if not isclose(
            self.unobserved + self.ambiguous + self.discriminable,
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("reveal belief must sum to one")

    @property
    def state(self) -> RevealState:
        values = {
            RevealState.UNOBSERVED: self.unobserved,
            RevealState.AMBIGUOUS: self.ambiguous,
            RevealState.DISCRIMINABLE: self.discriminable,
        }
        return max(values, key=values.__getitem__)


@dataclass
class BranchCandidate:
    """A recoverable option attached to a decision checkpoint.

    The scores are model outputs or oracle labels.  Entropy is intentionally
    absent: a confident distribution over an incomplete candidate set is not
    evidence that committing is safe.
    """

    branch_id: str
    target_probability: float
    information_gain: float
    constraint_coverage: float
    travel_cost: float
    return_cost: float
    irreversible_risk: float
    status: BranchStatus = BranchStatus.UNTRIED
    visits: int = 0

    def __post_init__(self) -> None:
        if not self.branch_id:
            raise ValueError("branch_id must not be empty")
        for name in (
            "target_probability",
            "information_gain",
            "constraint_coverage",
            "irreversible_risk",
        ):
            _probability(name, getattr(self, name))
        _nonnegative("travel_cost", self.travel_cost)
        _nonnegative("return_cost", self.return_cost)
        if self.visits < 0:
            raise ValueError("visits must be non-negative")


@dataclass(frozen=True)
class CheckpointProposal:
    """Sufficient statistics for deciding whether to retain a checkpoint."""

    checkpoint_id: str
    stable_branch_count: int
    stable_observations: int
    need_return_probability: float
    best_alternative_value: float
    recovery_cost_without_checkpoint: float
    irreversible_risk: float
    memory_cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")
        if self.stable_branch_count < 0 or self.stable_observations < 0:
            raise ValueError("branch and observation counts must be non-negative")
        _probability("need_return_probability", self.need_return_probability)
        _probability("irreversible_risk", self.irreversible_risk)
        for name in (
            "best_alternative_value",
            "recovery_cost_without_checkpoint",
            "memory_cost",
        ):
            _nonnegative(name, getattr(self, name))


@dataclass(frozen=True)
class DecisionContext:
    current_checkpoint_id: str
    reveal_belief: RevealBelief
    evidence_complete_probability: float
    last_safe_margin: float
    can_inspect: bool
    goal_found: bool = False

    def __post_init__(self) -> None:
        if not self.current_checkpoint_id:
            raise ValueError("current_checkpoint_id must not be empty")
        _probability(
            "evidence_complete_probability", self.evidence_complete_probability
        )
        _nonnegative("last_safe_margin", self.last_safe_margin)


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    reason: str
    checkpoint_id: Optional[str] = None
    branch_id: Optional[str] = None
    path: tuple[str, ...] = ()

