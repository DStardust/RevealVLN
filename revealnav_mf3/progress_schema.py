"""Minimal, outcome-blind schemas for the MF3ZV progress support audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Optional


class ProgressFamily(str, Enum):
    ORDINAL = "ORDINAL"
    PASSED_LANDMARK = "PASSED_LANDMARK"


class AtomReviewStatus(str, Enum):
    VALID_PROGRESS_ATOM = "VALID_PROGRESS_ATOM"
    AMBIGUOUS_PROGRESS_ATOM = "AMBIGUOUS_PROGRESS_ATOM"
    NOT_PROGRESS = "NOT_PROGRESS"


class SupportState(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProgressAtom:
    atom_id: str
    family: str
    subject: str
    relation: str
    target_value: str
    instruction_span: str

    def __post_init__(self) -> None:
        if self.family not in {item.value for item in ProgressFamily}:
            raise ValueError(f"unsupported progress family: {self.family}")
        for name in ("atom_id", "subject", "relation", "target_value", "instruction_span"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        expected = {
            ProgressFamily.ORDINAL.value: "COUNT_TARGET",
            ProgressFamily.PASSED_LANDMARK.value: "PASSED",
        }[self.family]
        if self.relation != expected:
            raise ValueError(f"{self.family} requires relation={expected}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrdinalProgress:
    encountered_count: int
    target_count: int

    def __post_init__(self) -> None:
        if self.encountered_count < 0:
            raise ValueError("encountered_count must be non-negative")
        if self.target_count < 1:
            raise ValueError("target_count must be positive")


@dataclass(frozen=True)
class LandmarkProgress:
    seen: Optional[bool]
    passed: Optional[bool]

    def __post_init__(self) -> None:
        if self.passed is True and self.seen is not True:
            raise ValueError("passed=true requires seen=true")


@dataclass(frozen=True)
class ProgressTransition:
    dataset: str
    episode_id: str
    scene_id: str
    atom_id: str
    before_step: int
    after_step: int
    state_before: str
    state_after: str
    evidence_paths: tuple[str, ...] = ()
    review_source: str = "AI_ASSISTED_REVIEW_NOT_HUMAN_GOLD"

    def __post_init__(self) -> None:
        if self.before_step < 0 or self.after_step <= self.before_step:
            raise ValueError("transition must move strictly forward in causal time")
        if not self.state_before or not self.state_after:
            raise ValueError("transition states must be explicit")
        if self.state_before == self.state_after:
            raise ValueError("a transition must change state")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_paths"] = list(self.evidence_paths)
        return data


FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "success",
        "reward",
        "spl",
        "ndtw",
        "sdtw",
        "utility",
        "delta_utility",
        "catastrophe",
        "car_result",
        "final_outcome",
    }
)
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {"val_seen", "val_unseen", "test", "test_challenge"}
)
FORBIDDEN_FUTURE_KEYS = frozenset(
    {"future_frame", "future_candidate", "future_candidates", "later_route_node"}
)


def reject_forbidden_progress_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if an MF3ZV input exposes outcome, public, or future data."""

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            lowered = {str(key).casefold() for key in value}
            forbidden = lowered & (
                FORBIDDEN_OUTCOME_KEYS | FORBIDDEN_PUBLIC_KEYS | FORBIDDEN_FUTURE_KEYS
            )
            if forbidden:
                raise ValueError(f"forbidden MF3ZV fields: {sorted(forbidden)}")
            for nested in value.values():
                walk(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                walk(nested)

    walk(payload)

