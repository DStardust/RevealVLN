"""Public schemas and legal high-level actions for RevealSkill."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class RevealSkillAction(str, Enum):
    FOLLOW = "FOLLOW"
    INSPECT = "INSPECT"
    EXPLORE = "EXPLORE"
    BACKTRACK = "BACKTRACK"
    COMMIT = "COMMIT"
    STOP = "STOP"


class Readiness(str, Enum):
    U = "U"
    A = "A"
    D = "D"


@dataclass(frozen=True)
class RevealSkillState:
    step: int
    checkpoint_id: str
    readiness_by_option: Mapping[str, Readiness | str]
    option_ids: tuple[str, ...]
    active_frontier: tuple[str, ...]
    reveal_after_expiry: bool

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")
        options = tuple(str(value) for value in self.option_ids)
        if len(options) != len(set(options)) or any(not value for value in options):
            raise ValueError("option_ids must be unique non-empty strings")
        readiness = {str(key): Readiness(value) for key, value in self.readiness_by_option.items()}
        if set(readiness) - set(options):
            raise ValueError("readiness contains an unknown option")
        frontier = tuple(str(value) for value in self.active_frontier)
        if len(frontier) != len(set(frontier)) or any(not value for value in frontier):
            raise ValueError("active_frontier must contain unique non-empty IDs")
        object.__setattr__(self, "option_ids", options)
        object.__setattr__(self, "readiness_by_option", readiness)
        object.__setattr__(self, "active_frontier", frontier)
        object.__setattr__(self, "reveal_after_expiry", bool(self.reveal_after_expiry))


def reject_forbidden_state_mapping(value: Mapping[str, object]) -> None:
    forbidden = {"target", "delta_utility", "reward", "success", "spl", "ndtw", "sdtw", "outcome", "future", "oracle", "pose", "navmesh"}
    for key in value:
        lowered = str(key).casefold()
        if lowered in forbidden or lowered.startswith(("future_", "outcome_", "oracle_", "treatment_")):
            raise ValueError(f"forbidden state field: {key}")


__all__ = ["Readiness", "RevealSkillAction", "RevealSkillState", "reject_forbidden_state_mapping"]
