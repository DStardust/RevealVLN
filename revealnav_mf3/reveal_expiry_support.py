"""Support-only Reveal/Expiry derivations for MF3ZR.

No policy is executed here.  Reveal can be derived only from independently
verified option bindings and factor states; Expiry can be derived only from a
real frozen-controller returnability trace.  Missing evidence is represented
explicitly as ``*_NOT_COMPUTABLE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .option_binding_schema import OptionEvidenceBinding


K_STABILITY = 3


class RevealSupportStatus(str, Enum):
    REVEAL_OBSERVED = "REVEAL_OBSERVED"
    REVEAL_INTERVAL_CENSORED = "REVEAL_INTERVAL_CENSORED"
    REVEAL_NOT_COMPUTABLE = "REVEAL_NOT_COMPUTABLE"


class ExpirySupportStatus(str, Enum):
    EXPIRY_OBSERVED = "EXPIRY_OBSERVED"
    EXPIRY_RIGHT_CENSORED = "EXPIRY_RIGHT_CENSORED"
    EXPIRY_NOT_COMPUTABLE = "EXPIRY_NOT_COMPUTABLE"


@dataclass(frozen=True)
class RevealSupport:
    option_id: str
    status: RevealSupportStatus | str
    reveal_step: int | None
    interval: tuple[int, int] | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.option_id, str) or not self.option_id:
            raise ValueError("reveal option_id is required")
        status = self.status if isinstance(self.status, RevealSupportStatus) else RevealSupportStatus(self.status)
        if self.reveal_step is not None and (isinstance(self.reveal_step, bool) or not isinstance(self.reveal_step, int) or self.reveal_step < 0):
            raise ValueError("reveal_step is invalid")
        if self.interval is not None:
            if len(self.interval) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in self.interval) or self.interval[0] > self.interval[1]:
                raise ValueError("reveal interval is invalid")
        if status is RevealSupportStatus.REVEAL_OBSERVED and self.reveal_step is None:
            raise ValueError("observed reveal needs a step")
        if status is RevealSupportStatus.REVEAL_INTERVAL_CENSORED and self.interval is None:
            raise ValueError("censored reveal needs an interval")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reveal reason is required")
        object.__setattr__(self, "status", status)

    def as_mapping(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "status": self.status.value,
            "reveal_step": self.reveal_step,
            "interval": list(self.interval) if self.interval is not None else None,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ExpirySupport:
    option_id: str
    status: ExpirySupportStatus | str
    expiry_step: int | None
    last_observed_step: int | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.option_id, str) or not self.option_id:
            raise ValueError("expiry option_id is required")
        status = self.status if isinstance(self.status, ExpirySupportStatus) else ExpirySupportStatus(self.status)
        for value, name in ((self.expiry_step, "expiry_step"), (self.last_observed_step, "last_observed_step")):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} is invalid")
        if status is ExpirySupportStatus.EXPIRY_OBSERVED and self.expiry_step is None:
            raise ValueError("observed expiry needs a step")
        if status is ExpirySupportStatus.EXPIRY_RIGHT_CENSORED and self.last_observed_step is None:
            raise ValueError("right-censored expiry needs a last step")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("expiry reason is required")
        object.__setattr__(self, "status", status)

    def as_mapping(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "status": self.status.value,
            "expiry_step": self.expiry_step,
            "last_observed_step": self.last_observed_step,
            "reason": self.reason,
        }


def _all_verified_for_option(bindings: Sequence[OptionEvidenceBinding], option_id: str, required: Iterable[str]) -> bool:
    by_constraint = {str(cid): False for cid in required}
    for edge in bindings:
        if edge.option_id == option_id and edge.constraint_id in by_constraint and edge.usable:
            by_constraint[edge.constraint_id] = True
    return bool(by_constraint) and all(by_constraint.values())


def derive_reveal_support(
    *,
    option_id: str,
    decisive_constraint_ids: Sequence[str],
    prerequisite_ids: Sequence[str],
    bindings: Sequence[OptionEvidenceBinding],
    decisive_states: Mapping[str, Sequence[str]] | None,
    prerequisite_satisfaction: Mapping[str, Sequence[bool]] | None,
    stability_k: int = K_STABILITY,
) -> RevealSupport:
    """Derive first hard-D only after option-specific binding is verified."""

    if stability_k != K_STABILITY:
        raise ValueError("MF3ZR K is fixed at 3")
    if not _all_verified_for_option(bindings, option_id, (*decisive_constraint_ids, *prerequisite_ids)):
        return RevealSupport(option_id, RevealSupportStatus.REVEAL_NOT_COMPUTABLE, None, None, "OPTION_BINDING_NOT_VERIFIED")
    if decisive_states is None or prerequisite_satisfaction is None:
        return RevealSupport(option_id, RevealSupportStatus.REVEAL_NOT_COMPUTABLE, None, None, "STATE_SEQUENCE_UNAVAILABLE")
    if not decisive_constraint_ids:
        return RevealSupport(option_id, RevealSupportStatus.REVEAL_NOT_COMPUTABLE, None, None, "NO_DECISIVE_CONSTRAINTS")
    lengths = [len(decisive_states.get(cid, ())) for cid in decisive_constraint_ids]
    lengths += [len(prerequisite_satisfaction.get(cid, ())) for cid in prerequisite_ids]
    if not lengths or len(set(lengths)) != 1:
        return RevealSupport(option_id, RevealSupportStatus.REVEAL_NOT_COMPUTABLE, None, None, "MISALIGNED_STATE_SEQUENCE")
    for step in range(lengths[0]):
        if not all(str(decisive_states[cid][step]) == "D" for cid in decisive_constraint_ids):
            continue
        if not all(bool(prerequisite_satisfaction[cid][step]) for cid in prerequisite_ids):
            continue
        # The state sequences are expected to have already applied K=3 to each
        # constraint; no post-hoc interval selection is performed here.
        return RevealSupport(option_id, RevealSupportStatus.REVEAL_OBSERVED, step, None, "VERIFIED_DEC_AND_PREREQUISITE_STATE")
    return RevealSupport(option_id, RevealSupportStatus.REVEAL_INTERVAL_CENSORED, None, (lengths[0], lengths[0]), "NO_REVEAL_WITHIN_CAUSAL_WINDOW")


def derive_expiry_support(
    option_id: str,
    returnability: Sequence[Mapping[str, object]],
) -> ExpirySupport:
    """Derive expiry from a monotone control witness or right-censor it."""

    if not returnability:
        return ExpirySupport(option_id, ExpirySupportStatus.EXPIRY_NOT_COMPUTABLE, None, None, "RETURNABILITY_SEQUENCE_EMPTY")
    rows = sorted(returnability, key=lambda row: int(row["from_step"]))
    steps = [int(row["from_step"]) for row in rows]
    if steps != list(range(steps[0], steps[-1] + 1)):
        return ExpirySupport(option_id, ExpirySupportStatus.EXPIRY_NOT_COMPUTABLE, None, None, "RETURNABILITY_SEQUENCE_NOT_CONTIGUOUS")
    statuses = [str(row.get("status", "")) for row in rows]
    available = {"RETURNABLE"}
    if any(status in {"EXECUTION_UNAVAILABLE", "INVALID_ANCHOR"} for status in statuses):
        return ExpirySupport(option_id, ExpirySupportStatus.EXPIRY_NOT_COMPUTABLE, None, None, "CONTROL_RETURNABILITY_UNAVAILABLE")
    true_positions = [index for index, status in enumerate(statuses) if status in available]
    if not true_positions:
        return ExpirySupport(option_id, ExpirySupportStatus.EXPIRY_NOT_COMPUTABLE, None, None, "NO_RETURNABLE_PREFIX")
    last_true = true_positions[-1]
    if last_true < len(statuses) - 1 and all(status not in available for status in statuses[last_true + 1:]):
        return ExpirySupport(option_id, ExpirySupportStatus.EXPIRY_OBSERVED, steps[last_true], None, "RETURNABLE_TO_NOT_RETURNABLE_TRANSITION")
    return ExpirySupport(option_id, ExpirySupportStatus.EXPIRY_RIGHT_CENSORED, None, steps[-1], "RETURNABLE_AT_OBSERVATION_END")


def reveal_expiry_status(reveal: RevealSupport, expiry: ExpirySupport) -> tuple[str, int | None]:
    if reveal.reveal_step is None or expiry.expiry_step is None:
        return "UNKNOWN", None
    delta = expiry.expiry_step - reveal.reveal_step
    if delta > 0:
        return "POSITIVE_SLACK", delta
    if delta == 0:
        return "TIGHT", delta
    return "UNRESOLVABLE", delta


__all__ = [
    "K_STABILITY", "RevealSupportStatus", "ExpirySupportStatus",
    "RevealSupport", "ExpirySupport", "derive_reveal_support",
    "derive_expiry_support", "reveal_expiry_status",
]
