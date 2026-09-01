"""Fail-closed schemas for MF3ZR evidence--option support.

MF3ZR is a data/observation-support revision.  These objects deliberately do
not contain a target, reward, route truth, or a correct-action label.  An
``UNRESOLVED`` edge is a representation of uncertainty, not an inferred
semantic match; an edge is usable for support only after an independent
review/verification explicitly marks it as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from collections.abc import Mapping, Sequence


OPTION_BINDING_SCHEMA = "revealnav-mf3zr-option-evidence-binding/1"
OPTION_IDENTITY_SCHEMA = "revealnav-mf3zr-option-identity/1"


class BindingState(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    UNRESOLVED = "UNRESOLVED"
    SHARED_CONTEXT = "SHARED_CONTEXT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


VALID_BINDING_STATES = frozenset(item.value for item in BindingState)

# These names are rejected recursively before a source/review payload reaches
# a support builder.  In particular, a free-form note cannot smuggle a target
# or a metric into what is supposed to be an outcome-blind artifact.
FORBIDDEN_BINDING_KEYS = frozenset({
    "reward", "success", "spl", "ndtw", "sdtw", "ne", "utility",
    "delta", "delta_utility", "outcome", "outcomes", "catastrophe",
    "catastrophic", "target", "correct_action", "best_action",
    "route_truth", "shortest_path", "navmesh", "pose", "simulator_pose",
    "car_result", "rcsp_result", "dsr_result", "ree_result",
    "prediction", "model_prediction", "qwen_result", "qwen_prediction",
})


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def reject_forbidden_binding_payload(value: object, *, path: str = "$") -> None:
    """Reject outcome/route-truth fields recursively.

    This intentionally treats suspicious field *names* as a hard error.  It
    is safer to reject a future annotation extension than to silently accept
    a leaked label.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if (
                key in FORBIDDEN_BINDING_KEYS
                or key.startswith(("future_", "outcome_", "reward_", "target_", "oracle_", "treatment_"))
                or key.endswith(("_outcome", "_reward", "_metric"))
            ):
                raise ValueError(f"forbidden MF3ZR field at {path}.{raw_key}")
            reject_forbidden_binding_payload(child, path=f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_forbidden_binding_payload(child, path=f"{path}[{index}]")


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _ids(value: Sequence[object] | tuple[object, ...], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(_nonempty(item, name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique values")
    return result


def _indices(value: Sequence[object], name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{name} must contain non-negative integers")
        result.append(int(item))
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique indices")
    return tuple(result)


@dataclass(frozen=True)
class OptionEvidenceBinding:
    """One causal prefix/constraint/option edge.

    ``verified`` defaults to false.  This prevents a machine-created review
    row from accidentally becoming support evidence merely because it has a
    syntactically valid state.
    """

    event_id: str
    prefix_step: int
    option_id: str
    candidate_id: str
    candidate_rank: int
    constraint_id: str
    binding_state: BindingState | str
    is_contextual: bool
    is_discriminative: bool
    evidence_image_indices: tuple[int, ...]
    evidence_ids: tuple[str, ...]
    source_sha256: str
    verified: bool = False
    verification_source: str = "UNVERIFIED"

    def __post_init__(self) -> None:
        for value, name in (
            (self.event_id, "event_id"),
            (self.option_id, "option_id"),
            (self.candidate_id, "candidate_id"),
            (self.constraint_id, "constraint_id"),
        ):
            _nonempty(value, name)
        if isinstance(self.prefix_step, bool) or not isinstance(self.prefix_step, int) or self.prefix_step < 0:
            raise ValueError("prefix_step must be a non-negative integer")
        if isinstance(self.candidate_rank, bool) or not isinstance(self.candidate_rank, int) or self.candidate_rank < 0:
            raise ValueError("candidate_rank must be a non-negative integer")
        state = self.binding_state if isinstance(self.binding_state, BindingState) else BindingState(self.binding_state)
        if type(self.is_contextual) is not bool or type(self.is_discriminative) is not bool:
            raise TypeError("binding role flags must be Boolean")
        if self.is_contextual and self.is_discriminative:
            raise ValueError("a binding cannot be both contextual and discriminative")
        if not is_sha256(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256")
        if type(self.verified) is not bool:
            raise TypeError("verified must be Boolean")
        _nonempty(self.verification_source, "verification_source")
        object.__setattr__(self, "binding_state", state)
        object.__setattr__(self, "evidence_image_indices", _indices(self.evidence_image_indices, "evidence_image_indices"))
        object.__setattr__(self, "evidence_ids", _ids(self.evidence_ids, "evidence_ids"))

    @property
    def usable(self) -> bool:
        """Whether an independently verified edge may support a computation."""

        return bool(self.verified and self.verification_source not in {"UNVERIFIED", "PENDING"})

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": OPTION_BINDING_SCHEMA,
            "event_id": self.event_id,
            "prefix_step": self.prefix_step,
            "option_id": self.option_id,
            "candidate_id": self.candidate_id,
            "candidate_rank": self.candidate_rank,
            "constraint_id": self.constraint_id,
            "binding_state": self.binding_state.value,
            "is_contextual": self.is_contextual,
            "is_discriminative": self.is_discriminative,
            "evidence_image_indices": list(self.evidence_image_indices),
            "evidence_ids": list(self.evidence_ids),
            "source_sha256": self.source_sha256,
            "verified": self.verified,
            "verification_source": self.verification_source,
        }


@dataclass(frozen=True)
class OptionIdentity:
    """Causal identity of an opaque executable candidate."""

    option_id: str
    event_id: str
    candidate_id: str
    first_seen_step: int
    last_seen_step: int
    anchor_checkpoint_id: str
    identity_status: str = "CAUSAL_OPAQUE_CANDIDATE"

    def __post_init__(self) -> None:
        for value, name in (
            (self.option_id, "option_id"),
            (self.event_id, "event_id"),
            (self.candidate_id, "candidate_id"),
            (self.anchor_checkpoint_id, "anchor_checkpoint_id"),
            (self.identity_status, "identity_status"),
        ):
            _nonempty(value, name)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (self.first_seen_step, self.last_seen_step)):
            raise ValueError("option identity steps must be non-negative integers")
        if self.first_seen_step > self.last_seen_step:
            raise ValueError("option identity interval is invalid")

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": OPTION_IDENTITY_SCHEMA,
            "option_id": self.option_id,
            "event_id": self.event_id,
            "candidate_id": self.candidate_id,
            "first_seen_step": self.first_seen_step,
            "last_seen_step": self.last_seen_step,
            "anchor_checkpoint_id": self.anchor_checkpoint_id,
            "identity_status": self.identity_status,
        }


def deterministic_option_id(event_id: str, first_seen_step: int, candidate_id: str) -> str:
    """Return the only option ID formula permitted by MF3ZR."""

    _nonempty(event_id, "event_id")
    _nonempty(candidate_id, "candidate_id")
    if isinstance(first_seen_step, bool) or not isinstance(first_seen_step, int) or first_seen_step < 0:
        raise ValueError("first_seen_step must be a non-negative integer")
    return _sha256({
        "event_id": event_id,
        "first_seen_step": int(first_seen_step),
        "candidate_id": candidate_id,
    })


def source_commitment(value: object) -> str:
    """Hash a safe, outcome-free source payload."""

    reject_forbidden_binding_payload(value)
    return _sha256(value)


__all__ = [
    "OPTION_BINDING_SCHEMA", "OPTION_IDENTITY_SCHEMA", "BindingState",
    "VALID_BINDING_STATES", "OptionEvidenceBinding", "OptionIdentity",
    "deterministic_option_id", "is_sha256", "reject_forbidden_binding_payload",
    "source_commitment",
]
