"""Strictly causal records and physically separate oracle labels for MF3ZO.

The pilot deliberately permits unavailable historical fields.  Missing values
are represented by ``None`` plus an explicit availability declaration; they
are never imputed from outcomes, future observations, geometry, or oracle
state.  Inference tensors can only be constructed from ``CausalTemporalRecord``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Iterable, Mapping

import numpy as np


EMBEDDING_DIM = 768
POLICY_FEATURE_DIM = 10
UAD_STABILITY_PREFIXES = 3
ORACLE_FIELDS = (
    "target_in_set",
    "candidate_separated",
    "evidence_closed",
    "reveal_interval",
    "expiry_step",
    "resolvable",
)
FORBIDDEN_INFERENCE_KEYS = frozenset({
    "target",
    "delta_utility",
    "counterfactual_outcome",
    "treatment_result",
    "navmesh",
    "pose",
    "oracle",
    "oracle_label",
})


class UADState(str, Enum):
    UNOBSERVED = "U"
    AMBIGUOUS = "A"
    DECISIVE = "D"


def _readonly_array(
    value: np.ndarray | Iterable[float], shape: tuple[int, ...], name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    result = np.array(array, dtype=np.float32, copy=True)
    result.flags.writeable = False
    return result


def _identity_tuple(
    values: Iterable[object], name: str, *, allow_empty: bool = False,
) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (
        (not result and not allow_empty)
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain unique nonempty identities")
    return result


@dataclass(frozen=True)
class CausalTemporalStep:
    """One prefix step containing only deployment-observable information."""

    step: int
    native_action_id: str | None
    candidate_action_ids: tuple[str, ...]
    policy_features: np.ndarray
    policy_feature_mask: np.ndarray
    instruction_embedding: np.ndarray
    checkpoint_embedding: np.ndarray | None
    embedded_action_ids: tuple[str, ...]
    action_embeddings: np.ndarray | None

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if self.native_action_id is not None and (
            not isinstance(self.native_action_id, str) or not self.native_action_id
        ):
            raise ValueError("native_action_id must be nonempty when available")
        candidates = _identity_tuple(
            self.candidate_action_ids,
            "candidate_action_ids",
            allow_empty=True,
        )
        policy = _readonly_array(
            self.policy_features, (POLICY_FEATURE_DIM,), "policy_features"
        )
        policy_mask = np.asarray(self.policy_feature_mask)
        if policy_mask.dtype != np.bool_ or policy_mask.shape != (POLICY_FEATURE_DIM,):
            raise ValueError("policy_feature_mask must be Boolean with shape (10,)")
        policy_mask = np.array(policy_mask, dtype=np.bool_, copy=True)
        policy_mask.flags.writeable = False
        instruction = _readonly_array(
            self.instruction_embedding,
            (EMBEDDING_DIM,),
            "instruction_embedding",
        )
        embedded_ids = tuple(str(value) for value in self.embedded_action_ids)
        if any(not value for value in embedded_ids) or len(set(embedded_ids)) != len(
            embedded_ids
        ):
            raise ValueError("embedded_action_ids must be unique and nonempty")
        if not set(embedded_ids).issubset(candidates):
            raise ValueError("embedded actions must be executable candidates")
        if self.checkpoint_embedding is None:
            checkpoint = None
        else:
            checkpoint = _readonly_array(
                self.checkpoint_embedding,
                (EMBEDDING_DIM,),
                "checkpoint_embedding",
            )
        if self.action_embeddings is None:
            if embedded_ids:
                raise ValueError("embedded action IDs require action embeddings")
            actions = None
        else:
            actions = _readonly_array(
                self.action_embeddings,
                (len(embedded_ids), EMBEDDING_DIM),
                "action_embeddings",
            )
        object.__setattr__(self, "candidate_action_ids", candidates)
        object.__setattr__(self, "policy_features", policy)
        object.__setattr__(self, "policy_feature_mask", policy_mask)
        object.__setattr__(self, "instruction_embedding", instruction)
        object.__setattr__(self, "checkpoint_embedding", checkpoint)
        object.__setattr__(self, "embedded_action_ids", embedded_ids)
        object.__setattr__(self, "action_embeddings", actions)

    @property
    def embedding_complete(self) -> bool:
        return (
            self.checkpoint_embedding is not None
            and self.action_embeddings is not None
            and set(self.embedded_action_ids) == set(self.candidate_action_ids)
        )


def _array_digest(digest: "hashlib._Hash", value: np.ndarray | None) -> None:
    if value is None:
        digest.update(b"none\0")
        return
    contiguous = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    digest.update(contiguous.tobytes(order="C"))


def causal_prefix_sha256(
    dataset: str,
    scene_id: str,
    episode_id: str,
    decision_step: int,
    steps: Iterable[CausalTemporalStep],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"mf3zo-causal-temporal-record/1\0")
    metadata = {
        "dataset": dataset,
        "scene_id": scene_id,
        "episode_id": episode_id,
        "decision_step": decision_step,
    }
    digest.update(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii"))
    for value in steps:
        step_metadata = {
            "step": value.step,
            "native_action_id": value.native_action_id,
            "candidate_action_ids": value.candidate_action_ids,
            "embedded_action_ids": value.embedded_action_ids,
        }
        encoded = json.dumps(
            step_metadata, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        _array_digest(digest, value.policy_features)
        _array_digest(digest, value.policy_feature_mask)
        _array_digest(digest, value.instruction_embedding)
        _array_digest(digest, value.checkpoint_embedding)
        _array_digest(digest, value.action_embeddings)
    return digest.hexdigest()


@dataclass(frozen=True)
class CausalTemporalRecord:
    dataset: str
    scene_id: str
    episode_id: str
    decision_step: int
    steps: tuple[CausalTemporalStep, ...]
    prefix_sha256: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.dataset, self.scene_id, self.episode_id)
        ):
            raise ValueError("record identity fields must be nonempty")
        if isinstance(self.decision_step, bool) or not isinstance(
            self.decision_step, int
        ) or self.decision_step < 0:
            raise ValueError("decision_step must be a non-negative integer")
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("causal temporal record must have at least one step")
        indices = tuple(value.step for value in steps)
        if any(left >= right for left, right in zip(indices, indices[1:])):
            raise ValueError("causal steps must be strictly increasing")
        if any(value > self.decision_step for value in indices):
            raise ValueError("future step entered causal temporal record")
        if indices[-1] != self.decision_step:
            raise ValueError("causal record must end at decision_step")
        expected = causal_prefix_sha256(
            self.dataset,
            self.scene_id,
            self.episode_id,
            self.decision_step,
            steps,
        )
        if self.prefix_sha256 != expected:
            raise ValueError("causal prefix hash mismatch")
        object.__setattr__(self, "steps", steps)

    @property
    def full_prefix_embedding_complete(self) -> bool:
        return all(value.embedding_complete for value in self.steps)


@dataclass(frozen=True)
class TemporalOracleLabel:
    """Oracle-only supervision; ``None`` means explicitly unavailable."""

    event_id: str
    target_in_set: tuple[bool, ...] | None
    candidate_separated: tuple[bool, ...] | None
    evidence_closed: tuple[bool, ...] | None
    reveal_interval: tuple[int, int] | None
    expiry_step: int | None
    resolvable: bool | None
    unavailable_fields: tuple[str, ...]
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("oracle event_id must be nonempty")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("oracle provenance must be explicit")
        unavailable = tuple(str(value) for value in self.unavailable_fields)
        if any(value not in ORACLE_FIELDS for value in unavailable) or len(
            set(unavailable)
        ) != len(unavailable):
            raise ValueError("invalid unavailable oracle field declaration")
        sequences = (
            self.target_in_set,
            self.candidate_separated,
            self.evidence_closed,
        )
        lengths: list[int] = []
        for value, name in zip(
            sequences,
            ORACLE_FIELDS[:3],
            strict=True,
        ):
            if value is None:
                if name not in unavailable:
                    raise ValueError(f"missing oracle field not declared: {name}")
                continue
            if name in unavailable:
                raise ValueError(f"available oracle field declared unavailable: {name}")
            if not value or any(type(item) is not bool for item in value):
                raise ValueError(f"{name} must be a nonempty Boolean tuple")
            lengths.append(len(value))
        if lengths and len(set(lengths)) != 1:
            raise ValueError("oracle factor sequences must have equal lengths")
        if self.reveal_interval is None:
            if "reveal_interval" not in unavailable:
                raise ValueError("missing reveal_interval not declared unavailable")
        else:
            if "reveal_interval" in unavailable:
                raise ValueError("available reveal_interval declared unavailable")
            left, right = self.reveal_interval
            if any(type(value) is not int or value < 0 for value in (left, right)) or left > right:
                raise ValueError("invalid reveal interval")
        if self.expiry_step is None:
            if "expiry_step" not in unavailable:
                raise ValueError("missing expiry_step not declared unavailable")
        elif (
            "expiry_step" in unavailable
            or type(self.expiry_step) is not int
            or self.expiry_step < 0
        ):
            raise ValueError("invalid expiry_step")
        if self.resolvable is None:
            if "resolvable" not in unavailable:
                raise ValueError("missing resolvable not declared unavailable")
        elif "resolvable" in unavailable or type(self.resolvable) is not bool:
            raise ValueError("invalid resolvable")
        object.__setattr__(self, "unavailable_fields", unavailable)

    @property
    def complete(self) -> bool:
        return not self.unavailable_fields


def derive_uad(
    target_in_set: Iterable[bool],
    candidate_separated: Iterable[bool],
    evidence_closed: Iterable[bool],
    *,
    stability_prefixes: int = UAD_STABILITY_PREFIXES,
) -> tuple[UADState, ...]:
    """Derive U/A/D without learning or redefining the frozen semantics."""

    if stability_prefixes != UAD_STABILITY_PREFIXES:
        raise ValueError("MF3ZO UAD stability is frozen at three prefixes")
    factors = tuple(tuple(value) for value in (
        target_in_set, candidate_separated, evidence_closed,
    ))
    if not factors[0] or len({len(value) for value in factors}) != 1:
        raise ValueError("UAD factors must be aligned and nonempty")
    if any(any(type(item) is not bool for item in value) for value in factors):
        raise TypeError("UAD factors must be Boolean")
    in_set, separated, evidence = factors
    if any((separated[index] or evidence[index]) and not in_set[index] for index in range(len(in_set))):
        raise ValueError("separation/evidence cannot precede target presence")
    stable = 0
    result: list[UADState] = []
    for present, distinct, closed in zip(in_set, separated, evidence, strict=True):
        if not present:
            stable = 0
            result.append(UADState.UNOBSERVED)
        elif not distinct or not closed:
            stable = 0
            result.append(UADState.AMBIGUOUS)
        else:
            stable += 1
            result.append(
                UADState.DECISIVE
                if stable >= UAD_STABILITY_PREFIXES
                else UADState.AMBIGUOUS
            )
    return tuple(result)


def inference_tensors(record: CausalTemporalRecord) -> Mapping[str, np.ndarray]:
    """Return the complete causal tensor set and explicit availability masks."""

    rows = len(record.steps)
    policy = np.stack([value.policy_features for value in record.steps])
    policy_mask = np.stack([value.policy_feature_mask for value in record.steps])
    instruction = np.stack([value.instruction_embedding for value in record.steps])
    checkpoint = np.zeros((rows, EMBEDDING_DIM), dtype=np.float32)
    checkpoint_mask = np.zeros(rows, dtype=np.bool_)
    candidate_count = max(len(value.candidate_action_ids) for value in record.steps)
    actions = np.zeros((rows, candidate_count, EMBEDDING_DIM), dtype=np.float32)
    action_mask = np.zeros((rows, candidate_count), dtype=np.bool_)
    for row, value in enumerate(record.steps):
        if value.checkpoint_embedding is not None:
            checkpoint[row] = value.checkpoint_embedding
            checkpoint_mask[row] = True
        if value.action_embeddings is not None:
            indices = {
                action_id: index
                for index, action_id in enumerate(value.candidate_action_ids)
            }
            for action_id, embedding in zip(
                value.embedded_action_ids, value.action_embeddings, strict=True,
            ):
                column = indices[action_id]
                actions[row, column] = embedding
                action_mask[row, column] = True
    return {
        "policy_features": policy,
        "policy_feature_mask": policy_mask,
        "instruction_embedding": instruction,
        "checkpoint_embedding": checkpoint,
        "checkpoint_embedding_mask": checkpoint_mask,
        "action_embeddings": actions,
        "action_embedding_mask": action_mask,
    }


def reject_forbidden_inference_mapping(value: Mapping[str, object]) -> None:
    for key in value:
        lower = str(key).lower()
        if lower in FORBIDDEN_INFERENCE_KEYS or lower.startswith(("future_", "oracle_")):
            raise ValueError(f"forbidden inference field: {key}")


__all__ = [
    "CausalTemporalRecord",
    "CausalTemporalStep",
    "EMBEDDING_DIM",
    "FORBIDDEN_INFERENCE_KEYS",
    "ORACLE_FIELDS",
    "POLICY_FEATURE_DIM",
    "TemporalOracleLabel",
    "UAD_STABILITY_PREFIXES",
    "UADState",
    "causal_prefix_sha256",
    "derive_uad",
    "inference_tensors",
    "reject_forbidden_inference_mapping",
]
