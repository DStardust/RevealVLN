"""Leakage-closed causal temporal records for MF3ZN-TUAD v1.

Only quantities available at or before the decision may enter these records.
Oracle supervision and treatment outcomes intentionally live in a different
module and cannot be represented by this schema.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np


TEMPORAL_SCHEMA_VERSION = "revealnav-mf3zn-causal-temporal/1"
TEMPORAL_RECORD_LIST_SCHEMA = "revealnav-mf3zn-causal-temporal-record-list/1"
TEMPORAL_RECORD_LIST_STATUS = "SEALED_BEFORE_TREATMENT_OUTCOMES"
PREFIX_HASH_DOMAIN = "revealnav-mf3zn-causal-prefix/1"

CAUSAL_STEP_FIELDS = frozenset({
    "step",
    "native_action_id",
    "candidate_action_ids",
    "policy_features",
    "instruction_embedding",
    "checkpoint_embedding",
    "action_embeddings",
})
CAUSAL_SEQUENCE_FIELDS = frozenset({
    "dataset",
    "scene_id",
    "episode_id",
    "decision_step",
    "steps",
    "prefix_sha256",
})
CAUSAL_RECORD_LIST_FIELDS = frozenset({
    "schema_version",
    "status",
    "source_canonical_identity_sha256",
    "records",
    "public_split_access",
})

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_EXACT_FIELDS = frozenset({
    "target",
    "delta_utility",
    "utility",
    "outcome",
    "catastrophic",
    "catastrophe",
    "navmesh",
    "pose",
    "simulator_pose",
    "treatment_result",
})
_FORBIDDEN_PREFIXES = (
    "target_",
    "future_",
    "oracle_",
    "outcome_",
    "treatment_result_",
)


def _plain_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _readonly_float_array(
    value: object,
    name: str,
    *,
    ndim: int,
) -> np.ndarray:
    """Return an immutable defensive copy backed by immutable bytes."""

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim != ndim or any(size < 1 for size in value.shape):
        raise ValueError(f"{name} must be a non-empty {ndim}-D array")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must use a floating dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")

    contiguous = np.ascontiguousarray(value)
    # An ndarray that merely has write=False can often be made writable again
    # by its owner.  Rebuilding it over immutable bytes makes the public
    # read-only contract fail closed while also providing a defensive copy.
    copied = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=contiguous.dtype,
    ).reshape(contiguous.shape)
    copied.setflags(write=False)
    return copied


def _forbidden_field(name: str) -> bool:
    lowered = name.casefold()
    return (
        lowered in _FORBIDDEN_EXACT_FIELDS
        or lowered.startswith(_FORBIDDEN_PREFIXES)
    )


def _validate_mapping_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    name: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    forbidden = sorted(key for key in value if _forbidden_field(key))
    if forbidden:
        raise ValueError(
            f"{name} contains forbidden causal fields: {forbidden}"
        )
    observed = set(value)
    if observed != allowed:
        missing = sorted(allowed - observed)
        unexpected = sorted(observed - allowed)
        raise ValueError(
            f"{name} schema drift; missing={missing}, unexpected={unexpected}"
        )


@dataclass(frozen=True)
class CausalTemporalStep:
    """One model-visible prefix record.

    ``candidate_action_ids`` is in frozen policy rank order and every row of
    ``action_embeddings`` is aligned to the identifier at the same index.
    ``policy_features`` is a fixed-width per-prefix scalar vector.
    """

    step: int
    native_action_id: str
    candidate_action_ids: tuple[str, ...]
    policy_features: np.ndarray
    instruction_embedding: np.ndarray
    checkpoint_embedding: np.ndarray
    action_embeddings: np.ndarray

    def __post_init__(self) -> None:
        step = _plain_nonnegative_int(self.step, "step")
        native = _nonempty_string(self.native_action_id, "native_action_id")
        try:
            candidates = tuple(self.candidate_action_ids)
        except TypeError as error:
            raise TypeError("candidate_action_ids must be iterable") from error
        if not candidates or any(
            not isinstance(value, str) or not value.strip()
            for value in candidates
        ):
            raise ValueError(
                "candidate_action_ids must contain non-empty strings"
            )
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate_action_ids must be unique")
        if native not in candidates:
            raise ValueError("native action must be in the executable candidates")

        policy = _readonly_float_array(
            self.policy_features, "policy_features", ndim=1,
        )
        instruction = _readonly_float_array(
            self.instruction_embedding, "instruction_embedding", ndim=1,
        )
        checkpoint = _readonly_float_array(
            self.checkpoint_embedding, "checkpoint_embedding", ndim=1,
        )
        actions = _readonly_float_array(
            self.action_embeddings, "action_embeddings", ndim=2,
        )
        if actions.shape[0] != len(candidates):
            raise ValueError(
                "action_embeddings rows must align with candidate_action_ids"
            )

        object.__setattr__(self, "step", step)
        object.__setattr__(self, "native_action_id", native)
        object.__setattr__(self, "candidate_action_ids", candidates)
        object.__setattr__(self, "policy_features", policy)
        object.__setattr__(self, "instruction_embedding", instruction)
        object.__setattr__(self, "checkpoint_embedding", checkpoint)
        object.__setattr__(self, "action_embeddings", actions)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CausalTemporalStep":
        _validate_mapping_fields(value, CAUSAL_STEP_FIELDS, "causal step")
        return cls(**{key: value[key] for key in CAUSAL_STEP_FIELDS})

    @classmethod
    def from_serialized_mapping(
        cls, value: Mapping[str, Any],
    ) -> "CausalTemporalStep":
        """Parse a JSON-compatible step into the one canonical float64 form."""

        _validate_mapping_fields(value, CAUSAL_STEP_FIELDS, "serialized causal step")
        arrays = {}
        for field, ndim in (
            ("policy_features", 1),
            ("instruction_embedding", 1),
            ("checkpoint_embedding", 1),
            ("action_embeddings", 2),
        ):
            try:
                array = np.asarray(value[field], dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"serialized causal step {field} is not numeric"
                ) from error
            if array.ndim != ndim or any(size < 1 for size in array.shape):
                raise ValueError(
                    f"serialized causal step {field} has the wrong shape"
                )
            arrays[field] = array
        return cls(
            step=value["step"],
            native_action_id=value["native_action_id"],
            candidate_action_ids=tuple(value["candidate_action_ids"]),
            **arrays,
        )


def _validated_steps(
    steps: Iterable[CausalTemporalStep],
    decision_step: int,
) -> tuple[CausalTemporalStep, ...]:
    try:
        values = tuple(steps)
    except TypeError as error:
        raise TypeError("steps must be iterable") from error
    if not values or any(
        not isinstance(value, CausalTemporalStep) for value in values
    ):
        raise TypeError("steps must contain CausalTemporalStep values")
    indices = tuple(value.step for value in values)
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise ValueError("causal steps must be strictly increasing")
    if any(index > decision_step for index in indices):
        raise ValueError("causal sequence contains a future step")

    policy_widths = {value.policy_features.shape for value in values}
    instruction_widths = {value.instruction_embedding.shape for value in values}
    checkpoint_widths = {value.checkpoint_embedding.shape for value in values}
    action_widths = {value.action_embeddings.shape[1] for value in values}
    if len(policy_widths) != 1:
        raise ValueError("policy feature width changes within the sequence")
    if len(instruction_widths) != 1:
        raise ValueError("instruction embedding width changes within the sequence")
    if len(checkpoint_widths) != 1:
        raise ValueError("checkpoint embedding width changes within the sequence")
    if len(action_widths) != 1:
        raise ValueError("action embedding width changes within the sequence")
    return values


def _hash_chunk(digest: Any, name: str, payload: bytes) -> None:
    name_bytes = name.encode("utf-8")
    digest.update(len(name_bytes).to_bytes(4, "big"))
    digest.update(name_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _hash_array(digest: Any, name: str, value: np.ndarray) -> None:
    descriptor = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    _hash_chunk(digest, f"{name}.descriptor", descriptor)
    _hash_chunk(digest, f"{name}.bytes", value.tobytes(order="C"))


def causal_prefix_sha256(
    *,
    dataset: str,
    scene_id: str,
    episode_id: str,
    decision_step: int,
    steps: Iterable[CausalTemporalStep],
) -> str:
    """Hash exactly the model-visible causal prefix, never its labels/outcome."""

    dataset = _nonempty_string(dataset, "dataset")
    scene_id = _nonempty_string(scene_id, "scene_id")
    episode_id = _nonempty_string(episode_id, "episode_id")
    decision_step = _plain_nonnegative_int(decision_step, "decision_step")
    values = _validated_steps(steps, decision_step)
    digest = hashlib.sha256()
    _hash_chunk(digest, "domain", PREFIX_HASH_DOMAIN.encode("ascii"))
    metadata = json.dumps(
        {
            "dataset": dataset,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "decision_step": decision_step,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    _hash_chunk(digest, "metadata", metadata)
    for index, step in enumerate(values):
        identity = json.dumps(
            {
                "step": step.step,
                "native_action_id": step.native_action_id,
                "candidate_action_ids": list(step.candidate_action_ids),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        _hash_chunk(digest, f"steps[{index}].identity", identity)
        _hash_array(
            digest, f"steps[{index}].policy_features", step.policy_features,
        )
        _hash_array(
            digest,
            f"steps[{index}].instruction_embedding",
            step.instruction_embedding,
        )
        _hash_array(
            digest,
            f"steps[{index}].checkpoint_embedding",
            step.checkpoint_embedding,
        )
        _hash_array(
            digest, f"steps[{index}].action_embeddings", step.action_embeddings,
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class TemporalSequence:
    """A strictly causal sequence ending no later than ``decision_step``."""

    dataset: str
    scene_id: str
    episode_id: str
    decision_step: int
    steps: tuple[CausalTemporalStep, ...]
    prefix_sha256: str

    def __post_init__(self) -> None:
        dataset = _nonempty_string(self.dataset, "dataset")
        scene = _nonempty_string(self.scene_id, "scene_id")
        episode = _nonempty_string(self.episode_id, "episode_id")
        decision = _plain_nonnegative_int(self.decision_step, "decision_step")
        values = _validated_steps(self.steps, decision)
        if not isinstance(self.prefix_sha256, str):
            raise TypeError("prefix_sha256 must be a string")
        observed = self.prefix_sha256.casefold()
        if _SHA256_PATTERN.fullmatch(observed) is None:
            raise ValueError("prefix_sha256 must be 64 lowercase hex digits")
        expected = causal_prefix_sha256(
            dataset=dataset,
            scene_id=scene,
            episode_id=episode,
            decision_step=decision,
            steps=values,
        )
        if observed != expected:
            raise ValueError("prefix_sha256 does not match the causal inputs")

        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "scene_id", scene)
        object.__setattr__(self, "episode_id", episode)
        object.__setattr__(self, "decision_step", decision)
        object.__setattr__(self, "steps", values)
        object.__setattr__(self, "prefix_sha256", observed)

    @classmethod
    def create(
        cls,
        *,
        dataset: str,
        scene_id: str,
        episode_id: str,
        decision_step: int,
        steps: Iterable[CausalTemporalStep],
    ) -> "TemporalSequence":
        """Create a sequence and compute its content-bound prefix hash."""

        decision = _plain_nonnegative_int(decision_step, "decision_step")
        values = _validated_steps(steps, decision)
        prefix = causal_prefix_sha256(
            dataset=dataset,
            scene_id=scene_id,
            episode_id=episode_id,
            decision_step=decision,
            steps=values,
        )
        return cls(
            dataset=dataset,
            scene_id=scene_id,
            episode_id=episode_id,
            decision_step=decision,
            steps=values,
            prefix_sha256=prefix,
        )

    @classmethod
    def from_trace(
        cls,
        *,
        dataset: str,
        scene_id: str,
        episode_id: str,
        decision_step: int,
        trace_steps: Iterable[CausalTemporalStep],
    ) -> "TemporalSequence":
        """Strictly truncate a longer trace before hashing or tensorization."""

        decision = _plain_nonnegative_int(decision_step, "decision_step")
        try:
            trace = tuple(trace_steps)
        except TypeError as error:
            raise TypeError("trace_steps must be iterable") from error
        if any(not isinstance(value, CausalTemporalStep) for value in trace):
            raise TypeError("trace_steps must contain CausalTemporalStep values")
        prefix = tuple(value for value in trace if value.step <= decision)
        return cls.create(
            dataset=dataset,
            scene_id=scene_id,
            episode_id=episode_id,
            decision_step=decision,
            steps=prefix,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TemporalSequence":
        _validate_mapping_fields(
            value, CAUSAL_SEQUENCE_FIELDS, "causal temporal sequence",
        )
        raw_steps = value["steps"]
        if not isinstance(raw_steps, (tuple, list)):
            raise TypeError("causal temporal sequence steps must be a list/tuple")
        steps = tuple(
            item if isinstance(item, CausalTemporalStep)
            else CausalTemporalStep.from_mapping(item)
            for item in raw_steps
        )
        return cls(
            dataset=value["dataset"],
            scene_id=value["scene_id"],
            episode_id=value["episode_id"],
            decision_step=value["decision_step"],
            steps=steps,
            prefix_sha256=value["prefix_sha256"],
        )

    @classmethod
    def from_serialized_mapping(
        cls, value: Mapping[str, Any],
    ) -> "TemporalSequence":
        """Parse one JSON-compatible record and reverify its causal hash."""

        _validate_mapping_fields(
            value, CAUSAL_SEQUENCE_FIELDS, "serialized causal temporal sequence",
        )
        raw_steps = value["steps"]
        if not isinstance(raw_steps, list) or not raw_steps:
            raise TypeError("serialized causal steps must be a nonempty list")
        return cls(
            dataset=value["dataset"],
            scene_id=value["scene_id"],
            episode_id=value["episode_id"],
            decision_step=value["decision_step"],
            steps=tuple(
                CausalTemporalStep.from_serialized_mapping(item)
                for item in raw_steps
            ),
            prefix_sha256=value["prefix_sha256"],
        )


def temporal_record_list_from_mapping(
    value: Mapping[str, Any],
) -> tuple[tuple[TemporalSequence, ...], str]:
    """Parse the only production causal-record artifact accepted by TUAD."""

    _validate_mapping_fields(value, CAUSAL_RECORD_LIST_FIELDS, "causal record list")
    if value["schema_version"] != TEMPORAL_RECORD_LIST_SCHEMA:
        raise ValueError("causal record-list schema version drift")
    if value["status"] != TEMPORAL_RECORD_LIST_STATUS:
        raise ValueError("causal records were not sealed before treatment outcomes")
    if value["public_split_access"] is not False:
        raise ValueError("causal records report public split access")
    source = value["source_canonical_identity_sha256"]
    if not isinstance(source, str) or _SHA256_PATTERN.fullmatch(source) is None:
        raise ValueError("causal record list lacks a canonical source commitment")
    raw_records = value["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("causal record list is empty")
    records = tuple(
        TemporalSequence.from_serialized_mapping(record)
        for record in raw_records
    )
    identities = [
        (record.dataset, record.scene_id, record.episode_id, record.decision_step)
        for record in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("causal record list repeats a decision identity")
    if any(record.dataset not in {"R2R", "RxR"} for record in records):
        raise ValueError("causal record list contains an unknown dataset")
    return records, source


def causal_array_bytes(value: np.ndarray) -> bytes:
    """Canonical bytes for invariance tests and sealed tensor payloads."""

    if not isinstance(value, np.ndarray):
        raise TypeError("causal_array_bytes expects a numpy.ndarray")
    descriptor = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return len(descriptor).to_bytes(8, "big") + descriptor + value.tobytes(
        order="C"
    )
