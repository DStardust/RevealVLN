"""Outcome-blind exact action-lattice sealing for MF3ZN-TUAD v1.

This module is intentionally independent of the historical MF3ZL runner-up
replay code.  MF3ZN seals the native action plus at most two executable
non-native actions before any treatment outcome is available.  The validators
below bind that action list to a physical native prefix, fail closed on action
ID/index drift, and keep every arm of an event in one episode and scene fold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real

from revealnav_mf3.shadow import validate_action_identity


SNAPSHOT_SCHEMA = "revealnav-mf3zn-causal-action-snapshot/1"
SEAL_SCHEMA = "revealnav-mf3zn-temporal-exact-action-lattice/1"
PHYSICAL_PREFIX_FIELDS = (
    "act",
    "ghost_vp",
    "cur_vp",
    "front_vp",
    "back_path_len",
)

_SNAPSHOT_FIELDS = frozenset({
    "dataset",
    "scene_id",
    "episode_id",
    "decision_step",
    "native_action_id",
    "global_action_ids",
    "executable_action_indices",
    "policy_scores",
    "native_prefix_sha256",
})
_OUTCOME_FIELDS = frozenset({
    "target",
    "delta",
    "delta_utility",
    "utility",
    "catastrophic",
    "reward",
    "outcome",
    "outcomes",
    "label",
    "labels",
    "success",
    "spl",
    "ndtw",
    "sdtw",
    "baseline_metrics",
    "treatment_metrics",
    "treatment_result",
})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_forbidden_snapshot_key(value: object) -> bool:
    if not isinstance(value, str):
        return True
    key = value.lower()
    return (
        key in _OUTCOME_FIELDS
        or key.startswith("future_")
        or key.startswith("oracle_")
        or key.endswith("_outcome")
        or key.endswith("_reward")
    )


def _tuple_value(value: object, name: str) -> tuple:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence, not text")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence") from error


@dataclass(frozen=True)
class CausalActionSnapshot:
    """The outcome-free target-step state used to seal an action lattice."""

    dataset: str
    scene_id: str
    episode_id: str
    decision_step: int
    native_action_id: str
    global_action_ids: tuple[str, ...]
    executable_action_indices: tuple[int, ...]
    policy_scores: tuple[float, ...]
    native_prefix_sha256: str

    def __post_init__(self) -> None:
        if self.dataset not in {"RxR", "R2R"}:
            raise ValueError("causal action snapshot dataset drift")
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("causal action snapshot requires a scene ID")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("causal action snapshot requires an episode ID")
        if (
            not isinstance(self.decision_step, Integral)
            or isinstance(self.decision_step, bool)
            or int(self.decision_step) < 0
        ):
            raise ValueError("causal action snapshot decision step is invalid")
        object.__setattr__(self, "decision_step", int(self.decision_step))
        if not isinstance(self.native_action_id, str) or not self.native_action_id:
            raise ValueError("causal action snapshot requires a native action ID")
        if not _is_sha256(self.native_prefix_sha256):
            raise ValueError("native prefix must be a lowercase SHA-256 digest")

        if not isinstance(self.global_action_ids, tuple) or not self.global_action_ids:
            raise ValueError("global action IDs must be a non-empty tuple")
        if any(not isinstance(value, str) or not value for value in self.global_action_ids):
            raise ValueError("global action IDs must be non-empty strings")
        if len(self.global_action_ids) != len(set(self.global_action_ids)):
            raise ValueError("global action IDs must be unique")

        if not isinstance(self.executable_action_indices, tuple):
            raise ValueError("executable action indices must be a tuple")
        indices: list[int] = []
        for value in self.executable_action_indices:
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise ValueError("executable action indices must be integers")
            indices.append(int(value))
        object.__setattr__(self, "executable_action_indices", tuple(indices))

        if not isinstance(self.policy_scores, tuple):
            raise ValueError("policy scores must be a tuple")
        scores: list[float] = []
        for value in self.policy_scores:
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError("policy scores must be finite real values")
            scores.append(float(value))
        object.__setattr__(self, "policy_scores", tuple(scores))
        if len(scores) != len(self.global_action_ids):
            raise ValueError("policy scores must align with global action IDs")

        try:
            native_index = self.global_action_ids.index(self.native_action_id)
        except ValueError as error:
            raise ValueError("native action is absent from the global action set") from error
        validate_action_identity(
            self.global_action_ids,
            self.executable_action_indices,
            native_index,
            declared_native_id=self.native_action_id,
            require_non_stop=True,
        )
        if not any(
            index != native_index for index in self.executable_action_indices
        ):
            raise ValueError("action lattice requires an executable non-native action")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CausalActionSnapshot":
        """Parse a strict mapping without silently dropping outcome fields."""

        if not isinstance(value, Mapping):
            raise TypeError("causal action snapshot must be a mapping")
        forbidden = sorted(
            str(key) for key in value if _is_forbidden_snapshot_key(key)
        )
        if forbidden:
            raise ValueError(
                "causal action snapshot contains forbidden outcome/oracle fields: "
                + ", ".join(forbidden)
            )
        keys = set(value)
        if keys != _SNAPSHOT_FIELDS:
            missing = sorted(_SNAPSHOT_FIELDS - keys)
            extra = sorted(str(key) for key in keys - _SNAPSHOT_FIELDS)
            raise ValueError(
                f"causal action snapshot field drift; missing={missing}, extra={extra}"
            )
        return cls(
            dataset=value["dataset"],  # type: ignore[arg-type]
            scene_id=value["scene_id"],  # type: ignore[arg-type]
            episode_id=value["episode_id"],  # type: ignore[arg-type]
            decision_step=value["decision_step"],  # type: ignore[arg-type]
            native_action_id=value["native_action_id"],  # type: ignore[arg-type]
            global_action_ids=_tuple_value(
                value["global_action_ids"], "global_action_ids"
            ),
            executable_action_indices=_tuple_value(
                value["executable_action_indices"],
                "executable_action_indices",
            ),
            policy_scores=_tuple_value(value["policy_scores"], "policy_scores"),
            native_prefix_sha256=value["native_prefix_sha256"],  # type: ignore[arg-type]
        )

    @property
    def native_action_index(self) -> int:
        return self.global_action_ids.index(self.native_action_id)

    @property
    def frozen_candidate_action_ids(self) -> tuple[str, ...]:
        return tuple(
            self.global_action_ids[index]
            for index in self.executable_action_indices
        )

    def causal_payload(self) -> dict[str, object]:
        return {
            "schema_version": SNAPSHOT_SCHEMA,
            "dataset": self.dataset,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "decision_step": self.decision_step,
            "native_action_id": self.native_action_id,
            "global_action_ids": list(self.global_action_ids),
            "executable_action_indices": list(self.executable_action_indices),
            "policy_scores": list(self.policy_scores),
            "native_prefix_sha256": self.native_prefix_sha256,
        }


def _rank_non_native(snapshot: CausalActionSnapshot) -> tuple[str, ...]:
    candidates = [
        (
            -snapshot.policy_scores[index],
            snapshot.global_action_ids[index],
        )
        for index in snapshot.executable_action_indices
        if snapshot.global_action_ids[index] != snapshot.native_action_id
    ]
    candidates.sort()
    return tuple(action_id for _negative_score, action_id in candidates)


def _lattice_identity_payload(snapshot: CausalActionSnapshot) -> dict[str, object]:
    return {
        "dataset": snapshot.dataset,
        "scene_id": snapshot.scene_id,
        "episode_id": snapshot.episode_id,
        "decision_step": snapshot.decision_step,
        "native_prefix_sha256": snapshot.native_prefix_sha256,
    }


@dataclass(frozen=True)
class SealedActionLatticeEvent:
    """One sealed native-inclusive exact counterfactual action set."""

    snapshot: CausalActionSnapshot
    ranked_non_native_action_ids: tuple[str, ...]
    alternative_action_ids: tuple[str, ...]
    snapshot_commitment_sha256: str
    lattice_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, CausalActionSnapshot):
            raise TypeError("sealed lattice event requires a causal snapshot")
        expected_ranking = _rank_non_native(self.snapshot)
        if self.ranked_non_native_action_ids != expected_ranking:
            raise ValueError("sealed non-native action ranking drift")
        expected_alternatives = expected_ranking[:2]
        if self.alternative_action_ids != expected_alternatives:
            raise ValueError("sealed action lattice is not the fixed top two")
        if not 1 <= len(self.alternative_action_ids) <= 2:
            raise ValueError("sealed action lattice requires one or two alternatives")
        if self.snapshot_commitment_sha256 != _stable_hash(
            self.snapshot.causal_payload()
        ):
            raise ValueError("causal snapshot commitment drift")
        if self.lattice_id != _stable_hash(
            _lattice_identity_payload(self.snapshot)
        ):
            raise ValueError("action lattice identity drift")

    @classmethod
    def from_snapshot(
        cls, snapshot: CausalActionSnapshot,
    ) -> "SealedActionLatticeEvent":
        ranking = _rank_non_native(snapshot)
        return cls(
            snapshot=snapshot,
            ranked_non_native_action_ids=ranking,
            alternative_action_ids=ranking[:2],
            snapshot_commitment_sha256=_stable_hash(snapshot.causal_payload()),
            lattice_id=_stable_hash(_lattice_identity_payload(snapshot)),
        )

    @property
    def dataset(self) -> str:
        return self.snapshot.dataset

    @property
    def scene_id(self) -> str:
        return self.snapshot.scene_id

    @property
    def episode_id(self) -> str:
        return self.snapshot.episode_id

    @property
    def decision_step(self) -> int:
        return self.snapshot.decision_step

    @property
    def native_action_id(self) -> str:
        return self.snapshot.native_action_id

    @property
    def native_prefix_sha256(self) -> str:
        return self.snapshot.native_prefix_sha256

    @property
    def frozen_candidate_action_ids(self) -> tuple[str, ...]:
        return self.snapshot.frozen_candidate_action_ids

    @property
    def action_ids(self) -> tuple[str, ...]:
        return (self.native_action_id, *self.alternative_action_ids)

    @property
    def core_identity(self) -> tuple[str, str, str, int]:
        return (
            self.dataset,
            self.scene_id,
            self.episode_id,
            self.decision_step,
        )

    def manifest_payload(self) -> dict[str, object]:
        return {
            "lattice_id": self.lattice_id,
            "snapshot_commitment_sha256": self.snapshot_commitment_sha256,
            "snapshot": self.snapshot.causal_payload(),
            "frozen_candidate_action_ids": list(
                self.frozen_candidate_action_ids
            ),
            "ranked_non_native_action_ids": list(
                self.ranked_non_native_action_ids
            ),
            "native_action_id": self.native_action_id,
            "alternative_action_ids": list(self.alternative_action_ids),
        }


@dataclass(frozen=True)
class ActionLatticeSeal:
    """A deterministic action-list commitment made before treatment results."""

    events: tuple[SealedActionLatticeEvent, ...]
    action_list_commitment_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple) or not self.events:
            raise ValueError("action lattice seal requires at least one event")
        expected_order = tuple(sorted(
            self.events,
            key=lambda event: (*event.core_identity, event.native_prefix_sha256),
        ))
        if self.events != expected_order:
            raise ValueError("action lattice events are not in canonical order")
        core_identities = [event.core_identity for event in self.events]
        if len(core_identities) != len(set(core_identities)):
            raise ValueError("duplicate action lattice decision identity")
        lattice_ids = [event.lattice_id for event in self.events]
        if len(lattice_ids) != len(set(lattice_ids)):
            raise ValueError("duplicate action lattice ID")
        expected = _stable_hash([
            event.manifest_payload() for event in self.events
        ])
        if self.action_list_commitment_sha256 != expected:
            raise ValueError("sealed action-list commitment drift")

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_version": SEAL_SCHEMA,
            "status": "SEALED_BEFORE_TREATMENT_OUTCOMES",
            "selection_rule": (
                "native plus the top two executable non-native actions ordered "
                "by (-policy_score, action_id); one alternative when only one exists"
            ),
            "outcome_fields_used_for_selection": [],
            "treatment_results_read": False,
            "adaptive_collection_allowed": False,
            "action_list_commitment_sha256": (
                self.action_list_commitment_sha256
            ),
            "events": [event.manifest_payload() for event in self.events],
        }


def seal_action_lattice(
    snapshots: Sequence[CausalActionSnapshot | Mapping[str, object]],
) -> ActionLatticeSeal:
    """Seal a deterministic native-inclusive lattice from causal inputs only."""

    if isinstance(snapshots, (str, bytes, bytearray)) or not snapshots:
        raise ValueError("action lattice sealing requires causal snapshots")
    events = []
    for value in snapshots:
        snapshot = (
            value
            if isinstance(value, CausalActionSnapshot)
            else CausalActionSnapshot.from_mapping(value)
        )
        events.append(SealedActionLatticeEvent.from_snapshot(snapshot))
    ordered = tuple(sorted(
        events,
        key=lambda event: (*event.core_identity, event.native_prefix_sha256),
    ))
    return ActionLatticeSeal(
        events=ordered,
        action_list_commitment_sha256=_stable_hash([
            event.manifest_payload() for event in ordered
        ]),
    )


def _physical_scalar(value: object, field: str) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(float(value)):
        return float(value)
    raise ValueError(f"physical prefix field {field} is not a finite JSON scalar")


def _physical_prefix_payload(
    trace: Sequence[Mapping[str, object]], decision_step: int,
) -> dict[str, object]:
    if (
        not isinstance(decision_step, Integral)
        or isinstance(decision_step, bool)
        or int(decision_step) < 0
    ):
        raise ValueError("physical prefix decision step is invalid")
    decision_step = int(decision_step)
    if isinstance(trace, (str, bytes, bytearray)) or len(trace) <= decision_step:
        raise ValueError("physical action trace is shorter than the target step")
    rows = []
    for index, record in enumerate(trace[:decision_step]):
        if not isinstance(record, Mapping):
            raise ValueError(f"physical prefix row {index} is not a mapping")
        missing = [field for field in PHYSICAL_PREFIX_FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"physical prefix row {index} is missing fields: {missing}"
            )
        rows.append({
            field: _physical_scalar(record[field], field)
            for field in PHYSICAL_PREFIX_FIELDS
        })
    return {"decision_step": decision_step, "physical_prefix": rows}


def canonical_prefix_sha256(
    trace: Sequence[Mapping[str, object]], decision_step: int,
) -> str:
    """Hash only the physical native action prefix strictly before the target."""

    return _stable_hash(_physical_prefix_payload(trace, decision_step))


@dataclass(frozen=True)
class LatticeArmIdentity:
    dataset: str
    scene_id: str
    episode_id: str
    decision_step: int
    native_prefix_sha256: str
    action_id: str

    def __post_init__(self) -> None:
        if self.dataset not in {"RxR", "R2R"}:
            raise ValueError("lattice arm dataset drift")
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("lattice arm requires a scene ID")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("lattice arm requires an episode ID")
        if (
            not isinstance(self.decision_step, Integral)
            or isinstance(self.decision_step, bool)
            or int(self.decision_step) < 0
        ):
            raise ValueError("lattice arm decision step is invalid")
        object.__setattr__(self, "decision_step", int(self.decision_step))
        if not _is_sha256(self.native_prefix_sha256):
            raise ValueError("lattice arm prefix is not a SHA-256 digest")
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError("lattice arm requires an action ID")

    @property
    def lattice_id(self) -> str:
        return _stable_hash({
            "dataset": self.dataset,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "decision_step": self.decision_step,
            "native_prefix_sha256": self.native_prefix_sha256,
        })


def _validate_arm_identity(
    event: SealedActionLatticeEvent,
    arm: LatticeArmIdentity,
    *,
    native: bool,
) -> None:
    if arm.lattice_id != event.lattice_id:
        raise ValueError("lattice arm crossed episode, decision, or prefix identity")
    allowed = (
        (event.native_action_id,)
        if native
        else event.alternative_action_ids
    )
    if arm.action_id not in allowed:
        raise ValueError("lattice arm action is outside its sealed support")
    if arm.action_id not in event.frozen_candidate_action_ids:
        raise ValueError("lattice arm action was not executable at the target prefix")


def _decision_records(
    records: Sequence[Mapping[str, object]],
    name: str,
    *,
    physical_trace_length: int,
) -> dict[int, Mapping[str, object]]:
    if isinstance(records, (str, bytes, bytearray)) or not records:
        raise ValueError(f"{name} decision trace is empty")
    if len(records) != physical_trace_length:
        raise ValueError(
            f"{name} decision trace does not cover its complete physical trace"
        )
    result: dict[int, Mapping[str, object]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{name} decision record is not a mapping")
        step = record.get("step")
        if not isinstance(step, Integral) or isinstance(step, bool) or int(step) < 0:
            raise ValueError(f"{name} decision record has an invalid step")
        step = int(step)
        if step in result:
            raise ValueError(f"{name} decision trace repeats step {step}")
        if step != position:
            raise ValueError(
                f"{name} decision trace is not a complete ordered step sequence"
            )
        if type(record.get("action_changed")) is not bool:
            raise ValueError(f"{name} decision record lacks a Boolean change flag")
        _validate_decision_action_consistency(record, name=name, step=step)
        result[step] = record
    return result


def _decision_action_id(value: object, *, name: str, step: int, field: str) -> object:
    """Validate the only action-ID forms emitted by an exact decision trace."""

    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(
            f"{name} decision step {step} has an invalid {field}"
        )
    return value


def _validate_decision_action_consistency(
    record: Mapping[str, object], *, name: str, step: int,
) -> None:
    """Make the change flag agree with both the index and ID round trips.

    Non-target steps do not carry a frozen per-step candidate list, so their
    absolute ID-to-index mapping cannot be reconstructed here.  They can still
    be checked fail-closed: both indices must be valid integers, both IDs must
    be valid trace identities, and index equality, ID equality, and the
    ``action_changed`` flag must describe the same action-preserving decision.
    The target step receives the stronger frozen-support validation below.
    """

    indices: dict[str, int] = {}
    for field in ("native_action_index", "adapted_action_index"):
        value = record.get(field)
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or int(value) < 0
        ):
            raise ValueError(
                f"{name} decision step {step} has an invalid {field}"
            )
        indices[field] = int(value)
    native_id = _decision_action_id(
        record.get("native_action_id"),
        name=name,
        step=step,
        field="native_action_id",
    )
    adapted_id = _decision_action_id(
        record.get("adapted_action_id"),
        name=name,
        step=step,
        field="adapted_action_id",
    )
    index_changed = (
        indices["native_action_index"] != indices["adapted_action_index"]
    )
    identity_changed = native_id != adapted_id
    declared_changed = record["action_changed"] is True
    if index_changed != identity_changed:
        raise ValueError(
            f"{name} decision step {step} action ID/index roundtrip disagrees"
        )
    if declared_changed != index_changed:
        raise ValueError(
            f"{name} decision step {step} change flag hides action drift"
        )


def _decision_signature(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(record["native_action_index"]),
        int(record["adapted_action_index"]),
        record.get("native_action_id"),
        record.get("adapted_action_id"),
        record["action_changed"],
    )


def _validate_target_decision(
    event: SealedActionLatticeEvent,
    record: Mapping[str, object],
    adapted_action_id: str,
) -> None:
    for field in ("native_action_index", "adapted_action_index"):
        if not isinstance(record.get(field), Integral) or isinstance(
            record.get(field), bool
        ):
            raise ValueError(f"target decision has an invalid {field}")
    if record.get("native_action_id") != event.native_action_id:
        raise ValueError("target decision native action ID drift")
    if record.get("adapted_action_id") != adapted_action_id:
        raise ValueError("target decision adapted action ID drift")
    validate_action_identity(
        event.snapshot.global_action_ids,
        event.snapshot.executable_action_indices,
        int(record["native_action_index"]),
        int(record["adapted_action_index"]),
        declared_native_id=event.native_action_id,
        declared_adapted_id=adapted_action_id,
        require_non_stop=True,
    )


def _validate_physical_target_action(
    trace: Sequence[Mapping[str, object]], decision_step: int, action_id: str,
) -> None:
    record = trace[decision_step]
    if (
        not isinstance(record, Mapping)
        or not isinstance(record.get("act"), Integral)
        or int(record["act"]) != 4
        or str(record.get("ghost_vp")) != action_id
    ):
        raise ValueError("declared lattice action differs from physical execution")


def validate_exact_lattice_treatment(
    event: SealedActionLatticeEvent,
    native_arm: LatticeArmIdentity,
    treatment_arm: LatticeArmIdentity,
    native_physical_trace: Sequence[Mapping[str, object]],
    treatment_physical_trace: Sequence[Mapping[str, object]],
    native_decision_trace: Sequence[Mapping[str, object]],
    treatment_decision_trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate one exact alternative arm against its sealed native baseline."""

    _validate_arm_identity(event, native_arm, native=True)
    _validate_arm_identity(event, treatment_arm, native=False)
    native_prefix = canonical_prefix_sha256(
        native_physical_trace, event.decision_step
    )
    treatment_prefix = canonical_prefix_sha256(
        treatment_physical_trace, event.decision_step
    )
    if not (
        native_prefix
        == treatment_prefix
        == event.native_prefix_sha256
        == native_arm.native_prefix_sha256
        == treatment_arm.native_prefix_sha256
    ):
        raise ValueError("native and treatment target-step prefixes are not identical")

    native_records = _decision_records(
        native_decision_trace,
        "native",
        physical_trace_length=len(native_physical_trace),
    )
    treatment_records = _decision_records(
        treatment_decision_trace,
        "treatment",
        physical_trace_length=len(treatment_physical_trace),
    )
    if event.decision_step not in native_records:
        raise ValueError("native trace does not contain the target decision")
    if event.decision_step not in treatment_records:
        raise ValueError("treatment trace does not contain the target decision")
    if any(record["action_changed"] is not False for record in native_records.values()):
        raise ValueError("native lattice arm changed an action")
    changed = [
        record for record in treatment_records.values()
        if record["action_changed"] is True
    ]
    if len(changed) != 1 or int(changed[0]["step"]) != event.decision_step:
        raise ValueError("treatment did not make exactly one target-step switch")
    if any(
        record["action_changed"] is not False
        for step, record in treatment_records.items()
        if step != event.decision_step
    ):
        raise ValueError("treatment made a second intervention")
    for step in range(event.decision_step):
        if _decision_signature(native_records[step]) != _decision_signature(
            treatment_records[step]
        ):
            raise ValueError(
                "native and treatment decision traces differ before the target"
            )

    native_target = native_records[event.decision_step]
    treatment_target = treatment_records[event.decision_step]
    if native_target["action_changed"] is not False:
        raise ValueError("native target decision is not an abstention")
    if treatment_target["action_changed"] is not True:
        raise ValueError("treatment target decision did not switch")
    _validate_target_decision(event, native_target, event.native_action_id)
    _validate_target_decision(event, treatment_target, treatment_arm.action_id)
    if int(native_target["native_action_index"]) != int(
        native_target["adapted_action_index"]
    ):
        raise ValueError("native target decision changed its action index")
    if int(treatment_target["native_action_index"]) == int(
        treatment_target["adapted_action_index"]
    ):
        raise ValueError("treatment target decision did not change its action index")

    _validate_physical_target_action(
        native_physical_trace, event.decision_step, event.native_action_id
    )
    _validate_physical_target_action(
        treatment_physical_trace,
        event.decision_step,
        treatment_arm.action_id,
    )
    return {
        "lattice_id": event.lattice_id,
        "native_prefix_sha256": native_prefix,
        "native_action_id": event.native_action_id,
        "treatment_action_id": treatment_arm.action_id,
        "exact_prefix_verified": True,
        "exact_one_switch_verified": True,
        "candidate_executability_verified": True,
        "action_identity_roundtrip_verified": True,
        "complete_decision_traces_verified": True,
        "non_target_action_consistency_verified": True,
    }


@dataclass(frozen=True)
class LatticeArmFold:
    arm: LatticeArmIdentity
    fold: int

    def __post_init__(self) -> None:
        if not isinstance(self.arm, LatticeArmIdentity):
            raise TypeError("fold assignment requires a lattice arm identity")
        if (
            not isinstance(self.fold, Integral)
            or isinstance(self.fold, bool)
            or int(self.fold) < 0
        ):
            raise ValueError("lattice arm fold must be a non-negative integer")
        object.__setattr__(self, "fold", int(self.fold))


def validate_lattice_fold_integrity(
    seal: ActionLatticeSeal,
    assignments: Sequence[LatticeArmFold],
) -> dict[str, int]:
    """Keep scenes, episodes, and all native/treatment arms in one fold."""

    if not isinstance(seal, ActionLatticeSeal):
        raise TypeError("fold validation requires an action lattice seal")
    if isinstance(assignments, (str, bytes, bytearray)) or not assignments:
        raise ValueError("fold validation requires lattice arm assignments")
    events = {event.lattice_id: event for event in seal.events}
    expected = {
        (event.lattice_id, action_id)
        for event in seal.events
        for action_id in event.action_ids
    }
    observed: set[tuple[str, str]] = set()
    scene_folds: dict[str, set[int]] = {}
    episode_folds: dict[tuple[str, str, str], set[int]] = {}
    lattice_folds: dict[str, set[int]] = {}
    for assignment in assignments:
        if not isinstance(assignment, LatticeArmFold):
            raise TypeError("invalid lattice arm fold assignment")
        arm = assignment.arm
        event = events.get(arm.lattice_id)
        if event is None:
            raise ValueError("fold assignment references an unknown lattice/prefix")
        _validate_arm_identity(
            event,
            arm,
            native=(arm.action_id == event.native_action_id),
        )
        key = (arm.lattice_id, arm.action_id)
        if key in observed:
            raise ValueError("duplicate lattice arm fold assignment")
        observed.add(key)
        scene_folds.setdefault(arm.scene_id, set()).add(assignment.fold)
        episode_folds.setdefault(
            (arm.dataset, arm.scene_id, arm.episode_id), set()
        ).add(assignment.fold)
        lattice_folds.setdefault(arm.lattice_id, set()).add(assignment.fold)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"lattice arm fold coverage drift; missing={missing}, extra={extra}")
    if any(len(values) != 1 for values in scene_folds.values()):
        raise ValueError("one raw MP3D scene spans multiple folds")
    if any(len(values) != 1 for values in episode_folds.values()):
        raise ValueError("one episode's lattice arms span multiple folds")
    if any(len(values) != 1 for values in lattice_folds.values()):
        raise ValueError("one action lattice spans multiple folds")
    return {
        "scenes": len(scene_folds),
        "episodes": len(episode_folds),
        "lattices": len(lattice_folds),
        "arms": len(observed),
    }


__all__ = [
    "ActionLatticeSeal",
    "CausalActionSnapshot",
    "LatticeArmFold",
    "LatticeArmIdentity",
    "SealedActionLatticeEvent",
    "canonical_prefix_sha256",
    "seal_action_lattice",
    "validate_exact_lattice_treatment",
    "validate_lattice_fold_integrity",
]
