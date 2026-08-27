"""Event-level schema and leakage-resistant invariants for RevealBench-CE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from pathlib import Path
from typing import Iterable, Mapping, Optional

from .provenance import canonical_phase0_asset, regular_project_file, sha256_file
from .screening import iter_vlnce_episodes
from .types import RevealState


class ConstraintStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"


class Resolvability(str, Enum):
    RESOLVABLE_BEFORE_SPLIT = "resolvable_before_split"
    UNRESOLVABLE_BEFORE_SPLIT = "unresolvable_before_split"


class CandidateProvenance(str, Enum):
    ORACLE = "oracle"
    FROZEN = "frozen"


class SafeDestinationKind(str, Enum):
    TARGET_BRANCH = "target_branch"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class SafeControlWitness:
    """A replayable controller trace proving that one option is still safe."""

    witness_id: str
    destination_kind: SafeDestinationKind
    destination_id: str
    action_ids: tuple[str, ...]
    replay_ref: str
    replay_sha256: str
    path_cost: float

    def __post_init__(self) -> None:
        for name in ("witness_id", "destination_id", "replay_ref"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.replay_sha256, str)
            or len(self.replay_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.replay_sha256)
        ):
            raise ValueError("replay_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.destination_kind, SafeDestinationKind):
            raise TypeError("destination_kind must be a SafeDestinationKind")
        if not isinstance(self.action_ids, tuple):
            raise TypeError("action_ids must be a tuple")
        if not self.action_ids or any(
            not isinstance(action_id, str) or not action_id
            for action_id in self.action_ids
        ):
            raise ValueError("action_ids must contain non-empty strings")
        if (
            not isinstance(self.path_cost, (int, float))
            or isinstance(self.path_cost, bool)
            or not isfinite(self.path_cost)
            or self.path_cost < 0.0
        ):
            raise ValueError("path_cost must be finite and non-negative")


@dataclass(frozen=True)
class NoSafeOptionCertificate:
    """Hashed exhaustive-search report for a prefix with no safe option."""

    certificate_id: str
    search_ref: str
    search_sha256: str

    def __post_init__(self) -> None:
        for name in ("certificate_id", "search_ref"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.search_sha256, str)
            or len(self.search_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.search_sha256
            )
        ):
            raise ValueError("search_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class RevealPrefix:
    """Observable labels at one strictly truncated episode prefix."""

    prefix_index: int
    history_end_step: int
    observation_ref: str
    observation_sha256: str
    parent_observation_sha256: Optional[str]
    candidate_ids: tuple[str, ...]
    candidate_separable: bool
    language_constraints: Mapping[str, ConstraintStatus]
    safe_option_witnesses: tuple[SafeControlWitness, ...]
    no_safe_option_certificate: Optional[NoSafeOptionCertificate]

    def __post_init__(self) -> None:
        if not isinstance(self.prefix_index, int) or isinstance(self.prefix_index, bool):
            raise TypeError("prefix_index must be an integer")
        if self.prefix_index < 0:
            raise ValueError("prefix_index must be non-negative")
        if (
            not isinstance(self.history_end_step, int)
            or isinstance(self.history_end_step, bool)
            or self.history_end_step < 0
        ):
            raise ValueError("history_end_step must be a non-negative integer")
        if not self.observation_ref:
            raise ValueError("observation_ref must not be empty")
        for name in ("observation_sha256", "parent_observation_sha256"):
            digest = getattr(self, name)
            if digest is None and name == "parent_observation_sha256":
                continue
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not isinstance(self.candidate_ids, tuple):
            raise TypeError("candidate_ids must be a tuple")
        if not self.candidate_ids or any(
            not isinstance(candidate_id, str) or not candidate_id
            for candidate_id in self.candidate_ids
        ):
            raise ValueError("candidate_ids must not be empty")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if not self.language_constraints:
            raise ValueError("language_constraints must not be empty")
        if not isinstance(self.candidate_separable, bool):
            raise TypeError("candidate_separable must be a boolean")
        if not isinstance(self.safe_option_witnesses, tuple):
            raise TypeError("safe_option_witnesses must be a tuple")
        if any(
            not isinstance(witness, SafeControlWitness)
            for witness in self.safe_option_witnesses
        ):
            raise TypeError(
                "safe_option_witnesses must contain SafeControlWitness values"
            )
        witness_ids = tuple(
            witness.witness_id for witness in self.safe_option_witnesses
        )
        if len(witness_ids) != len(set(witness_ids)):
            raise ValueError("safe-option witness ids must be unique per prefix")
        if self.no_safe_option_certificate is not None and not isinstance(
            self.no_safe_option_certificate, NoSafeOptionCertificate
        ):
            raise TypeError(
                "no_safe_option_certificate must be a NoSafeOptionCertificate"
            )
        if bool(self.safe_option_witnesses) == (
            self.no_safe_option_certificate is not None
        ):
            raise ValueError(
                "each prefix requires either safe witnesses or one no-safe certificate"
            )
        for constraint_id, status in self.language_constraints.items():
            if not constraint_id:
                raise ValueError("constraint ids must not be empty")
            if not isinstance(status, ConstraintStatus):
                raise TypeError("constraint status must be a ConstraintStatus")
        object.__setattr__(
            self,
            "language_constraints",
            MappingProxyType(dict(self.language_constraints)),
        )

    @property
    def evidence_complete(self) -> bool:
        return all(
            status is not ConstraintStatus.UNRESOLVED
            for status in self.language_constraints.values()
        )

    def target_in_set(self, target_branch_id: str) -> bool:
        return target_branch_id in self.candidate_ids

    def reveal_state(self, target_branch_id: str) -> RevealState:
        """Instantaneous factors; event-level U/A/D applies K-prefix stability."""

        if not self.target_in_set(target_branch_id):
            return RevealState.UNOBSERVED
        if self.candidate_separable and self.evidence_complete:
            return RevealState.DISCRIMINABLE
        return RevealState.AMBIGUOUS


@dataclass(frozen=True)
class RevealEvent:
    """One scene-disjoint branch event with its complete prefix sequence."""

    dataset: str
    scene_id: str
    split: str
    episode_id: str
    event_id: str
    counterfactual_group_id: str
    annotation_ref: str
    annotation_sha256: str
    candidate_frontend_id: str
    sensor_protocol_id: str
    return_controller_id: str
    candidate_provenance: CandidateProvenance
    target_branch_id: str
    prefixes: tuple[RevealPrefix, ...]
    reveal_interval: Optional[tuple[int, int]]
    resolvability: Resolvability
    counterfactual_action_costs: Mapping[str, float]
    stability_k: int = 3

    def __post_init__(self) -> None:
        for name in (
            "dataset",
            "scene_id",
            "split",
            "episode_id",
            "event_id",
            "counterfactual_group_id",
            "annotation_ref",
            "candidate_frontend_id",
            "sensor_protocol_id",
            "return_controller_id",
            "target_branch_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.dataset not in {"rxr-ce", "r2r-ce"}:
            raise ValueError("dataset must be rxr-ce or r2r-ce")
        if self.split not in {"train", "val_seen"}:
            raise ValueError("Phase 0 events are restricted to train and val_seen")
        if (
            not isinstance(self.annotation_sha256, str)
            or len(self.annotation_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.annotation_sha256
            )
        ):
            raise ValueError("annotation_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.candidate_provenance, CandidateProvenance):
            raise TypeError("candidate_provenance must be a CandidateProvenance")
        if not isinstance(self.resolvability, Resolvability):
            raise TypeError("resolvability must be a Resolvability")
        if not isinstance(self.prefixes, tuple):
            raise TypeError("prefixes must be a tuple")
        if not isinstance(self.stability_k, int) or isinstance(self.stability_k, bool):
            raise TypeError("stability_k must be an integer")
        if self.stability_k < 1:
            raise ValueError("stability_k must be positive")
        if len(self.prefixes) < self.stability_k:
            raise ValueError("prefixes must cover at least one stability window")

        indices = tuple(prefix.prefix_index for prefix in self.prefixes)
        expected = tuple(range(indices[0], indices[0] + len(indices)))
        if indices != expected:
            raise ValueError("prefix indices must be unique, sorted, and contiguous")
        history_steps = tuple(prefix.history_end_step for prefix in self.prefixes)
        if any(
            current >= following
            for current, following in zip(history_steps, history_steps[1:])
        ):
            raise ValueError("history_end_step must increase strictly across prefixes")
        observation_refs = tuple(prefix.observation_ref for prefix in self.prefixes)
        observation_hashes = tuple(
            prefix.observation_sha256 for prefix in self.prefixes
        )
        if len(observation_refs) != len(set(observation_refs)):
            raise ValueError("observation_ref must be unique within an event")
        if len(observation_hashes) != len(set(observation_hashes)):
            raise ValueError("observation_sha256 must be unique within an event")
        if self.prefixes[0].parent_observation_sha256 is not None:
            raise ValueError("the first prefix must not declare a parent observation")
        for previous, current in zip(self.prefixes, self.prefixes[1:]):
            if current.parent_observation_sha256 != previous.observation_sha256:
                raise ValueError("observation hashes must form a strict prefix chain")
        safe_offsets = tuple(
            offset
            for offset, prefix in enumerate(self.prefixes)
            if prefix.safe_option_witnesses
        )
        if not safe_offsets:
            raise ValueError("an event requires at least one safe-option witness")
        last_safe_offset = safe_offsets[-1]
        if last_safe_offset == len(self.prefixes) - 1:
            raise ValueError(
                "expiry is right-censored; record the first prefix with no safe option"
            )
        post_expiry = self.prefixes[last_safe_offset + 1]
        if (
            post_expiry.safe_option_witnesses
            or post_expiry.no_safe_option_certificate is None
        ):
            raise ValueError(
                "the first post-expiry prefix requires a no-safe certificate"
            )
        for prefix in self.prefixes:
            for witness in prefix.safe_option_witnesses:
                if (
                    witness.destination_kind is SafeDestinationKind.TARGET_BRANCH
                    and witness.destination_id != self.target_branch_id
                ):
                    raise ValueError(
                        "target-branch witnesses must identify target_branch_id"
                    )

        constraint_ids = set(self.prefixes[0].language_constraints)
        if any(set(prefix.language_constraints) != constraint_ids for prefix in self.prefixes):
            raise ValueError("all prefixes must use the same decisive constraints")
        if not any(
            status is ConstraintStatus.RESOLVED
            for prefix in self.prefixes
            for status in prefix.language_constraints.values()
        ):
            raise ValueError("at least one decisive constraint must become resolved")

        if not self.counterfactual_action_costs:
            raise ValueError("counterfactual_action_costs must not be empty")
        for action, cost in self.counterfactual_action_costs.items():
            if not isinstance(action, str) or not action:
                raise ValueError("counterfactual action ids must not be empty")
            if (
                not isinstance(cost, (int, float))
                or isinstance(cost, bool)
                or not isfinite(cost)
                or cost < 0.0
            ):
                raise ValueError(
                    "counterfactual action costs must be finite and non-negative"
                )
        object.__setattr__(
            self,
            "counterfactual_action_costs",
            MappingProxyType(dict(self.counterfactual_action_costs)),
        )

        stable_onset = self.stable_reveal_onset
        if self.resolvability is Resolvability.RESOLVABLE_BEFORE_SPLIT:
            if self.reveal_interval is None:
                raise ValueError("a resolvable event requires a reveal interval")
            start, end = self.reveal_interval
            if start > end or start not in indices or end not in indices:
                raise ValueError("reveal_interval must be ordered and within prefixes")
            if stable_onset is None or not start <= stable_onset <= end:
                raise ValueError("reveal_interval must contain the stable D onset")
            if end > self.expiry_index:
                raise ValueError("a resolvable reveal interval must end by expiry")
        else:
            if self.reveal_interval is not None:
                start, end = self.reveal_interval
                if start > end or start not in indices or end not in indices:
                    raise ValueError("reveal_interval must be ordered and within prefixes")
                if stable_onset is None or not start <= stable_onset <= end:
                    raise ValueError("reveal_interval must contain the stable D onset")
                if start <= self.expiry_index:
                    raise ValueError(
                        "an unresolvable reveal interval must begin after expiry"
                    )
            if stable_onset is not None and stable_onset <= self.expiry_index:
                raise ValueError("an unresolvable event cannot reveal by expiry")

    @property
    def reveal_states(self) -> tuple[RevealState, ...]:
        instantaneous = self.instantaneous_reveal_states
        stable_offsets: set[int] = set()
        run_start = 0
        while run_start < len(instantaneous):
            if instantaneous[run_start] is not RevealState.DISCRIMINABLE:
                run_start += 1
                continue
            run_end = run_start
            while (
                run_end < len(instantaneous)
                and instantaneous[run_end] is RevealState.DISCRIMINABLE
            ):
                run_end += 1
            if run_end - run_start >= self.stability_k:
                stable_offsets.update(range(run_start, run_end))
            run_start = run_end
        return tuple(
            state
            if state is not RevealState.DISCRIMINABLE or offset in stable_offsets
            else RevealState.AMBIGUOUS
            for offset, state in enumerate(instantaneous)
        )

    @property
    def instantaneous_reveal_states(self) -> tuple[RevealState, ...]:
        return tuple(
            prefix.reveal_state(self.target_branch_id) for prefix in self.prefixes
        )

    @property
    def expiry_index(self) -> int:
        """Last prefix with a replayable safe option; never an annotator input."""

        return max(
            prefix.prefix_index
            for prefix in self.prefixes
            if prefix.safe_option_witnesses
        )

    @property
    def stable_reveal_onset(self) -> Optional[int]:
        states = self.reveal_states
        for offset, state in enumerate(states):
            if state is RevealState.DISCRIMINABLE:
                return self.prefixes[offset].prefix_index
        return None


def _load_hashed_json(ref: str, expected_hash: str, project_root: Path) -> dict[str, object]:
    artifact = regular_project_file(Path(ref), project_root)
    if sha256_file(artifact) != expected_hash:
        raise ValueError(f"artifact hash mismatch: {ref}")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact {ref}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {ref}")
    return payload


def _verify_log_reference(
    payload: dict[str, object], project_root: Path, artifact_kind: str
) -> None:
    log_ref = payload.get("simulator_log_ref")
    log_hash = payload.get("simulator_log_sha256")
    if not isinstance(log_ref, str) or not isinstance(log_hash, str):
        raise ValueError(f"{artifact_kind} must identify a hashed simulator log")
    log = regular_project_file(Path(log_ref), project_root)
    if sha256_file(log) != log_hash:
        raise ValueError(f"{artifact_kind} simulator log hash mismatch")


def _verify_observation_descriptor(
    event: RevealEvent, prefix: RevealPrefix, project_root: Path
) -> None:
    payload = _load_hashed_json(
        prefix.observation_ref, prefix.observation_sha256, project_root
    )
    expected_keys = {
        "schema_version",
        "dataset",
        "scene_id",
        "episode_id",
        "prefix_index",
        "history_end_step",
        "parent_observation_sha256",
        "frame_steps",
        "payload_ref",
        "payload_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("observation descriptor schema mismatch")
    expected_values = {
        "schema_version": 1,
        "dataset": event.dataset,
        "scene_id": event.scene_id,
        "episode_id": event.episode_id,
        "prefix_index": prefix.prefix_index,
        "history_end_step": prefix.history_end_step,
        "parent_observation_sha256": prefix.parent_observation_sha256,
    }
    if any(payload.get(name) != value for name, value in expected_values.items()):
        raise ValueError("observation descriptor does not match its event prefix")
    frame_steps = payload.get("frame_steps")
    if (
        not isinstance(frame_steps, list)
        or not frame_steps
        or any(
            not isinstance(step, int) or isinstance(step, bool) or step < 0
            for step in frame_steps
        )
        or frame_steps != sorted(set(frame_steps))
        or frame_steps[-1] != prefix.history_end_step
    ):
        raise ValueError("observation frame_steps must be unique, ordered and truncated")
    payload_ref = payload.get("payload_ref")
    payload_hash = payload.get("payload_sha256")
    if not isinstance(payload_ref, str) or not isinstance(payload_hash, str):
        raise ValueError("observation descriptor must identify a hashed payload")
    observation_payload = regular_project_file(Path(payload_ref), project_root)
    if sha256_file(observation_payload) != payload_hash:
        raise ValueError("observation payload hash mismatch")


def _verify_safe_witness(
    event: RevealEvent,
    prefix: RevealPrefix,
    witness: SafeControlWitness,
    project_root: Path,
) -> None:
    payload = _load_hashed_json(witness.replay_ref, witness.replay_sha256, project_root)
    expected_keys = {
        "schema_version",
        "controller_id",
        "sensor_protocol_id",
        "scene_id",
        "episode_id",
        "prefix_index",
        "observation_sha256",
        "witness_id",
        "destination_kind",
        "destination_id",
        "action_ids",
        "path_cost",
        "success",
        "collision_free",
        "simulator_log_ref",
        "simulator_log_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("safe-option replay schema mismatch")
    expected_values = {
        "schema_version": 1,
        "controller_id": event.return_controller_id,
        "sensor_protocol_id": event.sensor_protocol_id,
        "scene_id": event.scene_id,
        "episode_id": event.episode_id,
        "prefix_index": prefix.prefix_index,
        "observation_sha256": prefix.observation_sha256,
        "witness_id": witness.witness_id,
        "destination_kind": witness.destination_kind.value,
        "destination_id": witness.destination_id,
        "action_ids": list(witness.action_ids),
        "path_cost": witness.path_cost,
        "success": True,
        "collision_free": True,
    }
    if any(payload.get(name) != value for name, value in expected_values.items()):
        raise ValueError("safe-option replay does not prove the declared witness")
    _verify_log_reference(payload, project_root, "safe-option replay")


def _verify_no_safe_certificate(
    event: RevealEvent,
    prefix: RevealPrefix,
    certificate: NoSafeOptionCertificate,
    project_root: Path,
) -> None:
    payload = _load_hashed_json(
        certificate.search_ref, certificate.search_sha256, project_root
    )
    expected_keys = {
        "schema_version",
        "controller_id",
        "sensor_protocol_id",
        "candidate_frontend_id",
        "scene_id",
        "episode_id",
        "prefix_index",
        "observation_sha256",
        "certificate_id",
        "search_complete",
        "enumerated_option_ids",
        "feasible_option_ids",
        "safety_constraints_id",
        "simulator_log_ref",
        "simulator_log_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("no-safe-option certificate schema mismatch")
    expected_values = {
        "schema_version": 1,
        "controller_id": event.return_controller_id,
        "sensor_protocol_id": event.sensor_protocol_id,
        "candidate_frontend_id": event.candidate_frontend_id,
        "scene_id": event.scene_id,
        "episode_id": event.episode_id,
        "prefix_index": prefix.prefix_index,
        "observation_sha256": prefix.observation_sha256,
        "certificate_id": certificate.certificate_id,
        "search_complete": True,
        "feasible_option_ids": [],
    }
    if any(payload.get(name) != value for name, value in expected_values.items()):
        raise ValueError("no-safe-option search does not prove the declared prefix")
    enumerated = payload.get("enumerated_option_ids")
    constraints_id = payload.get("safety_constraints_id")
    if (
        not isinstance(enumerated, list)
        or not enumerated
        or any(not isinstance(option, str) or not option for option in enumerated)
        or not isinstance(constraints_id, str)
        or not constraints_id
    ):
        raise ValueError("no-safe-option search metadata is incomplete")
    _verify_log_reference(payload, project_root, "no-safe-option search")


def validate_event_collection(
    events: Iterable[RevealEvent], project_root: Path
) -> tuple[RevealEvent, ...]:
    """Validate official source identity, split isolation and replay artifacts."""

    items = tuple(events)
    identities: set[tuple[str, str]] = set()
    scene_splits: dict[tuple[str, str], str] = {}
    groups: dict[tuple[str, str], tuple[str, str, str, str]] = {}
    observation_owners: dict[str, tuple[str, str, str]] = {}
    source_episodes: dict[Path, dict[str, str]] = {}
    for event in items:
        identity = (event.dataset, event.event_id)
        if identity in identities:
            raise ValueError(f"duplicate event identity: {identity}")
        identities.add(identity)

        scene_key = (event.dataset, event.scene_id)
        existing_split = scene_splits.setdefault(scene_key, event.split)
        if existing_split != event.split:
            raise ValueError("a scene cannot occur in multiple event splits")

        group_identity = (
            event.dataset,
            event.scene_id,
            event.split,
            event.episode_id,
        )
        existing_group = groups.setdefault(
            (event.dataset, event.counterfactual_group_id), group_identity
        )
        if existing_group != group_identity:
            raise ValueError(
                "a counterfactual group must remain in one episode and split"
            )

        owner = (event.dataset, event.scene_id, event.episode_id)
        for prefix in event.prefixes:
            existing_owner = observation_owners.setdefault(
                prefix.observation_sha256, owner
            )
            if existing_owner != owner:
                raise ValueError(
                    "an observation hash cannot cross dataset/scene/episode boundaries"
                )
    for event in items:
        asset = canonical_phase0_asset(Path(event.annotation_ref), project_root)
        if (
            asset.dataset != event.dataset
            or asset.split != event.split
            or asset.sha256 != event.annotation_sha256
        ):
            raise ValueError("event source does not match canonical dataset/split/hash")
        if asset.path not in source_episodes:
            source_episodes[asset.path] = {
                str(episode["episode_id"]): Path(str(episode["scene_id"])).stem
                for episode in iter_vlnce_episodes(asset.path)
            }
        if source_episodes[asset.path].get(event.episode_id) != event.scene_id:
            raise ValueError("event episode/scene does not exist in its canonical source")
        for prefix in event.prefixes:
            _verify_observation_descriptor(event, prefix, project_root)
            for witness in prefix.safe_option_witnesses:
                _verify_safe_witness(event, prefix, witness, project_root)
            if prefix.no_safe_option_certificate is not None:
                _verify_no_safe_certificate(
                    event,
                    prefix,
                    prefix.no_safe_option_certificate,
                    project_root,
                )
    return items
