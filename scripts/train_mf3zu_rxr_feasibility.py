#!/usr/bin/env python3
"""Run the fixed three-arm, five-scene-fold MF3ZU RxR feasibility probe.

Population and evidence are joined while candidate targets remain unopened.
Only after the evidence manifest proves a frozen, target-blind memory artifact
does this entrypoint read the separate, sealed exact-target artifact.  Arm A is
the untouched frozen ETP score.  Arms B and C share architecture,
initialization, batches, optimizer, and schedule; C alone receives a train-fold
safe shuffled memory donor.  Complete OOF evidence is embedded in the result so
one atomic rename publishes the result and every input required by its audit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zu_evidence_memory import (  # noqa: E402
    CANDIDATE_EVIDENCE_FEATURE_DIM,
    ConfidenceClass,
    EvidenceRecord,
    EvidenceType,
    candidate_memory_feature,
    reject_sensitive_mapping,
)
from revealnav_mf3.mf3zu_evidence_memory_metrics import (  # noqa: E402
    ARM_CURRENT,
    ARM_MEMORY,
    ARM_SHUFFLED,
    ARMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    apply_fixed_rxr_gates,
    evaluate_three_arm_probe,
)
from revealnav_mf3.mf3zu_evidence_memory_reranker import (  # noqa: E402
    CANDIDATE_DIM,
    FIXED_SEED,
    FeatureNormalizer,
    common_initialized_rerankers,
    fit_feature_normalizer,
    masked_candidate_cross_entropy,
    parameter_sha256,
    shuffled_memory_donor_indices,
)
from revealnav_mf3.mf3zu_protocol import (  # noqa: E402
    EVIDENCE_MEMORY_MANIFEST_PATH,
    EVIDENCE_MEMORY_PATH,
    EXACT_TARGETS_PATH,
    EXPECTED_POPULATION_ROWS,
    EXPECTED_POPULATION_EPISODES,
    EXPECTED_POPULATION_SCENES,
    FOLDS,
    POPULATION_MANIFEST_PATH,
    POPULATION_PATH,
    PROTOCOL_PATH,
    PUBLIC_CLOSED,
    RESULT_PATH,
    REVISION,
    ProtocolError,
    scene_fold_mapping,
    sha256_file,
    verify_protocol,
)


EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
AUDIT_PATH = RESULT_PATH.with_name("MF3ZU_RXR_FEASIBILITY_RESULT_AUDIT.json")


class MF3ZUTrainingError(RuntimeError):
    """Raised before a malformed or leaking training run can write results."""


@dataclass(frozen=True)
class ProbeArrays:
    event_id: np.ndarray
    scene_id: np.ndarray
    episode_id: np.ndarray
    decision_step: np.ndarray
    scene_fold: np.ndarray
    candidate_action_ids: tuple[tuple[str, ...], ...]
    candidate_features: np.ndarray
    base_scores: np.ndarray
    candidate_mask: np.ndarray
    memory_features: np.ndarray
    memory_count: np.ndarray
    memory_required: np.ndarray
    target_index: np.ndarray
    source_feature_path: tuple[str, ...]
    source_feature_row: np.ndarray
    population_rows_before_target: int
    evidence_diagnostics: Mapping[str, object]

    def validate(self, *, require_targets: bool = True) -> None:
        rows = len(self.event_id)
        vectors = (
            self.scene_id,
            self.episode_id,
            self.decision_step,
            self.scene_fold,
            self.memory_count,
            self.memory_required,
            self.target_index,
            self.source_feature_row,
        )
        if rows == 0 or any(len(value) != rows for value in vectors):
            raise MF3ZUTrainingError("probe arrays have inconsistent row counts")
        if len(self.candidate_action_ids) != rows or len(self.source_feature_path) != rows:
            raise MF3ZUTrainingError("probe identity arrays have inconsistent lengths")
        if len(set(self.event_id.tolist())) != rows:
            raise MF3ZUTrainingError("probe event identity is not unique")
        expected_candidate_shape = self.candidate_features.shape[:2]
        if (
            self.candidate_features.ndim != 3
            or self.candidate_features.shape[-1] != CANDIDATE_DIM
            or self.base_scores.shape != expected_candidate_shape
            or self.candidate_mask.shape != expected_candidate_shape
            or self.memory_features.shape != (*expected_candidate_shape, CANDIDATE_EVIDENCE_FEATURE_DIM)
        ):
            raise MF3ZUTrainingError("probe candidate/memory tensor shape drift")
        if self.candidate_mask.dtype != bool or self.memory_required.dtype != bool:
            raise MF3ZUTrainingError("probe masks must be boolean")
        if np.any(self.candidate_mask.sum(axis=1) < 2):
            raise MF3ZUTrainingError("probe row has fewer than two candidates")
        if not np.isfinite(self.candidate_features[self.candidate_mask]).all():
            raise MF3ZUTrainingError("candidate feature is non-finite")
        if not np.isfinite(self.base_scores[self.candidate_mask]).all():
            raise MF3ZUTrainingError("base score is non-finite")
        if not np.isfinite(self.memory_features[self.candidate_mask]).all():
            raise MF3ZUTrainingError("memory feature is non-finite")
        if np.any((self.memory_count < 0) | (self.memory_count > 8)):
            raise MF3ZUTrainingError("retrieved memory count is outside [0,8]")
        for row, ids in enumerate(self.candidate_action_ids):
            if len(ids) != int(self.candidate_mask[row].sum()) or len(set(ids)) != len(ids):
                raise MF3ZUTrainingError("candidate identity/count drift")
        if set(self.scene_fold.tolist()) != set(range(FOLDS)):
            raise MF3ZUTrainingError("five scene folds are not all represented")
        seen_scene_folds: dict[str, set[int]] = {}
        for scene, fold in zip(self.scene_id, self.scene_fold, strict=True):
            seen_scene_folds.setdefault(str(scene), set()).add(int(fold))
        if any(len(values) != 1 for values in seen_scene_folds.values()):
            raise MF3ZUTrainingError("one raw MP3D scene crosses folds")
        if require_targets:
            if np.any((self.target_index < 0) | (self.target_index >= self.candidate_mask.shape[1])):
                raise MF3ZUTrainingError("activated target is out of bounds")
            if np.any(~self.candidate_mask[np.arange(rows), self.target_index]):
                raise MF3ZUTrainingError("activated target is not an executable candidate")


def _read_object(path: Path, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MF3ZUTrainingError(f"{name} must be a JSON object")
    return value


def _read_jsonl(path: Path, name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise MF3ZUTrainingError(f"malformed {name} row {line_number}")
        rows.append(value)
    if not rows:
        raise MF3ZUTrainingError(f"{name} is empty")
    return rows


def _safe_project_path(raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise MF3ZUTrainingError("source feature path is missing")
    path = ROOT / raw
    resolved = path.resolve()
    if path.is_symlink() or not path.is_file() or ROOT.resolve() not in resolved.parents:
        raise MF3ZUTrainingError(f"unsafe project-local source: {raw}")
    return resolved


def _manifest_payload(
    data_path: Path,
    manifest_path: Path,
    *,
    kind: str,
) -> dict[str, object]:
    manifest = _read_object(manifest_path, f"{kind} manifest")
    if manifest.get("revision") != REVISION:
        raise MF3ZUTrainingError(f"{kind} manifest revision drift")
    inventory = manifest.get(kind)
    if not isinstance(inventory, Mapping):
        # Evidence builders may use the explicit evidence_memory key.
        inventory = manifest.get("evidence_memory" if kind == "evidence" else kind)
    if not isinstance(inventory, Mapping):
        raise MF3ZUTrainingError(f"{kind} manifest inventory is missing")
    if int(inventory.get("bytes", -1)) != data_path.stat().st_size:
        raise MF3ZUTrainingError(f"{kind} artifact byte-count drift")
    if inventory.get("sha256") != sha256_file(data_path):
        raise MF3ZUTrainingError(f"{kind} artifact hash drift")
    return manifest


def _load_target_blind_population(
    population_path: Path,
    population_manifest_path: Path,
    *,
    enforce_frozen_counts: bool,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray | tuple[str, ...] | tuple[tuple[str, ...], ...]]]:
    manifest = _manifest_payload(
        population_path, population_manifest_path, kind="population"
    )
    if manifest.get("exact_target_accessed_for_support_eligibility") is not True:
        raise MF3ZUTrainingError("population lacks exact legal-target support gate")
    if manifest.get("target_value_in_sanitized_population") is not False:
        raise MF3ZUTrainingError("candidate target leaked into sanitized population")
    rows = _read_jsonl(population_path, "population")
    if enforce_frozen_counts and len(rows) != EXPECTED_POPULATION_ROWS:
        raise MF3ZUTrainingError("frozen population row count drift")
    event_ids = [str(row.get("event_id", "")) for row in rows]
    if not all(event_ids) or len(set(event_ids)) != len(event_ids):
        raise MF3ZUTrainingError("population event identity is missing or repeated")

    maximum_candidates = max(int(row.get("candidate_count", 0)) for row in rows)
    candidate_features = np.zeros(
        (len(rows), maximum_candidates, CANDIDATE_DIM), dtype=np.float32
    )
    base_scores = np.zeros((len(rows), maximum_candidates), dtype=np.float32)
    candidate_mask = np.zeros((len(rows), maximum_candidates), dtype=bool)
    source_paths: list[str] = []
    source_rows: list[int] = []
    candidate_ids: list[tuple[str, ...]] = []
    cache: dict[Path, dict[str, np.ndarray]] = {}
    verified_hashes: set[Path] = set()
    for index, row in enumerate(rows):
        if row.get("dataset") != "RxR" or row.get("revision") != REVISION:
            raise MF3ZUTrainingError("population escaped RxR MF3ZU scope")
        forbidden = {
            "target", "target_index", "correct_candidate", "teacher_action_id_label_only",
            "teacher_action_index_label_only", "reward", "utility", "outcome",
        }
        if forbidden & set(row):
            raise MF3ZUTrainingError("target/outcome leaked into frozen population")
        source_text = str(row.get("source_feature_path", ""))
        source_path = _safe_project_path(source_text)
        if source_path not in verified_hashes:
            if row.get("source_feature_sha256") != sha256_file(source_path):
                raise MF3ZUTrainingError("source NPZ hash drift")
            verified_hashes.add(source_path)
        if source_path not in cache:
            with np.load(source_path, allow_pickle=False) as arrays:
                required = {"candidate_embeddings", "candidate_mask", "native_scores"}
                if not required.issubset(arrays.files):
                    raise MF3ZUTrainingError("source NPZ candidate arrays are incomplete")
                cache[source_path] = {name: np.asarray(arrays[name]).copy() for name in required}
        arrays = cache[source_path]
        feature_row = int(row.get("feature_row_index", -1))
        slots = np.asarray(row.get("active_candidate_feature_slots", ()), dtype=np.int64)
        ids = tuple(str(value) for value in row.get("candidate_action_ids", ()))
        count = int(row.get("candidate_count", 0))
        if (
            feature_row < 0
            or feature_row >= len(arrays["candidate_mask"])
            or slots.shape != (count,)
            or len(ids) != count
            or count < 2
        ):
            raise MF3ZUTrainingError("population candidate/source coordinate drift")
        source_mask = np.asarray(arrays["candidate_mask"][feature_row], dtype=bool)
        if not np.array_equal(slots, np.flatnonzero(source_mask)):
            raise MF3ZUTrainingError("population candidate slots differ from source mask")
        features = np.asarray(arrays["candidate_embeddings"][feature_row, slots], dtype=np.float32)
        scores = np.asarray(arrays["native_scores"][feature_row, slots], dtype=np.float32)
        if features.shape != (count, CANDIDATE_DIM) or scores.shape != (count,):
            raise MF3ZUTrainingError("source candidate tensor dimension drift")
        candidate_features[index, :count] = features
        base_scores[index, :count] = scores
        candidate_mask[index, :count] = True
        source_paths.append(source_text)
        source_rows.append(feature_row)
        candidate_ids.append(ids)

    scenes = np.asarray([str(row["scene_id"]) for row in rows])
    expected_folds = scene_fold_mapping(scenes.tolist())
    folds = np.asarray([int(row["scene_fold"]) for row in rows], dtype=np.int64)
    if any(int(fold) != expected_folds[str(scene)] for scene, fold in zip(scenes, folds, strict=True)):
        raise MF3ZUTrainingError("population scene-fold assignment drift")
    return rows, {
        "event_id": np.asarray(event_ids),
        "scene_id": scenes,
        "episode_id": np.asarray([str(row["episode_id"]) for row in rows]),
        "decision_step": np.asarray([int(row["decision_step"]) for row in rows], dtype=np.int64),
        "scene_fold": folds,
        "candidate_action_ids": tuple(candidate_ids),
        "active_candidate_feature_slots": tuple(
            tuple(int(value) for value in row["active_candidate_feature_slots"])
            for row in rows
        ),
        "candidate_features": candidate_features,
        "base_scores": base_scores,
        "candidate_mask": candidate_mask,
        "source_feature_path": tuple(source_paths),
        "source_feature_row": np.asarray(source_rows, dtype=np.int64),
    }


def _record_from_mapping(value: Mapping[str, object]) -> EvidenceRecord:
    try:
        return EvidenceRecord(
            evidence_id=str(value["evidence_id"]),
            event_id=str(value["event_id"]),
            source_step=int(value["source_step"]),
            source_node_id=str(value["source_node_id"]),
            instruction_atom_id=str(value["instruction_atom_id"]),
            evidence_type=EvidenceType(str(value["evidence_type"])),
            semantic_value=str(value["semantic_value"]),
            confidence_class=ConfidenceClass(str(value["confidence_class"])),
            current_status=ConfidenceClass(str(value["current_status"])),
            candidate_ids=tuple(str(item) for item in value.get("candidate_ids", ())),
            source_observation_sha256=str(value["source_observation_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MF3ZUTrainingError("malformed frozen evidence record") from error


def _join_frozen_evidence(
    population_rows: Sequence[Mapping[str, object]],
    population: Mapping[str, object],
    evidence_path: Path,
    evidence_manifest_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    # This entire function completes before _activate_exact_targets is called.
    manifest = _manifest_payload(evidence_path, evidence_manifest_path, kind="evidence")
    status = str(manifest.get("status", ""))
    if status != "MF3ZU_RXR_EVIDENCE_MEMORY_FROZEN":
        raise MF3ZUTrainingError("evidence memory is not frozen before target activation")
    if manifest.get("candidate_target_accessed") is not False:
        raise MF3ZUTrainingError("evidence construction accessed candidate target")
    if manifest.get("outcome_or_utility_accessed", False) is not False:
        raise MF3ZUTrainingError("evidence construction accessed outcome or utility")
    if manifest.get("exact_target_artifact_opened") is not False:
        raise MF3ZUTrainingError("evidence construction opened exact target artifact")
    rows = _read_jsonl(evidence_path, "evidence memory")
    by_event: dict[str, dict[str, object]] = {}
    for row in rows:
        compliance_fields = {
            "exact_target_artifact_opened", "ranking_label_read",
            "task_metric_read", "public_split_access",
        }
        if any(row.get(field) is not False for field in compliance_fields):
            raise MF3ZUTrainingError("evidence row opened a forbidden information source")
        reject_sensitive_mapping({
            key: value for key, value in row.items()
            if key not in compliance_fields
        })
        event_id = str(row.get("event_id", ""))
        if not event_id or event_id in by_event:
            raise MF3ZUTrainingError("evidence event identity is missing or repeated")
        by_event[event_id] = row
    expected = set(np.asarray(population["event_id"]).tolist())
    if set(by_event) != expected:
        raise MF3ZUTrainingError("evidence memory does not exactly cover the population")

    candidate_mask = np.asarray(population["candidate_mask"], dtype=bool)
    candidate_features = np.asarray(population["candidate_features"])
    base_scores = np.asarray(population["base_scores"])
    authoritative_candidate_ids: list[tuple[str, ...]] = []
    authoritative_feature_slots: list[tuple[int, ...]] = []
    memory = np.zeros(
        (*candidate_mask.shape, CANDIDATE_EVIDENCE_FEATURE_DIM), dtype=np.float32
    )
    memory_count = np.zeros(len(population_rows), dtype=np.int64)
    memory_required = np.zeros(len(population_rows), dtype=bool)
    evidence_types: Counter[str] = Counter()
    ages: list[int] = []
    current_absent = 0
    retrieved_total = 0
    for index, population_row in enumerate(population_rows):
        event_id = str(population_row["event_id"])
        row = by_event[event_id]
        for name in ("scene_id", "episode_id", "decision_step"):
            if str(row.get(name)) != str(population_row[name]):
                raise MF3ZUTrainingError("evidence/population causal identity drift")
        if type(row.get("memory_required")) is not bool:
            raise MF3ZUTrainingError("memory_required must be an outcome-blind boolean")
        memory_required[index] = bool(row["memory_required"])
        active = row.get("active_instruction_atom_ids")
        retrieved_raw = row.get("retrieved_records")
        if not isinstance(active, list) or not isinstance(retrieved_raw, list):
            raise MF3ZUTrainingError("evidence retrieval fields are missing")
        if len(retrieved_raw) > 8:
            raise MF3ZUTrainingError("retrieval exceeds K_MEM=8")
        retrieved = tuple(
            _record_from_mapping(value)
            for value in retrieved_raw
            if isinstance(value, Mapping)
        )
        if len(retrieved) != len(retrieved_raw):
            raise MF3ZUTrainingError("retrieved evidence record schema drift")
        decision_step = int(population_row["decision_step"])
        if any(record.event_id != event_id or record.source_step >= decision_step for record in retrieved):
            raise MF3ZUTrainingError("retrieved evidence is non-causal or cross-event")
        memory_count[index] = len(retrieved)
        retrieved_total += len(retrieved)
        for record in retrieved:
            evidence_types[record.evidence_type.value] += 1
            ages.append(decision_step - record.source_step)
            current_absent += int(record.current_status is not ConfidenceClass.OBSERVED)

        ids = tuple(str(value) for value in row.get("candidate_action_ids", ()))
        active_slots = tuple(
            int(value) for value in row.get("active_candidate_feature_slots", ())
        )
        expected_ids = tuple(str(value) for value in population_row["candidate_action_ids"])
        expected_active_slots = tuple(
            int(value) for value in population_row["active_candidate_feature_slots"]
        )
        mapping_raw = row.get("candidate_id_to_feature_slot")
        if not isinstance(mapping_raw, Mapping):
            raise MF3ZUTrainingError("replay candidate/feature binding is missing")
        mapping = {str(key): int(value) for key, value in mapping_raw.items()}
        if (
            set(ids) != set(expected_ids)
            or active_slots != expected_active_slots
            or set(mapping) != set(ids)
            or set(mapping.values()) != set(active_slots)
            or len(mapping.values()) != len(set(mapping.values()))
        ):
            raise MF3ZUTrainingError("evidence candidate coordinate drift")
        slots = tuple(mapping[candidate_id] for candidate_id in ids)
        source_position = {
            slot: position for position, slot in enumerate(expected_active_slots)
        }
        permutation = [source_position[slot] for slot in slots]
        count = len(ids)
        source_candidate = candidate_features[index, :count].copy()
        source_scores = base_scores[index, :count].copy()
        candidate_features[index, :count] = source_candidate[permutation]
        base_scores[index, :count] = source_scores[permutation]
        authoritative_candidate_ids.append(ids)
        authoritative_feature_slots.append(slots)
        features_raw = row.get("candidate_memory_features_by_slot")
        if not isinstance(features_raw, list) or len(features_raw) != len(ids):
            raise MF3ZUTrainingError("candidate-specific memory features are incomplete")
        by_slot: dict[int, Mapping[str, object]] = {}
        for item in features_raw:
            if not isinstance(item, Mapping):
                raise MF3ZUTrainingError("candidate memory feature row is malformed")
            slot = int(item.get("feature_slot", -1))
            if slot in by_slot:
                raise MF3ZUTrainingError("candidate memory feature slot is repeated")
            by_slot[slot] = item
        for local, (slot, candidate_id) in enumerate(zip(slots, ids, strict=True)):
            item = by_slot.get(slot)
            if item is None or str(item.get("candidate_action_id")) != candidate_id:
                raise MF3ZUTrainingError("candidate memory feature binding drift")
            observed = np.asarray(item.get("feature"), dtype=np.float32)
            expected_feature = candidate_memory_feature(
                retrieved,
                active_instruction_atom_ids=tuple(str(value) for value in active),
                decision_step=decision_step,
                candidate_id=candidate_id,
            )
            if observed.shape != (CANDIDATE_EVIDENCE_FEATURE_DIM,) or not np.array_equal(observed, expected_feature):
                raise MF3ZUTrainingError("frozen candidate memory feature is not bit-exact")
            memory[index, local] = observed

    age_array = np.asarray(ages, dtype=np.float64)
    diagnostics = {
        "population_decisions": len(population_rows),
        "memory_required_decisions_before_target_activation": int(memory_required.sum()),
        "memory_not_required_decisions_before_target_activation": int((~memory_required).sum()),
        "retrieved_records": int(retrieved_total),
        "retrieved_records_per_decision_mean": float(retrieved_total / len(population_rows)),
        "evidence_type_distribution": dict(sorted(evidence_types.items())),
        "historical_evidence_age": {
            "mean": None if not ages else float(age_array.mean()),
            "median": None if not ages else float(np.median(age_array)),
            "p90": None if not ages else float(np.quantile(age_array, 0.9)),
        },
        "percentage_current_frame_absent": (
            None if not ages else float(100.0 * current_absent / len(ages))
        ),
        "percentage_retrieved_from_at_least_2_steps_ago": (
            None if not ages else float(100.0 * np.mean(age_array >= 2))
        ),
        "candidate_target_accessed": False,
    }
    population["candidate_action_ids"] = tuple(authoritative_candidate_ids)
    population["ordered_candidate_feature_slots"] = tuple(
        authoritative_feature_slots
    )
    return memory, memory_count, memory_required, diagnostics


def load_frozen_probe_inputs(
    population_path: Path = POPULATION_PATH,
    population_manifest_path: Path = POPULATION_MANIFEST_PATH,
    evidence_path: Path = EVIDENCE_MEMORY_PATH,
    evidence_manifest_path: Path = EVIDENCE_MEMORY_MANIFEST_PATH,
    *,
    enforce_frozen_counts: bool = True,
) -> ProbeArrays:
    """Load target-blind inputs, freeze evidence, then activate exact targets."""

    population_rows, population = _load_target_blind_population(
        population_path,
        population_manifest_path,
        enforce_frozen_counts=enforce_frozen_counts,
    )
    memory, memory_count, memory_required, diagnostics = _join_frozen_evidence(
        population_rows, population, evidence_path, evidence_manifest_path
    )

    # Target access begins here, after _join_frozen_evidence has fully verified
    # the target-blind artifact and its immutable manifest.  Read the separate
    # sealed target table, never the source NPZ target array.
    population_manifest = _read_object(
        population_manifest_path, "population manifest"
    )
    exact_inventory = population_manifest.get("exact_targets")
    if not isinstance(exact_inventory, Mapping):
        raise MF3ZUTrainingError("separate exact-target inventory is missing")
    exact_path = ROOT / str(exact_inventory.get("path", ""))
    if exact_path.resolve() != EXACT_TARGETS_PATH.resolve():
        raise MF3ZUTrainingError("exact-target artifact path drift")
    if (
        not exact_path.is_file()
        or exact_path.is_symlink()
        or int(exact_inventory.get("bytes", -1)) != exact_path.stat().st_size
        or exact_inventory.get("sha256") != sha256_file(exact_path)
    ):
        raise MF3ZUTrainingError("exact-target artifact inventory drift")
    exact_rows = _read_jsonl(exact_path, "exact target")
    if len(exact_rows) != len(population_rows):
        raise MF3ZUTrainingError("exact-target table does not cover the population")
    exact_by_event: dict[str, Mapping[str, object]] = {}
    for row in exact_rows:
        event_id = str(row.get("event_id", ""))
        if not event_id or event_id in exact_by_event:
            raise MF3ZUTrainingError("exact-target event identity is missing or repeated")
        exact_by_event[event_id] = row
    if set(exact_by_event) != set(np.asarray(population["event_id"]).tolist()):
        raise MF3ZUTrainingError("exact-target/population identity drift")
    local_target = np.full(len(population_rows), -1, dtype=np.int64)
    for index, population_row in enumerate(population_rows):
        target_row = exact_by_event[str(population_row["event_id"])]
        if (
            str(target_row.get("source_feature_path"))
            != str(population_row["source_feature_path"])
            or int(target_row.get("source_feature_row_index", -1))
            != int(population_row["feature_row_index"])
            or target_row.get("coordinate_system") != "MF3B_candidate_feature_slot"
        ):
            raise MF3ZUTrainingError("exact-target causal/source identity drift")
        raw_slot = int(target_row.get("target_feature_slot", -1))
        slots = [
            int(value)
            for value in population["ordered_candidate_feature_slots"][index]
        ]
        if raw_slot not in slots:
            raise MF3ZUTrainingError("exact target is outside frozen candidate support")
        local_target[index] = slots.index(raw_slot)
    indices = np.arange(len(population_rows), dtype=np.int64)
    if enforce_frozen_counts:
        episodes = set(np.asarray(population["episode_id"]).tolist())
        scenes = set(np.asarray(population["scene_id"]).tolist())
        observed = (len(indices), len(episodes), len(scenes))
        expected = (
            EXPECTED_POPULATION_ROWS,
            EXPECTED_POPULATION_EPISODES,
            EXPECTED_POPULATION_SCENES,
        )
        if observed != expected:
            raise MF3ZUTrainingError(f"exact target activation count drift: {observed}")

    arrays = ProbeArrays(
        event_id=np.asarray(population["event_id"])[indices],
        scene_id=np.asarray(population["scene_id"])[indices],
        episode_id=np.asarray(population["episode_id"])[indices],
        decision_step=np.asarray(population["decision_step"])[indices],
        scene_fold=np.asarray(population["scene_fold"])[indices],
        candidate_action_ids=tuple(population["candidate_action_ids"][index] for index in indices),
        candidate_features=np.asarray(population["candidate_features"])[indices],
        base_scores=np.asarray(population["base_scores"])[indices],
        candidate_mask=np.asarray(population["candidate_mask"])[indices],
        memory_features=memory[indices],
        memory_count=memory_count[indices],
        memory_required=memory_required[indices],
        target_index=local_target[indices],
        source_feature_path=tuple(population["source_feature_path"][index] for index in indices),
        source_feature_row=np.asarray(population["source_feature_row"])[indices],
        population_rows_before_target=len(population_rows),
        evidence_diagnostics={
            **diagnostics,
            "rankable_decisions_after_target_activation": int(len(indices)),
            "rankable_memory_required_decisions": int(memory_required[indices].sum()),
            "rankable_memory_not_required_decisions": int((~memory_required[indices]).sum()),
            "target_source": "separate_frozen_exact_target_artifact",
        },
    )
    arrays.validate()
    return arrays


def _fit_fold_normalizers(
    data: ProbeArrays,
    train_indices: np.ndarray,
) -> tuple[FeatureNormalizer, FeatureNormalizer]:
    train = np.asarray(train_indices, dtype=np.int64)
    if train.ndim != 1 or len(train) == 0:
        raise MF3ZUTrainingError("normalizer training fold is empty")
    candidate_values = data.candidate_features[train][data.candidate_mask[train]]
    memory_values = data.memory_features[train][data.candidate_mask[train]]
    return (
        fit_feature_normalizer(candidate_values),
        fit_feature_normalizer(memory_values),
    )


def _normalized_inputs(
    data: ProbeArrays,
    candidate_normalizer: FeatureNormalizer,
    memory_normalizer: FeatureNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = candidate_normalizer.transform(data.candidate_features)
    memory = memory_normalizer.transform(data.memory_features)
    candidate[~data.candidate_mask] = 0.0
    memory[~data.candidate_mask] = 0.0
    return candidate, memory


def _donor_memory(
    memory: np.ndarray,
    candidate_mask: np.ndarray,
    donor: np.ndarray,
    relevant_indices: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(memory)
    for index in relevant_indices.tolist():
        source = int(donor[index])
        if source < 0:
            raise MF3ZUTrainingError("missing shuffled memory donor")
        target_count = int(candidate_mask[index].sum())
        source_count = int(candidate_mask[source].sum())
        if source_count < 1:
            raise MF3ZUTrainingError("shuffled donor has no candidate memory")
        # Exact candidate-count matches are preferred by donor assignment.  A
        # deterministic cyclic rank mapping is the sealed fallback.
        for candidate in range(target_count):
            result[index, candidate] = memory[source, candidate % source_count]
    return result


def _batch_scores(
    model: torch.nn.Module,
    candidate: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
    memory: torch.Tensor,
    indices: np.ndarray,
) -> torch.Tensor:
    index = torch.as_tensor(indices, dtype=torch.long, device=candidate.device)
    return model(candidate[index], base[index], mask[index], memory[index])


def fixed_scene_oof_train(
    data: ProbeArrays,
    *,
    device: torch.device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Fit B/C with common randomization and return complete held-scene OOF."""

    data.validate()
    if epochs < 1 or batch_size < 1:
        raise MF3ZUTrainingError("training schedule must be positive")
    rows, candidates = data.candidate_mask.shape
    scores = {
        arm: np.full((rows, candidates), np.nan, dtype=np.float32) for arm in ARMS
    }
    fits: list[dict[str, object]] = []
    for held_fold in range(FOLDS):
        train_indices = np.flatnonzero(data.scene_fold != held_fold)
        held_indices = np.flatnonzero(data.scene_fold == held_fold)
        if len(train_indices) == 0 or len(held_indices) == 0:
            raise MF3ZUTrainingError("scene fold has an empty train or held partition")
        candidate_norm, memory_norm = _fit_fold_normalizers(data, train_indices)
        candidate_np, memory_np = _normalized_inputs(data, candidate_norm, memory_norm)
        donor, donor_diagnostics = shuffled_memory_donor_indices(
            data.event_id,
            data.memory_count,
            train_indices,
            held_indices,
            candidate_counts=data.candidate_mask.sum(axis=1),
            seed=FIXED_SEED,
        )
        relevant = np.concatenate((train_indices, held_indices))
        shuffled_np = _donor_memory(
            memory_np, data.candidate_mask, donor, relevant
        )

        candidate = torch.from_numpy(candidate_np).to(device)
        base = torch.from_numpy(data.base_scores.astype(np.float32)).to(device)
        mask = torch.from_numpy(data.candidate_mask).to(device)
        target = torch.from_numpy(data.target_index.astype(np.int64)).to(device)
        true_memory = torch.from_numpy(memory_np).to(device)
        shuffled_memory = torch.from_numpy(shuffled_np).to(device)
        model_b, model_c, initial_hash = common_initialized_rerankers(
            CANDIDATE_EVIDENCE_FEATURE_DIM, seed=FIXED_SEED
        )
        model_b.to(device)
        model_c.to(device)
        if parameter_sha256(model_b) != parameter_sha256(model_c):
            raise RuntimeError("B/C initialization changed during device transfer")
        optimizer_b = torch.optim.AdamW(
            model_b.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        optimizer_c = torch.optim.AdamW(
            model_c.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        rng = np.random.default_rng(FIXED_SEED + held_fold)
        batch_digest = hashlib.sha256()
        final_loss_b = final_loss_c = float("nan")
        model_b.train()
        model_c.train()
        for _epoch in range(int(epochs)):
            order = rng.permutation(train_indices)
            batch_digest.update(np.asarray(order, dtype=np.int64).tobytes())
            for start in range(0, len(order), int(batch_size)):
                batch = np.asarray(order[start:start + int(batch_size)], dtype=np.int64)
                batch_index = torch.as_tensor(batch, dtype=torch.long, device=device)

                optimizer_b.zero_grad(set_to_none=True)
                score_b = model_b(
                    candidate[batch_index], base[batch_index], mask[batch_index],
                    true_memory[batch_index],
                )
                loss_b = masked_candidate_cross_entropy(
                    score_b, target[batch_index], mask[batch_index]
                )
                loss_b.backward()
                optimizer_b.step()

                optimizer_c.zero_grad(set_to_none=True)
                score_c = model_c(
                    candidate[batch_index], base[batch_index], mask[batch_index],
                    shuffled_memory[batch_index],
                )
                loss_c = masked_candidate_cross_entropy(
                    score_c, target[batch_index], mask[batch_index]
                )
                loss_c.backward()
                optimizer_c.step()
                final_loss_b = float(loss_b.detach().cpu())
                final_loss_c = float(loss_c.detach().cpu())

        model_b.eval()
        model_c.eval()
        with torch.no_grad():
            held_tensor = torch.as_tensor(held_indices, dtype=torch.long, device=device)
            scores[ARM_CURRENT][held_indices] = data.base_scores[held_indices]
            scores[ARM_MEMORY][held_indices] = model_b(
                candidate[held_tensor], base[held_tensor], mask[held_tensor],
                true_memory[held_tensor],
            ).cpu().numpy()
            scores[ARM_SHUFFLED][held_indices] = model_c(
                candidate[held_tensor], base[held_tensor], mask[held_tensor],
                shuffled_memory[held_tensor],
            ).cpu().numpy()
        fits.append({
            "held_fold": held_fold,
            "train_decisions": int(len(train_indices)),
            "held_decisions": int(len(held_indices)),
            "train_raw_scenes": len(set(data.scene_id[train_indices].tolist())),
            "held_raw_scenes": len(set(data.scene_id[held_indices].tolist())),
            "normalization_fit_indices_sha256": hashlib.sha256(
                np.asarray(train_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
            "normalization_fit_train_fold_only": True,
            "B_C_initial_parameter_sha256": initial_hash,
            "B_C_common_initialization": True,
            "B_C_common_batch_order": True,
            "batch_order_sha256": batch_digest.hexdigest(),
            "final_train_loss_B": final_loss_b,
            "final_train_loss_C": final_loss_c,
            "shuffled_memory": donor_diagnostics,
        })

    for arm in ARMS:
        if not np.isfinite(scores[arm][data.candidate_mask]).all():
            raise MF3ZUTrainingError(f"incomplete/non-finite OOF scores for {arm}")
        scores[arm][~data.candidate_mask] = float("-inf")
    evaluation = evaluate_three_arm_probe(
        scores,
        data.target_index,
        data.candidate_mask,
        data.scene_id,
        data.memory_required,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    gates = apply_fixed_rxr_gates(evaluation)
    return scores, {
        "complete_five_fold_oof": True,
        "folds": fits,
        "fixed_schedule": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": FIXED_SEED,
            "early_stopping": False,
            "best_checkpoint_selection": False,
            "model_or_threshold_selection": False,
        },
        "ETP_frozen": True,
        "candidate_generator_frozen": True,
        "visual_backbone_frozen": True,
        "topology_encoder_frozen": True,
        "checkpoint_written": False,
        "public_split_access": dict(PUBLIC_CLOSED),
        "full_navigation_run": False,
        "evaluation": evaluation,
        "gates": gates,
    }


def _result(data: ProbeArrays, training: Mapping[str, object]) -> dict[str, object]:
    gates = training["gates"]
    evaluation = training["evaluation"]
    return {
        "schema_version": "revealnav-mf3zu-rxr-evidence-memory-feasibility-result/1",
        "revision": REVISION,
        "status": gates["status"],
        "final_PASS_FAIL": gates["final_PASS_FAIL"],
        "scope": {
            "dataset": "RxR",
            "R2R_evaluated": False,
            "train_development_only": True,
            "decision_feasibility_probe_only": True,
        },
        "population": {
            "target_blind_decisions": int(data.population_rows_before_target),
            "rankable_exact_target_decisions": int(len(data.event_id)),
            "episodes": len(set(data.episode_id.tolist())),
            "raw_scenes": len(set(data.scene_id.tolist())),
        },
        "memory_required": evaluation["subgroup_support"]["MEMORY_REQUIRED"],
        "arms": list(ARMS),
        "metrics_per_domain": {"RxR": evaluation["metrics"]},
        "pairwise_deltas": evaluation["pairwise_deltas"],
        "scene_bootstrap_CI": evaluation["scene_bootstrap_CI"],
        "evidence_diagnostics": dict(data.evidence_diagnostics),
        "training": {
            key: value for key, value in training.items()
            if key not in {"evaluation", "gates"}
        },
        "fixed_gates": gates,
        "public_split_access": dict(PUBLIC_CLOSED),
        "full_navigation_run": False,
        "checkpoint_generated": False,
        "checkpoint_for_deployment": False,
        "deployment_authorized": False,
        "MF3ZT_two_domain_pass_claimed": False,
    }


def _write_fsynced_part(path: Path, payload: bytes) -> None:
    """Write one non-authoritative result payload without exposing its final name."""

    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_oof_jsonl(oof_rows: Sequence[Mapping[str, object]]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                dict(row),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for row in oof_rows
        )
        + "\n"
    ).encode("utf-8")


def _commit_embedded_result(
    result_path: Path,
    result: Mapping[str, object],
    oof_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Atomically publish one immutable result containing complete OOF rows.

    Embedding OOF evidence avoids the unavoidable crash window between two
    independent final-name renames.  The only authoritative name appears after
    the complete, fsynced payload is ready.
    """

    if not oof_rows:
        raise MF3ZUTrainingError("cannot publish an empty OOF result")
    directory = result_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    result_part = result_path.with_name(result_path.name + ".part")
    if result_path.exists() or result_path.is_symlink():
        raise MF3ZUTrainingError("immutable result already exists")
    # A stale part is non-authoritative by construction and can only be from an
    # interrupted write before the single final rename.
    if result_part.is_symlink():
        raise MF3ZUTrainingError(f"unsafe stale result partial: {result_part}")
    if result_part.exists():
        result_part.unlink()

    canonical_oof = _canonical_oof_jsonl(oof_rows)
    finalized = dict(result)
    finalized["OOF_predictions"] = {
        "storage": "embedded_in_result",
        "rows": [dict(row) for row in oof_rows],
        "row_count": len(oof_rows),
        "canonical_jsonl_bytes": len(canonical_oof),
        "canonical_jsonl_sha256": hashlib.sha256(canonical_oof).hexdigest(),
        "complete": True,
    }
    result_payload = (
        json.dumps(finalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        _write_fsynced_part(result_part, result_payload)
        _fsync_directory(directory)
        os.replace(result_part, result_path)
        _fsync_directory(directory)
    except BaseException:
        try:
            result_part.unlink()
        except FileNotFoundError:
            pass
        try:
            _fsync_directory(directory)
        except OSError:
            pass
        raise
    return finalized


def _oof_rows(
    data: ProbeArrays,
    scores: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, event_id in enumerate(data.event_id):
        count = int(data.candidate_mask[index].sum())
        row = {
            "event_id": str(event_id),
            "scene_id": str(data.scene_id[index]),
            "episode_id": str(data.episode_id[index]),
            "decision_step": int(data.decision_step[index]),
            "scene_fold": int(data.scene_fold[index]),
            "memory_required": bool(data.memory_required[index]),
            "candidate_action_ids": list(data.candidate_action_ids[index]),
            "target_index": int(data.target_index[index]),
            "scores": {
                arm: [float(value) for value in scores[arm][index, :count]]
                for arm in ARMS
            },
        }
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    try:
        verify_protocol(PROTOCOL_PATH)
        if RESULT_PATH.exists() or AUDIT_PATH.exists():
            raise MF3ZUTrainingError("MF3ZU result/audit already exists")
        data = load_frozen_probe_inputs()
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise MF3ZUTrainingError("requested CUDA device is unavailable")
        torch.manual_seed(FIXED_SEED)
        np.random.seed(FIXED_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FIXED_SEED)
        scores, training = fixed_scene_oof_train(data, device=device)
        result = _result(data, training)
        result = _commit_embedded_result(
            RESULT_PATH,
            result,
            _oof_rows(data, scores),
        )
    except (OSError, KeyError, TypeError, ValueError, ProtocolError, MF3ZUTrainingError) as error:
        print(f"MF3ZU_RXR_TRAIN_FAIL_CLOSED: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "result": str(RESULT_PATH),
        "status": result["status"],
        "rankable_decisions": len(data.event_id),
        "memory_required": int(data.memory_required.sum()),
        "checkpoint_generated": False,
        "full_navigation_run": False,
        "public_split_access": dict(PUBLIC_CLOSED),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
