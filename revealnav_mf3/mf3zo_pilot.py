"""Outcome-blind MF3ZO pilot selection and causal-prefix reconstruction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .mf3zo_temporal_schema import (
    CausalTemporalRecord,
    CausalTemporalStep,
    ORACLE_FIELDS,
    TemporalOracleLabel,
    causal_prefix_sha256,
    inference_tensors,
)


PILOT_EVENTS = 150
EVENTS_PER_DOMAIN = 75
REQUIRED_DOMAINS = ("R2R", "RxR")
SELECTION_SALT = "mf3zo-temporal-oracle-gap-v1-pilot/1"
EXPECTED_CANONICAL_IDENTITY = (
    "7047fe8e3514d6037926f77a2883e9f0cdf094d5b077aa82febba64260b07bae"
)
POLICY_TRACE_NAMES = (
    "step",
    "policy_risk_adjusted_score",
    "native_margin",
    "minimum_top2_advantage",
    "median_top2_advantage",
    "robust_top2_advantage",
    "ensemble_mad",
    "cold_start_floor_ratio",
    "cold_start_relative_mad",
    "candidate_count",
)


class PilotDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class PilotCandidate:
    event_id: str
    dataset: str
    scene_id: str
    episode_id: str
    decision_step: int
    source: str
    feature_path: str

    def __post_init__(self) -> None:
        if self.dataset not in REQUIRED_DOMAINS:
            raise ValueError("MF3ZO pilot candidate has an unknown domain")
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.event_id,
                self.scene_id,
                self.episode_id,
                self.source,
                self.feature_path,
            )
        ):
            raise ValueError("MF3ZO pilot candidate identity is incomplete")
        if self.decision_step < 0:
            raise ValueError("MF3ZO decision step must be non-negative")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_file(path: Path, root: Path) -> dict:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink() or root.resolve() not in resolved.parents:
        raise PilotDataError(f"invalid project-local source: {path}")
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_event_id(
    dataset: object,
    scene_id: object,
    episode_id: object,
    decision_step: int,
) -> str:
    value = {
        "dataset": str(dataset),
        "scene_id": str(scene_id),
        "episode_id": str(episode_id),
        "decision_step": int(decision_step),
    }
    if any(not value[key] for key in ("dataset", "scene_id", "episode_id")):
        raise ValueError("empty MF3ZO event identity")
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()


def _key(*parts: object) -> str:
    return hashlib.sha256(
        "\0".join((SELECTION_SALT, *(str(part) for part in parts))).encode("utf-8")
    ).hexdigest()


def select_balanced_candidates(
    candidates: Sequence[PilotCandidate],
    *,
    events_per_domain: int = EVENTS_PER_DOMAIN,
) -> tuple[PilotCandidate, ...]:
    """Deterministically select events using identities only, never outcomes."""

    if events_per_domain != EVENTS_PER_DOMAIN:
        raise ValueError("MF3ZO pilot size is frozen at 75 events per domain")
    identities = [value.event_id for value in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError("MF3ZO candidate event identities are not unique")
    selected: list[PilotCandidate] = []
    for domain in REQUIRED_DOMAINS:
        domain_rows = [value for value in candidates if value.dataset == domain]
        by_scene: dict[str, list[PilotCandidate]] = {}
        for value in domain_rows:
            by_scene.setdefault(value.scene_id, []).append(value)
        if len(by_scene) < 5 or len(domain_rows) < events_per_domain:
            raise ValueError(f"insufficient outcome-blind pilot capacity: {domain}")
        ordered_scenes = sorted(
            by_scene,
            key=lambda scene: (_key("scene", domain, scene), scene),
        )
        for scene in ordered_scenes:
            by_scene[scene].sort(key=lambda value: (
                _key("event", domain, scene, value.event_id), value.event_id,
            ))
        offsets = {scene: 0 for scene in ordered_scenes}
        domain_selected: list[PilotCandidate] = []
        while len(domain_selected) < events_per_domain:
            progressed = False
            for scene in ordered_scenes:
                offset = offsets[scene]
                values = by_scene[scene]
                if offset == len(values):
                    continue
                domain_selected.append(values[offset])
                offsets[scene] += 1
                progressed = True
                if len(domain_selected) == events_per_domain:
                    break
            if not progressed:
                raise ValueError(f"unable to fill fixed pilot allocation: {domain}")
        selected.extend(domain_selected)
    if len(selected) != PILOT_EVENTS or len({value.event_id for value in selected}) != PILOT_EVENTS:
        raise RuntimeError("MF3ZO fixed pilot allocation drift")
    return tuple(selected)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PilotDataError(f"cannot load project source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonl_objects(path: Path) -> tuple[dict, ...]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise PilotDataError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {token}")
                ),
            )
        except (ValueError, json.JSONDecodeError) as error:
            raise PilotDataError(f"invalid trace row {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise PilotDataError(f"trace row is not an object: {path}:{line_number}")
        records.append(value)
    if not records:
        raise PilotDataError(f"empty JSONL source: {path}")
    return tuple(records)


def _strict_json_lines(path: Path) -> tuple[dict, ...]:
    records = _jsonl_objects(path)
    indices = [value.get("step") for value in records]
    if any(type(value) is not int or value < 0 for value in indices) or any(
        left >= right for left, right in zip(indices, indices[1:])
    ):
        raise PilotDataError(f"trace steps are not strictly increasing: {path}")
    return tuple(records)


def trace_path_for_row(row: Mapping[str, object], root: Path) -> Path:
    feature = root / str(row["feature"]["path"])
    source = str(row["source"])
    if source == "mf3zk_dsr_v1_existing_exact":
        trace = feature.parent / "controller_trace.jsonl"
    elif source in {"mf3zl_parent_dense_exact", "mf3zl_v1r1_variant_exact"}:
        trace = feature.parent.parent / "proposal_trace.jsonl"
    else:
        raise PilotDataError(f"unknown canonical source: {source}")
    if not trace.is_file() or trace.is_symlink() or root.resolve() not in trace.resolve().parents:
        raise PilotDataError(f"invalid causal trace: {trace}")
    return trace


def _candidate_ids(record: Mapping[str, object]) -> tuple[str, ...]:
    raw = record.get("current_local_action_ids")
    if not isinstance(raw, list):
        raise PilotDataError("trace candidate identities are unavailable")
    values = tuple(str(value) for value in raw)
    if any(not value for value in values) or len(set(values)) != len(values):
        raise PilotDataError("trace candidate identities are invalid")
    return values


def _finite_or_missing(value: object) -> tuple[float, bool]:
    if value is None:
        return 0.0, False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotDataError("causal scalar has invalid type")
    result = float(value)
    if not math.isfinite(result):
        raise PilotDataError("causal scalar is non-finite")
    return result, True


def _policy_trace_features(record: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    mask: list[bool] = []
    for name in POLICY_TRACE_NAMES:
        if name == "candidate_count":
            candidates = _candidate_ids(record)
            values.append(float(len(candidates)))
            mask.append(True)
        else:
            value, available = _finite_or_missing(record.get(name))
            values.append(value)
            mask.append(available)
    return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=np.bool_)


def _terminal_action_ids(record: Mapping[str, object]) -> tuple[str, str]:
    native = record.get("feature_native_action_id") or record.get("native_action_id")
    alternative = record.get("feature_alternative_action_id") or record.get(
        "adapted_action_id"
    )
    if not isinstance(native, str) or not native or not isinstance(
        alternative, str
    ) or not alternative or native == alternative:
        raise PilotDataError("terminal native/alternative identities are unavailable")
    candidates = _candidate_ids(record)
    if native not in candidates or alternative not in candidates:
        raise PilotDataError("terminal intervention action is not executable")
    return native, alternative


def _native_id(record: Mapping[str, object]) -> str | None:
    value = record.get("native_action_id")
    return value if isinstance(value, str) and value else None


def reconstruct_record(
    row: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> CausalTemporalRecord:
    decision = row["decision"]
    arrays = row["arrays"]
    decision_step = int(decision["step"])
    prefix = tuple(value for value in records if int(value["step"]) <= decision_step)
    if not prefix or int(prefix[-1]["step"]) != decision_step:
        raise PilotDataError("trace does not contain the selected decision prefix")
    current = prefix[-1]
    if _candidate_ids(current) != tuple(
        str(value) for value in decision["current_local_action_ids"]
    ):
        raise PilotDataError("decision candidate identities differ from trace")
    native, alternative = _terminal_action_ids(current)
    instruction = np.asarray(arrays["instruction"], dtype=np.float32)
    checkpoint = np.asarray(arrays["checkpoint"], dtype=np.float32)
    native_embedding = np.asarray(arrays["native"], dtype=np.float32)
    alternative_embedding = np.asarray(arrays["alternative"], dtype=np.float32)
    steps: list[CausalTemporalStep] = []
    for value in prefix:
        terminal = int(value["step"]) == decision_step
        policy, policy_mask = _policy_trace_features(value)
        steps.append(CausalTemporalStep(
            step=int(value["step"]),
            native_action_id=_native_id(value),
            candidate_action_ids=_candidate_ids(value),
            policy_features=policy,
            policy_feature_mask=policy_mask,
            instruction_embedding=instruction,
            checkpoint_embedding=checkpoint if terminal else None,
            embedded_action_ids=(native, alternative) if terminal else (),
            action_embeddings=(
                np.stack((native_embedding, alternative_embedding))
                if terminal else None
            ),
        ))
    prefix_sha = causal_prefix_sha256(
        str(row["dataset"]),
        str(row["scene_id"]),
        str(row["episode_id"]),
        decision_step,
        steps,
    )
    return CausalTemporalRecord(
        dataset=str(row["dataset"]),
        scene_id=str(row["scene_id"]),
        episode_id=str(row["episode_id"]),
        decision_step=decision_step,
        steps=tuple(steps),
        prefix_sha256=prefix_sha,
    )


def unavailable_oracle_label(event_id: str) -> TemporalOracleLabel:
    return TemporalOracleLabel(
        event_id=event_id,
        target_in_set=None,
        candidate_separated=None,
        evidence_closed=None,
        reveal_interval=None,
        expiry_step=None,
        resolvable=None,
        unavailable_fields=ORACLE_FIELDS,
        provenance=(
            "UNAVAILABLE: historical exact traces do not contain independently "
            "validated target-branch factors, evidence closure, or Reveal/Expiry "
            "oracle supervision; no surrogate was substituted"
        ),
    )


def _atomic_text(path: Path, payload: str) -> None:
    if path.exists() or path.is_symlink():
        raise PilotDataError(f"refusing to overwrite MF3ZO output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise PilotDataError(f"stale MF3ZO partial output: {partial}")
    partial.write_text(payload, encoding="utf-8")
    os.replace(partial, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise PilotDataError(f"refusing to overwrite MF3ZO output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise PilotDataError(f"stale MF3ZO partial output: {partial}")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(partial, path)


def _json_line(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _record_metadata(
    event_id: str,
    record: CausalTemporalRecord,
    array_inventory: Mapping[str, object],
    source: Mapping[str, object],
) -> dict:
    return {
        "schema_version": "revealnav-mf3zo-causal-temporal-record/1",
        "event_id": event_id,
        "dataset": record.dataset,
        "scene_id": record.scene_id,
        "episode_id": record.episode_id,
        "decision_step": record.decision_step,
        "prefix_sha256": record.prefix_sha256,
        "prefix_length": len(record.steps),
        "full_prefix_embedding_complete": record.full_prefix_embedding_complete,
        "arrays": dict(array_inventory),
        "source": dict(source),
        "steps": [
            {
                "step": value.step,
                "native_action_id": value.native_action_id,
                "candidate_action_ids": list(value.candidate_action_ids),
                "embedded_action_ids": list(value.embedded_action_ids),
                "policy_feature_mask": value.policy_feature_mask.tolist(),
                "checkpoint_embedding_available": value.checkpoint_embedding is not None,
                "action_embedding_coverage": (
                    len(value.embedded_action_ids) / len(value.candidate_action_ids)
                    if value.candidate_action_ids else 0.0
                ),
            }
            for value in record.steps
        ],
    }


def _oracle_mapping(value: TemporalOracleLabel) -> dict:
    return {
        "schema_version": "revealnav-mf3zo-temporal-oracle-label/1",
        "event_id": value.event_id,
        "status": "AVAILABLE_VERIFIED" if value.complete else "UNAVAILABLE",
        "target_in_set": value.target_in_set,
        "candidate_separated": value.candidate_separated,
        "evidence_closed": value.evidence_closed,
        "reveal_interval": value.reveal_interval,
        "expiry_step": value.expiry_step,
        "resolvable": value.resolvable,
        "unavailable_fields": list(value.unavailable_fields),
        "provenance": value.provenance,
    }


def load_causal_records(path: Path, root: Path) -> dict[str, CausalTemporalRecord]:
    """Load the model-visible record store without opening an outcome source."""

    result: dict[str, CausalTemporalRecord] = {}
    for raw in _jsonl_objects(path):
        event_id = str(raw.get("event_id", ""))
        if not event_id or event_id in result:
            raise PilotDataError("causal record event identity is missing or duplicated")
        arrays_meta = raw.get("arrays")
        if not isinstance(arrays_meta, dict):
            raise PilotDataError("causal record array inventory is missing")
        array_path = root / str(arrays_meta.get("path", ""))
        if (
            inventory_file(array_path, root).get("sha256")
            != arrays_meta.get("sha256")
            or array_path.stat().st_size != int(arrays_meta.get("bytes", -1))
        ):
            raise PilotDataError("causal record array inventory drift")
        with np.load(array_path, allow_pickle=False) as source:
            expected_keys = {
                "policy_features", "policy_feature_mask",
                "instruction_embedding", "checkpoint_embedding",
                "checkpoint_embedding_mask", "action_embeddings",
                "action_embedding_mask",
            }
            if set(source.files) != expected_keys:
                raise PilotDataError("causal record array schema drift")
            arrays = {key: np.asarray(source[key]) for key in source.files}
        steps_meta = raw.get("steps")
        if not isinstance(steps_meta, list) or not steps_meta:
            raise PilotDataError("causal record step metadata is missing")
        rows = len(steps_meta)
        if any(len(arrays[key]) != rows for key in arrays):
            raise PilotDataError("causal record arrays are not step-aligned")
        steps: list[CausalTemporalStep] = []
        for index, metadata in enumerate(steps_meta):
            candidates = tuple(str(value) for value in metadata["candidate_action_ids"])
            embedded = tuple(str(value) for value in metadata["embedded_action_ids"])
            candidate_index = {value: position for position, value in enumerate(candidates)}
            action_values = (
                np.stack([
                    arrays["action_embeddings"][index, candidate_index[action_id]]
                    for action_id in embedded
                ])
                if embedded else None
            )
            if embedded and not all(
                arrays["action_embedding_mask"][index, candidate_index[action_id]]
                for action_id in embedded
            ):
                raise PilotDataError("embedded action mask contradicts metadata")
            checkpoint = (
                arrays["checkpoint_embedding"][index]
                if bool(arrays["checkpoint_embedding_mask"][index]) else None
            )
            steps.append(CausalTemporalStep(
                step=int(metadata["step"]),
                native_action_id=metadata.get("native_action_id"),
                candidate_action_ids=candidates,
                policy_features=arrays["policy_features"][index],
                policy_feature_mask=arrays["policy_feature_mask"][index],
                instruction_embedding=arrays["instruction_embedding"][index],
                checkpoint_embedding=checkpoint,
                embedded_action_ids=embedded,
                action_embeddings=action_values,
            ))
        result[event_id] = CausalTemporalRecord(
            dataset=str(raw["dataset"]),
            scene_id=str(raw["scene_id"]),
            episode_id=str(raw["episode_id"]),
            decision_step=int(raw["decision_step"]),
            steps=tuple(steps),
            prefix_sha256=str(raw["prefix_sha256"]),
        )
    return result


def load_oracle_labels(path: Path) -> dict[str, TemporalOracleLabel]:
    result: dict[str, TemporalOracleLabel] = {}
    for raw in _jsonl_objects(path):
        event_id = str(raw.get("event_id", ""))
        if not event_id or event_id in result:
            raise PilotDataError("oracle event identity is missing or duplicated")
        def optional_boolean_tuple(name: str):
            value = raw.get(name)
            return None if value is None else tuple(value)
        reveal = raw.get("reveal_interval")
        result[event_id] = TemporalOracleLabel(
            event_id=event_id,
            target_in_set=optional_boolean_tuple("target_in_set"),
            candidate_separated=optional_boolean_tuple("candidate_separated"),
            evidence_closed=optional_boolean_tuple("evidence_closed"),
            reveal_interval=None if reveal is None else tuple(reveal),
            expiry_step=raw.get("expiry_step"),
            resolvable=raw.get("resolvable"),
            unavailable_fields=tuple(raw.get("unavailable_fields", ())),
            provenance=str(raw.get("provenance", "")),
        )
    return result


def build_pilot(root: Path, output: Path) -> dict:
    """Build the sealed-input pilot without reading outcomes for selection."""

    root = root.resolve()
    car_path = root / "scripts/train_mf3zm_car.py"
    car = _load_module(car_path, "mf3zo_sealed_car_source")
    protocol = car.verify_protocol()
    rows = car._canonical_rows()
    if (
        len(rows) != 1540
        or car._identity_hash(rows) != EXPECTED_CANONICAL_IDENTITY
        or dict(Counter(str(value["dataset"]) for value in rows))
        != {"R2R": 543, "RxR": 997}
    ):
        raise PilotDataError("sealed canonical exact population drift")
    blacklist = set(str(value) for value in protocol["known_consumed_scene_ids"])
    row_by_event: dict[str, Mapping[str, object]] = {}
    candidates: list[PilotCandidate] = []
    for row in rows:
        event_id = canonical_event_id(
            row["dataset"], row["scene_id"], row["episode_id"],
            int(row["decision"]["step"]),
        )
        if str(row["scene_id"]) in blacklist:
            raise PilotDataError("consumed confirmation scene entered MF3ZO source")
        candidate = PilotCandidate(
            event_id=event_id,
            dataset=str(row["dataset"]),
            scene_id=str(row["scene_id"]),
            episode_id=str(row["episode_id"]),
            decision_step=int(row["decision"]["step"]),
            source=str(row["source"]),
            feature_path=str(row["feature"]["path"]),
        )
        candidates.append(candidate)
        row_by_event[event_id] = row
    selected = select_balanced_candidates(candidates)

    trace_cache: dict[Path, tuple[dict, ...]] = {}
    record_lines: list[str] = []
    oracle_lines: list[str] = []
    selection_rows: list[dict] = []
    source_files: dict[str, dict] = {}
    prefix_lengths: list[int] = []
    complete_prefix_embeddings = 0
    terminal_embedding_complete = 0
    policy_observed = 0
    policy_total = 0
    for candidate in selected:
        row = row_by_event[candidate.event_id]
        feature = root / candidate.feature_path
        trace = trace_path_for_row(row, root)
        records = trace_cache.setdefault(trace, _strict_json_lines(trace))
        record = reconstruct_record(row, records)
        tensors = inference_tensors(record)
        array_path = output / "causal_records" / f"{candidate.event_id}.npz"
        _atomic_npz(array_path, tensors)
        array_inventory = inventory_file(array_path, root)
        feature_inventory = inventory_file(feature, root)
        trace_inventory = inventory_file(trace, root)
        source_files[feature_inventory["path"]] = feature_inventory
        source_files[trace_inventory["path"]] = trace_inventory
        source = {
            "canonical_source": candidate.source,
            "feature": feature_inventory,
            "trace": trace_inventory,
        }
        record_lines.append(_json_line(_record_metadata(
            candidate.event_id, record, array_inventory, source,
        )))
        oracle = unavailable_oracle_label(candidate.event_id)
        oracle_lines.append(_json_line(_oracle_mapping(oracle)))
        selection_rows.append({
            "event_id": candidate.event_id,
            "dataset": candidate.dataset,
            "scene_id": candidate.scene_id,
            "episode_id": candidate.episode_id,
            "decision_step": candidate.decision_step,
            "source": candidate.source,
            "feature_path": candidate.feature_path,
            "prefix_sha256": record.prefix_sha256,
        })
        prefix_lengths.append(len(record.steps))
        complete_prefix_embeddings += int(record.full_prefix_embedding_complete)
        terminal_embedding_complete += int(record.steps[-1].checkpoint_embedding is not None)
        for step in record.steps:
            policy_observed += int(step.policy_feature_mask.sum())
            policy_total += len(step.policy_feature_mask)

    records_path = output / "MF3ZO_CAUSAL_TEMPORAL_RECORDS.jsonl"
    oracle_path = output / "MF3ZO_TEMPORAL_ORACLE_LABELS.jsonl"
    selection_path = output / "MF3ZO_PILOT_SELECTION.json"
    audit_path = output / "MF3ZO_PILOT_DATA_AUDIT.json"
    _atomic_text(records_path, "\n".join(record_lines) + "\n")
    _atomic_text(oracle_path, "\n".join(oracle_lines) + "\n")
    selection_value = {
        "schema_version": "revealnav-mf3zo-pilot-selection/1",
        "status": "SEALED_INPUT_POPULATION_SELECTED_OUTCOME_BLIND",
        "revision": "mf3zo_temporal_oracle_gap_v1",
        "selection_salt": SELECTION_SALT,
        "selection_inputs": [
            "dataset", "raw_mp3d_scene_id", "episode_id", "decision_step",
            "canonical_event_id", "causal_source_availability",
        ],
        "selection_forbidden_inputs": [
            "delta_utility", "catastrophic", "CAR/RCSP/DSR error",
            "future outcome", "public split metric",
        ],
        "events": selection_rows,
        "source_inventory": [source_files[key] for key in sorted(source_files)],
    }
    _atomic_text(selection_path, json.dumps(
        selection_value, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    domain_counts = dict(Counter(value.dataset for value in selected))
    domain_scenes = {
        domain: len({value.scene_id for value in selected if value.dataset == domain})
        for domain in REQUIRED_DOMAINS
    }
    audit_value = {
        "schema_version": "revealnav-mf3zo-pilot-data-audit/1",
        "status": "PILOT_CAUSAL_RECONSTRUCTION_COMPLETE_ORACLE_UNAVAILABLE",
        "revision": "mf3zo_temporal_oracle_gap_v1",
        "events": len(selected),
        "raw_mp3d_scenes": len({value.scene_id for value in selected}),
        "domain_counts": domain_counts,
        "domain_scene_counts": domain_scenes,
        "prefix_rows": int(sum(prefix_lengths)),
        "prefix_length": {
            "minimum": min(prefix_lengths),
            "maximum": max(prefix_lengths),
            "mean": float(np.mean(prefix_lengths)),
        },
        "causal_coverage": {
            "terminal_snapshot_embedding_complete_events": terminal_embedding_complete,
            "full_prefix_embedding_complete_events": complete_prefix_embeddings,
            "policy_scalar_observed_fraction": (
                float(policy_observed / policy_total) if policy_total else 0.0
            ),
            "future_steps_saved": 0,
            "simulator_pose_saved": 0,
            "navmesh_features_saved": 0,
        },
        "oracle_coverage": {
            "complete_verified_labels": 0,
            "unavailable_labels": len(selected),
            "unavailable_fields": list(ORACLE_FIELDS),
            "surrogate_labels_substituted": False,
        },
        "probe_readiness": {
            "probe_a_oracle_relevance": False,
            "probe_b_temporal_observability": False,
            "probe_c_learned_state_relevance": False,
            "reason": (
                "complete verified UAD/Reveal/Expiry oracle supervision is absent; "
                "historical per-prefix checkpoint/action embeddings are incomplete"
            ),
        },
        "consumed_confirmation_intersection": [],
        "public_split_access": {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
        "files": {
            "selection": inventory_file(selection_path, root),
            "causal_records": inventory_file(records_path, root),
            "oracle_labels": inventory_file(oracle_path, root),
        },
    }
    _atomic_text(audit_path, json.dumps(
        audit_value, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    return audit_value


__all__ = [
    "EVENTS_PER_DOMAIN",
    "EXPECTED_CANONICAL_IDENTITY",
    "PILOT_EVENTS",
    "PilotCandidate",
    "PilotDataError",
    "build_pilot",
    "canonical_event_id",
    "inventory_file",
    "load_causal_records",
    "load_oracle_labels",
    "reconstruct_record",
    "select_balanced_candidates",
    "sha256_file",
    "trace_path_for_row",
    "unavailable_oracle_label",
]
