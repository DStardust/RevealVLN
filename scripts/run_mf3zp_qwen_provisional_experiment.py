#!/usr/bin/env python3
"""Run a fixed, Qwen-provisional MF3ZP training/OOF feasibility diagnostic.

The script deliberately has no public-evaluation or checkpoint command.  It
uses Qwen labels only as provisional semantic supervision/augmentation and
keeps exact utility evaluation on the strictly matched development subset.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORRECTION_SCRIPT = ROOT / "scripts/repair_mf3zp_qwen_evidence_v1_1.py"
METHOD = ROOT / "METHOD_REVISION_3ZP_QWEN_PROVISIONAL_TRAINING.md"
OUTPUT = ROOT / "artifacts/training/mf3zp_qwen_provisional_exploratory_v1"
PROTOCOL = OUTPUT / "MF3ZP_QWEN_PROVISIONAL_PROTOCOL.json"
RESULT = OUTPUT / "MF3ZP_QWEN_PROVISIONAL_TRAINING_RESULT.json"
FEATURES = OUTPUT / "MF3ZP_QWEN_PROVISIONAL_FEATURES.jsonl"

SCHEMA = "revealnav-mf3zp-qwen-provisional-exploratory/1"
MODEL = "qwen3.8-max"
FOLDS = 5
RIDGE_L2 = 1.0
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260901
UTILITY_KEYS = ("nDTW", "SDTW", "SPL")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


correction = load_module(CORRECTION_SCRIPT, "mf3zp_qwen_evidence_correction")
base = correction.base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def inventory(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in resolved.parents:
        raise RuntimeError(f"invalid project-local source: {path}")
    return {"path": str(resolved.relative_to(ROOT.resolve())), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: object, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise RuntimeError(f"refusing to overwrite {path}")
    part = path.with_name(path.name + ".part")
    if part.exists() or part.is_symlink():
        raise RuntimeError(f"stale partial output: {part}")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]], *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise RuntimeError(f"refusing to overwrite {path}")
    part = path.with_name(path.name + ".part")
    if part.exists() or part.is_symlink():
        raise RuntimeError(f"stale partial output: {part}")
    with part.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(part, path)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def scene_folds(events: Sequence[Mapping[str, object]]) -> np.ndarray:
    values = np.asarray([int(event["scene_fold"]) for event in events], dtype=np.int64)
    if len(values) != len(events) or set(values.tolist()) != set(range(FOLDS)):
        raise RuntimeError("pilot scene-fold assignment is incomplete")
    return values


def _heading_features(candidates: Sequence[Mapping[str, object]]) -> np.ndarray:
    headings = np.asarray([float(item["relative_heading_rad"]) for item in candidates], dtype=np.float64)
    if len(headings) == 0 or not np.isfinite(headings).all():
        raise RuntimeError("candidate headings are unavailable/non-finite")
    return np.asarray([
        float(len(headings)),
        float(np.mean(np.sin(headings))),
        float(np.mean(np.cos(headings))),
        float(np.std(headings)),
        float(np.min(np.abs(headings))),
        float(np.max(np.abs(headings))),
    ], dtype=np.float64)


def _causal_step_vector(task: Mapping[str, object], previous: Sequence[str] | None) -> np.ndarray:
    candidates = [dict(item) for item in task["contract"]["current_candidates"]]
    ids = [str(item["alias"]) for item in candidates]
    current = set(ids)
    prior = set(previous or ())
    union = current | prior
    jaccard = len(current & prior) / len(union) if union else 1.0
    heading = _heading_features(candidates)
    storyboard_steps = task["causal_storyboard"]["steps"]
    vector = np.concatenate((
        np.asarray([float(task["prefix_step"])], dtype=np.float64),
        heading,
        np.asarray([
            float(len(current - prior)),
            float(len(prior - current)),
            float(jaccard),
            float(len(storyboard_steps)),
        ], dtype=np.float64),
    ))
    if not np.isfinite(vector).all():
        raise RuntimeError("causal feature vector is non-finite")
    return vector


def _graph_static(graph) -> np.ndarray:
    kinds = ("ENTITY", "RELATION", "DIRECTION", "ORDINAL", "TEMPORAL_ORDER", "EXCLUSION", "GOAL")
    counts = Counter(constraint.kind.value for constraint in graph.constraints)
    dependencies = [len(constraint.dependencies) for constraint in graph.constraints]
    instruction = graph.instruction.strip()
    vector = np.asarray([
        float(len(graph.constraints)),
        *(float(counts.get(kind, 0)) for kind in kinds),
        float(np.mean(dependencies)) if dependencies else 0.0,
        float(max(dependencies, default=0)),
        float(len(instruction.split())),
        float(len(instruction)),
    ], dtype=np.float64)
    return vector


def _qwen_event_summary(event: Mapping[str, object], task_by_key: Mapping[tuple[str, str, str, str, int], Mapping[str, object]], graph) -> tuple[np.ndarray, dict[str, object]]:
    factors_by_constraint = {
        constraint.constraint_id: {"instantiated": [], "distinguishable": [], "resolved": []}
        for constraint in graph.constraints
    }
    per_step_rows = []
    for step in range(int(event["prefix_start"]), int(event["prefix_end"]) + 1):
        key = (str(event["dataset"]), str(event["scene_id"]), str(event["episode_id"]), str(event["source_observation_stream_id"]), step)
        task = task_by_key[key]
        record = correction.combined_record(task)
        if record is None:
            raise RuntimeError(f"missing Qwen evidence record for {task['request_id']}")
        normalized = record["normalized_constraints"]
        per_step_rows.append(normalized)
        for cid, factors in factors_by_constraint.items():
            item = normalized[cid]
            for name in factors:
                factors[name].append(bool(item[name]))
    final_values = [
        float(np.mean([values["instantiated"][-1] for values in factors_by_constraint.values()])),
        float(np.mean([values["distinguishable"][-1] for values in factors_by_constraint.values()])),
        float(np.mean([values["resolved"][-1] for values in factors_by_constraint.values()])),
    ]
    # U/A/D is derived mechanically from the three independent Qwen factors;
    # it is not a new learned semantic definition.
    from revealnav_mf3.evidence_uad import ConstraintState, derive_constraint_uad
    final_states = []
    first_decisive = []
    for factors in factors_by_constraint.values():
        states = derive_constraint_uad(factors["instantiated"], factors["distinguishable"], factors["resolved"])
        final_states.append(states[-1].value)
        first_decisive.append(next((index for index, state in enumerate(states) if state is ConstraintState.D), None))
    state_counts = Counter(final_states)
    available = [value for value in first_decisive if value is not None]
    qwen_reveal = max(available) if available and len(available) == len(first_decisive) else None
    vector = np.asarray([
        *final_values,
        float(state_counts.get("U", 0) / len(final_states)),
        float(state_counts.get("A", 0) / len(final_states)),
        float(state_counts.get("D", 0) / len(final_states)),
        float(sum(value is not None for value in first_decisive) / len(first_decisive)),
        float(qwen_reveal is not None),
        float(qwen_reveal - int(event["prefix_end"]) if qwen_reveal is not None else 0.0),
    ], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise RuntimeError("Qwen provisional feature vector is non-finite")
    return vector, {
        "final_state_counts": dict(sorted(state_counts.items())),
        "constraints": len(final_states),
        "qwen_reveal_step": qwen_reveal,
        "qwen_reveal_available": qwen_reveal is not None,
        "prefix_count": len(per_step_rows),
    }


def build_features(events: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    tasks = base.prefix_tasks(list(events))
    task_by_key = {
        (str(task["dataset"]), str(task["scene_id"]), str(task["episode_id"]), str(task["event_id"]), int(task["prefix_step"])): task
        for task in tasks
    }
    rows = []
    qwen_state_counts = Counter()
    for event in events:
        graph = base.load_graph(str(event["instruction"]))
        sequence = []
        previous = None
        for step in range(int(event["prefix_start"]), int(event["prefix_end"]) + 1):
            key = (str(event["dataset"]), str(event["scene_id"]), str(event["episode_id"]), str(event["source_observation_stream_id"]), step)
            task = task_by_key[key]
            sequence.append(_causal_step_vector(task, previous))
            previous = [str(item["alias"]) for item in task["contract"]["current_candidates"]]
        matrix = np.stack(sequence)
        final = matrix[-1]
        diffs = np.diff(matrix, axis=0) if len(matrix) > 1 else np.zeros((1, matrix.shape[1]), dtype=np.float64)
        slope = (matrix[-1] - matrix[0]) / max(1.0, float(len(matrix) - 1))
        causal_snapshot = np.concatenate((_graph_static(graph), final))
        causal_temporal = np.concatenate((
            causal_snapshot,
            matrix.mean(axis=0), matrix.std(axis=0), slope,
            np.asarray([float(len(matrix) - 1), float(np.abs(diffs[:, 0]).sum())], dtype=np.float64),
        ))
        qwen_vector, qwen_summary = _qwen_event_summary(event, task_by_key, graph)
        qwen_state_counts.update(qwen_summary["final_state_counts"])
        rows.append({
            "event_id": str(event["event_id"]),
            "dataset": str(event["dataset"]),
            "scene_id": str(event["scene_id"]),
            "episode_id": str(event["episode_id"]),
            "decision_step": int(event["prefix_end"]),
            "scene_fold": int(event["scene_fold"]),
            "snapshot_features": causal_snapshot.tolist(),
            "temporal_features": causal_temporal.tolist(),
            "qwen_features": qwen_vector.tolist(),
            "qwen_summary": qwen_summary,
        })
    return rows, {
        "events": len(rows),
        "qwen_final_state_counts": dict(sorted(qwen_state_counts.items())),
        "qwen_labels_provisional": True,
        "target_payload_read": False,
        "outcome_payload_read": False,
    }


def _ridge_fit(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(matrix)), matrix))
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_L2
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def ridge_oof(matrix: np.ndarray, target: np.ndarray, folds: np.ndarray) -> np.ndarray:
    prediction = np.full(len(matrix), np.nan, dtype=np.float64)
    for fold in range(FOLDS):
        fit = folds != fold
        held = folds == fold
        if not held.any() or not fit.any():
            raise RuntimeError("incomplete scene fold")
        mean = matrix[fit].mean(axis=0)
        scale = np.where(matrix[fit].std(axis=0) > 1e-12, matrix[fit].std(axis=0), 1.0)
        coefficient = _ridge_fit((matrix[fit] - mean) / scale, target[fit])
        prediction[held] = np.column_stack((np.ones(int(held.sum())), (matrix[held] - mean) / scale)) @ coefficient
    if not np.isfinite(prediction).all():
        raise RuntimeError("OOF prediction incomplete")
    return prediction


def huber(error: np.ndarray) -> np.ndarray:
    absolute = np.abs(error)
    return np.where(absolute <= 1.0, 0.5 * error * error, absolute - 0.5)


def bootstrap_mean(values: np.ndarray, scenes: np.ndarray, *, seed: int) -> dict[str, float | int]:
    unique = np.unique(scenes)
    if len(unique) < 2:
        raise RuntimeError("at least two scenes are required for bootstrap")
    by_scene = {scene: np.flatnonzero(scenes == scene) for scene in unique}
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_scene[scene] for scene in draw])
        samples[index] = float(values[indices].mean())
    return {
        "observed": float(values.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "scene_count": int(len(unique)),
    }


def utility(metrics: Mapping[str, object]) -> float:
    values = [float(metrics[key]) for key in UTILITY_KEYS]
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite exact utility")
    return 0.50 * values[0] + 0.25 * values[1] + 0.25 * values[2]


def evaluate_utility(rows: Sequence[Mapping[str, object]], feature_name: str, outcome_by_key: Mapping[tuple[str, str, str, int], float]) -> dict[str, object]:
    matched = [row for row in rows if (str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]), int(row["decision_step"])) in outcome_by_key]
    if len(matched) < 20:
        raise RuntimeError("too few exact development matches")
    matrix = np.asarray([row[feature_name] for row in matched], dtype=np.float64)
    target = np.asarray([outcome_by_key[(str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]), int(row["decision_step"]))] for row in matched], dtype=np.float64)
    folds = np.asarray([int(row["scene_fold"]) for row in matched], dtype=np.int64)
    prediction = ridge_oof(matrix, target, folds)
    result: dict[str, object] = {"feature": feature_name, "matched_events": len(matched), "domains": {}}
    for domain_index, domain in enumerate(("R2R", "RxR")):
        mask = np.asarray([str(row["dataset"]) == domain for row in matched], dtype=bool)
        if not mask.any():
            raise RuntimeError(f"missing exact domain: {domain}")
        selected = mask & (prediction > 0.0)
        itt = np.where(selected, target, 0.0)
        catastrophic = selected & (target <= -0.10)
        domain_result = {
            "events": int(mask.sum()),
            "coverage": float(selected[mask].mean()),
            "selected_count": int(selected[mask].sum()),
            "selected_mean_delta_utility": float(target[selected].mean()) if selected.any() else None,
            "itt_mean_delta_utility": float(itt[mask].mean()),
            "huber_oof": float(huber(prediction[mask] - target[mask]).mean()),
            "catastrophic_rate_selected": float(catastrophic[mask].sum() / selected[mask].sum()) if selected[mask].any() else None,
            "itt_scene_bootstrap": bootstrap_mean(itt[mask], np.asarray([row["scene_id"] for row in matched], dtype=str)[mask], seed=BOOTSTRAP_SEED + domain_index),
        }
        result["domains"][domain] = domain_result
    result["target_mean"] = float(target.mean())
    result["prediction_target_correlation"] = float(np.corrcoef(prediction, target)[0, 1]) if np.std(prediction) > 0 and np.std(target) > 0 else 0.0
    return result


def _load_exact_outcomes() -> dict[tuple[str, str, str, int], float]:
    # This boundary is intentionally called only after build_features().
    car_path = ROOT / "scripts/train_mf3zm_car.py"
    car = load_module(car_path, "mf3zp_provisional_car_source")
    rows = car._canonical_rows()
    result = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]), int(row["decision"]["step"]))
        if key in result and not math.isclose(result[key], float(row["target"]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("conflicting exact outcome identity")
        result[key] = float(row["target"])
    return result


def protocol_value() -> dict[str, object]:
    correction.verify()
    events = base.read_events()
    tasks = base.prefix_tasks(events)
    return {
        "schema_version": SCHEMA,
        "revision": "mf3zp_qwen_provisional_training_v1",
        "status": "SEALED_BEFORE_PROVISIONAL_OUTCOME_READ",
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip(),
        "method": inventory(METHOD),
        "entrypoint": inventory(Path(__file__).resolve()),
        "science_protocol": inventory(correction.SCIENCE_PROTOCOL),
        "correctness_protocol": inventory(correction.CORRECTNESS_PROTOCOL),
        "pilot_events": inventory(base.EVENTS),
        "source_requests": inventory(base.SOURCE_REQUESTS),
        "qwen_status": inventory(correction.STATUS_PATH),
        "model": MODEL,
        "population": {
            "events": len(events),
            "prefix_tasks": len(tasks),
            "event_ids_sha256": stable_hash([str(event["event_id"]) for event in events]),
            "scene_ids": sorted({str(event["scene_id"]) for event in events}),
            "domain_counts": dict(sorted(Counter(str(event["dataset"]) for event in events).items())),
        },
        "probe": {
            "folds": FOLDS,
            "ridge_l2": RIDGE_L2,
            "decision_rule": "OOF prediction > 0 selects provisional alternative",
            "feature_sets": ["snapshot_features", "temporal_features", "snapshot_plus_qwen_features"],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "hyperparameter_search": False,
            "human_review": False,
        },
        "boundary": {
            "qwen_labels_provisional": True,
            "target_payload_read_before_features": False,
            "outcome_payload_read": False,
            "checkpoint_generated": False,
            "formal_label_validity_pass": False,
            "oracle_headroom_authorized": False,
            "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
        },
    }


def seal() -> dict[str, object]:
    value = protocol_value()
    atomic_json(PROTOCOL, value, refuse_existing=True)
    return value


def verify_protocol() -> dict[str, object]:
    value = read_json(PROTOCOL)
    if value != protocol_value():
        raise RuntimeError("provisional protocol/source drift")
    if value["boundary"]["public_split_access"] != {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}:
        raise RuntimeError("public split boundary is open")
    return value


def run() -> dict[str, object]:
    protocol = verify_protocol()
    events = base.read_events()
    # Build every inference feature before opening exact outcomes.
    feature_rows, feature_audit = build_features(events)
    atomic_jsonl(FEATURES, feature_rows, refuse_existing=True)
    outcome_by_key = _load_exact_outcomes()
    rows_with_outcomes = sum((str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]), int(row["decision_step"])) in outcome_by_key for row in feature_rows)
    current = np.asarray([row["snapshot_features"] for row in feature_rows], dtype=np.float64)
    temporal = np.asarray([row["temporal_features"] for row in feature_rows], dtype=np.float64)
    qwen = np.asarray([row["qwen_features"] for row in feature_rows], dtype=np.float64)
    folds = scene_folds(events)
    # Fixed provisional label learnability check: event-level resolved fraction.
    resolved_fraction = qwen[:, 2]
    snapshot_pred = ridge_oof(current, resolved_fraction, folds)
    temporal_pred = ridge_oof(temporal, resolved_fraction, folds)
    label_probe = {
        "snapshot_mae": float(np.abs(snapshot_pred - resolved_fraction).mean()),
        "temporal_mae": float(np.abs(temporal_pred - resolved_fraction).mean()),
        "temporal_minus_snapshot_mae": float(np.abs(snapshot_pred - resolved_fraction).mean() - np.abs(temporal_pred - resolved_fraction).mean()),
        "provisional_target": "Qwen-derived final resolved-constraint fraction",
    }
    utility_results = {}
    if rows_with_outcomes:
        for name, matrix in (
            ("snapshot_features", current),
            ("temporal_features", temporal),
            ("snapshot_plus_qwen_features", np.concatenate((current, qwen), axis=1)),
        ):
            # evaluate_utility rebuilds the same OOF fit from the serialized feature rows.
            utility_results[name] = evaluate_utility(feature_rows, name if name != "snapshot_plus_qwen_features" else "snapshot_features", outcome_by_key) if name != "snapshot_plus_qwen_features" else _evaluate_augmented(feature_rows, current, qwen, outcome_by_key)
    result = {
        "schema_version": SCHEMA,
        "status": "EXPLORATORY_QWEN_PROVISIONAL_TRAINING_COMPLETE",
        "protocol": inventory(PROTOCOL),
        "model": MODEL,
        "pilot_events": len(feature_rows),
        "prefix_tasks": 538,
        "exact_outcome_matches": rows_with_outcomes,
        "feature_audit": feature_audit,
        "label_probe": label_probe,
        "utility_probe": utility_results,
        "qwen_labels_are_provisional": True,
        "human_verified": False,
        "gold": False,
        "formal_label_validity_pass": False,
        "checkpoint_generated": False,
        "oracle_headroom_authorized": False,
        "public_unseen_authorized": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
        "note": "Exploratory feasibility only; no formal Oracle/REE/skill gate is passed by this result.",
    }
    atomic_json(RESULT, result, refuse_existing=True)
    return result


def _evaluate_augmented(feature_rows: Sequence[Mapping[str, object]], current: np.ndarray, qwen: np.ndarray, outcomes: Mapping[tuple[str, str, str, int], float]) -> dict[str, object]:
    matched = [row for row in feature_rows if (str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]), int(row["decision_step"])) in outcomes]
    indices = [feature_rows.index(row) for row in matched]
    matrix = np.concatenate((current[indices], qwen[indices]), axis=1)
    target = np.asarray([outcomes[(str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]), int(row["decision_step"]))] for row in matched], dtype=np.float64)
    folds = np.asarray([int(row["scene_fold"]) for row in matched], dtype=np.int64)
    prediction = ridge_oof(matrix, target, folds)
    result = {"feature": "snapshot_plus_qwen_features", "matched_events": len(matched), "domains": {}}
    scenes = np.asarray([str(row["scene_id"]) for row in matched])
    for domain_index, domain in enumerate(("R2R", "RxR")):
        mask = np.asarray([str(row["dataset"]) == domain for row in matched])
        selected = mask & (prediction > 0.0)
        itt = np.where(selected, target, 0.0)
        result["domains"][domain] = {
            "events": int(mask.sum()),
            "coverage": float(selected[mask].mean()),
            "selected_count": int(selected[mask].sum()),
            "selected_mean_delta_utility": float(target[selected].mean()) if selected.any() else None,
            "itt_mean_delta_utility": float(itt[mask].mean()),
            "huber_oof": float(huber(prediction[mask] - target[mask]).mean()),
            "catastrophic_rate_selected": float((target[selected] <= -0.10).mean()) if selected.any() else None,
            "itt_scene_bootstrap": bootstrap_mean(itt[mask], scenes[mask], seed=BOOTSTRAP_SEED + domain_index),
        }
    result["target_mean"] = float(target.mean())
    result["prediction_target_correlation"] = float(np.corrcoef(prediction, target)[0, 1]) if np.std(prediction) > 0 and np.std(target) > 0 else 0.0
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "verify", "run", "status"))
    args = parser.parse_args()
    if args.command == "seal":
        value = seal()
        print(json.dumps({"status": value["status"], "protocol_sha256": sha256_file(PROTOCOL)}, indent=2))
        return 0
    if args.command == "verify":
        value = verify_protocol()
        print(json.dumps({"status": "MF3ZP_QWEN_PROVISIONAL_PROTOCOL_VERIFIED", "protocol_sha256": sha256_file(PROTOCOL)}, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(read_json(RESULT) if RESULT.is_file() else {"status": "NOT_RUN"}, indent=2, ensure_ascii=False))
        return 0
    value = run()
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
