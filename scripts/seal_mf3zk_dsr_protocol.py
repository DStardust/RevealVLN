#!/usr/bin/env python3
"""Create the immutable, pre-training source inventory for MF3ZK-DSR v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for directory in (SCRIPTS, ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from revealnav_mf2r6.protocol import scene_fold  # noqa: E402
from revealnav_mf3.action_aligned import FEATURE_NAMES  # noqa: E402
from train_mf3zk_joint_action_aligned_gate import sha256_file  # noqa: E402
from train_mf3zk_nested_pooled_gate import (  # noqa: E402
    HIERARCHY_GATE,
    PROTOCOL as OLD_PROTOCOL,
    R2R_MANIFEST,
    RXR_CORE,
    RXR_EXPANSION,
    _load_hierarchy,
    _vector,
    load_rows,
)


REVISION = "mf3zk_dsr_v1"
SCHEMA = "revealnav-mf3zk-dsr-protocol/1"
OUT = ROOT / "artifacts/training/mf3zk_dsr_v1"
PROTOCOL = OUT / "MF3ZK_DSR_PROTOCOL.json"
PREVIOUS_FAILURE = ROOT / (
    "artifacts/training/mf3zk_nested_pooled_v9/"
    "MF3ZK_NESTED_POOLED_TRAINING_RESULT.json"
)
IMPLEMENTATION_PATHS = (
    "METHOD_REVISION_3ZK_DSR.md",
    "revealnav_mf3/distributional_switch.py",
    "revealnav_mf3/dsr_selection.py",
    "scripts/seal_mf3zk_dsr_protocol.py",
    "scripts/train_mf3zk_dsr.py",
    "tests/test_mf3zk_dsr_model.py",
    "tests/test_mf3zk_dsr_selection.py",
)
SOURCE_PATHS = (
    OLD_PROTOCOL,
    R2R_MANIFEST,
    RXR_CORE,
    RXR_EXPANSION,
    HIERARCHY_GATE,
    PREVIOUS_FAILURE,
)
OUTER_FOLD_SALT = "v6_scene_fold"
INNER_FOLD_SALT = "mf3zk-dsr-v1-inner-scenes/1"
SEEDS = (20260830, 20260831, 20260832)


def stable_json_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def checked_file(path: Path) -> Path:
    path = Path(path)
    resolved = path.resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or resolved != path.absolute()
        or ROOT not in resolved.parents
    ):
        raise RuntimeError(f"unsafe DSR source file: {path}")
    return resolved


def file_fact(path: Path) -> dict:
    path = checked_file(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _json(path: Path) -> dict:
    value = json.loads(checked_file(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"DSR JSON source is not an object: {path}")
    return value


def _raw_record_map() -> dict[tuple[str, int], tuple[str, dict]]:
    result = {}
    for path, dataset, tier in (
        (RXR_CORE, "RxR", "core"),
        (RXR_EXPANSION, "RxR", "expansion"),
        (R2R_MANIFEST, "R2R", None),
    ):
        value = _json(path)
        relative = str(path.relative_to(ROOT))
        for record in value.get("records", []):
            key = (relative, int(record["row_index"]))
            if key in result:
                raise RuntimeError(f"duplicate manifest row identity: {key}")
            result[key] = (tier or str(record["tier"]), record)
    return result


def _label_content(dataset: str, raw: dict) -> dict:
    return {
        "dataset": dataset,
        "scene_id": str(raw["scene_id"]),
        "episode_id": str(raw["episode_id"]),
        "decision_step": int(raw.get("decision_step", raw["decision"]["step"])),
        "decision": raw["decision"],
        "delta": raw["delta"],
        "baseline_metrics": raw["baseline_metrics"],
        "treatment_metrics": raw["treatment_metrics"],
        "feature_content": {
            "bytes": int(raw["feature"]["bytes"]),
            "sha256": str(raw["feature"]["sha256"]),
        },
        "exact_pair_evidence": {
            key: raw[key] for key in (
                "baseline_prefix_verified", "future_frames_used", "split",
            ) if key in raw
        },
    }


def _verify_pointer(pointer: dict, *, role: str) -> dict:
    if not isinstance(pointer, dict) or not all(
        key in pointer for key in ("path", "bytes", "sha256")
    ):
        raise RuntimeError(f"invalid {role} provenance pointer")
    fact = file_fact(ROOT / str(pointer["path"]))
    if (
        fact["bytes"] != int(pointer["bytes"])
        or fact["sha256"] != str(pointer["sha256"])
    ):
        raise RuntimeError(f"{role} provenance drift: {pointer['path']}")
    return fact


def build_source_inventory() -> tuple[dict, list[dict], dict]:
    """Rebuild all source facts without fitting or evaluating a model."""

    old_protocol = _json(OLD_PROTOCOL)
    if old_protocol.get("status") != "SEALED_BEFORE_MF3ZK_JOINT_TRAINING":
        raise RuntimeError("old MF3ZK source protocol status drift")
    if any(old_protocol.get("public_split_access", {}).values()):
        raise RuntimeError("old MF3ZK protocol contains public-split access")
    previous = _json(PREVIOUS_FAILURE)
    if previous.get("status") != "TRAIN_DEVELOPMENT_FAIL":
        raise RuntimeError("MF3ZK-NP v9 negative result is not frozen")
    hierarchy = _load_hierarchy()
    rows, row_meta = load_rows(old_protocol, hierarchy)
    if len(rows) != 249:
        raise RuntimeError(f"DSR canonical row inventory drift: {len(rows)}")

    raw_map = _raw_record_map()
    feature_files: dict[str, dict] = {}
    auxiliary_files: dict[str, dict] = {}
    source_entries = []
    canonical_entries = []
    identities = set()
    grouped_label_hashes: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in rows:
        identity = (
            str(row["dataset"]), str(row["episode_id"]),
            int(row["decision"].get("step", row.get("decision_step", -1))),
        )
        if identity[2] < 0 or identity in identities:
            raise RuntimeError(f"invalid canonical DSR identity: {identity}")
        identities.add(identity)
        per_row_sources = []
        for source in row["source_records"]:
            key = (str(source["manifest"]), int(source["row_index"]))
            if key not in raw_map:
                raise RuntimeError(f"DSR source record is absent: {key}")
            raw_tier, raw = raw_map[key]
            if raw_tier != str(source["tier"]):
                raise RuntimeError(f"DSR source tier drift: {key}")
            if (
                str(raw["scene_id"]) != str(row["scene_id"])
                or str(raw["episode_id"]) != str(row["episode_id"])
                or int(raw.get("decision_step", raw["decision"]["step"]))
                != identity[2]
                or not math.isclose(
                    float(raw["delta"]["utility"]), float(row["target"]),
                    rel_tol=0.0, abs_tol=0.0,
                )
            ):
                raise RuntimeError(f"DSR source/canonical label drift: {key}")
            label_hash = stable_json_hash(_label_content(identity[0], raw))
            grouped_label_hashes[identity].add(label_hash)
            feature = _verify_pointer(raw["feature"], role="feature")
            existing = feature_files.setdefault(feature["path"], feature)
            if existing != feature:
                raise RuntimeError("conflicting DSR feature inventory")
            pointers = {}
            for role in (
                "run_summary", "controller_trace", "baseline_stats",
                "treatment_stats",
            ):
                if role in raw:
                    fact = _verify_pointer(raw[role], role=role)
                    existing = auxiliary_files.setdefault(fact["path"], fact)
                    if existing != fact:
                        raise RuntimeError("conflicting DSR auxiliary inventory")
                    pointers[role] = fact
            provenance = {
                "dataset": identity[0],
                "tier": raw_tier,
                "source_manifest": key[0],
                "source_row_index": key[1],
                "raw_record_sha256": stable_json_hash(raw),
                "label_content_sha256": label_hash,
                "feature": feature,
                "auxiliary": pointers,
            }
            source_entries.append(provenance)
            per_row_sources.append(provenance)
        if len(grouped_label_hashes[identity]) != 1:
            raise RuntimeError(
                f"duplicate DSR identity has conflicting full label content: {identity}"
            )
        vector = np.asarray(_vector(row), dtype="<f8")
        if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
            raise RuntimeError("DSR causal feature vector drift")
        canonical_entries.append({
            "identity": {
                "dataset": identity[0], "episode_id": identity[1],
                "decision_step": identity[2],
            },
            "identity_sha256": stable_json_hash(identity),
            "scene_id": str(row["scene_id"]),
            "outer_fold": int(scene_fold(str(row["scene_id"]))),
            "frozen_tier": str(row["tier"]),
            "label_content_sha256": next(iter(grouped_label_hashes[identity])),
            "causal_vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
            "source_provenance_sha256": stable_json_hash(per_row_sources),
            "source_records": [
                {
                    "tier": value["tier"],
                    "source_manifest": value["source_manifest"],
                    "source_row_index": value["source_row_index"],
                    "raw_record_sha256": value["raw_record_sha256"],
                }
                for value in per_row_sources
            ],
        })

    source_entries.sort(key=lambda value: (
        value["dataset"], value["source_manifest"], value["source_row_index"]
    ))
    canonical_entries.sort(key=lambda value: (
        value["identity"]["dataset"], value["identity"]["episode_id"],
        value["identity"]["decision_step"],
    ))
    if len(source_entries) != 299:
        raise RuntimeError(f"DSR source-row inventory drift: {len(source_entries)}")
    if any(len(value) != 1 for value in grouped_label_hashes.values()):
        raise RuntimeError("DSR full-label duplicate verification failed")
    fit_scenes = sorted({str(row["scene_id"]) for row in rows})
    if len(fit_scenes) != 39:
        raise RuntimeError(f"DSR fit-scene inventory drift: {len(fit_scenes)}")
    consumed = sorted(old_protocol["strict_scene_holdout"]["confirmation_scenes"])
    if set(fit_scenes) & set(consumed):
        raise RuntimeError("consumed confirmation scene entered DSR fit")
    outer_mapping = {scene: int(scene_fold(scene)) for scene in fit_scenes}
    if set(outer_mapping.values()) != set(range(5)):
        raise RuntimeError("DSR outer scene folds are incomplete")

    inventory = {
        "source_files": {
            fact["path"]: {"bytes": fact["bytes"], "sha256": fact["sha256"]}
            for fact in (file_fact(path) for path in SOURCE_PATHS)
        },
        "feature_files": dict(sorted(feature_files.items())),
        "feature_inventory_sha256": stable_json_hash(
            dict(sorted(feature_files.items()))
        ),
        "auxiliary_files": dict(sorted(auxiliary_files.items())),
        "auxiliary_inventory_sha256": stable_json_hash(
            dict(sorted(auxiliary_files.items()))
        ),
        "source_records": source_entries,
        "source_record_inventory_sha256": stable_json_hash(source_entries),
        "canonical_rows": canonical_entries,
        "canonical_row_inventory_sha256": stable_json_hash(canonical_entries),
        "full_label_content_inventory_sha256": stable_json_hash(sorted(
            value["label_content_sha256"] for value in canonical_entries
        )),
        "counts": {
            "source_records": len(source_entries),
            "canonical_rows": len(canonical_entries),
            "duplicate_source_rows_collapsed": len(source_entries) - len(canonical_entries),
            "feature_files": len(feature_files),
            "auxiliary_files": len(auxiliary_files),
            "fit_scenes": len(fit_scenes),
        },
        "fit_scene_ids": fit_scenes,
        "outer_scene_assignment": outer_mapping,
        "known_consumed_scene_ids": consumed,
        "canonicalization": row_meta["canonicalization"],
    }
    return inventory, rows, hierarchy


def build_protocol() -> dict:
    inventory, _, hierarchy = build_source_inventory()
    implementation = {}
    for relative in IMPLEMENTATION_PATHS:
        fact = file_fact(ROOT / relative)
        implementation[relative] = {
            "bytes": fact["bytes"], "sha256": fact["sha256"]
        }
    return {
        "schema_version": SCHEMA,
        "status": "SEALED_BEFORE_MF3ZK_DSR_TRAINING",
        "revision": REVISION,
        "method": "Policy-Anchored Distributional Switch Critic",
        "frozen_components": [
            "ETP-R1 policy and visual-language backbone",
            "MF3V proposal ranker",
            "MF3ZG core-preserving proposal hierarchy",
            "frozen runner-up action identity",
            "one-switch intervention budget",
            "utility weights",
            "28D action-aligned causal feature schema",
        ],
        "known_consumed_scene_ids": inventory["known_consumed_scene_ids"],
        "source_inventory": inventory,
        "feature_inventory_sha256": inventory["feature_inventory_sha256"],
        "implementation_files": implementation,
        "utility": {"ndtw": 0.50, "sdtw": 0.25, "spl": 0.25},
        "model": {
            "input_dim": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "benchmark_id_input": False,
            "tier_id_input": False,
            "hidden_dim": 24,
            "activation": "GELU",
            "native_margin_anchor": "softplus(beta)*-log1p(max(native_margin,0))",
            "quantiles": [0.20, 0.50, 0.80],
            "ordered_parameterization": True,
            "ensemble_seeds": list(SEEDS),
        },
        "loss": {
            "name": "weighted_multi_quantile_pinball",
            "domain_balanced": True,
            "scene_balanced_within_domain": True,
            "row_balanced_within_scene": True,
            "independent_harm_head": False,
        },
        "selection": {
            "outer_folds": 5,
            "outer_fold_salt": OUTER_FOLD_SALT,
            "outer_scene_assignment": inventory["outer_scene_assignment"],
            "inner_folds": 4,
            "inner_fold_salt": INNER_FOLD_SALT,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "selection_metric": "inner_scene_oof_quantile_loss",
            "learning_rate": 0.01,
            "training_steps": 300,
            "optimizer": "AdamW_full_batch",
            "common_random_numbers": True,
            "decision_threshold": 0.0,
            "decision_rule": "lower_q20_utility > 0",
            "threshold_search": False,
        },
        "proposal_support_audit": {
            "must_precede_fit": True,
            "fixed_coverages": [0.05, 0.10, 0.20],
            "fail_if_oracle_10_and_20_percent_nonpositive": True,
            "minimum_positive_scenes": "max(5,ceil(0.20*domain_scenes))",
            "model_selection_input": False,
        },
        "failure_criteria": [
            "source_or_feature_provenance_drift",
            "old_confirmation_scene_in_fit",
            "public_split_access",
            "duplicate_full_label_conflict",
            "non_exact_pair_or_nonfinite_input",
            "proposal_support_audit_fail",
            "joint_outer_fold_missing_prediction",
            "eligible_outer_fold_domain_zero_intervention",
            "joint_domain_nonpositive_utility",
            "joint_domain_nonpositive_leave_one_selected_scene_utility",
            "joint_domain_catastrophic_rate_above_ungated",
            "joint_catastrophic_rate_above_fold_domain_matched_low_margin",
            "joint_utility_not_above_fold_domain_matched_low_margin",
        ],
        "historical_evidence": {
            "mf3zk_np_v9_status": "TRAIN_DEVELOPMENT_FAIL",
            "old_confirmation_consumed": True,
            "old_confirmation_reused": False,
        },
        "public_split_access": {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
        "authorization": {
            "trainer_may_authorize_confirmation": False,
            "trainer_may_authorize_public_unseen": False,
        },
        "resource_ceiling": {
            "gpu_hours": 2.0,
            "maximum_gpus": 1,
            "new_habitat_rollouts": 0,
        },
        "hierarchy": hierarchy,
    }


def verify_protocol(path: Path) -> tuple[dict, list[dict], dict]:
    path = checked_file(path)
    protocol = _json(path)
    if (
        protocol.get("schema_version") != SCHEMA
        or protocol.get("status") != "SEALED_BEFORE_MF3ZK_DSR_TRAINING"
        or protocol.get("revision") != REVISION
        or any(protocol.get("public_split_access", {}).values())
        or protocol.get("authorization", {}).get("trainer_may_authorize_public_unseen")
        is not False
    ):
        raise RuntimeError("DSR sealed protocol contract drift")
    inventory, rows, hierarchy = build_source_inventory()
    if protocol.get("source_inventory") != inventory:
        raise RuntimeError("DSR sealed source inventory drift")
    current_implementation = {}
    for relative in IMPLEMENTATION_PATHS:
        fact = file_fact(ROOT / relative)
        current_implementation[relative] = {
            "bytes": fact["bytes"], "sha256": fact["sha256"]
        }
    if protocol.get("implementation_files") != current_implementation:
        raise RuntimeError("DSR sealed implementation drift")
    return protocol, rows, hierarchy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    output = args.output.resolve()
    if output != PROTOCOL.resolve():
        raise RuntimeError("DSR protocol output path is frozen")
    value = build_protocol()
    if output.exists():
        if json.loads(output.read_text()) != value:
            raise RuntimeError("sealed DSR protocol already exists with different bytes")
        print(output)
        return 0
    atomic_json(output, value)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
