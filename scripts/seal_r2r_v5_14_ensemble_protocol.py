#!/usr/bin/env python3
"""Seal V5.14 after the preregistered single-model robustness gate failed."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_13_net_advantage"
PREDECESSOR = OUT / "R2R_V5_13_1_NET_ADVANTAGE_PROTOCOL.json"
FAILED_TRAINING = ROOT / (
    "artifacts/phase1/r2r_train_net_advantage/full/training/"
    "R2R_SPARSE_NET_ADVANTAGE_TRAINING_RESULT.json"
)
PROTOCOL = OUT / "R2R_V5_14_ENSEMBLE_PROTOCOL.json"
SOURCES = (
    "scripts/revealnav_net_advantage.py",
    "scripts/train_r2r_sparse_net_advantage.py",
    "scripts/r2r_v5_6_net_advantage_controller.py",
    "scripts/r2r_v5_13_group_worker.py",
    "scripts/run_r2r_v5_13_1_paired.py",
    "scripts/evaluate_r2r_v5_13_paired.py",
    "scripts/monitor_r2r_v5_13_1_paired.py",
    "scripts/watch_r2r_v5_13_1_handoff.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_value() -> dict:
    predecessor = json.loads(PREDECESSOR.read_text())
    failed = json.loads(FAILED_TRAINING.read_text())
    if (
        predecessor.get("status")
        != "SEALED_V5_13_1_BEFORE_FULL_TRAINING_AND_UNSEEN_EVALUATION"
        or failed.get("status")
        != "R2R_SPARSE_NET_ADVANTAGE_LEARNABILITY_FAIL"
        or failed.get("unseen_or_test_read") is not False
        or failed.get("task_metric_payload_read") is not False
    ):
        raise RuntimeError("V5.14 predecessor evidence is invalid")
    value = copy.deepcopy(predecessor)
    value.update({
        "schema_version": "revealnav-r2r-v5.14-ensemble-protocol/2",
        "status": (
            "SEALED_V5_14_AFTER_TRAIN_ONLY_FEASIBILITY_"
            "BEFORE_BENCHMARK_VALIDATION"
        ),
        "supersedes": {
            "path": str(PREDECESSOR.relative_to(ROOT)),
            "sha256": sha256_file(PREDECESSOR),
            "failed_training_result": str(FAILED_TRAINING.relative_to(ROOT)),
            "failed_training_sha256": sha256_file(FAILED_TRAINING),
            "reason": (
                "all three single models had above-chance untouched-dev AUC, "
                "but the preregistered sparse-net robustness gate failed because "
                "extreme-tail activations were unstable across random initializations"
            ),
        },
    })
    value["method_revision"] = {
        "version": "V5.14",
        "frozen_method_claims_changed": False,
        "main": (
            "V5.6 Full OPP including ECOG plus one deterministic three-member "
            "causal Net-Advantage veto"
        ),
        "ensemble_aggregation": (
            "mean member probability and mean member positive gain before the "
            "unchanged causal online penalty"
        ),
        "online_score": (
            "mean(p_better)*mean(positive_gain)-(1-mean(p_better))*2*"
            "checkpoint_to_alternative_euclidean_distance"
        ),
        "future_or_oracle_online_inputs": False,
    }
    value["training_lock"] = {
        "source_split": "R2R train only",
        "partition": "unchanged scene-disjoint train/calibration/dev",
        "member_seeds": [20260826, 20260827, 20260828],
        # Evaluator-compatible alias. Both fields are required to be identical.
        "seeds": [20260826, 20260827, 20260828],
        "deployment": "all three members in one deterministic ensemble",
        "aggregation_selected_without_internal_dev_metrics": False,
        "selection_disclosure": (
            "aggregation was selected after a post-hoc feasibility calculation "
            "on the R2R-train-only scene holdout; that holdout is method-development "
            "evidence, not confirmatory evidence or a paper benchmark result"
        ),
        "benchmark_validation_metrics_seen_at_seal": False,
        "threshold_selection": "calibration only on ensemble online score",
        "required_training_gate": (
            "finite member AUCs; median dev AUC>=0.55; calibration and internal-dev "
            "ensemble net positive; >=5 activations in each; activation rates<=0.20"
        ),
        "failed_single_model_result_retained": True,
    }
    value["correctness_revision"] = {
        "version": "V5.14.1",
        "pre_correction_protocol_sha256": (
            "41b80a2f442035db912e609f8df48e9124a51cbfd98ff6485f3f88e0b2c7f8d2"
        ),
        "timing": (
            "after complete val_seen episode execution, before successful val_seen "
            "metric aggregation, and before any val_unseen access"
        ),
        "change": (
            "add training_lock.seeds as an exact schema alias of member_seeds for "
            "the already-frozen evaluator"
        ),
        "method_matrix_seeds_thresholds_and_gates_changed": False,
        "val_seen_metrics_seen_before_revision": False,
        "val_unseen_payload_opened_before_revision": False,
    }
    for group in value["groups"]:
        if group["id"] != "etp_r1":
            group["checkpoint_policy"] = (
                "same frozen three-member ensemble; listed seeds are controller "
                "seeds, not checkpoint-selection alternatives"
            )
    value["sources"] = {
        relative: sha256_file(ROOT / relative) for relative in SOURCES
    }
    value["unseen_metrics_opened"] = False
    value["paper_result"] = False
    return value


def main() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.14 protocol drift")
    if not PROTOCOL.exists():
        part = PROTOCOL.with_name(PROTOCOL.name + ".part")
        part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(part, PROTOCOL)
    print(json.dumps({
        "status": value["status"], "groups": len(value["groups"]),
        "comparisons": len(value["comparisons"]),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
