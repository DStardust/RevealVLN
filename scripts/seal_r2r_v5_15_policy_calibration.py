#!/usr/bin/env python3
"""Seal policy-induced calibration before full R2R-train collection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/design/R2R_V5_15_POLICY_CALIBRATION_PROTOCOL.json"
SELECTION = ROOT / (
    "artifacts/phase1/r2r_train_policy_calibration_v5_15/"
    "R2R_TRAIN_V5_15_POLICY_SELECTION.json"
)
NEGATIVE_RESULT = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_14_net_advantage/val_seen/"
    "R2R_V5_13_1_PAIRED_RESULT.json"
)
TRAINING_RESULT = ROOT / (
    "artifacts/phase1/r2r_train_net_advantage/full/training_v5_14/"
    "R2R_SPARSE_NET_ADVANTAGE_TRAINING_RESULT.json"
)
SOURCES = (
    "scripts/r2r_train_v5_15_policy_proposal_worker.py",
    "scripts/run_r2r_v5_15_policy_calibration_pipeline.py",
    "scripts/build_r2r_train_net_advantage_labels.py",
    "scripts/calibrate_r2r_v5_15_policy_threshold.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def value() -> dict:
    selection = json.loads(SELECTION.read_text())
    negative = json.loads(NEGATIVE_RESULT.read_text())
    training = json.loads(TRAINING_RESULT.read_text())
    if (
        selection.get("status")
        != "SEALED_R2R_TRAIN_V5_15_POLICY_SELECTION"
        or selection.get("selected_runs") != 10809
        or selection.get("unseen_or_test_read") is not False
        or negative.get("status") != "FAIL"
        or negative.get("split") != "val_seen"
        or negative.get("paper_result") is not False
        or training.get("status")
        != "R2R_SPARSE_NET_ADVANTAGE_LEARNABILITY_PASS"
        or training.get("unseen_or_test_read") is not False
    ):
        raise RuntimeError("V5.15 predecessor evidence is invalid")
    return {
        "schema_version": "revealnav-r2r-v5.15-policy-calibration-protocol/1",
        "status": "SEALED_V5_15_BEFORE_POLICY_INDUCED_COLLECTION",
        "revision_scope": (
            "policy-induced score calibration only; candidate-level ensemble "
            "weights, V5.6, REE, ECOG, and OPP remain frozen"
        ),
        "trigger": {
            "v5_14_val_seen_result": str(NEGATIVE_RESULT.relative_to(ROOT)),
            "sha256": sha256_file(NEGATIVE_RESULT),
            "finding": (
                "304 of 304 main Net-Advantage decisions were vetoed because "
                "the +3.641m all-candidate threshold exceeded every online score"
            ),
        },
        "collection": {
            "selection": str(SELECTION.relative_to(ROOT)),
            "selection_sha256": sha256_file(SELECTION),
            "split": "R2R train only",
            "unique_episodes": 3603,
            "controller_seeds": [20260826, 20260827, 20260828],
            "runs": 10809,
            "mode": "V5.6 shadow proposals over the native ETP-R1 trajectory",
            "native_action_overridden": False,
            "proposal_filter": (
                "commit or explore, proposed differs from native, and both have "
                "complete causal online embeddings and graph positions"
            ),
            "label": (
                "unchanged shortest-geodesic merge into the remaining R2R reference "
                "route; reference and geodesics are labels only"
            ),
            "future_or_oracle_online_inputs": False,
            "metric_or_label_dependent_early_stopping": False,
        },
        "calibration": {
            "frozen_ensemble_training_result": str(TRAINING_RESULT.relative_to(ROOT)),
            "frozen_ensemble_training_sha256": sha256_file(TRAINING_RESULT),
            "member_seeds": [20260826, 20260827, 20260828],
            "scene_partition": "unchanged deterministic 70/15/15 train/calibration/dev",
            "fit_scope": (
                "recalibrate the frozen ensemble score on policy-induced calibration "
                "rows; no validation score or task metric selects the threshold"
            ),
            "threshold_selection": (
                "maximize calibration realized mean net per event, then precision, "
                "then sparsity; require activation rate <=0.10 and >=5 activations"
            ),
            "minimum_dataset_gate": (
                ">=300 finite policy rows, all 61 train scenes represented in run "
                "coverage, and >=5 positive rows in calibration and dev each"
            ),
            "internal_dev_gate": (
                ">=5 activations, positive realized mean net per event, positive "
                "precision >0.5, activation rate <=0.10, and finite ensemble scores"
            ),
        },
        "benchmark_boundary": {
            "completed_v5_14_val_seen_is_development_evidence": True,
            "post_hoc_threshold_search_on_v5_14_val_seen_forbidden": True,
            "rerun_val_seen_only_after_v5_15_internal_gate": True,
            "val_unseen_payload_opened": False,
            "test_or_challenge_access": False,
            "paper_result": False,
        },
        "sources": {relative: sha256_file(ROOT / relative) for relative in SOURCES},
    }


def main() -> int:
    result = value()
    if OUT.is_file() and json.loads(OUT.read_text()) != result:
        raise RuntimeError("sealed V5.15 calibration protocol drift")
    if not OUT.is_file():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        part = OUT.with_name(OUT.name + ".part")
        part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(part, OUT)
    print(json.dumps({
        "status": result["status"], "runs": result["collection"]["runs"],
        "protocol": str(OUT.relative_to(ROOT)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
