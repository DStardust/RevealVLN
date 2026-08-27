#!/usr/bin/env python3
"""Seal the corrected V5.13.1 five-group protocol before evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_13_net_advantage"
LEGACY = OUT / "R2R_V5_13_NET_ADVANTAGE_PROTOCOL.json"
PROTOCOL = OUT / "R2R_V5_13_1_NET_ADVANTAGE_PROTOCOL.json"
SEEDS = [20260826, 20260827, 20260828]
SOURCES = (
    "scripts/revealnav_net_advantage.py",
    "scripts/train_r2r_sparse_net_advantage.py",
    "scripts/r2r_v5_6_net_advantage_controller.py",
    "scripts/r2r_v5_13_group_worker.py",
    "scripts/run_r2r_v5_13_1_paired.py",
    "scripts/evaluate_r2r_v5_13_paired.py",
    "scripts/monitor_r2r_v5_13_1_paired.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_value() -> dict:
    legacy = json.loads(LEGACY.read_text())
    if legacy.get("status") != "SEALED_BEFORE_FULL_TRAINING_AND_UNSEEN_EVALUATION":
        raise RuntimeError("sealed V5.13 predecessor is absent")
    return {
        "schema_version": "revealnav-r2r-v5.13.1-net-advantage-protocol/1",
        "status": "SEALED_V5_13_1_BEFORE_FULL_TRAINING_AND_UNSEEN_EVALUATION",
        "supersedes": {
            "path": str(LEGACY.relative_to(ROOT)),
            "sha256": sha256_file(LEGACY),
            "reason": (
                "V5.13 main and reversible groups both inherited V5.6 ECOG; "
                "V5.13.1 replaces the duplicate with an explicit no-return ablation"
            ),
        },
        "method_revision": {
            "version": "V5.13.1",
            "frozen_method_claims_changed": False,
            "main": "V5.6 Full OPP including ECOG plus causal Net-Advantage veto",
            "reversibility_ablation": (
                "same V5.6 proposal and Net-Advantage veto, but every ECOG "
                "checkpointed-exploration proposal is delegated to native ETP"
            ),
            "online_score": (
                "p_better*positive_gain-(1-p_better)*2*"
                "checkpoint_to_alternative_euclidean_distance"
            ),
            "future_or_oracle_online_inputs": False,
        },
        "training_lock": {
            "source_split": "R2R train only",
            "partition": "scene-disjoint train/calibration/dev",
            "checkpoint_and_threshold_selection": "calibration only",
            "untouched_internal_dev_role": "learnability gate only",
            "required_training_gate": "R2R_SPARSE_NET_ADVANTAGE_LEARNABILITY_PASS",
            "seeds": SEEDS,
            "pilot_checkpoint_role": "shape smoke only and never evaluation",
        },
        "groups": [
            {
                "id": "etp_r1", "seeds": [0],
                "description": "deterministic frozen official ETP-R1",
                "seed_policy": "one execution per episode, broadcast as paired control",
            },
            {
                "id": "v5_6", "seeds": SEEDS,
                "description": "validated V5.6 Full OPP including ECOG",
            },
            {
                "id": "net_advantage_only", "seeds": SEEDS,
                "description": "causal sparse ranker without a V5.6 proposal",
            },
            {
                "id": "v5_6_net_advantage", "seeds": SEEDS,
                "description": "main V5.6 Full OPP plus conservative causal veto",
            },
            {
                "id": "v5_6_net_advantage_no_return", "seeds": SEEDS,
                "description": "main inputs and veto with ECOG trials suppressed",
            },
        ],
        "comparisons": [
            {"treatment": "v5_6", "baseline": "etp_r1"},
            {"treatment": "net_advantage_only", "baseline": "etp_r1"},
            {"treatment": "v5_6_net_advantage", "baseline": "v5_6"},
            {"treatment": "v5_6_net_advantage", "baseline": "etp_r1"},
            {
                "treatment": "v5_6_net_advantage",
                "baseline": "v5_6_net_advantage_no_return",
            },
            {
                "treatment": "v5_6_net_advantage_no_return",
                "baseline": "etp_r1",
            },
        ],
        "evaluation": {
            "development": "complete R2R val_seen; diagnostic only",
            "primary_benchmark": (
                "complete R2R-CE val_unseen after full training gate passes"
            ),
            "cross_benchmark": (
                "RxR-CE English val_unseen only after the R2R operating point is locked"
            ),
            "test_or_challenge_split_access": False,
            "selection_rule": "all episodes in the authorized validation split",
            "paired_unit": "episode mean across the three locked treatment seeds",
            "bootstrap": "10000 deterministic paired episode replicates",
            "metrics": [
                "success", "oracle_success", "spl", "ndtw", "sdtw",
                "distance_to_goal", "path_length", "steps_taken", "collisions",
            ],
            "controller_metrics": [
                "net_advantage_decisions", "net_advantage_approvals",
                "net_advantage_vetoes", "checkpointed_excursions",
                "backtrack_decisions", "successful_returns", "failed_returns",
                "no_return_suppressions",
            ],
        },
        "primary_scientific_gate": {
            "comparison": "v5_6_net_advantage minus etp_r1",
            "directional": "mean SPL>0, mean nDTW>0, mean Success>=0",
            "statistical": "paired-bootstrap lower bounds for SPL and nDTW >0",
            "incremental": "v5_6_net_advantage mean SPL must exceed v5_6",
            "ecog_ablation": (
                "main minus no-return is reported independently and does not "
                "retroactively change the preregistered main variant"
            ),
        },
        "reporting": {
            "all_five_groups_reported": True,
            "all_three_treatment_seeds_reported": True,
            "deterministic_baseline_reuse_disclosed": True,
            "zero_activation_and_failed_runs_retained": True,
            "threshold_or_metric_search_on_val_unseen": False,
        },
        "sources": {
            relative: sha256_file(ROOT / relative) for relative in SOURCES
        },
        "unseen_metrics_opened": False,
        "paper_result": False,
    }


def main() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.13.1 protocol drift")
    if not PROTOCOL.exists():
        PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
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
