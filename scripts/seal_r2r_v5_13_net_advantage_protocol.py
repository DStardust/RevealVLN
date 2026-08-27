#!/usr/bin/env python3
"""Seal the post-training R2R paired evaluation protocol before results exist."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_13_net_advantage"
PROTOCOL = OUT / "R2R_V5_13_NET_ADVANTAGE_PROTOCOL.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_value() -> dict:
    return {
        "schema_version": "revealnav-r2r-v5.13-net-advantage-protocol/1",
        "status": "SEALED_BEFORE_FULL_TRAINING_AND_UNSEEN_EVALUATION",
        "method_revision": {
            "version": "V5.13",
            "scope": (
                "correct the sparse trigger's deployment score and add it as a "
                "conservative veto; Method-Freeze-2 REE/ECOG/OPP claims are unchanged"
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
            "checkpoint_selection": "calibration only; untouched internal dev is gate only",
            "threshold_selection": "calibration only using the same online score used at inference",
            "seeds": [20260826, 20260827, 20260828],
            "required_training_gate": "R2R_SPARSE_NET_ADVANTAGE_LEARNABILITY_PASS",
            "pilot_checkpoint_role": "shape smoke only; legacy offline threshold is rejected",
        },
        "groups": [
            {"id": "etp_r1", "description": "frozen official ETP-R1"},
            {"id": "v5_6", "description": "validated Full OPP V5.6"},
            {"id": "net_advantage_only", "description": "learned causal sparse branch trigger"},
            {"id": "v5_6_net_advantage", "description": "V5.6 proposal plus learned conservative veto"},
            {
                "id": "v5_6_net_advantage_reversible",
                "description": "V5.6 plus learned veto plus reversible ECOG trial",
            },
        ],
        "comparisons": [
            {"treatment": "v5_6", "baseline": "etp_r1"},
            {"treatment": "net_advantage_only", "baseline": "etp_r1"},
            {"treatment": "v5_6_net_advantage", "baseline": "v5_6"},
            {"treatment": "v5_6_net_advantage", "baseline": "etp_r1"},
            {"treatment": "v5_6_net_advantage_reversible", "baseline": "v5_6_net_advantage"},
            {"treatment": "v5_6_net_advantage_reversible", "baseline": "etp_r1"},
        ],
        "evaluation": {
            "development": "val_seen; diagnostic only and never fresh confirmation",
            "primary_benchmark": "R2R-CE val_unseen complete authorized split after lock",
            "cross_benchmark": "RxR-CE English val_unseen after R2R operating point is locked",
            "test_or_challenge_split_access": False,
            "paired_key": ["episode_id", "seed"],
            "paired_unit": "episode mean across three locked seeds",
            "bootstrap": "10000 deterministic paired episode replicates",
            "metrics": [
                "success", "oracle_success", "spl", "ndtw", "sdtw",
                "distance_to_goal", "path_length", "steps_taken", "collisions",
            ],
            "controller_metrics": [
                "activation_rate", "mean_net_per_activation_m", "backtrack_success",
                "return_distance", "repeated_branch_rate", "checkpoint_count",
            ],
        },
        "primary_scientific_gate": {
            "comparison": "v5_6_net_advantage minus etp_r1",
            "directional": "mean SPL>0, mean nDTW>0, mean Success>=0",
            "statistical": "paired-bootstrap lower bounds for SPL and nDTW >0",
            "v5_6_gain": "v5_6_net_advantage must improve mean SPL over v5_6",
            "reversible_ablation_retention": (
                "the reversible addition is retained in the deployed main variant only "
                "if it does not reduce mean SPL; it is always reported as an ablation"
            ),
        },
        "reporting": {
            "all_five_groups_reported": True,
            "all_three_seeds_reported": True,
            "zero-activation_runs_retained": True,
            "failed_runs_retained": True,
            "threshold_or_metric_search_on_val_unseen": False,
        },
        "paper_result": False,
        "unseen_metrics_opened": False,
        "sources": {
            "scripts/revealnav_net_advantage.py": sha256_file(
                ROOT / "scripts/revealnav_net_advantage.py"
            ),
            "scripts/train_r2r_sparse_net_advantage.py": sha256_file(
                ROOT / "scripts/train_r2r_sparse_net_advantage.py"
            ),
            "scripts/r2r_v5_6_net_advantage_controller.py": sha256_file(
                ROOT / "scripts/r2r_v5_6_net_advantage_controller.py"
            ),
        },
    }


def main() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.13 protocol drift")
    if not PROTOCOL.exists():
        PROTOCOL.parent.mkdir(parents=True)
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
