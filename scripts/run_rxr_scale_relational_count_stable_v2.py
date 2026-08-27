#!/usr/bin/env python3
"""Retrain scale relational REE with batch-invariant candidate-count input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2 import RevealFeatureDataset  # noqa: E402
from run_rxr_balanced_tuning_v2 import training_weights  # noqa: E402
from run_rxr_scale_relational_training import (  # noqa: E402
    EPOCHS, HIDDEN_DIM, MANIFEST, PATIENCE, SEEDS, train_relational,
)


AUTHORIZATION = (
    ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2/model_training/"
    "RXR_SCALE_AUTOMATIC_TRAINING_AUTHORIZATION.json"
)
LEGACY_ROOT = ROOT / "artifacts/evaluation/mf2_scale_relational_v1"
LEGACY_COMPARISON = LEGACY_ROOT / "RXR_SCALE_RELATIONAL_COMPARISON_V1.json"
OUT = ROOT / "artifacts/evaluation/mf2_scale_relational_v2_count_stable"
PROTOCOL = OUT / "RXR_SCALE_RELATIONAL_COUNT_STABLE_PROTOCOL_V2.json"
COMPARISON = OUT / "RXR_SCALE_RELATIONAL_COUNT_STABLE_COMPARISON_V2.json"
MODEL_SOURCE = ROOT / "revealnav_mf2r2/model.py"
TRAINING_SOURCE = ROOT / "scripts/run_rxr_scale_relational_training.py"
METRICS = (
    "accuracy", "macro_f1", "nll", "brier", "ece_10bin",
    "false_ready_rate", "missed_ready_rate",
)
COUNT_ENCODING = "saturating"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def protocol_value() -> dict:
    authorization = json.loads(AUTHORIZATION.read_text())
    legacy = json.loads(LEGACY_COMPARISON.read_text())
    if not (
        authorization.get("status") == "AUTOMATIC_SCALE_TRAINING_AUTHORIZED"
        and authorization.get("training_authorized") is True
        and authorization["manifest"]["sha256"] == sha256_file(MANIFEST)
        and legacy.get("status") == "SCALE_RELATIONAL_SCORE_GATE_PASS"
        and legacy.get("selected_model") == "scale_relational"
        and legacy.get("gold_feature_payload_read") is False
    ):
        raise RuntimeError("count-stable training preconditions failed")
    return {
        "schema_version": "revealnav-mf2-scale-relational-count-stable-protocol/2",
        "status": "SEALED_BEFORE_COUNT_STABLE_RETRAINING",
        "reason": (
            "correct batch-partition dependence in the legacy candidate-count "
            "feature without changing model parameters or training population"
        ),
        "seeds": list(SEEDS),
        "model": "relational_ree_count_stable",
        "candidate_count_encoding": "count/(count+1)",
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "source_sampling": {
            "human_audited_probability": 2 / 3,
            "automatic_probability": 1 / 3,
            "identical_to_legacy_scale_training": True,
        },
        "development": "unchanged 68 human-audited scene-heldout events",
        "success_gates": {
            "macro_f1_degradation_vs_legacy_max": 0.02,
            "false_ready_degradation_vs_legacy_max": 0.02,
            "macro_f1_margin_over_history_min": 0.10,
            "false_ready_no_higher_than_history": True,
            "all_seed_outputs_finite": True,
        },
        "sources": {
            "manifest": {
                "path": str(MANIFEST.relative_to(ROOT)),
                "sha256": sha256_file(MANIFEST),
            },
            "authorization": {
                "path": str(AUTHORIZATION.relative_to(ROOT)),
                "sha256": sha256_file(AUTHORIZATION),
            },
            "legacy_comparison": {
                "path": str(LEGACY_COMPARISON.relative_to(ROOT)),
                "sha256": sha256_file(LEGACY_COMPARISON),
            },
            "implementation": {
                str(MODEL_SOURCE.relative_to(ROOT)): sha256_file(MODEL_SOURCE),
                str(TRAINING_SOURCE.relative_to(ROOT)): sha256_file(TRAINING_SOURCE),
            },
        },
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed count-stable protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def train(seed: int, device: torch.device) -> int:
    if seed not in SEEDS or not PROTOCOL.is_file():
        raise RuntimeError("count-stable training request is not sealed")
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("count-stable protocol drift")
    run_dir = OUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    train_set = RevealFeatureDataset(MANIFEST, "train")
    development_set = RevealFeatureDataset(MANIFEST, "development")
    state_weights, class_weights, binary_counts, uad_counts = training_weights(
        train_set
    )
    model, metrics, history = train_relational(
        train_set, development_set, seed, state_weights, class_weights, device,
        candidate_count_encoding=COUNT_ENCODING,
    )
    checkpoint = run_dir / "relational_ree_count_stable.pt"
    torch.save({
        "schema_version": "revealnav-mf2-scale-model-checkpoint/2",
        "model_name": "relational_ree_count_stable",
        "seed": seed,
        "hidden_dim": HIDDEN_DIM,
        "candidate_count_encoding": COUNT_ENCODING,
        "model_state_dict": model.state_dict(),
        "manifest_sha256": sha256_file(MANIFEST),
        "protocol_sha256": sha256_file(PROTOCOL),
    }, checkpoint)
    value = {
        "schema_version": "revealnav-mf2-scale-count-stable-run/2",
        "status": "COUNT_STABLE_MODEL_RUN_COMPLETE",
        "seed": seed,
        "results": metrics,
        "training_history": history,
        "train_counts": {
            "events": len(train_set), "binary": binary_counts,
            "uad": dict(zip(("U", "A", "D"), uad_counts.tolist())),
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "gold_feature_payload_read": False,
        "paper_result": False,
    }
    atomic_json(run_dir / "result.json", value)
    print(json.dumps({
        "status": value["status"], "seed": seed,
        "macro_f1": metrics["macro_f1"],
        "false_ready_rate": metrics["false_ready_rate"],
    }, indent=2, sort_keys=True))
    return 0


def summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "values": values,
    }


def aggregate() -> int:
    if json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("count-stable protocol drift")
    runs = [json.loads((OUT / f"seed_{seed}/result.json").read_text()) for seed in SEEDS]
    if any(row.get("status") != "COUNT_STABLE_MODEL_RUN_COMPLETE" for row in runs):
        raise RuntimeError("one or more count-stable runs are incomplete")
    corrected = {
        metric: summary([row["results"][metric] for row in runs])
        for metric in METRICS
    }
    legacy = json.loads(LEGACY_COMPARISON.read_text())
    legacy_relational = legacy["results"]["relational_ree"]
    history = legacy["results"]["history_direct_uad"]
    finite = all(
        np.isfinite(row["results"][metric])
        for row in runs for metric in METRICS
    )
    gates = {
        "macro_f1_noninferior_to_legacy": (
            legacy_relational["macro_f1"]["mean"] - corrected["macro_f1"]["mean"]
            <= 0.02
        ),
        "false_ready_noninferior_to_legacy": (
            corrected["false_ready_rate"]["mean"]
            - legacy_relational["false_ready_rate"]["mean"] <= 0.02
        ),
        "macro_f1_margin_over_history": (
            corrected["macro_f1"]["mean"] - history["macro_f1"]["mean"] >= 0.10
        ),
        "false_ready_no_higher_than_history": (
            corrected["false_ready_rate"]["mean"]
            <= history["false_ready_rate"]["mean"]
        ),
        "all_seed_outputs_finite": finite,
    }
    value = {
        "schema_version": "revealnav-mf2-scale-count-stable-comparison/2",
        "status": (
            "COUNT_STABLE_SCORE_GATE_PASS"
            if all(gates.values()) else "COUNT_STABLE_SCORE_GATE_FAIL_RETAIN_LEGACY"
        ),
        "results": {"relational_ree_count_stable": corrected},
        "references": {
            "legacy_relational_ree": legacy_relational,
            "history_direct_uad": history,
        },
        "deltas": {
            "macro_f1_vs_legacy": (
                corrected["macro_f1"]["mean"] - legacy_relational["macro_f1"]["mean"]
            ),
            "false_ready_vs_legacy": (
                corrected["false_ready_rate"]["mean"]
                - legacy_relational["false_ready_rate"]["mean"]
            ),
            "macro_f1_vs_history": (
                corrected["macro_f1"]["mean"] - history["macro_f1"]["mean"]
            ),
        },
        "gates": gates,
        "selected_model": (
            "relational_ree_count_stable" if all(gates.values())
            else "legacy_scale_relational"
        ),
        "gold_feature_payload_read": False,
        "paper_result": False,
    }
    atomic_json(COMPARISON, value)
    print(json.dumps({
        "status": value["status"], "selected_model": value["selected_model"],
        "deltas": value["deltas"], "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--seed", type=int)
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if args.aggregate:
        return aggregate()
    return train(args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
