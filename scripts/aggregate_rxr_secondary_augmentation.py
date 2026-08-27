#!/usr/bin/env python3
"""Compare fixed primary-only and augmented runs on the unchanged development set."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/daiyang/vla").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from revealnav_mf2 import (  # noqa: E402
    RevealFeatureDataset,
    RevealOptionHeads,
    collate_reveal_examples,
)
from evaluate_rxr_balanced_risk_delay_v2 import (  # noqa: E402
    BUDGETS,
    matched_point,
    roc_auc,
)
from run_rxr_representation_comparison_v2 import (  # noqa: E402
    DirectBaseline,
    classification_metrics,
    collect_probabilities,
)


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
MANIFEST = BASE / "RXR_SECONDARY_AUGMENTED_FEATURE_MANIFEST_V1.json"
AUTH = BASE / "RXR_SECONDARY_AUGMENTED_TRAINING_AUTHORIZATION_V1.json"
PRIMARY = ROOT / "artifacts/evaluation/mf2_balanced_tuning_v2"
OUT = ROOT / "artifacts/evaluation/mf2_secondary_augmentation_v1"
PROTOCOL = OUT / "RXR_SECONDARY_AUGMENTATION_PROTOCOL_V1.json"
OUTPUT = OUT / "RXR_SECONDARY_AUGMENTATION_COMPARISON_V1.json"
SEEDS = (20260826, 20260827, 20260828)
MODELS = ("balanced_full_ree", "balanced_history_direct_uad")
CONDITIONS = ("primary_only", "primary_plus_automatic_secondary")
METRICS = (
    "accuracy", "macro_f1", "nll", "brier", "ece_10bin",
    "false_ready_rate", "missed_ready_rate", "ready_roc_auc",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def source_run(condition: str, seed: int) -> tuple[Path, dict]:
    directory = (
        PRIMARY / f"h128_seed_{seed}"
        if condition == "primary_only" else OUT / f"seed_{seed}"
    )
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    expected = (
        "BALANCED_TUNING_RUN_COMPLETE"
        if condition == "primary_only"
        else "AUGMENTED_DEVELOPMENT_RUN_COMPLETE"
    )
    if result.get("status") != expected or result.get("hidden_dim") != 128:
        raise RuntimeError(f"invalid {condition} run for seed {seed}")
    return directory, result


def model_probabilities(
    condition: str,
    model_name: str,
    seed: int,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    directory, result = source_run(condition, seed)
    checkpoint_info = result["checkpoints"][model_name]
    checkpoint_path = ROOT / checkpoint_info["path"]
    if (
        checkpoint_path.parent != directory
        or checkpoint_path.stat().st_size != checkpoint_info["bytes"]
        or sha256_file(checkpoint_path) != checkpoint_info["sha256"]
    ):
        raise RuntimeError("checkpoint provenance mismatch")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if checkpoint["seed"] != seed or checkpoint["hidden_dim"] != 128:
        raise RuntimeError("checkpoint metadata mismatch")
    if model_name == "balanced_full_ree":
        model = RevealOptionHeads(768, 128, 4).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        labels, probabilities = collect_probabilities(
            "full_ree", model, loader, device,
            full_checkpoint={"normalized_budgets": [1.5, 2.0, 3.0, 4.0]},
        )
    else:
        model = DirectBaseline(
            history_aware=True, output_dim=3, hidden_dim=128
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        labels, probabilities = collect_probabilities(
            "history_direct_uad", model, loader, device
        )
    metrics = classification_metrics(labels, probabilities)
    metrics["ready_roc_auc"] = roc_auc(labels, probabilities[:, 2])
    metrics["matched_missed_ready"] = {
        str(budget): matched_point(labels, probabilities[:, 2], budget)
        for budget in BUDGETS
    }
    stored = result["results"][model_name]
    for key in ("accuracy", "macro_f1", "false_ready_rate", "missed_ready_rate"):
        if abs(metrics[key] - stored[key]) > 1e-12:
            raise RuntimeError(
                f"stored metric mismatch: {condition} {model_name} {seed} {key}"
            )
    return labels, probabilities, metrics


def summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "values": values,
    }


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    authorization = json.loads(AUTH.read_text())
    if not (
        protocol.get("status") == "FIXED_BEFORE_AUGMENTED_TRAINING"
        and protocol.get("seeds") == list(SEEDS)
        and protocol.get("hidden_dim") == 128
        and protocol.get("gold_access_allowed") is False
        and authorization.get("status")
        == "AUGMENTED_DEVELOPMENT_EXPERIMENT_AUTHORIZED"
        and authorization["manifest"]["sha256"] == sha256_file(MANIFEST)
    ):
        raise RuntimeError("augmentation aggregate precondition failed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    development = RevealFeatureDataset(MANIFEST, "development")
    loader = DataLoader(
        development, batch_size=16, shuffle=False,
        collate_fn=collate_reveal_examples,
    )
    seed_results = []
    reference_labels = None
    for seed in SEEDS:
        row = {"seed": seed, "conditions": {}}
        for condition in CONDITIONS:
            row["conditions"][condition] = {}
            for model in MODELS:
                labels, _, metrics = model_probabilities(
                    condition, model, seed, loader, device
                )
                if reference_labels is None:
                    reference_labels = labels
                elif not np.array_equal(reference_labels, labels):
                    raise RuntimeError("development label order drift")
                row["conditions"][condition][model] = metrics
        seed_results.append(row)

    aggregate = {}
    for condition in CONDITIONS:
        aggregate[condition] = {}
        for model in MODELS:
            aggregate[condition][model] = {
                metric: summary([
                    row["conditions"][condition][model][metric]
                    for row in seed_results
                ])
                for metric in METRICS
            }
    paired_deltas = {}
    for model in MODELS:
        paired_deltas[model] = {}
        for metric in METRICS:
            values = [
                row["conditions"]["primary_plus_automatic_secondary"][model][metric]
                - row["conditions"]["primary_only"][model][metric]
                for row in seed_results
            ]
            paired_deltas[model][metric] = summary(values)

    primary_gap = (
        aggregate["primary_only"]["balanced_full_ree"]["macro_f1"]["mean"]
        - aggregate["primary_only"]["balanced_history_direct_uad"]["macro_f1"]["mean"]
    )
    augmented_gap = (
        aggregate["primary_plus_automatic_secondary"]["balanced_full_ree"]["macro_f1"]["mean"]
        - aggregate["primary_plus_automatic_secondary"]["balanced_history_direct_uad"]["macro_f1"]["mean"]
    )
    full_macro_delta = paired_deltas["balanced_full_ree"]["macro_f1"]
    full_false_ready_delta = paired_deltas["balanced_full_ree"]["false_ready_rate"]
    interaction = (
        full_macro_delta["mean"]
        - paired_deltas["balanced_history_direct_uad"]["macro_f1"]["mean"]
    )
    criteria = {
        "full_macro_f1_mean_improves": full_macro_delta["mean"] > 0,
        "full_macro_f1_improves_at_least_two_seeds": sum(
            value > 0 for value in full_macro_delta["values"]
        ) >= 2,
        "full_false_ready_not_worse_by_more_than_0_02": (
            full_false_ready_delta["mean"] <= 0.02
        ),
        "full_vs_history_macro_f1_gap_narrows": augmented_gap > primary_gap,
        "augmentation_benefits_full_more_than_history": interaction > 0,
    }
    positive = all(criteria.values())
    result = {
        "schema_version": "revealnav-mf2-secondary-augmentation-comparison/1",
        "status": (
            "POSITIVE_DEVELOPMENT_AUGMENTATION_SIGNAL"
            if positive else "NO_CLEAR_DEVELOPMENT_AUGMENTATION_SIGNAL"
        ),
        "scope": (
            "fixed three-seed development-only data ablation; not an unbiased "
            "test, navigation result, or paper claim"
        ),
        "sources": {
            "protocol_sha256": sha256_file(PROTOCOL),
            "authorization_sha256": sha256_file(AUTH),
            "manifest_sha256": sha256_file(MANIFEST),
        },
        "development_prefixes": int(len(reference_labels)),
        "seeds": list(SEEDS),
        "hidden_dim": 128,
        "seed_results": seed_results,
        "aggregate": aggregate,
        "paired_augmented_minus_primary": paired_deltas,
        "representation_interaction": {
            "primary_full_minus_history_macro_f1": primary_gap,
            "augmented_full_minus_history_macro_f1": augmented_gap,
            "full_minus_history_augmentation_gain": interaction,
        },
        "predeclared_signal_criteria": criteria,
        "topology_only_training_included": False,
        "secondary_evaluation_labels_used": False,
        "gold_payload_read": False,
        "paper_result": False,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "criteria": criteria,
        "full_macro_f1_delta": full_macro_delta,
        "full_false_ready_delta": full_false_ready_delta,
        "representation_interaction": result["representation_interaction"],
        "output": str(OUTPUT.relative_to(ROOT)),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
