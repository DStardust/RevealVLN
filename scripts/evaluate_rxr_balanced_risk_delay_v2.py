#!/usr/bin/env python3
"""Evaluate matched false-ready/missed-ready tradeoffs after balanced tuning."""

from __future__ import annotations

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
from run_rxr_representation_comparison_v2 import (  # noqa: E402
    DirectBaseline,
    collect_probabilities,
)
from run_rxr_balanced_tuning_v2 import sha256_file  # noqa: E402


V2 = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2"
MANIFEST = V2 / "RXR_MULTIBRANCH_FEATURE_MANIFEST_V2_AUTHORIZED.json"
BASE = ROOT / "artifacts/evaluation/mf2_balanced_tuning_v2"
PROTOCOL = BASE / "RXR_BALANCED_TUNING_PROTOCOL_V2.json"
AGGREGATE = BASE / "RXR_BALANCED_TUNING_AGGREGATE_V2.json"
OUTPUT = BASE / "RXR_BALANCED_RISK_DELAY_V2.json"
SEEDS = (20260826, 20260827, 20260828)
BUDGETS = (0.05, 0.10, 0.20, 0.30)


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
        start = stop
    positive = labels == 2
    positive_count = int(positive.sum())
    negative_count = len(labels) - positive_count
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def matched_point(labels: np.ndarray, scores: np.ndarray, budget: float) -> dict:
    ready = labels == 2
    best = None
    for threshold in np.r_[np.inf, np.unique(scores)[::-1], -np.inf]:
        prediction = scores >= threshold
        missed = float((~prediction[ready]).mean())
        false = float(prediction[~ready].mean())
        candidate = (false, missed, float(threshold))
        if missed <= budget + 1e-12 and (best is None or candidate < best):
            best = candidate
    return {
        "false_ready_rate": best[0],
        "missed_ready_rate": best[1],
        "threshold": best[2],
    }


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text())
    aggregate = json.loads(AGGREGATE.read_text())
    if not (
        protocol.get("status")
        == "DIAGNOSTIC_INFORMED_PROTOCOL_FROZEN_BEFORE_TUNING_RUNS"
        and aggregate.get("status")
        == "DEVELOPMENT_TUNING_COMPLETE_GOLD_UNTOUCHED"
    ):
        raise RuntimeError("balanced tuning evidence is incomplete")
    full_hidden = aggregate["selected"]["balanced_full_ree_hidden_dim"]
    history_hidden = aggregate["selected"][
        "balanced_history_direct_uad_hidden_dim"
    ]
    if full_hidden != history_hidden:
        raise RuntimeError("current evaluator expects the selected dimensions to match")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RevealFeatureDataset(MANIFEST, "development")
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False,
        collate_fn=collate_reveal_examples,
    )
    seed_results = []
    for seed in SEEDS:
        run = BASE / f"h{full_hidden}_seed_{seed}"
        history_checkpoint = torch.load(
            run / "balanced_history_direct_uad.pt",
            map_location=device,
            weights_only=True,
        )
        history = DirectBaseline(
            history_aware=True, output_dim=3, hidden_dim=history_hidden
        ).to(device)
        history.load_state_dict(
            history_checkpoint["model_state_dict"], strict=True
        )
        labels, history_probability = collect_probabilities(
            "history_direct_uad", history, loader, device
        )
        full_checkpoint = torch.load(
            run / "balanced_full_ree.pt",
            map_location=device,
            weights_only=True,
        )
        full = RevealOptionHeads(768, full_hidden, 4).to(device)
        full.load_state_dict(full_checkpoint["model_state_dict"], strict=True)
        full_labels, full_probability = collect_probabilities(
            "full_ree", full, loader, device,
            full_checkpoint={"normalized_budgets": [1.5, 2.0, 3.0, 4.0]},
        )
        if not np.array_equal(labels, full_labels):
            raise RuntimeError("model evaluation label order mismatch")
        seed_results.append({
            "seed": seed,
            "ready_roc_auc": {
                "history": roc_auc(labels, history_probability[:, 2]),
                "full": roc_auc(labels, full_probability[:, 2]),
            },
            "matched_missed_ready": {
                str(budget): {
                    "history": matched_point(
                        labels, history_probability[:, 2], budget
                    ),
                    "full": matched_point(
                        labels, full_probability[:, 2], budget
                    ),
                }
                for budget in BUDGETS
            },
        })
    matched = {}
    for budget in BUDGETS:
        key = str(budget)
        history_values = [
            row["matched_missed_ready"][key]["history"]["false_ready_rate"]
            for row in seed_results
        ]
        full_values = [
            row["matched_missed_ready"][key]["full"]["false_ready_rate"]
            for row in seed_results
        ]
        history_mean = statistics.mean(history_values)
        full_mean = statistics.mean(full_values)
        matched[key] = {
            "history_false_ready_mean": history_mean,
            "full_false_ready_mean": full_mean,
            "relative_false_ready_reduction": (
                history_mean - full_mean
            ) / history_mean,
            "full_reaches_frozen_25pct_proxy": (
                history_mean - full_mean
            ) / history_mean >= 0.25,
            "history_values": history_values,
            "full_values": full_values,
        }
    auc_history = [row["ready_roc_auc"]["history"] for row in seed_results]
    auc_full = [row["ready_roc_auc"]["full"] for row in seed_results]
    result = {
        "schema_version": "revealnav-mf2-balanced-risk-delay/2",
        "status": "DEVELOPMENT_RISK_DELAY_BELOW_FROZEN_25PCT_PROXY",
        "scope": (
            "development representation proxy; not navigation PCR, an unbiased "
            "test, a confidence interval, or a paper result"
        ),
        "sources": {
            "protocol_sha256": sha256_file(PROTOCOL),
            "aggregate_sha256": sha256_file(AGGREGATE),
            "manifest_sha256": sha256_file(MANIFEST),
        },
        "selected_hidden_dim": full_hidden,
        "budgets": list(BUDGETS),
        "seed_results": seed_results,
        "aggregate": {
            "matched_missed_ready": matched,
            "ready_roc_auc": {
                "history_mean": statistics.mean(auc_history),
                "full_mean": statistics.mean(auc_full),
                "full_minus_history": (
                    statistics.mean(auc_full) - statistics.mean(auc_history)
                ),
                "history_values": auc_history,
                "full_values": auc_full,
            },
        },
        "gold_read": False,
        "navigation_pcr_measured": False,
        "paper_result": False,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "matched_missed_ready": matched,
        "ready_roc_auc": result["aggregate"]["ready_roc_auc"],
        "output": str(OUTPUT.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
