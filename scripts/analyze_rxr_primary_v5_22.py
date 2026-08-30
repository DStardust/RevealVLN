#!/usr/bin/env python3
"""Produce the immutable scientific diagnosis for the RxR V5.22 screen."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/evaluation/mf2_rxr_primary_v5_22_seen_dev/full/runs"
RESULT = ROOT / (
    "artifacts/evaluation/mf2_rxr_primary_v5_22_seen_dev/"
    "RXR_PRIMARY_SCREEN_RESULT_V5_22.json"
)
OUTPUT = ROOT / (
    "artifacts/evaluation/mf2_rxr_primary_v5_22_seen_dev/"
    "RXR_PRIMARY_DIAGNOSIS_V5_22.json"
)
SEEDS = (20260826, 20260827, 20260828)
METRICS = ("success", "spl", "ndtw", "distance_to_goal", "path_length", "steps_taken")
HIGHER = {"success", "spl", "ndtw"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def main() -> int:
    frozen_result = json.loads(RESULT.read_text())
    if frozen_result.get("status") != "RXR_V5_22_ENGINEERING_PASS_SCIENTIFIC_FAIL":
        raise RuntimeError("unexpected V5.22 result status")
    summaries = [
        json.loads(path.read_text())
        for path in sorted(RUNS.glob("*/RUN_SUMMARY.json"))
    ]
    baseline = {row["episode_id"]: row for row in summaries if row["mode"] == "baseline"}
    treatment = [row for row in summaries if row["mode"] == "revealnav"]
    if len(baseline) != 24 or len(treatment) != 72:
        raise RuntimeError("V5.22 paired run inventory drift")

    funnel = Counter()
    changed = set()
    active = set()
    effective = set()
    per_seed = {}
    for row in treatment:
        episode = row["episode_id"]
        if row["controller"]["checkpointed_excursions"]:
            active.add(episode)
        for key, value in row.get("safety_funnel", {}).items():
            if isinstance(value, int):
                funnel[key] += value
        if any(
            abs(float(row["metrics"][metric]) - float(baseline[episode]["metrics"][metric])) > 1e-9
            for metric in METRICS
        ):
            changed.add(episode)
            effective.add((row["seed"], episode))
    for seed in SEEDS:
        rows = [row for row in treatment if row["seed"] == seed]
        deltas = {}
        for metric in METRICS:
            values = []
            for row in rows:
                delta = float(row["metrics"][metric]) - float(
                    baseline[row["episode_id"]]["metrics"][metric]
                )
                values.append(delta if metric in HIGHER else -delta)
            deltas[metric] = sum(values) / len(values)
        per_seed[str(seed)] = {
            "active_episodes": sum(
                row["controller"]["checkpointed_excursions"] > 0 for row in rows
            ),
            "checkpointed_excursions": sum(
                row["controller"]["checkpointed_excursions"] for row in rows
            ),
            "effective_metric_change_episodes": sum(
                (seed, row["episode_id"]) in effective for row in rows
            ),
            "mean_benefit_deltas": deltas,
        }
    value = {
        "schema_version": "revealnav-rxr-primary-diagnosis/5.22",
        "status": "V5_22_BOTTLENECK_LOCALIZED",
        "source_result": str(RESULT.relative_to(ROOT)),
        "source_result_sha256": sha256_file(RESULT),
        "runs": {"baseline": len(baseline), "treatment": len(treatment)},
        "unique_active_episodes": len(active),
        "unique_effective_metric_change_episodes": len(changed),
        "effective_seed_episode_pairs": len(effective),
        "funnel": dict(sorted(funnel.items())),
        "per_seed": per_seed,
        "diagnosis": {
            "initial_checkpoint_gate_is_not_the_task_bottleneck": True,
            "checkpointed_excursions": funnel["topology_snapshots"],
            "native_first_trials": funnel["native_first_trials"],
            "successful_returns": sum(
                row["controller"]["successful_returns"] for row in treatment
            ),
            "remaining_set_probes": funnel["remaining_set_probe_count"],
            "reason": (
                "Most checkpoints preserve the native ETP action and therefore cannot change "
                "task metrics. Only the frozen post-excursion decision can trigger a return; "
                "all three returns occur on one episode and produce the only route change."
            ),
            "required_correction": (
                "Train a policy-induced, baseline-relative reversible advantage model on RxR "
                "train states; do not repair V5.22 with an activation threshold or a selected cohort."
            ),
        },
        "val_unseen_or_test_accessed": False,
        "paper_result": False,
    }
    atomic_json(OUTPUT, value)
    print(json.dumps({
        "status": value["status"],
        "unique_active_episodes": value["unique_active_episodes"],
        "unique_effective_metric_change_episodes": value["unique_effective_metric_change_episodes"],
        "funnel": {
            key: value["funnel"].get(key, 0)
            for key in (
                "topology_snapshots", "native_first_trials", "robust_median_returns",
                "remaining_set_probe_count", "alternative_commits",
            )
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
