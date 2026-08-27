#!/usr/bin/env python3
"""Aggregate sealed V5.13 groups with paired episode bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_13_net_advantage/"
    "R2R_V5_13_NET_ADVANTAGE_PROTOCOL.json"
)
DEFAULT_TRAINING_RESULT = ROOT / (
    "artifacts/phase1/r2r_train_net_advantage/full/training/"
    "R2R_SPARSE_NET_ADVANTAGE_TRAINING_RESULT.json"
)
HIGHER_IS_BETTER = {"success", "oracle_success", "spl", "ndtw", "sdtw"}


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def load_group(path: Path, metrics: list[str]) -> dict[tuple[str, int], dict]:
    rows = {}
    for summary_path in path.rglob("RUN_SUMMARY.json"):
        row = json.loads(summary_path.read_text())
        if row.get("status") != "PASS" or row.get("metrics") is None:
            raise RuntimeError(f"invalid paired run: {summary_path}")
        key = (str(row["episode_id"]), int(row["seed"]))
        if key in rows:
            raise RuntimeError(f"duplicate paired key in {path}: {key}")
        if not all(math.isfinite(float(row["metrics"][metric])) for metric in metrics):
            raise RuntimeError(f"non-finite task metric: {summary_path}")
        rows[key] = row
    if not rows:
        raise RuntimeError(f"no completed runs in {path}")
    return rows


def validate_training_result(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "R2R_SPARSE_NET_ADVANTAGE_LEARNABILITY_PASS":
        raise RuntimeError("full training learnability gate did not pass")
    if value.get("unseen_or_test_read") is not False:
        raise RuntimeError("training result does not prove unseen/test isolation")
    if value.get("task_metric_payload_read") is not False:
        raise RuntimeError("training result used forbidden task-metric payload")
    if sorted(row.get("seed") for row in value.get("results", [])) != [
        20260826, 20260827, 20260828
    ]:
        raise RuntimeError("training result does not contain the three locked seeds")
    return value


def paired_comparison(
    treatment: dict, baseline: dict, metrics: list[str], replicates: int,
) -> dict:
    if set(treatment) != set(baseline):
        raise RuntimeError("treatment and baseline paired keys differ")
    seeds = sorted({key[1] for key in treatment})
    episodes = sorted({key[0] for key in treatment})
    if any((episode, seed) not in treatment for episode in episodes for seed in seeds):
        raise RuntimeError("paired episode/seed matrix is incomplete")
    per_episode = {metric: {} for metric in metrics}
    for episode in episodes:
        for metric in metrics:
            direction = 1.0 if metric in HIGHER_IS_BETTER else -1.0
            per_episode[metric][episode] = sum(
                direction * (
                    float(treatment[(episode, seed)]["metrics"][metric])
                    - float(baseline[(episode, seed)]["metrics"][metric])
                ) for seed in seeds
            ) / len(seeds)
    rng = random.Random(20260827)
    boots = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sample = [rng.choice(episodes) for _ in episodes]
        for metric in metrics:
            boots[metric].append(sum(
                per_episode[metric][episode] for episode in sample
            ) / len(sample))
    return {
        "paired_episodes": len(episodes),
        "seeds": seeds,
        "benefit_treatment_minus_baseline": {
            metric: {
                "mean": sum(values.values()) / len(values),
                "median": quantile(list(values.values()), 0.5),
                "minimum": min(values.values()),
                "maximum": max(values.values()),
                "episode_bootstrap_95pct": [
                    quantile(boots[metric], 0.025),
                    quantile(boots[metric], 0.975),
                ],
            } for metric, values in per_episode.items()
        },
    }


def evaluate(
    protocol_path: Path, training_result_path: Path, runs_root: Path,
    replicates: int,
) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "SEALED_BEFORE_FULL_TRAINING_AND_UNSEEN_EVALUATION":
        raise RuntimeError("V5.13 protocol is not sealed")
    training_result = validate_training_result(training_result_path)
    metrics = protocol["evaluation"]["metrics"]
    groups = {
        row["id"]: load_group(runs_root / row["id"], metrics)
        for row in protocol["groups"]
    }
    keys = [set(rows) for rows in groups.values()]
    if any(value != keys[0] for value in keys[1:]):
        raise RuntimeError("five-group paired coverage differs")
    comparisons = {}
    for row in protocol["comparisons"]:
        name = f"{row['treatment']}_minus_{row['baseline']}"
        comparisons[name] = paired_comparison(
            groups[row["treatment"]], groups[row["baseline"]],
            metrics, replicates,
        )
    primary = comparisons["v5_6_net_advantage_minus_etp_r1"][
        "benefit_treatment_minus_baseline"
    ]
    v56_gain = comparisons["v5_6_net_advantage_minus_v5_6"][
        "benefit_treatment_minus_baseline"
    ]["spl"]["mean"]
    reversible_gain = comparisons[
        "v5_6_net_advantage_reversible_minus_v5_6_net_advantage"
    ]["benefit_treatment_minus_baseline"]["spl"]["mean"]
    directional = (
        primary["spl"]["mean"] > 0 and primary["ndtw"]["mean"] > 0
        and primary["success"]["mean"] >= 0
    )
    statistical = (
        primary["spl"]["episode_bootstrap_95pct"][0] > 0
        and primary["ndtw"]["episode_bootstrap_95pct"][0] > 0
    )
    main_gates = {
        "all_five_groups_complete_and_paired": True,
        "three_locked_seeds": sorted({key[1] for key in keys[0]})
        == protocol["training_lock"]["seeds"],
        "primary_directionally_positive": directional,
        "primary_statistically_positive": statistical,
        "net_advantage_improves_v5_6_mean_spl": v56_gain > 0,
    }
    ablation_gates = {
        "reversible_module_nonnegative_mean_spl": reversible_gain >= 0,
    }
    splits = {
        row["split"] for group in groups.values() for row in group.values()
    }
    if len(splits) != 1 or not splits <= {"val_seen", "val_unseen"}:
        raise RuntimeError("paired evaluation must use one authorized validation split")
    split = next(iter(splits))
    return {
        "schema_version": "revealnav-r2r-v5.13-paired-result/1",
        "status": "PASS" if all(main_gates.values()) else "FAIL",
        "main_gates": main_gates,
        "ablation_gates": ablation_gates,
        "deployed_main_variant": (
            "v5_6_net_advantage_reversible"
            if ablation_gates["reversible_module_nonnegative_mean_spl"]
            else "v5_6_net_advantage"
        ),
        "comparisons": comparisons,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "training_result": str(training_result_path.relative_to(ROOT)),
        "selected_training_seed": training_result["selected_seed"],
        "bootstrap_replicates": replicates,
        "split": split,
        "paper_result": split == "val_unseen",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--training-result", type=Path, default=DEFAULT_TRAINING_RESULT
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    paths = [
        args.protocol.resolve(), args.training_result.resolve(),
        args.runs_root.resolve(), args.output.resolve(),
    ]
    if any(path != ROOT and ROOT not in path.parents for path in paths):
        raise SystemExit("evaluation paths must remain inside the project")
    if args.bootstrap_replicates < 1000:
        raise SystemExit("paired evaluation requires at least 1000 bootstrap replicates")
    value = evaluate(paths[0], paths[1], paths[2], args.bootstrap_replicates)
    paths[3].parent.mkdir(parents=True, exist_ok=True)
    part = paths[3].with_name(paths[3].name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, paths[3])
    print(json.dumps({
        "status": value["status"], "main_gates": value["main_gates"],
        "ablation_gates": value["ablation_gates"],
    }, sort_keys=True))
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
