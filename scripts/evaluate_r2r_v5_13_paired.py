#!/usr/bin/env python3
"""Aggregate sealed V5.13 groups with paired episode bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_13_net_advantage/"
    "R2R_V5_14_ENSEMBLE_PROTOCOL.json"
)
DEFAULT_TRAINING_RESULT = ROOT / (
    "artifacts/phase1/r2r_train_net_advantage/full/training_v5_14/"
    "R2R_SPARSE_NET_ADVANTAGE_TRAINING_RESULT.json"
)
HIGHER_IS_BETTER = {"success", "oracle_success", "spl", "ndtw", "sdtw"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def load_group(
    path: Path, metrics: list[str], expected_group: str,
) -> dict[tuple[str, int], dict]:
    rows = {}
    for summary_path in path.rglob("RUN_SUMMARY.json"):
        row = json.loads(summary_path.read_text())
        if (
            row.get("status") != "PASS" or row.get("metrics") is None
            or row.get("group") != expected_group
        ):
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
    results = value.get("results", [])
    if sorted(row.get("seed") for row in results) != [
        20260826, 20260827, 20260828
    ]:
        raise RuntimeError("training result does not contain the three locked seeds")
    manifest = (ROOT / value["dataset_manifest"]).resolve()
    if (
        ROOT not in manifest.parents or manifest.is_symlink()
        or not manifest.is_file()
        or sha256_file(manifest) != value["dataset_manifest_sha256"]
    ):
        raise RuntimeError("training dataset manifest provenance drift")
    for row in results:
        checkpoint = (ROOT / row["checkpoint"]["path"]).resolve()
        if (
            ROOT not in checkpoint.parents or checkpoint.is_symlink()
            or not checkpoint.is_file()
            or checkpoint.stat().st_size != row["checkpoint"]["bytes"]
            or sha256_file(checkpoint) != row["checkpoint"]["sha256"]
        ):
            raise RuntimeError("training checkpoint provenance drift")
    deployment = value.get("deployment_checkpoint", {})
    checkpoint = (ROOT / deployment.get("path", "")).resolve()
    if (
        value.get("schema_version")
        != "revealnav-r2r-sparse-net-advantage-training/3"
        or value.get("deployment") != "three-member deterministic ensemble"
        or deployment.get("member_seeds") != [20260826, 20260827, 20260828]
        or ROOT not in checkpoint.parents or checkpoint.is_symlink()
        or not checkpoint.is_file()
        or checkpoint.stat().st_size != deployment.get("bytes")
        or sha256_file(checkpoint) != deployment.get("sha256")
    ):
        raise RuntimeError("ensemble deployment checkpoint provenance drift")
    return value


def paired_comparison(
    treatment: dict, baseline: dict, metrics: list[str], replicates: int,
) -> dict:
    seeds = sorted({key[1] for key in treatment})
    episodes = sorted({key[0] for key in treatment})
    if any((episode, seed) not in treatment for episode in episodes for seed in seeds):
        raise RuntimeError("paired episode/seed matrix is incomplete")
    if set(treatment) != set(baseline):
        baseline_seeds = {key[1] for key in baseline}
        baseline_episodes = {key[0] for key in baseline}
        if baseline_seeds != {0} or baseline_episodes != set(episodes):
            raise RuntimeError("treatment and baseline paired coverage differ")
        baseline = {
            (episode, seed): baseline[(episode, 0)]
            for episode in episodes for seed in seeds
        }
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
    if protocol.get("status") != (
        "SEALED_V5_14_AFTER_TRAIN_ONLY_FEASIBILITY_BEFORE_BENCHMARK_VALIDATION"
    ):
        raise RuntimeError("V5.14 protocol is not sealed")
    training_result = validate_training_result(training_result_path)
    metrics = protocol["evaluation"]["metrics"]
    groups = {
        row["id"]: load_group(runs_root / row["id"], metrics, row["id"])
        for row in protocol["groups"]
    }
    episodes = [{key[0] for key in rows} for rows in groups.values()]
    if any(value != episodes[0] for value in episodes[1:]):
        raise RuntimeError("five-group episode coverage differs")
    seed_policy = {row["id"]: row["seeds"] for row in protocol["groups"]}
    for group, rows in groups.items():
        expected = {
            (episode, seed) for episode in episodes[0]
            for seed in seed_policy[group]
        }
        if set(rows) != expected:
            raise RuntimeError(f"{group} seed coverage differs from protocol")
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
        "v5_6_net_advantage_minus_v5_6_net_advantage_no_return"
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
        "deterministic_baseline_seed_zero": seed_policy["etp_r1"] == [0],
        "three_locked_treatment_seeds": all(
            seed_policy[group] == protocol["training_lock"]["seeds"]
            for group in groups if group != "etp_r1"
        ),
        "primary_directionally_positive": directional,
        "primary_statistically_positive": statistical,
        "net_advantage_improves_v5_6_mean_spl": v56_gain > 0,
    }
    ablation_gates = {
        "ecog_reversibility_nonnegative_mean_spl": reversible_gain >= 0,
    }
    splits = {
        row["split"] for group in groups.values() for row in group.values()
    }
    if len(splits) != 1 or not splits <= {"val_seen", "val_unseen"}:
        raise RuntimeError("paired evaluation must use one authorized validation split")
    split = next(iter(splits))
    return {
        "schema_version": "revealnav-r2r-v5.14-paired-result/1",
        "status": "PASS" if all(main_gates.values()) else "FAIL",
        "main_gates": main_gates,
        "ablation_gates": ablation_gates,
        "deployed_main_variant": "v5_6_net_advantage",
        "group_metrics": {
            group: {
                metric: sum(
                    float(row["metrics"][metric]) for row in rows.values()
                ) / len(rows)
                for metric in metrics
            }
            for group, rows in groups.items()
        },
        "controller_totals": {
            group: {
                key: sum(
                    int((row.get("controller") or {}).get(key, 0))
                    for row in rows.values()
                )
                for key in (
                    "net_advantage_decisions", "net_advantage_approvals",
                    "net_advantage_vetoes", "checkpointed_excursions",
                    "backtrack_decisions", "successful_returns",
                    "failed_returns", "no_return_suppressions",
                )
            }
            for group, rows in groups.items()
        },
        "comparisons": comparisons,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "training_result": str(training_result_path.relative_to(ROOT)),
        "training_deployment": training_result["deployment"],
        "bootstrap_replicates": replicates,
        "split": split,
        "paper_result": split == "val_unseen",
    }


def write_tables(value: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ("success", "spl", "ndtw", "sdtw", "distance_to_goal")
    csv_path = output_dir / "R2R_V5_13_1_GROUP_METRICS.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("group", *metrics))
        for group, row in value["group_metrics"].items():
            writer.writerow((group, *(f"{row[key]:.6f}" for key in metrics)))
    controller_path = output_dir / "R2R_V5_13_1_CONTROLLER_METRICS.csv"
    with controller_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "group", "decisions", "approvals", "vetoes", "approval_rate",
            "checkpointed_excursions", "successful_returns", "failed_returns",
            "return_success_rate", "no_return_suppressions",
        ))
        for group, row in value["controller_totals"].items():
            decisions = row["net_advantage_decisions"]
            returns = row["successful_returns"] + row["failed_returns"]
            writer.writerow((
                group, decisions, row["net_advantage_approvals"],
                row["net_advantage_vetoes"],
                f"{row['net_advantage_approvals'] / decisions:.6f}"
                if decisions else "",
                row["checkpointed_excursions"], row["successful_returns"],
                row["failed_returns"],
                f"{row['successful_returns'] / returns:.6f}" if returns else "",
                row["no_return_suppressions"],
            ))
    lines = [
        "# R2R V5.13.1 paired results", "",
        "| Group | Success | SPL | nDTW | SDTW | Distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, row in value["group_metrics"].items():
        lines.append(
            f"| {group} | {row['success']:.4f} | {row['spl']:.4f} | "
            f"{row['ndtw']:.4f} | {row['sdtw']:.4f} | "
            f"{row['distance_to_goal']:.4f} |"
        )
    lines.extend([
        "", "## Paired benefit", "",
        "| Comparison | SPL mean [95% CI] | nDTW mean [95% CI] |",
        "|---|---:|---:|",
    ])
    for name, row in value["comparisons"].items():
        spl = row["benefit_treatment_minus_baseline"]["spl"]
        ndtw = row["benefit_treatment_minus_baseline"]["ndtw"]
        spl_low, spl_high = spl["episode_bootstrap_95pct"]
        ndtw_low, ndtw_high = ndtw["episode_bootstrap_95pct"]
        lines.append(
            f"| {name} | {spl['mean']:.4f} [{spl_low:.4f}, {spl_high:.4f}] | "
            f"{ndtw['mean']:.4f} [{ndtw_low:.4f}, {ndtw_high:.4f}] |"
        )
    (output_dir / "R2R_V5_13_1_PAPER_TABLES.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--training-result", type=Path, default=DEFAULT_TRAINING_RESULT
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    paths = [
        args.protocol.resolve(), args.training_result.resolve(),
        args.runs_root.resolve(), args.output.resolve(),
    ]
    tables_dir = (
        args.tables_dir.resolve() if args.tables_dir else paths[3].parent / "tables"
    )
    paths.append(tables_dir)
    if any(path != ROOT and ROOT not in path.parents for path in paths):
        raise SystemExit("evaluation paths must remain inside the project")
    if args.bootstrap_replicates < 1000:
        raise SystemExit("paired evaluation requires at least 1000 bootstrap replicates")
    value = evaluate(paths[0], paths[1], paths[2], args.bootstrap_replicates)
    paths[3].parent.mkdir(parents=True, exist_ok=True)
    part = paths[3].with_name(paths[3].name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, paths[3])
    write_tables(value, paths[4])
    print(json.dumps({
        "status": value["status"], "main_gates": value["main_gates"],
        "ablation_gates": value["ablation_gates"],
    }, sort_keys=True))
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
