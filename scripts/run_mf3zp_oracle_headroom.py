#!/usr/bin/env python3
"""Evaluate pre-authorized Oracle RevealSkill rollouts, never create them."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.revealskill_protocol import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, OUTPUT, verify_protocol  # noqa: E402


LABEL_RESULT = OUTPUT / "MF3ZP_LABEL_VALIDITY_RESULT.json"
ROLLOUTS = OUTPUT / "MF3ZP_ORACLE_SKILL_ROLLOUTS.jsonl"
RESULT = OUTPUT / "MF3ZP_ORACLE_HEADROOM_RESULT.json"


class OracleHeadroomError(RuntimeError):
    pass


def utility(metrics: Mapping[str, object]) -> float:
    values = [float(metrics[key]) for key in ("nDTW", "SDTW", "SPL")]
    if any(not math.isfinite(value) for value in values):
        raise OracleHeadroomError("nonfinite navigation metric")
    return 0.50 * values[0] + 0.25 * values[1] + 0.25 * values[2]


def premature_bounds(commit_step: int, reveal_interval: tuple[int, int]) -> tuple[int, int]:
    lower, upper = reveal_interval
    if commit_step < 0 or lower < 0 or upper < lower:
        raise OracleHeadroomError("invalid commit/reveal interval")
    # definite when commitment predates the earliest possible Reveal; possible
    # when it predates the latest possible Reveal.
    return (int(commit_step < lower), int(commit_step < upper))


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.floor(q * (len(ordered) - 1)))))
    return ordered[index]


def scene_bootstrap(rows: list[dict[str, object]], *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict[str, object]:
    by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene_id"])].append(row)
    scenes = sorted(by_scene)
    if len(scenes) < 2:
        raise OracleHeadroomError("at least two raw scenes required")
    rng = random.Random(seed)
    delta_samples: list[float] = []
    pcr_samples: list[float] = []
    for _ in range(replicates):
        sample = [item for _scene in (rng.choice(scenes) for _ in scenes) for item in by_scene[_scene]]
        delta_samples.append(sum(float(item["delta_utility"]) for item in sample) / len(sample))
        base_lower = sum(int(item["base_premature_lower"]) for item in sample) / len(sample)
        oracle_upper = sum(int(item["oracle_premature_upper"]) for item in sample) / len(sample)
        pcr_samples.append((base_lower - oracle_upper) / base_lower if base_lower > 0 else float("-inf"))
    return {
        "delta_utility_lower_95": _percentile(delta_samples, 0.025),
        "delta_utility_upper_95": _percentile(delta_samples, 0.975),
        "pcr_relative_reduction_conservative_lower_95": _percentile(pcr_samples, 0.025),
        "replicates": replicates,
        "cluster": "raw_mp3d_scene",
        "seed": seed,
    }


def evaluate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    normalized: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    for row in rows:
        required = {"dataset", "scene_id", "episode_id", "reveal_interval", "base", "oracle", "controller_frozen", "teleport", "public_split"}
        if set(row) != required:
            raise OracleHeadroomError("oracle rollout schema mismatch")
        if row["dataset"] not in {"R2R", "RxR"} or row["public_split"] is not False or row["controller_frozen"] is not True or row["teleport"] is not False:
            raise OracleHeadroomError("oracle rollout boundary violation")
        identity = (str(row["dataset"]), str(row["episode_id"]))
        if identity in identities:
            raise OracleHeadroomError("duplicate oracle rollout identity")
        identities.add(identity)
        reveal = tuple(int(value) for value in row["reveal_interval"])
        if len(reveal) != 2:
            raise OracleHeadroomError("Reveal interval must contain two endpoints")
        base = row["base"]
        oracle = row["oracle"]
        base_bounds = premature_bounds(int(base["commit_step"]), reveal)
        oracle_bounds = premature_bounds(int(oracle["commit_step"]), reveal)
        normalized.append({
            "dataset": row["dataset"], "scene_id": row["scene_id"], "episode_id": row["episode_id"],
            "delta_utility": utility(oracle["metrics"]) - utility(base["metrics"]),
            "base_premature_lower": base_bounds[0], "base_premature_upper": base_bounds[1],
            "oracle_premature_lower": oracle_bounds[0], "oracle_premature_upper": oracle_bounds[1],
        })
    domains: dict[str, object] = {}
    failures: list[str] = []
    for domain in ("R2R", "RxR"):
        subset = [row for row in normalized if row["dataset"] == domain]
        if not subset:
            failures.append(f"{domain}:no_rows")
            continue
        base_lower = sum(int(row["base_premature_lower"]) for row in subset) / len(subset)
        oracle_upper = sum(int(row["oracle_premature_upper"]) for row in subset) / len(subset)
        reduction = (base_lower - oracle_upper) / base_lower if base_lower > 0 else float("-inf")
        delta = sum(float(row["delta_utility"]) for row in subset) / len(subset)
        bootstrap = scene_bootstrap(subset, seed=BOOTSTRAP_SEED + (0 if domain == "R2R" else 1))
        domain_failures = []
        if reduction < 0.25:
            domain_failures.append("pcr_relative_reduction_below_25pct")
        if delta <= 0 or bootstrap["delta_utility_lower_95"] <= 0:
            domain_failures.append("navigation_utility_not_positive_with_ci")
        if bootstrap["pcr_relative_reduction_conservative_lower_95"] <= 0:
            domain_failures.append("pcr_bootstrap_lower_not_positive")
        failures.extend(f"{domain}:{reason}" for reason in domain_failures)
        domains[domain] = {
            "episodes": len(subset), "raw_scenes": len({row["scene_id"] for row in subset}),
            "mean_delta_utility": delta,
            "base_pcr_conservative_lower": base_lower,
            "oracle_pcr_conservative_upper": oracle_upper,
            "pcr_relative_reduction_conservative": reduction,
            "bootstrap": bootstrap, "failures": domain_failures,
        }
    return {"domains": domains, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "evaluate"))
    args = parser.parse_args()
    verify_protocol()
    if not LABEL_RESULT.is_file() or json.loads(LABEL_RESULT.read_text()).get("status") != "MF3ZP_LABEL_VALIDITY_PASS":
        print(json.dumps({"status": "MF3ZP_ORACLE_HEADROOM_NOT_AUTHORIZED", "reason": "formal_label_validity_not_passed", "checkpoint_generated": False}, indent=2))
        return 3
    if args.command == "check":
        print(json.dumps({"status": "MF3ZP_ORACLE_HEADROOM_INPUT_REQUIRED", "rollouts": str(ROLLOUTS.relative_to(ROOT))}, indent=2))
        return 0
    rows = [json.loads(line) for line in ROLLOUTS.read_text(encoding="utf-8").splitlines()]
    evidence = evaluate_rows(rows)
    result = {
        "schema_version": "revealnav-mf3zp-oracle-headroom/1",
        "status": "MF3ZP_ORACLE_HEADROOM_PASS" if not evidence["failures"] else "MF3ZP_ORACLE_HEADROOM_FAIL",
        **evidence,
        "checkpoint_generated": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }
    if RESULT.exists() or RESULT.is_symlink():
        raise OracleHeadroomError("refusing to overwrite oracle result")
    partial = RESULT.with_name(RESULT.name + ".part")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, RESULT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not evidence["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
