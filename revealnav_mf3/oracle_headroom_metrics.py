"""Metric and scene-cluster bootstrap helpers for MF3ZQ.

The helpers are deliberately independent of Habitat.  A rollout worker writes
one row per unique episode after the frozen controller exits; this module then
computes paired utility/PCR/option metrics without ever selecting a policy from
the observed outcomes.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


UTILITY_WEIGHTS = {"nDTW": 0.50, "SDTW": 0.25, "SPL": 0.25}
CATASTROPHIC_THRESHOLD = -0.10
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260901


class HeadroomMetricError(ValueError):
    pass


def utility(metrics: Mapping[str, object]) -> float:
    try:
        values = [float(metrics[name]) for name in ("nDTW", "SDTW", "SPL")]
    except (KeyError, TypeError, ValueError) as error:
        raise HeadroomMetricError("metrics must contain nDTW, SDTW and SPL") from error
    if any(not math.isfinite(value) for value in values):
        raise HeadroomMetricError("navigation metrics must be finite")
    return sum(weight * value for weight, value in zip(UTILITY_WEIGHTS.values(), values, strict=True))


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def scene_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    value_fn,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Bootstrap a row-level statistic by raw MP3D scene clusters."""

    if replicates != BOOTSTRAP_REPLICATES:
        raise HeadroomMetricError("MF3ZQ bootstrap replicates are fixed at 10000")
    by_scene: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        scene = str(row.get("scene_id", ""))
        if not scene:
            raise HeadroomMetricError("scene_id is required for clustered bootstrap")
        by_scene[scene].append(row)
    scenes = tuple(sorted(by_scene))
    if len(scenes) < 2:
        raise HeadroomMetricError("at least two raw scenes are required")
    observed = float(value_fn(tuple(rows)))
    rng = np.random.default_rng(int(seed))
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draw = rng.choice(scenes, size=len(scenes), replace=True)
        sampled = tuple(item for scene in draw for item in by_scene[str(scene)])
        samples[index] = float(value_fn(sampled))
    return {
        "observed": observed,
        "lower_95": _percentile(samples, 0.025),
        "upper_95": _percentile(samples, 0.975),
        "scene_count": len(scenes),
        "row_count": len(rows),
        "cluster": "raw_mp3d_scene",
        "replicates": replicates,
        "seed": int(seed),
    }


def _metric_value(row: Mapping[str, object], arm: str) -> float:
    metrics = row.get(arm)
    if not isinstance(metrics, Mapping):
        raise HeadroomMetricError(f"missing metrics for arm {arm}")
    return utility(metrics)


def summarize_arm(rows: Sequence[Mapping[str, object]], arm: str) -> dict[str, object]:
    if not rows:
        return {"episodes": 0, "utility": None, "metrics": {}, "unsupported": True}
    values = [
        _metric_value(row, arm)
        for row in rows
    ]
    names = ("SR", "SPL", "nDTW", "SDTW", "NE")
    aggregate: dict[str, float] = {}
    for name in names:
        observed = [float(row[arm][name]) for row in rows if name in row.get(arm, {})]
        if observed and all(math.isfinite(value) for value in observed):
            aggregate[name] = float(np.mean(observed))
    catastrophes = sum(
        int(float(row.get("delta_utility", {}).get(arm, 0.0)) <= CATASTROPHIC_THRESHOLD)
        for row in rows
        if isinstance(row.get("delta_utility"), Mapping)
    )
    return {
        "episodes": len(rows),
        "utility": float(np.mean(values)),
        "metrics": aggregate,
        "catastrophe_count": catastrophes,
        "catastrophe_rate": float(catastrophes / len(rows)),
        "scene_count": len({str(row["scene_id"]) for row in rows}),
        "unsupported": False,
    }


def paired_delta_rows(
    rows: Sequence[Mapping[str, object]],
    arm: str,
    *,
    baseline_arm: str = "baseline",
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        delta = _metric_value(row, arm) - _metric_value(row, baseline_arm)
        result.append({
            "dataset": str(row["dataset"]),
            "scene_id": str(row["scene_id"]),
            "episode_id": str(row["episode_id"]),
            "delta_utility": float(delta),
        })
    return result


def _rate(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = [row.get(field) for row in rows]
    if not values or any(value is None for value in values):
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def pcr_relative_reduction(rows: Sequence[Mapping[str, object]], base_field: str, arm_field: str) -> float | None:
    base = _rate(rows, base_field)
    arm = _rate(rows, arm_field)
    if base is None or arm is None:
        return None
    if base == 0.0:
        return 0.0 if arm == 0.0 else float("-inf")
    return float((base - arm) / base)


def summarize_pair(
    rows: Sequence[Mapping[str, object]],
    arm: str,
    *,
    baseline_arm: str = "baseline",
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    deltas = paired_delta_rows(rows, arm, baseline_arm=baseline_arm)
    utility_ci = scene_cluster_bootstrap(deltas, lambda values: float(np.mean([float(v["delta_utility"]) for v in values])), seed=seed)
    delta_values = [float(value["delta_utility"]) for value in deltas]
    catastrophes = sum(value <= CATASTROPHIC_THRESHOLD for value in delta_values)
    result = {
        "arm": arm,
        "baseline_arm": baseline_arm,
        "episodes": len(rows),
        "mean_delta_utility": float(np.mean(delta_values)) if delta_values else None,
        "delta_utility_bootstrap": utility_ci,
        "catastrophe_count": int(catastrophes),
        "catastrophe_rate": float(catastrophes / len(delta_values)) if delta_values else None,
    }
    for name in ("pcr", "olr", "mor"):
        base_field = f"baseline_{name}"
        arm_field = f"{arm}_{name}"
        reduction = pcr_relative_reduction(rows, base_field, arm_field)
        if reduction is not None:
            result[f"{name}_relative_reduction"] = reduction
    return result


def apply_pcr_bounds(commit_step: int | None, reveal_step: int | None) -> tuple[int, int] | None:
    if commit_step is None or reveal_step is None:
        return None
    if int(commit_step) < 0 or int(reveal_step) < 0:
        raise HeadroomMetricError("steps must be non-negative")
    premature = int(int(commit_step) < int(reveal_step))
    return premature, premature


__all__ = [
    "UTILITY_WEIGHTS", "CATASTROPHIC_THRESHOLD", "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED", "HeadroomMetricError", "utility", "scene_cluster_bootstrap",
    "summarize_arm", "paired_delta_rows", "summarize_pair", "pcr_relative_reduction",
    "apply_pcr_bounds",
]
