#!/usr/bin/env python3
"""Run the fixed MF3ZN temporal pretest without human label review.

This is an explicitly exploratory, fail-closed bridge requested after the
formal MF3ZN-TUAD protocol was sealed.  It reuses the 1,540 existing exact
native-versus-runner outcomes and strictly causal scalar proposal traces.  It
does *not* manufacture UAD/oracle labels, authorize formal TEAL collection, or
access a public validation/test split.

The one fixed comparison is a pooled five-fold whole-scene ridge probe:

* current-only: the ten frozen policy scalars plus six causal semantic cosines;
* temporal: current-only plus a predeclared trace-history summary;
* deployment: choose the existing runner iff its OOF predicted delta is > 0.

The result is written once and never treated as the sealed identifiability
gate.  A failure stops the exploratory progression before new collection.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.rcsp_v1_1 import (  # noqa: E402
    POLICY_FEATURE_NAMES,
    policy_features,
)
from revealnav_mf3.tuad_identifiability import (  # noqa: E402
    canonical_audit_event_id,
)
from revealnav_mf3.tuad_protocol import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    CATASTROPHIC_THRESHOLD,
    HUBER_DELTA,
    IDENTIFIABILITY_CANONICAL_IDENTITY_SHA256,
    IDENTIFIABILITY_EXPECTED_DOMAIN_COUNTS,
    IDENTIFIABILITY_EXPECTED_ROWS,
    IDENTIFIABILITY_EXPECTED_SCENES,
    sha256_file,
    verify_protocol,
)
from revealnav_mf3.tuad_selection import (  # noqa: E402
    REQUIRED_DOMAINS,
    TUAD_OUTER_FOLDS,
    assign_tuad_scene_folds,
    matched_budget_baselines,
    scene_cluster_difference,
)


FORMAL_PROTOCOL = (
    ROOT / "artifacts/training/mf3zn_tuad_v1/MF3ZN_TUAD_PROTOCOL.json"
)
CAR_TRAINER = ROOT / "scripts/train_mf3zm_car.py"
RESULT = ROOT / (
    "artifacts/training/mf3zn_tuad_exploratory_no_review_v1/"
    "MF3ZN_NO_REVIEW_TEMPORAL_PRETEST.json"
)
RIDGE_L2 = 1.0
BOOTSTRAP_SEED = 20260831

TRACE_SCALAR_NAMES = (
    "policy_risk_adjusted_score",
    "native_margin",
    "robust_top2_advantage",
    "ensemble_mad",
)
TRACE_SCALAR_STATISTICS = (
    "slope",
    "mean",
    "std",
    "last_minus_first",
    "observed_fraction",
)
STRUCTURAL_TEMPORAL_NAMES = (
    "prefix_record_count",
    "decision_step_span",
    "candidate_birth_count",
    "candidate_expiry_count",
    "candidate_count_mean",
    "candidate_count_std",
    "candidate_count_slope",
    "native_persistence",
    "runner_persistence",
    "native_switch_count",
    "runner_switch_count",
    "candidate_set_jaccard",
)
TEMPORAL_FEATURE_NAMES = tuple(
    f"{scalar}_{statistic}"
    for scalar in TRACE_SCALAR_NAMES
    for statistic in TRACE_SCALAR_STATISTICS
) + STRUCTURAL_TEMPORAL_NAMES
SEMANTIC_FEATURE_NAMES = (
    "instruction_checkpoint_cosine",
    "instruction_native_cosine",
    "instruction_runner_cosine",
    "checkpoint_native_cosine",
    "checkpoint_runner_cosine",
    "native_runner_cosine",
)
CURRENT_FEATURE_NAMES = POLICY_FEATURE_NAMES + SEMANTIC_FEATURE_NAMES


class ExploratoryPretestError(RuntimeError):
    """The no-review pretest cannot be evaluated without changing its scope."""


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExploratoryPretestError(f"cannot load project module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strict_json(payload: str, source: Path) -> object:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExploratoryPretestError(
                    f"duplicate JSON key in {source}: {key}"
                )
            result[key] = value
        return result

    def reject_constant(token: str):
        raise ExploratoryPretestError(
            f"non-finite JSON constant in {source}: {token}"
        )

    return json.loads(
        payload,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


def _atomic_json(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise ExploratoryPretestError(f"refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ExploratoryPretestError(f"stale atomic output: {partial}")
    partial.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _trace_path(row: dict) -> Path:
    feature = ROOT / str(row["feature"]["path"])
    if row["source"] == "mf3zk_dsr_v1_existing_exact":
        trace = feature.parent / "controller_trace.jsonl"
    elif row["source"] in {
        "mf3zl_parent_dense_exact", "mf3zl_v1r1_variant_exact",
    }:
        trace = feature.parent.parent / "proposal_trace.jsonl"
    else:
        raise ExploratoryPretestError(f"unknown exact source: {row['source']}")
    if not trace.is_file() or trace.is_symlink():
        raise ExploratoryPretestError(f"invalid causal trace: {trace}")
    resolved = trace.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise ExploratoryPretestError(f"causal trace escaped project root: {trace}")
    return trace


def _load_trace(path: Path) -> tuple[dict, ...]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        value = _strict_json(line, path)
        if not isinstance(value, dict):
            raise ExploratoryPretestError(
                f"trace row is not an object: {path}:{line_number}"
            )
        records.append(value)
    if not records:
        raise ExploratoryPretestError(f"empty causal trace: {path}")
    steps = [value.get("step") for value in records]
    if any(
        isinstance(step, bool) or not isinstance(step, int) or step < 0
        for step in steps
    ) or any(left >= right for left, right in zip(steps, steps[1:])):
        raise ExploratoryPretestError(f"trace steps are not strictly increasing: {path}")
    return tuple(records)


def _finite_optional(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExploratoryPretestError(f"non-numeric causal scalar: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ExploratoryPretestError(f"non-finite causal scalar: {name}")
    return result


def _candidate_ids(
    record: dict, *, require_nonempty: bool = False,
) -> tuple[str, ...]:
    raw = record.get("current_local_action_ids")
    if not isinstance(raw, list):
        raise ExploratoryPretestError("causal trace candidate list is not a list")
    result = tuple(str(value) for value in raw)
    if (
        (require_nonempty and not result)
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ExploratoryPretestError("invalid causal candidate identity list")
    return result


def _runner_identity(record: dict) -> str | None:
    value = record.get("runner_local_index")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExploratoryPretestError("invalid causal runner index")
    return f"runner-index:{value}"


def _native_identity(record: dict) -> str | None:
    value = record.get("native_action_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ExploratoryPretestError("invalid causal native identity")
    return value


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    centered = times - times.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(centered, values - values.mean()) / denominator)


def _persistence(values: Sequence[str | None]) -> float:
    if len(values) == 1:
        return float(values[0] is not None)
    return float(sum(
        left is not None and right is not None and left == right
        for left, right in zip(values, values[1:])
    ) / (len(values) - 1))


def _switch_count(values: Sequence[str | None]) -> float:
    return float(sum(
        left is not None and right is not None and left != right
        for left, right in zip(values, values[1:])
    ))


def _candidate_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _temporal_summary(records: Sequence[dict], decision_step: int) -> np.ndarray:
    """Summarize only rows at or before the target decision step."""

    prefix = tuple(record for record in records if int(record["step"]) <= decision_step)
    if not prefix or int(prefix[-1]["step"]) != decision_step:
        raise ExploratoryPretestError("causal trace does not end at decision step")
    times = np.asarray([int(value["step"]) for value in prefix], dtype=np.float64)
    result: list[float] = []
    for scalar in TRACE_SCALAR_NAMES:
        observed = [
            (int(record["step"]), _finite_optional(record.get(scalar), name=scalar))
            for record in prefix
        ]
        valid = [(step, value) for step, value in observed if value is not None]
        if not valid:
            raise ExploratoryPretestError(
                f"no observed causal values for temporal scalar: {scalar}"
            )
        scalar_times = np.asarray([value[0] for value in valid], dtype=np.float64)
        scalar_values = np.asarray([value[1] for value in valid], dtype=np.float64)
        result.extend((
            _slope(scalar_times, scalar_values),
            float(scalar_values.mean()),
            float(scalar_values.std()),
            float(scalar_values[-1] - scalar_values[0]),
            float(len(valid) / len(prefix)),
        ))

    candidates = tuple(_candidate_ids(record) for record in prefix)
    candidate_counts = np.asarray([len(value) for value in candidates], dtype=np.float64)
    births = 0
    expiries = 0
    previous: set[str] = set()
    for value in candidates:
        current = set(value)
        births += len(current - previous)
        expiries += len(previous - current)
        previous = current
    natives = tuple(_native_identity(record) for record in prefix)
    runners = tuple(_runner_identity(record) for record in prefix)
    jaccards = [
        _candidate_jaccard(left, right)
        for left, right in zip(candidates, candidates[1:])
    ]
    result.extend((
        float(len(prefix)),
        float(times[-1] - times[0]),
        float(births),
        float(expiries),
        float(candidate_counts.mean()),
        float(candidate_counts.std()),
        _slope(times, candidate_counts),
        _persistence(natives),
        _persistence(runners),
        _switch_count(natives),
        _switch_count(runners),
        float(np.mean(jaccards)) if jaccards else 1.0,
    ))
    array = np.asarray(result, dtype=np.float64)
    if array.shape != (len(TEMPORAL_FEATURE_NAMES),) or not np.isfinite(array).all():
        raise ExploratoryPretestError("temporal feature contract drift")
    return array


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != (768,) or right.shape != (768,):
        raise ExploratoryPretestError("semantic embedding shape drift")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ExploratoryPretestError("semantic embedding has zero norm")
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))


def _current_features(decision: dict, arrays: dict) -> np.ndarray:
    instruction = arrays["instruction"]
    checkpoint = arrays["checkpoint"]
    native = arrays["native"]
    runner = arrays["alternative"]
    semantic = np.asarray([
        _cosine(instruction, checkpoint),
        _cosine(instruction, native),
        _cosine(instruction, runner),
        _cosine(checkpoint, native),
        _cosine(checkpoint, runner),
        _cosine(native, runner),
    ], dtype=np.float64)
    result = np.concatenate((policy_features(decision), semantic))
    if result.shape != (len(CURRENT_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ExploratoryPretestError("current feature contract drift")
    return result


def _validate_current_trace(record: dict, decision: dict) -> None:
    if int(record["step"]) != int(decision["step"]):
        raise ExploratoryPretestError("target trace step mismatch")
    if _candidate_ids(record, require_nonempty=True) != tuple(
        str(value) for value in decision["current_local_action_ids"]
    ):
        raise ExploratoryPretestError("target trace candidate set mismatch")
    for name in TRACE_SCALAR_NAMES + (
        "minimum_top2_advantage",
        "median_top2_advantage",
        "cold_start_floor_ratio",
        "cold_start_relative_mad",
    ):
        observed = _finite_optional(record.get(name), name=name)
        expected = float(decision[name])
        if observed is None or not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ExploratoryPretestError(f"target trace scalar mismatch: {name}")


def _future_mutation_check(records: Sequence[dict], decision_step: int) -> bool:
    expected = _temporal_summary(records, decision_step)
    changed = []
    for record in records:
        value = dict(record)
        if int(value["step"]) > decision_step:
            value["policy_risk_adjusted_score"] = 1.0e12
            value["native_margin"] = 1.0e12
            value["robust_top2_advantage"] = -1.0e12
            value["ensemble_mad"] = 1.0e12
            value["current_local_action_ids"] = ["future-mutated"]
            value["native_action_id"] = "future-native-mutated"
            value["runner_local_index"] = 987654321
        changed.append(value)
    observed = _temporal_summary(changed, decision_step)
    return expected.tobytes(order="C") == observed.tobytes(order="C")


def _ridge_oof(
    matrix: np.ndarray, target: np.ndarray, folds: np.ndarray,
) -> np.ndarray:
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    for fold in range(TUAD_OUTER_FOLDS):
        fit = folds != fold
        held = folds == fold
        if not fit.any() or not held.any():
            raise ExploratoryPretestError("incomplete whole-scene OOF partition")
        mean = matrix[fit].mean(axis=0)
        scale = matrix[fit].std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        fit_matrix = (matrix[fit] - mean) / scale
        held_matrix = (matrix[held] - mean) / scale
        design = np.column_stack((np.ones(int(fit.sum())), fit_matrix))
        penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_L2
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target[fit],
        )
        prediction[held] = np.column_stack((
            np.ones(int(held.sum())), held_matrix,
        )) @ coefficient
    if not np.isfinite(prediction).all():
        raise ExploratoryPretestError("OOF prediction is incomplete")
    return prediction


def _huber(error: np.ndarray) -> np.ndarray:
    absolute = np.abs(error)
    return np.where(
        absolute <= HUBER_DELTA,
        0.5 * error * error,
        HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
    )


def _scene_bootstrap_mean(
    values: np.ndarray,
    scenes: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
) -> dict:
    population = np.unique(scenes[mask])
    if len(population) < 2:
        raise ExploratoryPretestError("scene bootstrap requires two scenes")
    totals = np.asarray([
        values[mask & (scenes == scene)].sum() for scene in population
    ], dtype=np.float64)
    counts = np.asarray([
        int((mask & (scenes == scene)).sum()) for scene in population
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(population), size=(BOOTSTRAP_REPLICATES, len(population)),
    )
    replicates = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "observed": float(values[mask].mean()),
        "lower_95": float(np.quantile(replicates, 0.025)),
        "upper_95": float(np.quantile(replicates, 0.975)),
    }


def _catastrophic_rate(selected: np.ndarray, target: np.ndarray) -> float | None:
    return (
        float((target[selected] <= CATASTROPHIC_THRESHOLD).mean())
        if bool(selected.any()) else None
    )


def _policy_evidence(
    selected: np.ndarray,
    target: np.ndarray,
    scenes: np.ndarray,
    datasets: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, dict]:
    utility = np.where(selected, target, 0.0)
    by_domain = {}
    for domain in REQUIRED_DOMAINS:
        domain_mask = datasets == domain
        selected_scenes = sorted(set(scenes[domain_mask & selected].tolist()))
        leave_one = [
            float(utility[domain_mask & (scenes != scene)].sum())
            for scene in selected_scenes
        ]
        fold_values = {}
        for fold in range(TUAD_OUTER_FOLDS):
            stratum = domain_mask & (folds == fold)
            if not bool(stratum.any()):
                raise ExploratoryPretestError(
                    f"missing fold={fold},domain={domain} stratum"
                )
            fold_values[str(fold)] = {
                "utility": float(utility[stratum].sum()),
                "selected": int(selected[stratum].sum()),
                "catastrophic_rate": _catastrophic_rate(
                    selected[stratum], target[stratum],
                ),
            }
        by_domain[domain] = {
            "utility": float(utility[domain_mask].sum()),
            "selected": int(selected[domain_mask].sum()),
            "catastrophic_rate": _catastrophic_rate(
                selected[domain_mask], target[domain_mask],
            ),
            "minimum_leave_one_selected_scene_out_total": (
                min(leave_one) if leave_one else None
            ),
            "folds": fold_values,
        }
    return utility, {"by_domain": by_domain}


def _provenance_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(b"mf3zn-no-review-causal-traces/1\0")
    for path in sorted(set(paths), key=lambda value: str(value)):
        relative = str(path.relative_to(ROOT))
        for value in (relative, str(path.stat().st_size), sha256_file(path)):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run() -> dict:
    formal = verify_protocol(FORMAL_PROTOCOL, ROOT)
    car = _load_module(CAR_TRAINER, "mf3zn_no_review_car_source")
    car.verify_protocol()
    rows = car._canonical_rows()
    domains = dict(Counter(str(row["dataset"]) for row in rows))
    scenes_set = {str(row["scene_id"]) for row in rows}
    if (
        len(rows) != IDENTIFIABILITY_EXPECTED_ROWS
        or len(scenes_set) != IDENTIFIABILITY_EXPECTED_SCENES
        or domains != IDENTIFIABILITY_EXPECTED_DOMAIN_COUNTS
        or car._identity_hash(rows) != IDENTIFIABILITY_CANONICAL_IDENTITY_SHA256
    ):
        raise ExploratoryPretestError("sealed CAR population drift")

    trace_cache: dict[Path, tuple[dict, ...]] = {}
    current_rows = []
    temporal_rows = []
    trace_paths = []
    prefix_lengths = []
    future_mutation_rows = 0
    event_ids = []
    for row in rows:
        trace_path = _trace_path(row)
        trace_paths.append(trace_path)
        records = trace_cache.setdefault(trace_path, _load_trace(trace_path))
        decision_step = int(row["decision"]["step"])
        current_matches = [
            value for value in records if int(value["step"]) == decision_step
        ]
        if len(current_matches) != 1:
            raise ExploratoryPretestError("target step is not unique in causal trace")
        _validate_current_trace(current_matches[0], row["decision"])
        if not _future_mutation_check(records, decision_step):
            raise ExploratoryPretestError("future mutation changed causal tensor")
        future_mutation_rows += int(any(
            int(value["step"]) > decision_step for value in records
        ))
        prefix_lengths.append(sum(
            int(value["step"]) <= decision_step for value in records
        ))
        current_rows.append(_current_features(row["decision"], row["arrays"]))
        temporal_rows.append(_temporal_summary(records, decision_step))
        event_ids.append(canonical_audit_event_id(
            row["dataset"], row["scene_id"], row["episode_id"], decision_step,
        ))

    if len(set(event_ids)) != len(rows):
        raise ExploratoryPretestError("exploratory event identities are not unique")
    current = np.stack(current_rows)
    temporal = np.stack(temporal_rows)
    augmented = np.concatenate((current, temporal), axis=1)
    # Outcome is intentionally read only after all inference tensors exist.
    target = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    scenes = np.asarray([str(row["scene_id"]) for row in rows])
    datasets = np.asarray([str(row["dataset"]) for row in rows])
    identities = np.asarray(event_ids)
    if not np.isfinite(target).all():
        raise ExploratoryPretestError("exact outcome contains non-finite values")
    folds, scene_fold_mapping = assign_tuad_scene_folds(scenes)

    current_prediction = _ridge_oof(current, target, folds)
    temporal_prediction = _ridge_oof(augmented, target, folds)
    predictive_improvement = (
        _huber(current_prediction - target)
        - _huber(temporal_prediction - target)
    )
    predictive = {}
    failures: list[str] = []
    for index, domain in enumerate(REQUIRED_DOMAINS):
        interval = _scene_bootstrap_mean(
            predictive_improvement,
            scenes,
            datasets == domain,
            seed=BOOTSTRAP_SEED + index,
        )
        passed = interval["observed"] > 0.0 and interval["lower_95"] > 0.0
        predictive[domain] = {"delta_huber": interval, "pass": passed}
        if not passed:
            failures.append(f"{domain}:temporal_delta_huber_not_positive")

    current_selected = current_prediction > 0.0
    temporal_selected = temporal_prediction > 0.0
    current_utility, current_evidence = _policy_evidence(
        current_selected, target, scenes, datasets, folds,
    )
    temporal_utility, temporal_evidence = _policy_evidence(
        temporal_selected, target, scenes, datasets, folds,
    )
    baselines = matched_budget_baselines(
        temporal_selected,
        np.asarray([
            float(row["decision"]["policy_risk_adjusted_score"])
            for row in rows
        ]),
        np.asarray([float(row["decision"]["native_margin"]) for row in rows]),
        folds,
        datasets,
        identities,
        random_seed=BOOTSTRAP_SEED,
    )
    policy_utility = {
        "temporal": temporal_utility,
        "current-only": current_utility,
    }
    evidence = {
        "temporal": temporal_evidence,
        "current-only": current_evidence,
    }
    for name, mask in baselines.items():
        utility, value = _policy_evidence(mask, target, scenes, datasets, folds)
        policy_utility[name] = utility
        evidence[name] = value

    temporal_contribution = scene_cluster_difference(
        temporal_utility,
        current_utility,
        scenes,
        datasets,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED + 10,
    )
    strongest_by_domain = {}
    for domain in REQUIRED_DOMAINS:
        item = temporal_evidence["by_domain"][domain]
        if not item["utility"] > 0.0:
            failures.append(f"{domain}:utility_nonpositive")
        if (
            item["minimum_leave_one_selected_scene_out_total"] is None
            or not item["minimum_leave_one_selected_scene_out_total"] > 0.0
        ):
            failures.append(f"{domain}:leave_one_scene_nonpositive")
        for fold, fold_item in item["folds"].items():
            if fold_item["utility"] < 0.0:
                failures.append(f"{domain}:fold_{fold}:utility_negative")
        simple = (
            "matched-high-proposal-score", "matched-low-native-margin",
        )
        strongest = max(
            simple,
            key=lambda name: evidence[name]["by_domain"][domain]["utility"],
        )
        strongest_by_domain[domain] = strongest
        comparator = evidence[strongest]["by_domain"][domain]
        if not item["utility"] > comparator["utility"]:
            failures.append(f"{domain}:not_above_strongest_simple_baseline")
        if item["catastrophic_rate"] is None or comparator["catastrophic_rate"] is None:
            failures.append(f"{domain}:undefined_catastrophic_rate")
        elif item["catastrophic_rate"] > comparator["catastrophic_rate"]:
            failures.append(f"{domain}:catastrophic_rate_above_{strongest}")
        contribution = temporal_contribution[domain]
        if contribution["observed"] <= 0.0:
            failures.append(f"{domain}:temporal_utility_delta_nonpositive")
        if contribution["lower_95"] <= 0.0:
            failures.append(f"{domain}:temporal_utility_bootstrap_lower_nonpositive")

    status = (
        "EXPLORATORY_NO_REVIEW_PRETEST_PASS"
        if not failures else "EXPLORATORY_NO_REVIEW_PRETEST_FAIL"
    )
    result = {
        "schema_version": "revealnav-mf3zn-no-review-temporal-pretest/1",
        "method_id": "mf3zn_tuad_v1",
        "status": status,
        "scope": (
            "exploratory stop-on-first-failed-stage bridge; not the sealed "
            "MF3ZN identifiability audit or TUAD development result"
        ),
        "protocol": {
            "formal_protocol_sha256": sha256_file(FORMAL_PROTOCOL),
            "formal_protocol_status": formal["status"],
            "formal_protocol_unchanged": True,
            "post_seal_exploratory_override": True,
        },
        "population": {
            "rows": len(rows),
            "scenes": len(scenes_set),
            "domains": domains,
            "canonical_identity_sha256": car._identity_hash(rows),
            "event_identity_count": len(set(event_ids)),
            "source_counts": dict(Counter(str(row["source"]) for row in rows)),
        },
        "causal_trace": {
            "unique_files": len(set(trace_paths)),
            "aggregate_inventory_sha256": _provenance_digest(trace_paths),
            "prefix_rows": int(sum(prefix_lengths)),
            "minimum_prefix_length": int(min(prefix_lengths)),
            "maximum_prefix_length": int(max(prefix_lengths)),
            "mean_prefix_length": float(np.mean(prefix_lengths)),
            "future_mutation_invariance": True,
            "events_with_mutated_future_suffix": future_mutation_rows,
            "treatment_outcome_isolation": (
                "target is read only after current and temporal tensors are built"
            ),
            "fields_consumed": {
                "scalar": list(TRACE_SCALAR_NAMES),
                "structural": [
                    "step", "current_local_action_ids", "native_action_id",
                    "runner_local_index",
                ],
            },
        },
        "fixed_probe": {
            "model": "pooled ridge with fold-fit standardization",
            "ridge_l2": RIDGE_L2,
            "folds": TUAD_OUTER_FOLDS,
            "fold_unit": "raw MP3D scene",
            "scene_fold_mapping": scene_fold_mapping,
            "current_feature_names": list(CURRENT_FEATURE_NAMES),
            "temporal_feature_names": list(TEMPORAL_FEATURE_NAMES),
            "current_width": int(current.shape[1]),
            "temporal_added_width": int(temporal.shape[1]),
            "decision_rule": "OOF predicted runner-native delta utility > 0",
            "model_search": False,
            "threshold_search": False,
            "seed_selection": False,
        },
        "predictive_temporal_relevance": predictive,
        "scientific_effect": {
            "policies": evidence,
            "strongest_simple_baseline_by_domain": strongest_by_domain,
            "temporal_minus_current_scene_bootstrap": temporal_contribution,
            "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
        },
        "failures": failures,
        "stop_rule_triggered": bool(failures),
        "next_exploratory_stage_authorized": not failures,
        "manual_review_performed": False,
        "oracle_uad_audit_performed": False,
        "formal_identifiability_pass": False,
        "formal_collection_authorized": False,
        "scientific_claim_authorized": False,
        "checkpoint_written": False,
        "public_split_access": {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
    }
    return _jsonable(result)


def main() -> int:
    result = run()
    _atomic_json(RESULT, result)
    print(json.dumps({
        "status": result["status"],
        "failures": result["failures"],
        "result": str(RESULT.relative_to(ROOT)),
        "next_exploratory_stage_authorized": (
            result["next_exploratory_stage_authorized"]
        ),
    }, indent=2, sort_keys=True))
    return 0 if not result["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
