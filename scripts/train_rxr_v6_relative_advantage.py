#!/usr/bin/env python3
"""Five-fold scene-block training and calibration for RxR V6."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2r6 import (  # noqa: E402
    ReversibleAdvantageHead,
    ReversibleAdvantageHeadV631,
    ReversibleAdvantageLoss,
    ReversibleAdvantageLossV631,
    outer_scene_partition,
)


SEEDS = (20260826, 20260827, 20260828)
V631_SELECTION = ROOT / (
    "artifacts/phase1/rxr_v6/full_v6_2/RXR_V6_EPISODE_SELECTION.json"
)
V631_CORRECTION = ROOT / (
    "artifacts/design/MF2_V6_3_1_1_ZERO_CANDIDATE_CORRECTION.md"
)
INPUT_KEYS = (
    "instruction", "post_observation", "temporal_history", "checkpoint",
    "native", "alternative", "scalars",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def to_device(
    arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device,
    outer_fold: int | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    inputs = []
    for key in INPUT_KEYS:
        values = arrays[key][indices]
        if key == "scalars" and values.ndim == 3:
            if outer_fold is None:
                raise RuntimeError("fold-specific V6.3.1 scalars need outer fold")
            values = values[:, outer_fold, :]
        inputs.append(torch.from_numpy(values.astype(np.float32)).to(device))
    target = torch.from_numpy(
        arrays["target"][indices].astype(np.float32)
    ).to(device)
    return inputs, target


def predict(
    model: ReversibleAdvantageHead, arrays: dict[str, np.ndarray],
    indices: np.ndarray, device: torch.device, outer_fold: int | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    keys = ["lower", "median", "upper", "sign_score"]
    if isinstance(model, ReversibleAdvantageHeadV631):
        keys.append("native_failure_score")
    values = {key: [] for key in keys}
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            local = indices[start:start + 256]
            inputs, _ = to_device(arrays, local, device, outer_fold)
            output = model(*inputs)
            values["lower"].append(output.lower.cpu().numpy())
            values["median"].append(output.median.cpu().numpy())
            values["upper"].append(output.upper.cpu().numpy())
            values["sign_score"].append(
                torch.sigmoid(output.positive_logit).cpu().numpy()
            )
            if isinstance(model, ReversibleAdvantageHeadV631):
                values["native_failure_score"].append(
                    torch.sigmoid(output.native_failure_logit).cpu().numpy()
                )
    return {key: np.concatenate(rows) for key, rows in values.items()}


def train_model(
    arrays: dict[str, np.ndarray], indices: np.ndarray, seed: int,
    device: torch.device, method_revision: str, outer_fold: int,
) -> tuple[ReversibleAdvantageHead, dict]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    scalar_dim = int(arrays["scalars"].shape[-1])
    positives = int(np.sum(arrays["target"][indices] > 0.0))
    negatives = len(indices) - positives
    if not positives or not negatives:
        raise RuntimeError("V6 fit partition lacks both advantage signs")
    if method_revision == "v6_3_1":
        failures = int(np.sum(arrays["native_failure"][indices] > 0.5))
        successes = len(indices) - failures
        if not failures or not successes:
            raise RuntimeError("V6.3.1 fit partition lacks both native outcomes")
        model = ReversibleAdvantageHeadV631(
            projection_dim=32, scalar_dim=scalar_dim
        ).to(device)
        objective = ReversibleAdvantageLossV631(
            positive_weight=negatives / positives,
            native_failure_positive_weight=successes / failures,
        )
    else:
        failures = None
        successes = None
        model = ReversibleAdvantageHead(
            projection_dim=32, scalar_dim=scalar_dim
        ).to(device)
        balanced_sign = method_revision == "v6_3"
        objective = ReversibleAdvantageLoss(
            sign_weight=1.0 if balanced_sign else 0.25,
            positive_weight=(
                negatives / positives if balanced_sign else None
            ),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3
    )
    generator = torch.Generator().manual_seed(seed)
    for _ in range(80):
        model.train()
        order = torch.randperm(len(indices), generator=generator).numpy()
        for start in range(0, len(order), 64):
            local = indices[order[start:start + 64]]
            inputs, target = to_device(
                arrays, local, device,
                outer_fold if method_revision == "v6_3_1" else None,
            )
            output = model(*inputs)
            if method_revision == "v6_3_1":
                native_failure = torch.from_numpy(
                    arrays["native_failure"][local].astype(np.float32)
                ).to(device)
                loss = objective(output, target, native_failure)["total"]
            else:
                loss = objective(output, target)["total"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite V6 training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model, {
        "advantage_positive_rows": positives,
        "advantage_negative_or_tied_rows": negatives,
        "advantage_positive_weight": negatives / positives,
        "native_failure_rows": failures,
        "native_success_rows": successes,
        "native_failure_positive_weight": (
            None if failures is None else successes / failures
        ),
        "auxiliary_scores_used_for_authorization": False,
    }


def conformal_offset(lower: np.ndarray, target: np.ndarray) -> float:
    if len(target) < 1:
        raise RuntimeError("empty V6 conformal partition")
    nonconformity = np.sort(lower - target)
    rank = min(len(nonconformity) - 1, math.ceil(
        0.9 * (len(nonconformity) + 1)
    ) - 1)
    return max(0.0, float(nonconformity[rank]))


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float,
) -> float:
    if not len(values) or values.shape != weights.shape:
        raise RuntimeError("invalid V6.3.1 weighted quantile inputs")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(sorted_weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("invalid V6.3.1 calibration weights")
    cumulative = np.cumsum(sorted_weights) / total
    index = min(
        len(sorted_values) - 1,
        int(np.searchsorted(cumulative, quantile, side="left")),
    )
    return float(sorted_values[index])


def scene_equal_empirical_offset(
    lower: np.ndarray, target: np.ndarray, indices: np.ndarray,
    records: list[dict],
) -> float:
    counts = {}
    for index in indices:
        scene = str(records[int(index)]["scene_id"])
        counts[scene] = counts.get(scene, 0) + 1
    weights = np.asarray([
        1.0 / (len(counts) * counts[str(records[int(index)]["scene_id"])])
        for index in indices
    ], dtype=np.float64)
    return max(0.0, weighted_quantile(
        lower.astype(np.float64) - target.astype(np.float64), weights, 0.9
    ))


def scene_macro_mean(
    values: np.ndarray, indices: np.ndarray, records: list[dict],
) -> float:
    grouped: dict[str, list[float]] = {}
    for value, index in zip(values, indices):
        grouped.setdefault(
            str(records[int(index)]["scene_id"]), []
        ).append(float(value))
    return float(np.mean([np.mean(rows) for rows in grouped.values()]))


def partition_indices(
    records: list[dict], heldout: int, method_revision: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if method_revision == "v6_3_1":
        roles = outer_scene_partition(
            {str(row["scene_id"]) for row in records}, heldout
        )
        partitions = {role: [] for role in ("fit", "calibration", "evaluation")}
        for row in records:
            role = roles[str(row["scene_id"])]
            if (role == "evaluation") != (int(row["scene_fold"]) == heldout):
                raise RuntimeError("V6.3.1 outer scene-fold drift")
            partitions[role].append(int(row["row_index"]))
        evidence = {
            f"{role}_scene_count": len({
                str(records[index]["scene_id"])
                for index in partitions[role]
            })
            for role in partitions
        }
        evidence.update({
            f"{role}_scene_ids_sha256": stable_hash(sorted({
                str(records[index]["scene_id"])
                for index in partitions[role]
            }))
            for role in partitions
        })
        return (
            np.asarray(partitions["fit"], dtype=np.int64),
            np.asarray(partitions["calibration"], dtype=np.int64),
            np.asarray(partitions["evaluation"], dtype=np.int64),
            evidence,
        )
    evaluation = []
    fit = []
    calibration = []
    for row in records:
        index = int(row["row_index"])
        if int(row["scene_fold"]) == heldout:
            evaluation.append(index)
        elif int(stable_hash({
            "v6_conformal": row["scene_id"], "heldout": heldout,
        }), 16) % 5 == 0:
            calibration.append(index)
        else:
            fit.append(index)
    if min(len(fit), len(calibration), len(evaluation)) < 1:
        raise RuntimeError(f"empty V6 cross-fit partition for fold {heldout}")
    arrays = tuple(np.asarray(values, dtype=np.int64) for values in (
        fit, calibration, evaluation,
    ))
    return (*arrays, {})


def earliest_authorized_policy(
    lower: np.ndarray, target: np.ndarray, indices: np.ndarray,
    records: list[dict], cohort_episodes: list[dict], heldout_fold: int,
) -> dict:
    episode_rows = [
        row for row in cohort_episodes
        if int(row["scene_fold"]) == heldout_fold
    ]
    by_episode: dict[str, list[int]] = {
        str(row["episode_id"]): [] for row in episode_rows
    }
    episode_scenes = {
        str(row["episode_id"]): str(row["scene_id"]) for row in episode_rows
    }
    if len(by_episode) != len(episode_rows):
        raise RuntimeError("duplicate V6.3.1 sealed evaluation episode")
    local_by_index = {int(index): local for local, index in enumerate(indices)}
    for index in indices:
        row = records[int(index)]
        episode_id = str(row["episode_id"])
        if (
            episode_id not in by_episode
            or episode_scenes[episode_id] != str(row["scene_id"])
        ):
            raise RuntimeError("V6.3.1 record is outside sealed evaluation episodes")
        by_episode[episode_id].append(int(index))
    scene_episode_benefits: dict[str, list[float]] = {}
    selected_rows = []
    zero_candidate_episodes = 0
    for episode_id, candidate_rows in by_episode.items():
        ordered = sorted(candidate_rows, key=lambda index: (
            int(records[index]["post_navigation_step"]),
            str(records[index]["event_id"]),
        ))
        authorized = [
            index for index in ordered
            if float(lower[local_by_index[index]]) > 0.0
        ]
        selected = authorized[0] if authorized else None
        benefit = (
            float(target[local_by_index[selected]])
            if selected is not None else 0.0
        )
        scene = episode_scenes[episode_id]
        scene_episode_benefits.setdefault(scene, []).append(benefit)
        if not ordered:
            zero_candidate_episodes += 1
        if selected is not None:
            selected_rows.append({
                "event_id": str(records[selected]["event_id"]),
                "episode_id": str(records[selected]["episode_id"]),
                "scene_id": scene,
                "realized_benefit": benefit,
            })
    scene_benefits = {
        scene: float(np.mean(values))
        for scene, values in scene_episode_benefits.items()
    }
    return {
        "episodes": len(by_episode),
        "zero_candidate_episodes": zero_candidate_episodes,
        "selected_episodes": len(selected_rows),
        "selected_scenes": len({row["scene_id"] for row in selected_rows}),
        "selected_positive_precision": (
            float(np.mean([row["realized_benefit"] > 0 for row in selected_rows]))
            if selected_rows else 0.0
        ),
        "selected_realized_benefit_mean": (
            float(np.mean([row["realized_benefit"] for row in selected_rows]))
            if selected_rows else 0.0
        ),
        "selected_realized_benefit_sum": float(sum(
            row["realized_benefit"] for row in selected_rows
        )),
        "scene_macro_policy_benefit": float(np.mean(list(scene_benefits.values()))),
        "scene_benefits": scene_benefits,
        "selected_event_ids": [row["event_id"] for row in selected_rows],
    }


def scene_cluster_bootstrap(
    folds: list[dict], resamples: int = 10_000, seed: int = 20260831,
) -> dict:
    scene_values = {
        scene: float(value)
        for fold in folds
        for scene, value in fold["scene_policy_benefits"].items()
    }
    if len(scene_values) != sum(
        fold["partition_evidence"]["evaluation_scene_count"]
        for fold in folds
    ):
        raise RuntimeError("V6.3.1 evaluation scene appeared in multiple folds")
    values = np.asarray(list(scene_values.values()), dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = float(np.mean(generator.choice(
            values, size=len(values), replace=True
        )))
    return {
        "unit": "scene",
        "scenes": len(values),
        "resamples": resamples,
        "seed": seed,
        "point_estimate": float(values.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def fold_run(
    fold: int, arrays: dict[str, np.ndarray], records: list[dict],
    output_dir: Path, device: torch.device, method_revision: str,
    cohort_episodes: list[dict],
) -> dict:
    fit, calibration, evaluation, partition_evidence = partition_indices(
        records, fold, method_revision
    )
    members = []
    states = []
    training_evidence = []
    for seed in SEEDS:
        model, evidence = train_model(
            arrays, fit, seed + fold * 100, device,
            method_revision, fold,
        )
        members.append(model)
        training_evidence.append(evidence)
        states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
    model_fold = fold if method_revision == "v6_3_1" else None
    cal_predictions = [
        predict(model, arrays, calibration, device, model_fold)
        for model in members
    ]
    eval_predictions = [
        predict(model, arrays, evaluation, device, model_fold)
        for model in members
    ]
    cal_lower = np.median(np.stack([
        value["lower"] for value in cal_predictions
    ]), axis=0)
    offset = (
        scene_equal_empirical_offset(
            cal_lower, arrays["target"][calibration], calibration, records
        )
        if method_revision == "v6_3_1"
        else conformal_offset(cal_lower, arrays["target"][calibration])
    )
    prediction = {
        key: np.median(np.stack([value[key] for value in eval_predictions]), axis=0)
        for key in eval_predictions[0]
    }
    lower = prediction["lower"] - offset
    target = arrays["target"][evaluation].astype(np.float64)
    selected = lower > 0.0
    coverage = (
        scene_macro_mean(target >= lower, evaluation, records)
        if method_revision == "v6_3_1"
        else float(np.mean(target >= lower))
    )
    median_mae = (
        scene_macro_mean(
            np.abs(target - prediction["median"]), evaluation, records
        )
        if method_revision == "v6_3_1"
        else float(np.mean(np.abs(target - prediction["median"])))
    )
    zero_mae = (
        scene_macro_mean(np.abs(target), evaluation, records)
        if method_revision == "v6_3_1"
        else float(np.mean(np.abs(target)))
    )
    selected_sum = float(target[selected].sum()) if selected.any() else 0.0
    selected_mean = float(target[selected].mean()) if selected.any() else 0.0
    checkpoint = output_dir / f"fold_{fold}" / "crossfit_ensemble.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "revealnav-rxr-v6-crossfit-ensemble/1",
        "fold": fold,
        "member_base_seeds": list(SEEDS),
        "member_effective_seeds": [seed + fold * 100 for seed in SEEDS],
        "model_state_dicts": states,
        "projection_dim": 32,
        "scalar_dim": int(arrays["scalars"].shape[-1]),
        "method_revision": method_revision,
        "quantiles": [0.1, 0.5, 0.9],
        "conformal_lower_offset": offset,
        "calibration_kind": (
            "scene-blocked scene-equal-weighted empirical"
            if method_revision == "v6_3_1" else "event-level conformal"
        ),
        "training_evidence": training_evidence,
        "partition_evidence": partition_evidence,
    }
    part = checkpoint.with_name(checkpoint.name + ".part")
    torch.save(payload, part); os.replace(part, checkpoint)
    result = {
        "fold": fold,
        "fit_rows": len(fit), "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
        "conformal_lower_offset": offset,
        "lower_coverage": coverage,
        "median_mae": median_mae,
        "zero_predictor_mae": zero_mae,
        "median_beats_zero": median_mae < zero_mae,
        "selected": int(selected.sum()),
        "selected_realized_benefit_sum": selected_sum,
        "selected_realized_benefit_mean": selected_mean,
        "selected_positive_precision": (
            float(np.mean(target[selected] > 0.0)) if selected.any() else 0.0
        ),
        "beneficial_candidate_recall": (
            float(np.sum(selected & (target > 0.0)) / np.sum(target > 0.0))
            if np.any(target > 0.0) else 0.0
        ),
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "partition_evidence": partition_evidence,
        "auxiliary_scores_used_for_authorization": False,
    }
    if method_revision == "v6_3_1":
        policy = earliest_authorized_policy(
            lower, target, evaluation, records, cohort_episodes, fold,
        )
        result.update({
            "selected": policy["selected_episodes"],
            "selected_scenes": policy["selected_scenes"],
            "selected_positive_precision": policy[
                "selected_positive_precision"
            ],
            "selected_realized_benefit_sum": policy[
                "selected_realized_benefit_sum"
            ],
            "selected_realized_benefit_mean": policy[
                "selected_realized_benefit_mean"
            ],
            "scene_macro_policy_benefit": policy[
                "scene_macro_policy_benefit"
            ],
            "scene_policy_benefits": policy["scene_benefits"],
            "selected_event_ids": policy["selected_event_ids"],
            "policy_accounting": "earliest authorized event per episode; otherwise zero",
            "evaluation_episodes": policy["episodes"],
            "zero_candidate_episodes": policy["zero_candidate_episodes"],
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--method-revision", choices=("v6", "v6_3", "v6_3_1"),
        default="v6"
    )
    args = parser.parse_args()
    manifest_raw = (
        args.manifest if args.manifest.is_absolute()
        else Path.cwd() / args.manifest
    )
    output_raw = (
        args.output_dir if args.output_dir.is_absolute()
        else Path.cwd() / args.output_dir
    )
    if manifest_raw.is_symlink() or output_raw.is_symlink():
        raise SystemExit("V6 manifest/output symlinks are forbidden")
    manifest_path = manifest_raw.resolve()
    output_dir = output_raw.resolve()
    if (
        ROOT not in manifest_path.parents
        or ROOT not in output_dir.parents
        or not manifest_path.is_file()
        or output_raw.exists()
    ):
        raise SystemExit("V6 paths must remain inside the project")
    manifest = json.loads(manifest_path.read_text())
    expected_status = (
        "RXR_V6_3_1_PAIRED_DATASET_READY"
        if args.method_revision == "v6_3_1"
        else "RXR_V6_PAIRED_DATASET_READY"
    )
    if manifest.get("status") != expected_status:
        raise RuntimeError("V6 paired dataset is not ready")
    array_relative = Path(manifest["arrays"]["path"])
    if array_relative.is_absolute() or ".." in array_relative.parts:
        raise RuntimeError("unsafe V6 paired array path")
    array_raw = ROOT / array_relative
    array_path = array_raw.resolve()
    if (
        ROOT not in array_path.parents
        or array_raw.is_symlink()
        or not array_path.is_file()
        or array_path.stat().st_size != manifest["arrays"]["bytes"]
        or sha256_file(array_path) != manifest["arrays"]["sha256"]
    ):
        raise RuntimeError("V6 paired array provenance drift")
    with np.load(array_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    records = manifest["records"]
    cohort_episodes = []
    if len(records) != len(arrays["target"]):
        raise RuntimeError("V6 record/array alignment drift")
    if any(int(row["row_index"]) != index for index, row in enumerate(records)):
        raise RuntimeError("V6 row order drift")
    if args.method_revision == "v6_3_1":
        protocol_relative = Path(manifest["protocol"]["path"])
        if protocol_relative.is_absolute() or ".." in protocol_relative.parts:
            raise RuntimeError("unsafe V6.3.1 protocol path")
        protocol_raw = ROOT / protocol_relative
        protocol_path = protocol_raw.resolve()
        if not (
            ROOT in protocol_path.parents
            and not protocol_raw.is_symlink()
            and protocol_path.is_file()
            and sha256_file(protocol_path) == manifest["protocol"]["sha256"]
        ):
            raise RuntimeError("V6.3.1 feature protocol provenance drift")
        feature_protocol = json.loads(protocol_path.read_text())
        selection_evidence = feature_protocol.get("sources", {}).get(
            str(V631_SELECTION.relative_to(ROOT)), {}
        )
        if not (
            not V631_SELECTION.is_symlink()
            and V631_SELECTION.is_file()
            and V631_SELECTION.stat().st_size == selection_evidence.get("bytes")
            and sha256_file(V631_SELECTION) == selection_evidence.get("sha256")
        ):
            raise RuntimeError("V6.3.1 sealed episode selection drift")
        selection = json.loads(V631_SELECTION.read_text())
        cohort_episodes = selection.get("episodes", [])
        if not (
            selection.get("episode_count") == 120 == len(cohort_episodes)
            and selection.get("split") == "train"
            and selection.get("unseen_or_test_read") is False
            and len({str(row["episode_id"]) for row in cohort_episodes}) == 120
        ):
            raise RuntimeError("V6.3.1 sealed episode population drift")
        if not (
            arrays["scalars"].shape == (len(records), 5, 20)
            and arrays["native_failure"].shape == (len(records),)
            and np.isfinite(arrays["scalars"]).all()
            and set(np.unique(arrays["native_failure"])) <= {0.0, 1.0}
        ):
            raise RuntimeError("V6.3.1 scalar or auxiliary target drift")
        expected_failure = np.asarray([
            float(row["native_metrics"]["success"] <= 0.0)
            for row in records
        ], dtype=np.float32)
        if not np.array_equal(arrays["native_failure"], expected_failure):
            raise RuntimeError("V6.3.1 native-failure target alignment drift")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    folds = [
        fold_run(
            fold, arrays, records, output_dir, device,
            args.method_revision, cohort_episodes,
        )
        for fold in range(5)
    ]
    bootstrap = (
        scene_cluster_bootstrap(folds)
        if args.method_revision == "v6_3_1" else None
    )
    if args.method_revision == "v6_3_1":
        aggregate_median = sum(
            row["median_mae"]
            * row["partition_evidence"]["evaluation_scene_count"]
            for row in folds
        )
        aggregate_zero = sum(
            row["zero_predictor_mae"]
            * row["partition_evidence"]["evaluation_scene_count"]
            for row in folds
        )
        gates = {
            "at_least_100_same_state_pairs": len(records) >= 100,
            "at_least_20_positive_pairs": int((arrays["target"] > 0).sum()) >= 20,
            "all_folds_scene_macro_lower_coverage_at_least_0_85": all(
                row["lower_coverage"] >= 0.85 for row in folds
            ),
            "aggregate_scene_macro_median_beats_zero_predictor": (
                aggregate_median < aggregate_zero
            ),
            "at_least_ten_selected_episodes": sum(
                row["selected"] for row in folds
            ) >= 10,
            "selected_episodes_cover_at_least_five_scenes": sum(
                row["selected_scenes"] for row in folds
            ) >= 5,
            "positive_scene_macro_policy_benefit_in_all_folds": all(
                row["scene_macro_policy_benefit"] > 0.0 for row in folds
            ),
            "scene_cluster_bootstrap_95_lcb_above_zero": (
                bootstrap["lower_95"] > 0.0
            ),
        }
    else:
        gates = {
        "at_least_100_same_state_pairs": len(records) >= 100,
        "at_least_20_positive_pairs": int((arrays["target"] > 0).sum()) >= 20,
        "all_folds_have_nominal_lower_coverage": all(
            row["lower_coverage"] >= 0.85 for row in folds
        ),
        "aggregate_median_beats_zero_predictor": (
            sum(row["median_mae"] * row["evaluation_rows"] for row in folds)
            < sum(row["zero_predictor_mae"] * row["evaluation_rows"] for row in folds)
        ),
        "positive_selected_benefit_in_four_of_five_folds": sum(
            row["selected_realized_benefit_sum"] > 0.0 for row in folds
        ) >= 4,
        "at_least_ten_crossfit_interventions": sum(
            row["selected"] for row in folds
        ) >= 10,
        }
    value = {
        "schema_version": "revealnav-rxr-v6-relative-advantage-crossfit/1",
        "method_revision": args.method_revision,
        "status": (
            "RXR_V6_OFFLINE_GATE_PASS" if all(gates.values())
            else "RXR_V6_OFFLINE_GATE_FAIL"
        ),
        "dataset_manifest": str(manifest_path.relative_to(ROOT)),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "pairs": len(records),
        "positive_pairs": int((arrays["target"] > 0).sum()),
        "folds": folds,
        "gates": gates,
        "scene_cluster_bootstrap": bootstrap,
        "online_gate": (
            "calibrated_lower_quantile > 0 and return executable and option live"
        ),
        "val_seen_authorized": all(gates.values()),
        "unseen_or_test_read": False,
        "paper_result": False,
        "development_only_due_to_prior_method_inspection": (
            args.method_revision == "v6_3_1"
        ),
        "correctness_revision": (
            {
                "name": "V6.3.1.1 zero-candidate episode accounting",
                "path": str(V631_CORRECTION.relative_to(ROOT)),
                "sha256": sha256_file(V631_CORRECTION),
            }
            if args.method_revision == "v6_3_1" else None
        ),
        "sources": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                Path(__file__).resolve(),
                ROOT / "revealnav_mf2r6/model.py",
                ROOT / "revealnav_mf2r6/protocol.py",
                *(
                    (V631_SELECTION, V631_CORRECTION)
                    if args.method_revision == "v6_3_1" else ()
                ),
            )
        },
    }
    atomic_json(output_dir / "RXR_V6_OFFLINE_CROSSFIT_RESULT.json", value)
    print(json.dumps({
        "status": value["status"], "pairs": value["pairs"],
        "positive_pairs": value["positive_pairs"], "gates": gates,
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
