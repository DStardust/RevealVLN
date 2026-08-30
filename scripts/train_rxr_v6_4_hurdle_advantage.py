#!/usr/bin/env python3
"""Locked five-fold development test for V6.4 hurdle advantage."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r6 import (  # noqa: E402
    FailureConditionedHurdleAdvantage,
    FailureConditionedHurdleLoss,
    V64_TRAINING_CONTRACT,
    validate_hurdle_checkpoint_payload,
)
from train_rxr_v6_relative_advantage import (  # noqa: E402
    atomic_json, earliest_authorized_policy, partition_indices,
    scene_cluster_bootstrap, scene_equal_empirical_offset, scene_macro_mean,
    sha256_file,
)


SEEDS = (20260826, 20260827, 20260828)
SELECTION = ROOT / (
    "artifacts/phase1/rxr_v6/full_v6_2/RXR_V6_EPISODE_SELECTION.json"
)
CANONICAL_MANIFEST = ROOT / (
    "artifacts/phase1/rxr_v6/full_v6_3_1/"
    "RXR_V6_3_1_PAIRED_DATASET_MANIFEST.json"
)
FEATURE_PROTOCOL = ROOT / (
    "artifacts/phase1/rxr_v6/full_v6_3_1/"
    "RXR_V6_3_1_FEATURE_PROTOCOL.json"
)
DESIGN = ROOT / (
    "artifacts/design/MF2_FAILURE_CONDITIONED_HURDLE_ADVANTAGE_V6_4.md"
)
CORRECTION = ROOT / (
    "artifacts/design/MF2_V6_4_0_1_PROTOCOL_CORRECTION.md"
)
MODEL_SOURCE = ROOT / "revealnav_mf2r6/hurdle.py"
PARTITION_SOURCE = ROOT / "revealnav_mf2r6/protocol.py"
HELPER_SOURCE = ROOT / "scripts/train_rxr_v6_relative_advantage.py"
LOCKED_INPUTS = {
    str(CANONICAL_MANIFEST.relative_to(ROOT)): {
        "bytes": 2_232_892,
        "sha256": "49c65d0daa9477e84e47581ad7dee49d8112ffa250dc59c6164952c783474185",
    },
    str(FEATURE_PROTOCOL.relative_to(ROOT)): {
        "bytes": 6_151,
        "sha256": "6b7ea62927d939b119d8742db09e37a2748a55843208618af0af06e7a2d24872",
    },
    str(SELECTION.relative_to(ROOT)): {
        "bytes": 21_882,
        "sha256": "5c8d9683ed72ac1da1719f339409226623214f7e42c566387a4b0ce9449f13b6",
    },
    "artifacts/phase1/rxr_v6/full_v6_3_1/RXR_V6_3_1_PAIRED_DATASET.npz": {
        "bytes": 2_878_603,
        "sha256": "806e5b7910097fd3cd53372e61c3ed96298a5af2860cfc8980cc0c2aebe4d2d5",
    },
}
INPUT_KEYS = (
    "instruction", "post_observation", "temporal_history", "checkpoint",
    "native", "alternative",
)


def safe_inside(path: Path, must_exist: bool) -> Path:
    raw = path if path.is_absolute() else Path.cwd() / path
    if raw.is_symlink():
        raise RuntimeError("V6.4 path symlinks are forbidden")
    resolved = raw.resolve()
    if ROOT not in resolved.parents:
        raise RuntimeError("V6.4 path escaped project")
    if must_exist and not resolved.is_file():
        raise RuntimeError("V6.4 input is not a file")
    return resolved


def verify_locked_file(path: Path) -> None:
    relative = str(path.relative_to(ROOT))
    expected = LOCKED_INPUTS.get(relative)
    if not (
        expected is not None and not path.is_symlink() and path.is_file()
        and path.stat().st_size == expected["bytes"]
        and sha256_file(path) == expected["sha256"]
    ):
        raise RuntimeError(f"V6.4 locked input drift: {relative}")


def load_inputs(manifest_path: Path) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    if manifest_path != CANONICAL_MANIFEST.resolve():
        raise RuntimeError("V6.4 requires the canonical sealed manifest")
    for relative in LOCKED_INPUTS:
        verify_locked_file(ROOT / relative)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "RXR_V6_3_1_PAIRED_DATASET_READY":
        raise RuntimeError("V6.4 requires the audited V6.3.1 paired dataset")
    relative = Path(manifest["arrays"]["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("unsafe V6.4 array path")
    raw = ROOT / relative
    path = raw.resolve()
    if not (
        ROOT in path.parents and not raw.is_symlink() and path.is_file()
        and relative.as_posix() in LOCKED_INPUTS
        and path.stat().st_size == manifest["arrays"]["bytes"]
        and sha256_file(path) == manifest["arrays"]["sha256"]
    ):
        raise RuntimeError("V6.4 paired-array provenance drift")
    protocol = manifest.get("protocol", {})
    if not (
        protocol.get("path") == str(FEATURE_PROTOCOL.relative_to(ROOT))
        and protocol.get("sha256")
        == LOCKED_INPUTS[str(FEATURE_PROTOCOL.relative_to(ROOT))]["sha256"]
    ):
        raise RuntimeError("V6.4 feature-protocol link drift")
    feature_protocol = json.loads(FEATURE_PROTOCOL.read_text())
    selection_evidence = feature_protocol.get("sources", {}).get(
        str(SELECTION.relative_to(ROOT)), {}
    )
    if selection_evidence != LOCKED_INPUTS[str(SELECTION.relative_to(ROOT))]:
        raise RuntimeError("V6.4 selection provenance-chain drift")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    records = manifest["records"]
    expected_shapes = {
        **{key: (339, 768) for key in INPUT_KEYS},
        "scalars": (339, 5, 20),
        "target": (339,),
        "native_failure": (339,),
    }
    if not (
        len(records) == 339 == len(arrays["target"])
        and set(arrays) == set(expected_shapes)
        and all(arrays[key].shape == shape
                for key, shape in expected_shapes.items())
        and all(np.issubdtype(arrays[key].dtype, np.number)
                and np.isfinite(arrays[key]).all() for key in arrays)
        and np.all(arrays["scalars"][..., 2] >= 0.0)
        and np.array_equal(
            arrays["scalars"][..., 2],
            np.repeat(arrays["scalars"][:, :1, 2], 5, axis=1),
        )
        and all(int(row["row_index"]) == index
                for index, row in enumerate(records))
    ):
        raise RuntimeError("V6.4 paired dataset shape/order drift")
    expected_failure = np.asarray([
        float(row["native_metrics"]["success"] <= 0.0) for row in records
    ], dtype=np.float32)
    if not np.array_equal(expected_failure, arrays["native_failure"]):
        raise RuntimeError("V6.4 native-failure target drift")
    return manifest, arrays, records


def locked_provenance(manifest_path: Path) -> dict[str, dict]:
    paths = (
        manifest_path,
        ROOT / "artifacts/phase1/rxr_v6/full_v6_3_1/"
        "RXR_V6_3_1_PAIRED_DATASET.npz",
        FEATURE_PROTOCOL, SELECTION, DESIGN, CORRECTION, MODEL_SOURCE,
        PARTITION_SOURCE, HELPER_SOURCE, Path(__file__).resolve(),
    )
    value = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("V6.4 provenance source is not a regular file")
        value[str(path.relative_to(ROOT))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return value


def tensors(
    arrays: dict[str, np.ndarray], indices: np.ndarray, fold: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    inputs = [
        torch.from_numpy(arrays[key][indices].astype(np.float32)).to(device)
        for key in INPUT_KEYS
    ]
    inputs.append(torch.from_numpy(
        arrays["scalars"][indices, fold].astype(np.float32)
    ).to(device))
    target = torch.from_numpy(
        arrays["target"][indices].astype(np.float32)
    ).to(device)
    failure = torch.from_numpy(
        arrays["native_failure"][indices].astype(np.float32)
    ).to(device)
    return inputs, target, failure


def scene_episode_event_weights(
    indices: np.ndarray, records: list[dict], device: torch.device,
) -> torch.Tensor:
    scenes = {str(records[int(index)]["scene_id"]) for index in indices}
    episode_scenes = {}
    for index in indices:
        row = records[int(index)]
        episode = str(row["episode_id"])
        scene = str(row["scene_id"])
        if episode in episode_scenes and episode_scenes[episode] != scene:
            raise RuntimeError("V6.4 episode crossed scenes")
        episode_scenes[episode] = scene
    scene_episodes = Counter(episode_scenes.values())
    episode_events = Counter(
        str(records[int(index)]["episode_id"]) for index in indices
    )
    weights = np.asarray([
        1.0 / (
            len(scenes)
            * scene_episodes[str(records[int(index)]["scene_id"])]
            * episode_events[str(records[int(index)]["episode_id"])]
        )
        for index in indices
    ], dtype=np.float32)
    weights /= weights.mean()
    return torch.from_numpy(weights).to(device)


def class_ratio(label: torch.Tensor, weight: torch.Tensor) -> float:
    positive = float(weight[label > 0.5].sum().cpu())
    negative = float(weight[label <= 0.5].sum().cpu())
    if positive <= 0.0 or negative <= 0.0:
        raise RuntimeError("V6.4 fit partition lacks both classes")
    return negative / positive


def train_member(
    arrays: dict[str, np.ndarray], records: list[dict], indices: np.ndarray,
    fold: int, seed: int, device: torch.device,
) -> tuple[FailureConditionedHurdleAdvantage, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    inputs, target, failure = tensors(arrays, indices, fold, device)
    weight = scene_episode_event_weights(indices, records, device)
    failure_ratio = class_ratio(failure, weight)
    sign_ratio = class_ratio((target > 0.0).to(target.dtype), weight)
    model = FailureConditionedHurdleAdvantage().to(device)
    objective = FailureConditionedHurdleLoss(failure_ratio, sign_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    model.train()
    final = None
    for _ in range(200):
        losses = objective(model(*inputs), target, failure, weight)
        if not torch.isfinite(losses["total"]):
            raise RuntimeError("non-finite V6.4 training loss")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        final = {key: float(value.detach().cpu()) for key, value in losses.items()}
    return model.eval(), {
        "effective_seed": seed,
        "fit_rows": len(indices),
        "failure_positive_weight": failure_ratio,
        "sign_positive_weight": sign_ratio,
        "final_fit_loss": final,
        "failure_probability_used_as_mixture_weight": True,
        "failure_probability_used_as_independent_gate": False,
        "sign_score_used_as_independent_gate": False,
    }


def predict(
    model: FailureConditionedHurdleAdvantage,
    arrays: dict[str, np.ndarray], indices: np.ndarray, fold: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    values = {key: [] for key in (
        "expected", "failure_probability", "failure_expert",
        "success_expert", "sign_score",
    )}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            local = indices[start:start + 256]
            inputs, _, _ = tensors(arrays, local, fold, device)
            output = model(*inputs)
            rows = {
                "expected": output.expected_advantage,
                "failure_probability": torch.sigmoid(output.failure_logit),
                "failure_expert": output.failure_expert,
                "success_expert": output.success_expert,
                "sign_score": torch.sigmoid(output.sign_logit),
            }
            for key, value in rows.items():
                values[key].append(value.cpu().numpy())
    return {key: np.concatenate(rows) for key, rows in values.items()}


def binary_auc(score: np.ndarray, label: np.ndarray) -> float | None:
    positive = score[label > 0.5]
    negative = score[label <= 0.5]
    if not len(positive) or not len(negative):
        return None
    return float(
        np.mean(positive[:, None] > negative[None, :])
        + 0.5 * np.mean(positive[:, None] == negative[None, :])
    )


def fold_run(
    fold: int, arrays: dict[str, np.ndarray], records: list[dict],
    episodes: list[dict], output: Path, device: torch.device,
    provenance: dict[str, dict],
) -> dict:
    fit, calibration, evaluation, partition = partition_indices(
        records, fold, "v6_3_1"
    )
    members = []
    evidence = []
    states = []
    for base_seed in SEEDS:
        model, row = train_member(
            arrays, records, fit, fold, base_seed + 100 * fold, device
        )
        members.append(model)
        evidence.append(row)
        states.append({
            key: value.detach().cpu() for key, value in model.state_dict().items()
        })
    cal = [predict(model, arrays, calibration, fold, device) for model in members]
    eva = [predict(model, arrays, evaluation, fold, device) for model in members]
    cal_expected = np.median(np.stack([
        row["expected"] for row in cal
    ]), axis=0)
    offset = scene_equal_empirical_offset(
        cal_expected, arrays["target"][calibration], calibration, records
    )
    prediction = {
        key: np.median(np.stack([row[key] for row in eva]), axis=0)
        for key in eva[0]
    }
    lower = prediction["expected"] - offset
    target = arrays["target"][evaluation].astype(np.float64)
    coverage = scene_macro_mean(target >= lower, evaluation, records)
    median_mae = scene_macro_mean(
        np.abs(target - prediction["expected"]), evaluation, records
    )
    zero_mae = scene_macro_mean(np.abs(target), evaluation, records)
    policy = earliest_authorized_policy(
        lower, target, evaluation, records, episodes, fold
    )
    checkpoint_partition = {
        **partition,
        "fit_rows": len(fit),
        "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
    }
    checkpoint = output / f"fold_{fold}/hurdle_ensemble.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    part = checkpoint.with_name(checkpoint.name + ".part")
    payload = {
        "schema_version": "revealnav-rxr-v6.4-hurdle-ensemble/1",
        "method_revision": "v6_4_failure_conditioned_hurdle",
        "fold": fold,
        "member_base_seeds": list(SEEDS),
        "member_effective_seeds": [seed + 100 * fold for seed in SEEDS],
        "model_state_dicts": states,
        "training_contract": V64_TRAINING_CONTRACT,
        "locked_provenance": provenance,
        "empirical_lower_offset": offset,
        "partition_evidence": checkpoint_partition,
        "training_evidence": evidence,
    }
    validate_hurdle_checkpoint_payload(
        payload, provenance, fold, checkpoint_partition
    )
    torch.save(payload, part)
    os.replace(part, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_hurdle_checkpoint_payload(
        saved, provenance, fold, checkpoint_partition
    )
    return {
        "fold": fold,
        "fit_rows": len(fit), "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
        "empirical_lower_offset": offset,
        "scene_macro_lower_coverage": coverage,
        "scene_macro_expected_mae": median_mae,
        "scene_macro_zero_mae": zero_mae,
        "expected_beats_zero": median_mae < zero_mae,
        "failure_auc": binary_auc(
            prediction["failure_probability"],
            arrays["native_failure"][evaluation],
        ),
        "selected_episodes": policy["selected_episodes"],
        "selected_scenes": policy["selected_scenes"],
        "selected_positive_precision": policy["selected_positive_precision"],
        "selected_realized_benefit_sum": policy[
            "selected_realized_benefit_sum"
        ],
        "scene_macro_policy_benefit": policy["scene_macro_policy_benefit"],
        "scene_policy_benefits": policy["scene_benefits"],
        "selected_event_ids": policy["selected_event_ids"],
        "evaluation_episodes": policy["episodes"],
        "zero_candidate_episodes": policy["zero_candidate_episodes"],
        "partition_evidence": checkpoint_partition,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "failure_probability_used_as_mixture_weight": True,
        "failure_probability_used_as_independent_gate": False,
        "sign_score_used_as_independent_gate": False,
        "offline_liveness_scope": (
            "accepted same-state pairs had executable return at generation; "
            "online integration must re-check return executable and option live"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest_path = safe_inside(args.manifest, True)
    output = safe_inside(args.output_dir, False)
    if output.exists():
        raise RuntimeError("refusing to overwrite V6.4 output")
    manifest, arrays, records = load_inputs(manifest_path)
    provenance = locked_provenance(manifest_path)
    selection = json.loads(SELECTION.read_text())
    episodes = selection.get("episodes", [])
    if not (
        not SELECTION.is_symlink() and selection.get("episode_count") == 120
        and len(episodes) == 120
        and len({str(row["episode_id"]) for row in episodes}) == 120
        and selection.get("split") == "train"
        and selection.get("unseen_or_test_read") is False
    ):
        raise RuntimeError("V6.4 sealed episode selection drift")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    working = output.with_name(output.name + ".part")
    if working.exists():
        raise RuntimeError("refusing stale V6.4 partial output")
    folds = [
        fold_run(fold, arrays, records, episodes, working, device, provenance)
        for fold in range(5)
    ]
    bootstrap = scene_cluster_bootstrap(folds)
    scene_count = sum(
        row["partition_evidence"]["evaluation_scene_count"] for row in folds
    )
    gates = {
        "at_least_100_same_state_pairs": len(records) >= 100,
        "at_least_20_positive_pairs": int(
            np.sum(arrays["target"] > 0.0)
        ) >= 20,
        "exactly_120_sealed_evaluation_episodes": sum(
            row["evaluation_episodes"] for row in folds
        ) == 120,
        "exactly_one_zero_candidate_episode_counted_as_zero": sum(
            row["zero_candidate_episodes"] for row in folds
        ) == 1,
        "all_folds_scene_macro_lower_coverage_at_least_0_85": all(
            row["scene_macro_lower_coverage"] >= 0.85 for row in folds
        ),
        "aggregate_scene_macro_expected_beats_zero": (
            sum(row["scene_macro_expected_mae"] * row[
                "partition_evidence"]["evaluation_scene_count"] for row in folds)
            < sum(row["scene_macro_zero_mae"] * row[
                "partition_evidence"]["evaluation_scene_count"] for row in folds)
        ),
        "at_least_ten_selected_episodes": sum(
            row["selected_episodes"] for row in folds
        ) >= 10,
        "selected_episodes_cover_at_least_five_scenes": sum(
            row["selected_scenes"] for row in folds
        ) >= 5,
        "positive_scene_macro_policy_benefit_in_all_folds": all(
            row["scene_macro_policy_benefit"] > 0.0 for row in folds
        ),
        "scene_cluster_bootstrap_95_lcb_above_zero": bootstrap[
            "lower_95"
        ] > 0.0,
    }
    value = {
        "schema_version": "revealnav-rxr-v6.4-hurdle-crossfit/1",
        "status": (
            "RXR_V6_4_OFFLINE_GATE_PASS" if all(gates.values())
            else "RXR_V6_4_OFFLINE_GATE_FAIL"
        ),
        "method_revision": "v6_4_failure_conditioned_hurdle",
        "dataset_manifest": str(manifest_path.relative_to(ROOT)),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "pairs": len(records), "episodes": len(episodes),
        "scenes": scene_count,
        "positive_pairs": int(np.sum(arrays["target"] > 0.0)),
        "native_failure_pairs": int(np.sum(arrays["native_failure"])),
        "folds": folds, "gates": gates,
        "scene_cluster_bootstrap": bootstrap,
        "online_gate": (
            "scene-calibrated lower expected advantage > 0 AND return "
            "executable AND option live"
        ),
        "failure_probability_used_as_mixture_weight": True,
        "failure_probability_used_as_independent_gate": False,
        "sign_score_used_as_independent_gate": False,
        "offline_liveness_scope": (
            "accepted same-state pairs had executable return at generation; "
            "online integration must re-check return executable and option live"
        ),
        "training_contract": V64_TRAINING_CONTRACT,
        "runtime": {
            "requested_device": args.device,
            "effective_device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "protocol_correction": {
            "path": str(CORRECTION.relative_to(ROOT)),
            "bytes": CORRECTION.stat().st_size,
            "sha256": sha256_file(CORRECTION),
        },
        "development_only_due_to_method_selection": True,
        "val_seen_authorized": all(gates.values()),
        "unseen_or_test_read": False, "paper_result": False,
        "sources": provenance,
    }
    atomic_json(working / "RXR_V6_4_HURDLE_RESULT.json", value)
    os.replace(working, output)
    print(json.dumps({
        "status": value["status"], "pairs": value["pairs"],
        "gates": gates, "bootstrap": bootstrap,
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
