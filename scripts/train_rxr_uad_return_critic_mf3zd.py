#!/usr/bin/env python3
"""Train scene-cross-fitted pre-decision critics on paired final returns."""

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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2r6.protocol import scene_fold
from revealnav_mf3 import CausalReturnSafetyCritic
from train_rxr_uad_correction_mf3e import atomic_json, sha256_file


MANIFEST = ROOT / (
    "artifacts/phase1/rxr_v6/full_v6_3_1/"
    "RXR_V6_3_1_PAIRED_DATASET_MANIFEST.json"
)
ARRAYS = ROOT / (
    "artifacts/phase1/rxr_v6/full_v6_3_1/"
    "RXR_V6_3_1_PAIRED_DATASET.npz"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZD_CAUSAL_RETURN_VETO.md"
OUT = ROOT / "artifacts/training/mf3zd_causal_return_critic_v1"
SEEDS = (20260826, 20260827, 20260828)
INPUT_KEYS = ("instruction", "checkpoint", "native", "alternative")
STEPS = 400


def load() -> tuple[dict[str, np.ndarray], list[dict]]:
    manifest = json.loads(MANIFEST.read_text())
    if not (
        manifest.get("status") == "RXR_V6_3_1_PAIRED_DATASET_READY"
        and manifest.get("metadata", {}).get("unseen_or_test_read") is False
        and manifest.get("metadata", {}).get("pairs") == 339
        and manifest.get("arrays", {}).get("path") == str(ARRAYS.relative_to(ROOT))
        and manifest["arrays"]["bytes"] == ARRAYS.stat().st_size
        and manifest["arrays"]["sha256"] == sha256_file(ARRAYS)
    ):
        raise RuntimeError("MF3ZD paired-return provenance drift")
    with np.load(ARRAYS, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in (*INPUT_KEYS, "target")}
    records = manifest["records"]
    if not (
        len(records) == 339
        and all(arrays[key].shape == (339, 768) for key in INPUT_KEYS)
        and arrays["target"].shape == (339,)
        and all(np.isfinite(value).all() for value in arrays.values())
        and all(int(row["row_index"]) == index for index, row in enumerate(records))
    ):
        raise RuntimeError("MF3ZD paired-return array drift")
    return arrays, records


def scene_weights(indices: np.ndarray, records: list[dict], device: torch.device) -> torch.Tensor:
    counts = Counter(str(records[int(index)]["scene_id"]) for index in indices)
    values = np.asarray([
        1.0 / counts[str(records[int(index)]["scene_id"])] for index in indices
    ], dtype=np.float32)
    values /= values.mean()
    return torch.from_numpy(values).to(device)


def tensors(arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device):
    inputs = [torch.from_numpy(arrays[key][indices].astype(np.float32)).to(device)
              for key in INPUT_KEYS]
    target = torch.from_numpy(arrays["target"][indices].astype(np.float32)).to(device)
    return inputs, target


def train_member(
    arrays: dict[str, np.ndarray], records: list[dict], indices: np.ndarray,
    seed: int, device: torch.device,
) -> tuple[CausalReturnSafetyCritic, dict]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
    model = CausalReturnSafetyCritic(768, 32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    inputs, target = tensors(arrays, indices, device)
    weights = scene_weights(indices, records, device)
    labels = (target > 0.0).to(target.dtype)
    history = []
    for _ in range(STEPS):
        output = model(*inputs)
        utility = F.smooth_l1_loss(output.expected_utility, target, reduction="none")
        sign = F.binary_cross_entropy_with_logits(
            output.beneficial_logit, labels, reduction="none"
        )
        loss = ((utility + 0.5 * sign) * weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite MF3ZD loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        history.append(float(loss.detach()))
    return model.eval(), {
        "seed": seed, "fit_rows": len(indices), "optimizer_steps": STEPS,
        "final_loss": history[-1], "minimum_loss": min(history),
        "positive_rows": int(labels.sum()), "negative_or_tied_rows": int((1-labels).sum()),
    }


def predict(model, arrays, indices, device):
    inputs, _ = tensors(arrays, indices, device)
    with torch.no_grad():
        output = model(*inputs)
    return output.expected_utility.cpu().numpy(), torch.sigmoid(
        output.beneficial_logit
    ).cpu().numpy()


def train_crossfit(device: torch.device) -> int:
    arrays, records = load()
    folds = np.asarray([scene_fold(str(row["scene_id"])) for row in records])
    oof_utility = np.zeros((len(records), len(SEEDS)), dtype=np.float32)
    oof_probability = np.zeros_like(oof_utility)
    fold_evidence = []
    for fold in range(5):
        fit = np.flatnonzero(folds != fold); evaluate = np.flatnonzero(folds == fold)
        states = []; members = []
        for member, seed in enumerate(SEEDS):
            model, evidence = train_member(
                arrays, records, fit, seed + fold * 100, device
            )
            utility, probability = predict(model, arrays, evaluate, device)
            oof_utility[evaluate, member] = utility
            oof_probability[evaluate, member] = probability
            states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
            members.append(evidence)
        fold_dir = OUT / f"crossfit/fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = fold_dir / "return_critic_mf3zd.pt"
        torch.save({
            "schema_version": "revealnav-mf3zd-return-critic-crossfit/1",
            "fold": fold, "seeds": list(SEEDS), "projection_dim": 32,
            "model_state_dicts": states,
        }, checkpoint)
        fold_evidence.append({
            "fold": fold, "fit_rows": len(fit), "evaluation_rows": len(evaluate),
            "fit_scenes": len({records[int(i)]["scene_id"] for i in fit}),
            "evaluation_scenes": len({records[int(i)]["scene_id"] for i in evaluate}),
            "scene_overlap": 0, "members": members,
            "checkpoint": {"path": str(checkpoint.relative_to(ROOT)),
                           "bytes": checkpoint.stat().st_size,
                           "sha256": sha256_file(checkpoint)},
        })
    rows = []
    for index, record in enumerate(records):
        expected = oof_utility[index].tolist()
        probability = oof_probability[index].tolist()
        median_expected = float(np.median(oof_utility[index]))
        mad_expected = float(np.median(np.abs(oof_utility[index] - median_expected)))
        rows.append({
            "row_index": index, "scene_id": str(record["scene_id"]),
            "episode_id": str(record["episode_id"]), "event_id": record["event_id"],
            "target_utility": float(arrays["target"][index]),
            "member_expected_utility": expected,
            "member_beneficial_probability": probability,
            "robust_expected_utility": median_expected - mad_expected,
            "minimum_beneficial_probability": min(probability),
        })
    atomic_json(OUT / "MF3ZD_CROSSFIT_PREDICTIONS.json", {
        "schema_version": "revealnav-mf3zd-crossfit-predictions/1",
        "status": "CROSSFIT_COMPLETE", "rows": rows, "folds": fold_evidence,
        "sources": {
            "manifest": sha256_file(MANIFEST), "arrays": sha256_file(ARRAYS),
            "design": sha256_file(DESIGN),
        }, "unseen_or_test_read": False,
    })
    return 0


def train_final(device: torch.device) -> int:
    arrays, records = load(); indices = np.arange(len(records))
    states = []; members = []
    for seed in SEEDS:
        model, evidence = train_member(arrays, records, indices, seed, device)
        states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        members.append(evidence)
    final = OUT / "final"; final.mkdir(parents=True, exist_ok=False)
    checkpoint = final / "return_critic_mf3zd.pt"
    torch.save({
        "schema_version": "revealnav-mf3zd-return-critic-final/1",
        "seeds": list(SEEDS), "projection_dim": 32,
        "optimizer_steps": STEPS, "model_state_dicts": states,
        "source_manifest_sha256": sha256_file(MANIFEST),
        "source_arrays_sha256": sha256_file(ARRAYS),
    }, checkpoint)
    atomic_json(final / "RESULT.json", {
        "status": "FINAL_TRAINING_COMPLETE", "members": members,
        "checkpoint": {"path": str(checkpoint.relative_to(ROOT)),
                       "bytes": checkpoint.stat().st_size,
                       "sha256": sha256_file(checkpoint)},
        "unseen_or_test_read": False,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("crossfit", "final"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    return (train_crossfit if args.command == "crossfit" else train_final)(
        torch.device(args.device)
    )


if __name__ == "__main__":
    raise SystemExit(main())
