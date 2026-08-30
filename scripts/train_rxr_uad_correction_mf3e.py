#!/usr/bin/env python3
"""Train three native-conditioned UAD correction seeds on RxR-train fit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import (  # noqa: E402
    MF3B_SCOPE,
    NativeConditionedUAD,
    OnlineUADFeatureDataset,
    collate_online_uad,
    native_conditioned_uad_loss,
)


SEEDS = (20260826, 20260827, 20260828)
DATA = ROOT / (
    "artifacts/phase1/mf3b_uad_online/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
DESIGN = ROOT / (
    "artifacts/design/METHOD_FREEZE_3E_NATIVE_CONDITIONED_UAD_CORRECTION.md"
)
OUT = ROOT / "artifacts/training/mf3e_uad_correction_v1"
EPOCHS = 30
PATIENCE = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def protocol() -> dict:
    manifest = json.loads(DATA.read_text())
    if manifest.get("status") != "PASS":
        raise RuntimeError("exact-online dataset is not complete")
    return {
        "schema_version": "revealnav-mf3e-uad-training-protocol/1",
        "status": "SEALED_BEFORE_TRAINING",
        "seeds": list(SEEDS),
        "model": "NativeConditionedUAD",
        "hidden_dim": 128,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "training_split": "scene-disjoint RxR-train fit only",
        "selection": (
            "maximize calibration native-error average precision times "
            "wrong-native alternative accuracy, then their mean"
        ),
        "data_sha256": sha256_file(DATA),
        "design_sha256": sha256_file(DESIGN),
        "uses_native_policy_logits_as_inputs": True,
        "uses_teacher_at_inference": False,
        **MF3B_SCOPE,
    }


def move(batch: dict[str, torch.Tensor], device: torch.device):
    return {name: value.to(device) for name, value in batch.items()}


def error_positive_weight(dataset: OnlineUADFeatureDataset) -> float:
    positive = negative = 0
    for row in dataset:
        valid = (
            (row["native_index"] >= 0)
            & (row["target_index"] >= 0)
            & (row["candidate_mask"].sum(-1) >= 2)
        )
        wrong = valid & (row["native_index"] != row["target_index"])
        positive += int(wrong.sum())
        negative += int((valid & ~wrong).sum())
    if min(positive, negative) < 1:
        raise RuntimeError("native correction training lacks a class")
    return negative / positive


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = int(ranked.sum())
    if positives == 0:
        raise RuntimeError("calibration has no native errors")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def metrics(model, loader, device) -> dict:
    labels = []
    probabilities = []
    alternative_correct = alternative_total = native_correct = total = 0
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = move(cpu, device)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            valid = (
                batch["step_mask"] & (batch["native_index"] >= 0)
                & (batch["target_index"] >= 0)
                & (batch["candidate_mask"].sum(-1) >= 2)
            )
            wrong = valid & (batch["native_index"] != batch["target_index"])
            labels.append(wrong[valid].cpu().numpy().astype(np.int64))
            probabilities.append(torch.sigmoid(
                output.native_error_logit[valid]
            ).cpu().numpy())
            native_correct += int((valid & ~wrong).sum())
            total += int(valid.sum())
            alternative_correct += int((
                output.alternative_logits.argmax(-1)[wrong]
                == batch["target_index"][wrong]
            ).sum())
            alternative_total += int(wrong.sum())
    labels_array = np.concatenate(labels)
    probabilities_array = np.concatenate(probabilities)
    return {
        "eligible": total,
        "native_accuracy": native_correct / total,
        "native_error_average_precision": average_precision(
            labels_array, probabilities_array
        ),
        "wrong_native_alternative_accuracy": (
            alternative_correct / alternative_total
        ),
        "native_errors": alternative_total,
    }


def seal() -> int:
    value = protocol()
    path = OUT / "MF3E_UAD_TRAINING_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed MF3E training protocol drift")
    if not path.exists():
        atomic_json(path, value)
    return 0


def train(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError("unsealed MF3E seed")
    protocol_path = OUT / "MF3E_UAD_TRAINING_PROTOCOL.json"
    if json.loads(protocol_path.read_text()) != protocol():
        raise RuntimeError("MF3E training protocol drift")
    run_dir = OUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    fit = OnlineUADFeatureDataset(DATA, "fit")
    calibration = OnlineUADFeatureDataset(DATA, "calibration")
    fit_loader = DataLoader(
        fit, batch_size=8, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_online_uad,
    )
    calibration_loader = DataLoader(
        calibration, batch_size=4, shuffle=False, collate_fn=collate_online_uad
    )
    positive_weight = error_positive_weight(fit)
    model = NativeConditionedUAD(768, 128).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    best_key = best_state = None
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = steps = 0
        for cpu in fit_loader:
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            losses = native_conditioned_uad_loss(
                output, batch, error_positive_weight=positive_weight
            )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite MF3E training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(batch["step_mask"].sum())
            total_loss += float(losses["total"].detach()) * count
            steps += count
        calibration_metrics = metrics(model, calibration_loader, device)
        mean_loss = total_loss / steps
        ap = calibration_metrics["native_error_average_precision"]
        alternative = calibration_metrics["wrong_native_alternative_accuracy"]
        key = (ap * alternative, (ap + alternative) / 2.0, -mean_loss)
        history.append({
            "epoch": epoch, "train_total": mean_loss,
            **calibration_metrics,
        })
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state, strict=True)
    final_metrics = metrics(model, calibration_loader, device)
    checkpoint = run_dir / "uad_correction_mf3e.pt"
    torch.save({
        "schema_version": "revealnav-mf3e-uad-checkpoint/1",
        "seed": seed, "hidden_dim": 128,
        "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(protocol_path),
        **MF3B_SCOPE,
    }, checkpoint)
    atomic_json(run_dir / "RESULT.json", {
        "status": "TRAINING_COMPLETE", "seed": seed,
        "calibration": final_metrics, "history": history,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        **MF3B_SCOPE,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return seal() if args.seal else train(args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
