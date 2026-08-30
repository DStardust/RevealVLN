#!/usr/bin/env python3
"""Train the bounded native-residual UAD on expanded exact-online data."""

from __future__ import annotations

import argparse
import copy
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
    native_residual_logits,
    native_residual_uad_loss,
)
from scripts.train_rxr_uad_correction_mf3e import (  # noqa: E402
    atomic_json,
    average_precision,
    move,
    sha256_file,
)


SEEDS = (20260826, 20260827, 20260828)
DATA = ROOT / (
    "artifacts/phase1/mf3g_uad_online_expanded/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3G_BOUNDED_NATIVE_RESIDUAL.md"
OUT = ROOT / "artifacts/training/mf3g_uad_residual_v1"
CORRECTION_BOUND = 2.0
EPOCHS = 30
PATIENCE = 6


def protocol() -> dict:
    manifest = json.loads(DATA.read_text())
    if manifest.get("status") != "PASS" or manifest.get("counts") != {
        "fit": 519, "calibration": 112, "shadow": 56,
    }:
        raise RuntimeError("MF3G expanded dataset is not complete")
    return {
        "schema_version": "revealnav-mf3g-uad-training-protocol/1",
        "status": "SEALED_BEFORE_TRAINING",
        "seeds": list(SEEDS), "model": "NativeConditionedUAD",
        "hidden_dim": 128, "epochs": EPOCHS, "patience": PATIENCE,
        "correction_bound": CORRECTION_BOUND,
        "loss": {
            "fused_policy_cross_entropy": 1.0,
            "native_error_bce": 0.25,
            "correction_l2": 0.01,
        },
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "selection": (
            "maximize calibration fused-minus-native accuracy, then net "
            "rescue, lower harm, and native-error average precision"
        ),
        "data_sha256": sha256_file(DATA),
        "design_sha256": sha256_file(DESIGN),
        **MF3B_SCOPE,
    }


def metrics(model, loader, device) -> dict:
    labels = []
    probabilities = []
    counts = {"eligible": 0, "native_correct": 0, "fused_correct": 0,
              "rescues": 0, "harms": 0, "interventions": 0}
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = move(cpu, device)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            fused, _ = native_residual_logits(
                output, batch["native_scores"], batch["candidate_mask"],
                correction_bound=CORRECTION_BOUND,
            )
            valid = (
                batch["step_mask"] & (batch["native_index"] >= 0)
                & (batch["target_index"] >= 0)
                & (batch["candidate_mask"].sum(-1) >= 2)
            )
            native = batch["native_index"]
            target = batch["target_index"]
            adapted = fused.argmax(-1)
            native_correct = native == target
            adapted_correct = adapted == target
            wrong = valid & ~native_correct
            labels.append(wrong[valid].cpu().numpy().astype(np.int64))
            probabilities.append(torch.sigmoid(
                output.native_error_logit[valid]
            ).cpu().numpy())
            counts["eligible"] += int(valid.sum())
            counts["native_correct"] += int((valid & native_correct).sum())
            counts["fused_correct"] += int((valid & adapted_correct).sum())
            counts["rescues"] += int((wrong & adapted_correct).sum())
            counts["harms"] += int((valid & native_correct & ~adapted_correct).sum())
            counts["interventions"] += int((valid & (adapted != native)).sum())
    labels_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    eligible = counts["eligible"]
    return {
        **counts,
        "native_accuracy": counts["native_correct"] / eligible,
        "fused_accuracy": counts["fused_correct"] / eligible,
        "net_rescues": counts["rescues"] - counts["harms"],
        "native_error_average_precision": average_precision(
            labels_array, probability_array
        ),
    }


def seal() -> int:
    value = protocol()
    path = OUT / "MF3G_UAD_TRAINING_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed MF3G training protocol drift")
    if not path.exists():
        atomic_json(path, value)
    return 0


def train(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError("unsealed MF3G seed")
    protocol_path = OUT / "MF3G_UAD_TRAINING_PROTOCOL.json"
    if json.loads(protocol_path.read_text()) != protocol():
        raise RuntimeError("MF3G training protocol drift")
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
        calibration, batch_size=4, shuffle=False,
        collate_fn=collate_online_uad,
    )
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
            losses = native_residual_uad_loss(
                output, batch, correction_bound=CORRECTION_BOUND
            )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite MF3G training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(batch["step_mask"].sum())
            total_loss += float(losses["total"].detach()) * count
            steps += count
        calibration_metrics = metrics(model, calibration_loader, device)
        mean_loss = total_loss / steps
        key = (
            calibration_metrics["fused_accuracy"]
            - calibration_metrics["native_accuracy"],
            calibration_metrics["net_rescues"],
            -calibration_metrics["harms"],
            calibration_metrics["native_error_average_precision"],
            -mean_loss,
        )
        history.append({"epoch": epoch, "train_total": mean_loss,
                        **calibration_metrics})
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
    checkpoint = run_dir / "uad_residual_mf3g.pt"
    torch.save({
        "schema_version": "revealnav-mf3g-uad-checkpoint/1",
        "seed": seed, "hidden_dim": 128,
        "correction_bound": CORRECTION_BOUND,
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
