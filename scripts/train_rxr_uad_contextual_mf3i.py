#!/usr/bin/env python3
"""Train the frozen-backbone contextual-token MF3I UAD adapter."""

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
    native_alternative_posterior_gain,
    native_conditioned_uad_loss,
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
    "artifacts/phase1/mf3i_policy_token_uad/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3I_CONTEXTUAL_TOKEN_UAD.md"
OUT = ROOT / "artifacts/training/mf3i_policy_token_uad_v1"
HIDDEN_DIM = 64
CORRECTION_BOUND = 1.0
EPOCHS = 30
PATIENCE = 6


def manifest() -> dict:
    value = json.loads(DATA.read_text())
    if value.get("status") != "PASS" or value.get("counts") != {
        "fit": 519, "calibration": 112, "diagnostic": 112, "shadow": 56,
    }:
        raise RuntimeError("MF3I contextual dataset is not complete")
    if any(
        row.get("observation_frontend")
        != "frozen_etp_r1_policy_fusion_token"
        for row in value.get("records", [])
    ):
        raise RuntimeError("MF3I contextual observation provenance drift")
    return value


def protocol() -> dict:
    manifest()
    return {
        "schema_version": "revealnav-mf3i-uad-training-protocol/1",
        "status": "SEALED_BEFORE_TRAINING",
        "seeds": list(SEEDS),
        "model": "NativeConditionedUAD",
        "feature": "frozen_etp_r1_policy_fusion_token",
        "candidate_feature_dim": 1536,
        "hidden_dim": HIDDEN_DIM,
        "correction_bound": CORRECTION_BOUND,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "native_error_positive_weight": 1.0,
        "loss": {
            "fused_policy_cross_entropy": 1.0,
            "proper_native_error_bce": 0.5,
            "wrong_step_alternative_cross_entropy": 1.0,
            "correction_l2": 0.02,
        },
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "epoch_selection": (
            "maximize calibration posterior-gated net rescue, then minimize "
            "harm, maximize rescue and fused accuracy"
        ),
        "posterior_gain_rule": (
            "p(native_wrong)*p(proposed_alternative|native_wrong) "
            "- p(native_correct) > 0"
        ),
        "data_sha256": sha256_file(DATA),
        "design_sha256": sha256_file(DESIGN),
        **MF3B_SCOPE,
    }


def metrics(model, loader, device) -> dict:
    labels = []
    probabilities = []
    counts = {
        "eligible": 0, "native_correct": 0, "fused_correct": 0,
        "posterior_correct": 0, "posterior_rescues": 0,
        "posterior_harms": 0, "posterior_interventions": 0,
    }
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
            error_probability = torch.sigmoid(output.native_error_logit)
            posterior_gain = native_alternative_posterior_gain(
                output, adapted
            )
            authorized = valid & (adapted != native) & (posterior_gain > 0)
            selected = torch.where(authorized, adapted, native)
            native_correct = native == target
            adapted_correct = adapted == target
            selected_correct = selected == target
            wrong = valid & ~native_correct
            labels.append(wrong[valid].cpu().numpy().astype(np.int64))
            probabilities.append(error_probability[valid].cpu().numpy())
            counts["eligible"] += int(valid.sum())
            counts["native_correct"] += int((valid & native_correct).sum())
            counts["fused_correct"] += int((valid & adapted_correct).sum())
            counts["posterior_correct"] += int((valid & selected_correct).sum())
            counts["posterior_rescues"] += int(
                (authorized & wrong & selected_correct).sum()
            )
            counts["posterior_harms"] += int(
                (authorized & native_correct & ~selected_correct).sum()
            )
            counts["posterior_interventions"] += int(authorized.sum())
    labels_array = np.concatenate(labels)
    probability_array = np.concatenate(probabilities)
    eligible = counts["eligible"]
    return {
        **counts,
        "native_accuracy": counts["native_correct"] / eligible,
        "fused_accuracy": counts["fused_correct"] / eligible,
        "posterior_accuracy": counts["posterior_correct"] / eligible,
        "posterior_net_rescues": (
            counts["posterior_rescues"] - counts["posterior_harms"]
        ),
        "native_error_average_precision": average_precision(
            labels_array, probability_array
        ),
    }


def seal() -> int:
    value = protocol()
    path = OUT / "MF3I_UAD_TRAINING_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed MF3I training protocol drift")
    if not path.exists():
        atomic_json(path, value)
    return 0


def train(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError("unsealed MF3I seed")
    protocol_path = OUT / "MF3I_UAD_TRAINING_PROTOCOL.json"
    frozen_protocol = protocol()
    if json.loads(protocol_path.read_text()) != frozen_protocol:
        raise RuntimeError("MF3I training protocol drift")
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
    model = NativeConditionedUAD(
        768, HIDDEN_DIM, candidate_feature_dim=1536
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    positive_weight = float(frozen_protocol["native_error_positive_weight"])
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
            residual = native_residual_uad_loss(
                output, batch, correction_bound=CORRECTION_BOUND,
                error_weight=0.0, regularization_weight=0.02,
            )
            factorized = native_conditioned_uad_loss(
                output, batch, error_positive_weight=positive_weight,
            )
            loss = (
                residual["total"] + 0.5 * factorized["native_error"]
                + factorized["alternative"]
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite MF3I training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(batch["step_mask"].sum())
            total_loss += float(loss.detach()) * count
            steps += count
        calibration_metrics = metrics(model, calibration_loader, device)
        mean_loss = total_loss / steps
        key = (
            calibration_metrics["posterior_net_rescues"],
            -calibration_metrics["posterior_harms"],
            calibration_metrics["posterior_rescues"],
            calibration_metrics["fused_accuracy"],
            calibration_metrics["native_error_average_precision"],
            -mean_loss,
        )
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
    checkpoint = run_dir / "uad_contextual_mf3i.pt"
    torch.save({
        "schema_version": "revealnav-mf3i-uad-checkpoint/1",
        "seed": seed,
        "hidden_dim": HIDDEN_DIM,
        "correction_bound": CORRECTION_BOUND,
        "feature": "frozen_etp_r1_policy_fusion_token",
        "candidate_feature_dim": 1536,
        "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(protocol_path),
        **MF3B_SCOPE,
    }, checkpoint)
    atomic_json(run_dir / "RESULT.json", {
        "status": "TRAINING_COMPLETE",
        "seed": seed,
        "calibration": final_metrics,
        "history": history,
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
