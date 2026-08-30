#!/usr/bin/env python3
"""Train the sealed MF3K policy-anchored target-posterior ensemble."""

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
    OnlineUADFeatureDataset,
    PolicyAnchoredTop2UAD,
    collate_online_uad,
    policy_anchored_target_loss,
    top2_posterior_advantage,
    top2_switch_targets,
)
from scripts.train_rxr_uad_correction_mf3e import (  # noqa: E402
    atomic_json,
    move,
    sha256_file,
)

SEEDS = (20260826, 20260827, 20260828)
ARCHITECTURES = ((64, 1.0), (64, 2.0), (128, 1.0), (128, 2.0))
DATA = ROOT / (
    "artifacts/phase1/mf3i_policy_token_uad/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3K_POLICY_ANCHORED_TOP2.md"
MF3J_GATE = ROOT / (
    "artifacts/evaluation/mf3j_switch_utility_shadow_gate_v1/"
    "MF3J_SHADOW_GATE.json"
)
OUT = ROOT / "artifacts/training/mf3k_policy_top2_v1"
EPOCHS = 30
PATIENCE = 6


def validate_data() -> None:
    value = json.loads(DATA.read_text())
    if value.get("status") != "PASS" or value.get("counts") != {
        "fit": 519, "calibration": 112, "diagnostic": 112, "shadow": 56,
    }:
        raise RuntimeError("MF3K source data is incomplete")
    if any(
        row.get("observation_frontend")
        != "frozen_etp_r1_policy_fusion_token"
        or int(row.get("candidate_feature_dim")) != 1536
        for row in value["records"]
    ):
        raise RuntimeError("MF3K policy-token provenance drift")
    gate = json.loads(MF3J_GATE.read_text())
    if not (
        gate.get("status") == "SHADOW_GATE_FAIL"
        and gate.get("rank14_payload_read") is True
        and gate.get("task_metric_run_authorized") is False
    ):
        raise RuntimeError("MF3J negative-evidence precondition failed")


def protocol() -> dict:
    validate_data()
    return {
        "schema_version": "revealnav-mf3k-training-protocol/1",
        "status": "SEALED_BEFORE_TRAINING",
        "architectures": [
            {"hidden_dim": hidden, "correction_bound": bound,
             "seeds": list(SEEDS)}
            for hidden, bound in ARCHITECTURES
        ],
        "model": "PolicyAnchoredTop2UAD",
        "candidate_feature_dim": 1536,
        "loss": "unweighted candidate-level cross-entropy",
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "epoch_selection": "minimum calibration target NLL",
        "optimizer": {"name": "AdamW", "lr": 3e-4,
                      "weight_decay": 1e-4},
        "development_reclassification": {
            "diagnostic_a": "episode ranks 12-13",
            "diagnostic_b": "consumed MF3J rank 14",
            "fresh_shadow": "uncollected episode ranks 15-17",
        },
        "data_sha256": sha256_file(DATA),
        "design_sha256": sha256_file(DESIGN),
        "mf3j_gate_sha256": sha256_file(MF3J_GATE),
        **MF3B_SCOPE,
    }


def evaluate(model, loader, device) -> dict:
    model.eval()
    total_loss = total_steps = correct = 0
    rescues = harms = neither = eligible = 0
    with torch.no_grad():
        for cpu in loader:
            batch = move(cpu, device)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            valid_target = batch["step_mask"] & (batch["target_index"] >= 0)
            count = int(valid_target.sum())
            loss = policy_anchored_target_loss(output, batch)
            total_loss += float(loss) * count
            total_steps += count
            correct += int((
                output.target_logits.argmax(-1)[valid_target]
                == batch["target_index"][valid_target]
            ).sum())
            advantage, runner, valid_pair = top2_posterior_advantage(
                output, batch["native_scores"], batch["candidate_mask"],
                batch["native_index"],
            )
            valid_pair &= batch["step_mask"] & (batch["target_index"] >= 0)
            proposed = valid_pair & (advantage > 0)
            native = batch["native_index"]
            target = batch["target_index"]
            eligible += int(valid_pair.sum())
            rescues += int((proposed & (native != target) & (runner == target)).sum())
            harms += int((proposed & (native == target)).sum())
            neither += int((proposed & (native != target) & (runner != target)).sum())
    if total_steps < 1:
        raise RuntimeError("MF3K evaluation contains no target-labelled steps")
    return {
        "target_steps": total_steps,
        "target_nll": total_loss / total_steps,
        "target_accuracy": correct / total_steps,
        "eligible_top2_steps": eligible,
        "zero_threshold_rescues": rescues,
        "zero_threshold_harms": harms,
        "zero_threshold_neither": neither,
    }


def architecture_name(hidden: int, bound: float) -> str:
    return f"hidden_{hidden}_bound_{str(bound).replace('.', 'p')}"


def seal() -> int:
    value = protocol()
    path = OUT / "MF3K_TRAINING_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed MF3K training protocol drift")
    if not path.exists():
        atomic_json(path, value)
    return 0


def train(hidden: int, bound: float, seed: int, device: torch.device) -> int:
    if (hidden, bound) not in ARCHITECTURES or seed not in SEEDS:
        raise ValueError("unsealed MF3K architecture or seed")
    protocol_path = OUT / "MF3K_TRAINING_PROTOCOL.json"
    if json.loads(protocol_path.read_text()) != protocol():
        raise RuntimeError("MF3K training protocol drift")
    run_dir = OUT / architecture_name(hidden, bound) / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    fit = DataLoader(
        OnlineUADFeatureDataset(DATA, "fit"), batch_size=8, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_online_uad,
    )
    calibration = DataLoader(
        OnlineUADFeatureDataset(DATA, "calibration"), batch_size=4,
        shuffle=False, collate_fn=collate_online_uad,
    )
    model = PolicyAnchoredTop2UAD(
        768, 1536, hidden, correction_bound=bound
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    best_key = best_state = None
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = steps = 0
        for cpu in fit:
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            loss = policy_anchored_target_loss(output, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite MF3K training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int((batch["step_mask"] & (batch["target_index"] >= 0)).sum())
            total += float(loss.detach()) * count
            steps += count
        metrics = evaluate(model, calibration, device)
        key = (-metrics["target_nll"], metrics["target_accuracy"],
               -total / steps)
        history.append({"epoch": epoch, "train_target_nll": total / steps,
                        **metrics})
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state, strict=True)
    final = evaluate(model, calibration, device)
    checkpoint = run_dir / "policy_top2_mf3k.pt"
    torch.save({
        "schema_version": "revealnav-mf3k-checkpoint/1",
        "hidden_dim": hidden,
        "correction_bound": bound,
        "seed": seed,
        "candidate_feature_dim": 1536,
        "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(protocol_path),
        **MF3B_SCOPE,
    }, checkpoint)
    atomic_json(run_dir / "RESULT.json", {
        "status": "TRAINING_COMPLETE",
        "hidden_dim": hidden,
        "correction_bound": bound,
        "seed": seed,
        "calibration": final,
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
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--bound", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if None in (args.hidden, args.bound, args.seed):
        raise SystemExit("training requires --hidden, --bound, and --seed")
    return train(args.hidden, args.bound, args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
