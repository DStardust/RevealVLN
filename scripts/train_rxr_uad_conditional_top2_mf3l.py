#!/usr/bin/env python3
"""Train the sealed MF3L conditional Top-2 UAD ensemble."""

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
    policy_anchored_conditional_top2_loss,
    top2_conditional_advantage,
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
    "artifacts/phase1/mf3k_policy_top2_rank17/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3L_CONDITIONAL_TOP2.md"
MF3K_GATE = ROOT / (
    "artifacts/evaluation/mf3k_policy_top2_shadow_gate_v1/"
    "MF3K_SHADOW_GATE.json"
)
OUT = ROOT / "artifacts/training/mf3l_conditional_top2_v1"
EPOCHS = 30
PATIENCE = 6


def architecture_name(hidden: int, bound: float) -> str:
    return f"hidden_{hidden}_bound_{str(bound).replace('.', 'p')}"


def validate_data() -> None:
    manifest = json.loads(DATA.read_text())
    if manifest.get("status") != "PASS" or manifest.get("counts") != {
        "fit": 519, "calibration": 112,
        "diagnostic": 168, "shadow": 168,
    }:
        raise RuntimeError("MF3L source data is incomplete")
    if any(
        row.get("observation_frontend")
        != "frozen_etp_r1_policy_fusion_token"
        or int(row.get("candidate_feature_dim")) != 1536
        for row in manifest["records"]
    ):
        raise RuntimeError("MF3L policy-token provenance drift")
    gate = json.loads(MF3K_GATE.read_text())
    if not (
        gate.get("status") == "SHADOW_GATE_FAIL"
        and gate.get("ranks15_17_payload_read") is True
        and gate.get("shadow", {}).get("net_rescues") == 2
        and gate.get("task_metric_run_authorized") is False
    ):
        raise RuntimeError("MF3K frozen-evidence precondition failed")


def protocol() -> dict:
    validate_data()
    return {
        "schema_version": "revealnav-mf3l-training-protocol/1",
        "status": "SEALED_BEFORE_TRAINING",
        "architectures": [
            {"hidden_dim": hidden, "correction_bound": bound,
             "seeds": list(SEEDS)}
            for hidden, bound in ARCHITECTURES
        ],
        "model": "PolicyAnchoredTop2UAD",
        "score": "tanh((runner_logit-native_logit)/2)",
        "loss": "candidate CE + conditional native-vs-runner BCE",
        "loss_weights": {"candidate": 1.0, "conditional_top2": 1.0},
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "epoch_selection": "minimum calibration combined NLL",
        "optimizer": {"name": "AdamW", "lr": 3e-4,
                      "weight_decay": 1e-4},
        "development_reclassification": {
            "stratum_a": "consumed episode ranks 12-14",
            "stratum_b": "consumed episode ranks 15-17",
            "fresh_shadow": "uncollected episode ranks 18-23",
        },
        "data_sha256": sha256_file(DATA),
        "design_sha256": sha256_file(DESIGN),
        "mf3k_gate_sha256": sha256_file(MF3K_GATE),
        **MF3B_SCOPE,
    }


def evaluate(model, loader, device) -> dict:
    model.eval()
    totals = {"total": 0.0, "target": 0.0, "top2": 0.0}
    target_steps = top2_steps = target_correct = 0
    rescues = harms = neither = 0
    with torch.no_grad():
        for cpu in loader:
            batch = move(cpu, device)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            losses = policy_anchored_conditional_top2_loss(output, batch)
            valid_target = batch["step_mask"] & (batch["target_index"] >= 0)
            labels, runner, valid_pair = top2_switch_targets(batch)
            comparable = valid_pair & (
                (batch["target_index"] == batch["native_index"])
                | (batch["target_index"] == runner)
            )
            target_count = int(valid_target.sum())
            pair_count = int(comparable.sum())
            totals["target"] += float(losses["target"]) * target_count
            totals["top2"] += float(losses["top2"]) * pair_count
            target_steps += target_count
            top2_steps += pair_count
            target_correct += int((
                output.target_logits.argmax(-1)[valid_target]
                == batch["target_index"][valid_target]
            ).sum())
            advantage, _, valid = top2_conditional_advantage(
                output, batch["native_scores"], batch["candidate_mask"],
                batch["native_index"],
            )
            proposed = valid & valid_target & (advantage > 0)
            rescues += int((proposed & (labels == 1)).sum())
            harms += int((proposed & (labels == 2)).sum())
            neither += int((proposed & (labels == 0)).sum())
    if target_steps < 1 or top2_steps < 1:
        raise RuntimeError("MF3L evaluation has no valid supervision")
    target_nll = totals["target"] / target_steps
    top2_nll = totals["top2"] / top2_steps
    return {
        "target_steps": target_steps,
        "top2_steps": top2_steps,
        "target_nll": target_nll,
        "top2_nll": top2_nll,
        "combined_nll": target_nll + top2_nll,
        "target_accuracy": target_correct / target_steps,
        "zero_threshold_rescues": rescues,
        "zero_threshold_harms": harms,
        "zero_threshold_neither": neither,
    }


def seal() -> int:
    value = protocol()
    path = OUT / "MF3L_TRAINING_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed MF3L training protocol drift")
    if not path.exists():
        atomic_json(path, value)
    return 0


def train(hidden: int, bound: float, seed: int, device: torch.device) -> int:
    if (hidden, bound) not in ARCHITECTURES or seed not in SEEDS:
        raise ValueError("unsealed MF3L architecture or seed")
    protocol_path = OUT / "MF3L_TRAINING_PROTOCOL.json"
    if json.loads(protocol_path.read_text()) != protocol():
        raise RuntimeError("MF3L training protocol drift")
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
    model = PolicyAnchoredTop2UAD(768, 1536, hidden, bound).to(device)
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
            losses = policy_anchored_conditional_top2_loss(output, batch)
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite MF3L training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int((batch["step_mask"] & (batch["target_index"] >= 0)).sum())
            total += float(loss.detach()) * count
            steps += count
        metrics = evaluate(model, calibration, device)
        key = (-metrics["combined_nll"], metrics["target_accuracy"],
               -total / steps)
        history.append({"epoch": epoch, "train_combined_loss": total / steps,
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
    checkpoint = run_dir / "conditional_top2_mf3l.pt"
    torch.save({
        "schema_version": "revealnav-mf3l-checkpoint/1",
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
