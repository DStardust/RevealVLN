#!/usr/bin/env python3
"""Train the bounded MF3J direct switch-outcome models."""

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
    PairwiseSwitchUtility,
    collate_online_uad,
    pairwise_expected_utility,
    pairwise_switch_targets,
    pairwise_switch_utility_loss,
)
from scripts.train_rxr_uad_correction_mf3e import (  # noqa: E402
    atomic_json,
    move,
    sha256_file,
)

SEEDS = (20260826, 20260827, 20260828)
HIDDEN_DIMS = (64, 128)
DATA = ROOT / (
    "artifacts/phase1/mf3i_policy_token_uad/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3J_DIRECT_SWITCH_UTILITY.md"
OUT = ROOT / "artifacts/training/mf3j_switch_utility_v1"
EPOCHS = 30
PATIENCE = 6


def validate_data() -> None:
    value = json.loads(DATA.read_text())
    if value.get("status") != "PASS" or value.get("counts") != {
        "fit": 519, "calibration": 112, "diagnostic": 112, "shadow": 56,
    }:
        raise RuntimeError("MF3J source data is incomplete")
    if any(
        row.get("observation_frontend")
        != "frozen_etp_r1_policy_fusion_token"
        or int(row.get("candidate_feature_dim")) != 1536
        for row in value["records"]
    ):
        raise RuntimeError("MF3J policy-token provenance drift")
    prior_gate = ROOT / (
        "artifacts/evaluation/mf3i_contextual_uad_shadow_gate_v1/"
        "MF3I_UAD_SHADOW_GATE.json"
    )
    prior = json.loads(prior_gate.read_text())
    if prior.get("shadow") != {}:
        raise RuntimeError("rank-14 was already opened by the prior gate")


def protocol() -> dict:
    validate_data()
    return {
        "schema_version": "revealnav-mf3j-training-protocol/1",
        "status": "SEALED_BEFORE_TRAINING",
        "architectures": [
            {"hidden_dim": hidden, "seeds": list(SEEDS)}
            for hidden in HIDDEN_DIMS
        ],
        "model": "PairwiseSwitchUtility",
        "candidate_feature_dim": 1536,
        "outcome_classes": ["NEITHER", "RESCUE", "HARM"],
        "loss": "unweighted three-class cross-entropy",
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "epoch_selection": "minimum calibration pairwise NLL",
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "data_sha256": sha256_file(DATA),
        "design_sha256": sha256_file(DESIGN),
        **MF3B_SCOPE,
    }


def evaluate(model, loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    pairs = correct = steps = rescues = harms = neither = 0
    with torch.no_grad():
        for cpu in loader:
            batch = move(cpu, device)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            labels, pair_mask = pairwise_switch_targets(batch)
            count = int(pair_mask.sum())
            loss = pairwise_switch_utility_loss(output, batch)
            total_loss += float(loss) * count
            pairs += count
            correct += int(
                (output.outcome_logits.argmax(-1)[pair_mask]
                 == labels[pair_mask]).sum()
            )
            utility = pairwise_expected_utility(output).masked_fill(
                ~pair_mask, -torch.inf
            )
            proposed = utility.argmax(-1)
            valid_steps = pair_mask.any(-1)
            native = batch["native_index"]
            target = batch["target_index"]
            proposal_correct = proposed == target
            native_correct = native == target
            steps += int(valid_steps.sum())
            rescues += int(
                (valid_steps & ~native_correct & proposal_correct).sum()
            )
            harms += int(
                (valid_steps & native_correct & ~proposal_correct).sum()
            )
            neither += int(
                (valid_steps & ~native_correct & ~proposal_correct).sum()
            )
    if pairs < 1:
        raise RuntimeError("MF3J evaluation contains no valid pairs")
    return {
        "pairs": pairs,
        "steps": steps,
        "pairwise_nll": total_loss / pairs,
        "pairwise_accuracy": correct / pairs,
        "ungated_proposal_rescues": rescues,
        "ungated_proposal_harms": harms,
        "ungated_proposal_neither": neither,
    }


def seal() -> int:
    value = protocol()
    path = OUT / "MF3J_TRAINING_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed MF3J training protocol drift")
    if not path.exists():
        atomic_json(path, value)
    return 0


def train(hidden: int, seed: int, device: torch.device) -> int:
    if hidden not in HIDDEN_DIMS or seed not in SEEDS:
        raise ValueError("unsealed MF3J architecture or seed")
    protocol_path = OUT / "MF3J_TRAINING_PROTOCOL.json"
    if json.loads(protocol_path.read_text()) != protocol():
        raise RuntimeError("MF3J training protocol drift")
    run_dir = OUT / f"hidden_{hidden}" / f"seed_{seed}"
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
    model = PairwiseSwitchUtility(768, 1536, hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    best_key = best_state = None
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = pairs = 0
        for cpu in fit:
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            _, pair_mask = pairwise_switch_targets(batch)
            loss = pairwise_switch_utility_loss(output, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite MF3J training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(pair_mask.sum())
            total += float(loss.detach()) * count
            pairs += count
        calibration_metrics = evaluate(model, calibration, device)
        key = (-calibration_metrics["pairwise_nll"],
               calibration_metrics["pairwise_accuracy"], -total / pairs)
        history.append({
            "epoch": epoch, "train_pairwise_nll": total / pairs,
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
    final = evaluate(model, calibration, device)
    checkpoint = run_dir / "switch_utility_mf3j.pt"
    torch.save({
        "schema_version": "revealnav-mf3j-checkpoint/1",
        "hidden_dim": hidden,
        "seed": seed,
        "candidate_feature_dim": 1536,
        "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(protocol_path),
        **MF3B_SCOPE,
    }, checkpoint)
    atomic_json(run_dir / "RESULT.json", {
        "status": "TRAINING_COMPLETE",
        "hidden_dim": hidden,
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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seal:
        if args.hidden is not None or args.seed is not None:
            raise SystemExit("seal does not accept a run")
        return seal()
    if args.hidden is None or args.seed is None:
        raise SystemExit("training requires --hidden and --seed")
    return train(args.hidden, args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
