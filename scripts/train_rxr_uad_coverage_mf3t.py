#!/usr/bin/env python3
"""Train the frozen MF3T cross-fitted rescue/harm rankers."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import (
    MF3B_SCOPE,
    OnlineUADFeatureDataset,
    PairwiseSwitchUtility,
    collate_online_uad,
    top2_rescue_harm_ranked_loss,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json, move, sha256_file
from scripts.train_rxr_uad_crossfit_mf3q import FOLDS, manifest_path


DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3T_COVERAGE_CONSTRAINED_UAD.md"
MF3S_RESULT = ROOT / (
    "artifacts/evaluation/mf3s_uad_rxr_val_seen_v1/"
    "MF3S_RXR_VAL_SEEN_RESULT.json"
)
OUT = ROOT / "artifacts/training/mf3t_coverage_ranker_v2"
SEEDS = (20260826, 20260827, 20260828)
HIDDEN_DIMS = (64, 128, 256)
STEPS = 800
RANKING_WEIGHT = 0.25
POSITIVE_WEIGHT = 7700 / 1437


def protocol() -> dict:
    result = json.loads(MF3S_RESULT.read_text())
    failed = [name for name, passed in result["gates"].items() if not passed]
    if result.get("status") != "TASK_METRIC_GATE_FAIL" or failed != [
        "utility_lower_95_positive"
    ]:
        raise RuntimeError("MF3T trigger evidence drift")
    return {
        "schema_version": "revealnav-mf3t-training-protocol/2",
        "status": "SEALED_BEFORE_MF3T_TRAINING",
        "folds": list(FOLDS),
        "final_model": True,
        "hidden_dims": list(HIDDEN_DIMS),
        "seeds": list(SEEDS),
        "optimizer_steps": STEPS,
        "rescue_positive_weight": POSITIVE_WEIGHT,
        "ranking_weight": RANKING_WEIGHT,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "design_sha256": sha256_file(DESIGN),
        "mf3s_result_sha256": sha256_file(MF3S_RESULT),
        "public_unseen_authorized": False,
        **MF3B_SCOPE,
    }


def seal() -> int:
    value = protocol()
    path = OUT / "MF3T_TRAINING_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed MF3T protocol drift")
    if not path.exists():
        atomic_json(path, value)
    return 0


def train(fold: int | str, hidden: int, seed: int, device: torch.device) -> int:
    if fold not in (*FOLDS, "final") or hidden not in HIDDEN_DIMS or seed not in SEEDS:
        raise ValueError("unsealed MF3T run")
    protocol_path = OUT / "MF3T_TRAINING_PROTOCOL.json"
    if json.loads(protocol_path.read_text()) != protocol():
        raise RuntimeError("MF3T protocol drift")
    run = OUT / f"hidden_{hidden}/fold_{fold}/seed_{seed}"
    if run.exists():
        raise RuntimeError(f"refusing to overwrite {run}")
    run.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    loader = DataLoader(
        OnlineUADFeatureDataset(manifest_path(fold), "fit"),
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_online_uad,
    )
    iterator = iter(loader)
    model = PairwiseSwitchUtility(768, 1536, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = {"total": [], "binary": [], "ranking": []}
    for _ in range(STEPS):
        try:
            cpu = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            cpu = next(iterator)
        batch = move(cpu, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        losses = top2_rescue_harm_ranked_loss(
            output, batch,
            rescue_positive_weight=POSITIVE_WEIGHT,
            ranking_weight=RANKING_WEIGHT,
        )
        if not all(torch.isfinite(value) for value in losses.values()):
            raise RuntimeError("non-finite MF3T loss")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for name, value in losses.items():
            history[name].append(float(value.detach()))
    checkpoint = run / "coverage_ranker_mf3t.pt"
    torch.save({
        "schema_version": "revealnav-mf3t-checkpoint/2",
        "fold": fold,
        "seed": seed,
        "hidden_dim": hidden,
        "optimizer_steps": STEPS,
        "rescue_positive_weight": POSITIVE_WEIGHT,
        "ranking_weight": RANKING_WEIGHT,
        "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(protocol_path),
        **MF3B_SCOPE,
    }, checkpoint)
    atomic_json(run / "RESULT.json", {
        "status": "TRAINING_COMPLETE",
        "fold": fold,
        "seed": seed,
        "hidden_dim": hidden,
        "optimizer_steps": STEPS,
        "mean_losses": {
            name: statistics.mean(values) for name, values in history.items()
        },
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
    parser.add_argument("--fold")
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seal:
        return seal()
    fold = "final" if args.fold == "final" else int(args.fold)
    return train(fold, args.hidden, args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
