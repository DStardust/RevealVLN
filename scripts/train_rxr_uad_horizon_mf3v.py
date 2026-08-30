#!/usr/bin/env python3
"""Train MF3V rankers with fixed three-step trajectory-consistent labels."""

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

from revealnav_mf3 import (  # noqa: E402
    MF3B_SCOPE, OnlineUADFeatureDataset, PairwiseSwitchUtility,
    collate_online_uad, top2_horizon_rescue_harm_ranked_loss,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json, move, sha256_file  # noqa: E402
from scripts.train_rxr_uad_crossfit_mf3q import FOLDS, manifest_path  # noqa: E402


DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3V_HORIZON_CONSISTENT_UAD.md"
OUT = ROOT / "artifacts/training/mf3v_horizon_ranker_v1"
SEEDS = (20260826, 20260827, 20260828)
HIDDEN = 128
STEPS = 800
HORIZON = 3
RANKING_WEIGHT = 0.25
POSITIVE_WEIGHT = 7700 / 1437


def protocol() -> dict:
    return {
        "schema_version": "revealnav-mf3v-training-protocol/1",
        "status": "SEALED_BEFORE_MF3V_TRAINING",
        "folds": list(FOLDS), "final_model": True,
        "hidden_dim": HIDDEN, "seeds": list(SEEDS),
        "optimizer_steps": STEPS, "horizon": HORIZON,
        "rescue_positive_weight": POSITIVE_WEIGHT,
        "ranking_weight": RANKING_WEIGHT,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "design_sha256": sha256_file(DESIGN),
        "public_unseen_authorized": False, **MF3B_SCOPE,
    }


def seal() -> int:
    path = OUT / "MF3V_TRAINING_PROTOCOL.json"
    value = protocol()
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed MF3V protocol drift")
    if not path.exists():
        atomic_json(path, value)
    return 0


def train(fold: int | str, seed: int, device: torch.device) -> int:
    if fold not in (*FOLDS, "final") or seed not in SEEDS:
        raise ValueError("unsealed MF3V run")
    protocol_path = OUT / "MF3V_TRAINING_PROTOCOL.json"
    if json.loads(protocol_path.read_text()) != protocol():
        raise RuntimeError("MF3V protocol drift")
    run = OUT / f"fold_{fold}/seed_{seed}"
    if run.exists():
        raise RuntimeError(f"refusing to overwrite {run}")
    run.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
    loader = DataLoader(
        OnlineUADFeatureDataset(manifest_path(fold), "fit"), batch_size=8,
        shuffle=True, generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_online_uad,
    )
    iterator = iter(loader)
    model = PairwiseSwitchUtility(768, 1536, HIDDEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = {"total": [], "binary": [], "ranking": []}
    for _ in range(STEPS):
        try:
            cpu = next(iterator)
        except StopIteration:
            iterator = iter(loader); cpu = next(iterator)
        batch = move(cpu, device); optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        losses = top2_horizon_rescue_harm_ranked_loss(
            output, batch, rescue_positive_weight=POSITIVE_WEIGHT,
            ranking_weight=RANKING_WEIGHT, horizon=HORIZON,
        )
        if not all(torch.isfinite(value) for value in losses.values()):
            raise RuntimeError("non-finite MF3V loss")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for name, value in losses.items(): history[name].append(float(value.detach()))
    checkpoint = run / "horizon_ranker_mf3v.pt"
    torch.save({
        "schema_version": "revealnav-mf3v-checkpoint/1", "fold": fold,
        "seed": seed, "hidden_dim": HIDDEN, "optimizer_steps": STEPS,
        "horizon": HORIZON, "rescue_positive_weight": POSITIVE_WEIGHT,
        "ranking_weight": RANKING_WEIGHT, "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(protocol_path), **MF3B_SCOPE,
    }, checkpoint)
    atomic_json(run / "RESULT.json", {
        "status": "TRAINING_COMPLETE", "fold": fold, "seed": seed,
        "hidden_dim": HIDDEN, "optimizer_steps": STEPS, "horizon": HORIZON,
        "mean_losses": {k: statistics.mean(v) for k, v in history.items()},
        "checkpoint": {"path": str(checkpoint.relative_to(ROOT)),
                        "bytes": checkpoint.stat().st_size,
                        "sha256": sha256_file(checkpoint)}, **MF3B_SCOPE,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--fold")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seal: return seal()
    return train("final" if args.fold == "final" else int(args.fold), args.seed,
                 torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
