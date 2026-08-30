#!/usr/bin/env python3
"""Build the consumed-data view and train the sealed MF3N ensemble."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import sys
from collections import defaultdict
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
    top2_expected_switch_utility,
    top2_switch_targets,
    top2_switch_utility_loss,
)
from scripts.train_rxr_uad_correction_mf3e import (  # noqa: E402
    atomic_json,
    move,
    sha256_file,
)

SEED = 20260828
SEEDS = (20260826, 20260827, 20260828)
HIDDEN_DIMS = (32, 64)
SOURCE = ROOT / (
    "artifacts/phase1/mf3m_robust_top2_rank23/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
DATA = ROOT / (
    "artifacts/phase1/mf3n_top2_utility_rank23/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3N_DIRECT_TOP2_UTILITY.md"
MF3M_GATE = ROOT / (
    "artifacts/evaluation/mf3m_robust_top2_shadow_gate_v1/"
    "MF3M_SHADOW_GATE.json"
)
OUT = ROOT / "artifacts/training/mf3n_top2_utility_v1"
EPOCHS = 24
PATIENCE = 5


def rank_key(episode_id: str) -> str:
    return hashlib.sha256(
        f"{SEED}:episode:{episode_id}".encode()
    ).hexdigest()


def build_data_view() -> dict:
    source = json.loads(SOURCE.read_text())
    if source.get("status") != "PASS" or source.get("counts") != {
        "fit": 519, "calibration": 112,
        "diagnostic": 336, "shadow": 336,
    }:
        raise RuntimeError("MF3N source manifest is incomplete")
    by_scene = defaultdict(list)
    for row in source["records"]:
        by_scene[row["scene_id"]].append(row)
    records = []
    for scene in sorted(by_scene):
        ranked = sorted(
            by_scene[scene], key=lambda row: rank_key(str(row["episode_id"]))
        )
        if len(ranked) > 23:
            raise RuntimeError("MF3N source contains an episode rank above 23")
        for rank, source_row in enumerate(ranked, 1):
            row = dict(source_row)
            row["source_split"] = row["split"]
            row["episode_rank"] = rank
            row["split"] = (
                "fit" if rank <= 17 else
                "calibration" if rank <= 20 else "diagnostic"
            )
            records.append(row)
    counts = {
        split: sum(row["split"] == split for row in records)
        for split in ("fit", "calibration", "diagnostic")
    }
    if counts != {"fit": 967, "calibration": 168, "diagnostic": 168}:
        raise RuntimeError(f"MF3N derived rank counts drift: {counts}")
    return {
        "schema_version": "revealnav-mf3b-online-manifest/1",
        "status": "PASS",
        "counts": counts,
        "failures": [],
        "records": records,
        "source_manifest": {
            "path": str(SOURCE.relative_to(ROOT)),
            "bytes": SOURCE.stat().st_size,
            "sha256": sha256_file(SOURCE),
        },
        "repartition": {
            "seed": SEED,
            "fit_ranks": [1, 17],
            "calibration_ranks": [18, 20],
            "diagnostic_ranks": [21, 23],
            "payload_copied": False,
        },
        "public_unseen_authorized": False,
    }


def validate_data() -> None:
    expected = build_data_view()
    if not DATA.is_file() or json.loads(DATA.read_text()) != expected:
        raise RuntimeError("MF3N consumed-data view drift")
    if any(
        row.get("observation_frontend")
        != "frozen_etp_r1_policy_fusion_token"
        or int(row.get("candidate_feature_dim")) != 1536
        for row in expected["records"]
    ):
        raise RuntimeError("MF3N policy-token provenance drift")
    gate = json.loads(MF3M_GATE.read_text())
    if not (
        gate.get("status") == "SHADOW_GATE_FAIL"
        and gate.get("ranks18_23_payload_read") is True
        and gate.get("task_metric_run_authorized") is False
    ):
        raise RuntimeError("MF3N prior-gate boundary drift")


def protocol() -> dict:
    validate_data()
    return {
        "schema_version": "revealnav-mf3n-training-protocol/1",
        "status": "SEALED_BEFORE_TRAINING",
        "architectures": [
            {"hidden_dim": hidden, "seeds": list(SEEDS)}
            for hidden in HIDDEN_DIMS
        ],
        "model": "PairwiseSwitchUtility runner-up projection",
        "outcomes": ["NEITHER", "RESCUE", "HARM"],
        "score": "P(RESCUE)-P(HARM)",
        "loss": "episode-balanced unweighted top-2 three-class CE",
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "epoch_selection": "minimum ranks18-20 top-2 NLL",
        "optimizer": {"name": "AdamW", "lr": 3e-4,
                      "weight_decay": 1e-4},
        "data_sha256": sha256_file(DATA),
        "design_sha256": sha256_file(DESIGN),
        "mf3m_gate_sha256": sha256_file(MF3M_GATE),
        **MF3B_SCOPE,
    }


def evaluate(model, loader, device) -> dict:
    model.eval()
    episode_losses = []
    steps = correct = rescues = harms = neither = 0
    with torch.no_grad():
        for cpu in loader:
            batch = move(cpu, device)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            loss = top2_switch_utility_loss(output, batch)
            episode_losses.append(float(loss))
            score, runner, valid = top2_expected_switch_utility(output, batch)
            labels, _, _ = top2_switch_targets(batch)
            logits = output.outcome_logits.gather(
                2, runner[..., None, None].expand(
                    *runner.shape, 1, output.outcome_logits.shape[-1]
                )
            ).squeeze(2)
            steps += int(valid.sum())
            correct += int((logits.argmax(-1)[valid] == labels[valid]).sum())
            proposed = valid & (score > 0)
            rescues += int((proposed & (labels == 1)).sum())
            harms += int((proposed & (labels == 2)).sum())
            neither += int((proposed & (labels == 0)).sum())
    if not episode_losses or steps < 1:
        raise RuntimeError("MF3N evaluation has no top-2 supervision")
    return {
        "episodes": len(episode_losses),
        "steps": steps,
        "top2_nll": statistics.mean(episode_losses),
        "top2_accuracy": correct / steps,
        "zero_threshold_rescues": rescues,
        "zero_threshold_harms": harms,
        "zero_threshold_neither": neither,
    }


def seal() -> int:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    view = build_data_view()
    if DATA.exists() and json.loads(DATA.read_text()) != view:
        raise RuntimeError("refusing to overwrite a drifted MF3N data view")
    if not DATA.exists():
        atomic_json(DATA, view)
    value = protocol()
    path = OUT / "MF3N_TRAINING_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed MF3N training protocol drift")
    if not path.exists():
        atomic_json(path, value)
    return 0


def train(hidden: int, seed: int, device: torch.device) -> int:
    if hidden not in HIDDEN_DIMS or seed not in SEEDS:
        raise ValueError("unsealed MF3N architecture or seed")
    protocol_path = OUT / "MF3N_TRAINING_PROTOCOL.json"
    if json.loads(protocol_path.read_text()) != protocol():
        raise RuntimeError("MF3N training protocol drift")
    run_dir = OUT / f"hidden_{hidden}/seed_{seed}"
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
        OnlineUADFeatureDataset(DATA, "calibration"), batch_size=1,
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
        train_losses = []
        for cpu in fit:
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            loss = top2_switch_utility_loss(output, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite MF3N training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        metrics = evaluate(model, calibration, device)
        key = (-metrics["top2_nll"], metrics["top2_accuracy"],
               -statistics.mean(train_losses))
        history.append({"epoch": epoch,
                        "train_top2_nll": statistics.mean(train_losses),
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
    checkpoint = run_dir / "top2_utility_mf3n.pt"
    torch.save({
        "schema_version": "revealnav-mf3n-checkpoint/1",
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
        return seal()
    if args.hidden is None or args.seed is None:
        raise SystemExit("training requires --hidden and --seed")
    return train(args.hidden, args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
