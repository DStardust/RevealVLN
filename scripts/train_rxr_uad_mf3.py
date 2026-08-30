#!/usr/bin/env python3
"""Train UAD-only MF3 heads from audited semantics and exact-online features."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2 import RevealFeatureDataset  # noqa: E402
from revealnav_mf2r3 import RevealExpiryFeatureDataset  # noqa: E402
from revealnav_mf3 import (  # noqa: E402
    MF3B_SCOPE,
    OnlineUADFeatureDataset,
    StructuredUADHeads,
    StructuredUADLoss,
    StructuredUADLossConfig,
    collate_online_uad,
)
from scripts.run_rxr_representation_comparison_v2 import (  # noqa: E402
    classification_metrics,
)


SEEDS = (20260826, 20260827, 20260828)
ONLINE = ROOT / (
    "artifacts/phase1/mf3b_uad_online/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
LEGACY = ROOT / (
    "artifacts/phase1/rxr_train_expansion/"
    "RXR_SCALE_AUTOMATIC_TRAINING_MANIFEST.json"
)
EXPIRY = ROOT / (
    "artifacts/phase1/rxr_train_expansion/expiry_r3/"
    "RXR_EXPIRY_R3_FEATURE_MANIFEST.json"
)
OUT = ROOT / "artifacts/training/mf3b_uad_online_v1"
EPOCHS = 20
PATIENCE = 4


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


class LegacyAdapter(Dataset):
    def __init__(self, source: Dataset) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = dict(self.source[index])
        steps, candidates = row["candidate_mask"].shape
        row.setdefault("expiry_hazard", torch.full((steps,), -1.0))
        row["native_index"] = torch.full((steps,), -1, dtype=torch.long)
        row["native_scores"] = torch.full((steps, candidates), -torch.inf)
        row["outside_score"] = torch.full((steps,), -torch.inf)
        return {
            name: row[name] for name in (
                "instruction_embedding", "history_embeddings",
                "candidate_embeddings", "candidate_mask", "target_index",
                "native_index", "native_scores", "outside_score", "target_in_set",
                "separation", "evidence_complete", "reveal_hazard",
                "expiry_hazard",
            )
        }


def protocol() -> dict:
    online = json.loads(ONLINE.read_text())
    if online.get("status") != "PASS":
        raise RuntimeError("exact-online dataset is not complete")
    return {
        "schema_version": "revealnav-mf3b-uad-training-protocol/1",
        "status": "SEALED_BEFORE_TRAINING",
        "seeds": list(SEEDS),
        "model": "StructuredUADHeads",
        "hidden_dim": 128,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "sampling": {
            "legacy_semantic": 0.4,
            "legacy_expiry": 0.2,
            "exact_online": 0.4,
        },
        "selection": (
            "maximize the worse of exact-online calibration target accuracy "
            "and legacy scene-heldout UAD macro-F1, then their mean, then "
            "lower total loss"
        ),
        "sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (ONLINE, LEGACY, EXPIRY)
        },
        "online_label_boundary": {
            "target_and_target_in_set": "native RxR nDTW teacher",
            "separation": "fixed 0.5m teacher geodesic margin",
            "evidence_reveal_expiry": "masked, not geometrically fabricated",
        },
        **MF3B_SCOPE,
    }


def seal() -> int:
    value = protocol()
    path = OUT / "MF3B_UAD_TRAINING_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed MF3B training protocol drift")
    if not path.exists():
        atomic_json(path, value)
    return 0


def move(batch: dict[str, torch.Tensor], device: torch.device):
    return {name: value.to(device) for name, value in batch.items()}


def label_weights(datasets: list[Dataset]) -> tuple[tuple, tuple]:
    binary = {
        name: [0, 0] for name in (
            "target_in_set", "separation", "evidence_complete",
            "reveal_hazard", "expiry_hazard",
        )
    }
    uad = np.zeros(3, dtype=np.int64)
    for dataset in datasets:
        for row in dataset:
            for name in binary:
                values = row[name].numpy()
                valid = values >= 0
                binary[name][0] += int((values[valid] < 0.5).sum())
                binary[name][1] += int((values[valid] >= 0.5).sum())
            valid = (
                (row["target_in_set"] >= 0)
                & (row["separation"] >= 0)
                & (row["evidence_complete"] >= 0)
            )
            labels = torch.where(
                row["target_in_set"] < 0.5,
                torch.zeros_like(row["target_index"]),
                torch.where(
                    (row["separation"] < 0.5)
                    | (row["evidence_complete"] < 0.5),
                    torch.ones_like(row["target_index"]),
                    torch.full_like(row["target_index"], 2),
                ),
            )
            uad += np.bincount(labels[valid].numpy(), minlength=3)
    if any(min(value) < 1 for value in binary.values()) or np.any(uad == 0):
        raise RuntimeError("MF3B training labels lack a required class")
    positive = tuple(value[0] / value[1] for value in binary.values())
    classes = tuple((uad.sum() / (3.0 * uad)).tolist())
    return positive, classes


def semantic_metrics(model, loader, device) -> dict:
    labels, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = move(cpu, device)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
            )
            valid = (
                batch["step_mask"]
                & (batch["target_in_set"] >= 0)
                & (batch["separation"] >= 0)
                & (batch["evidence_complete"] >= 0)
            )
            label = torch.where(
                batch["target_in_set"] < 0.5,
                torch.zeros_like(batch["target_index"]),
                torch.where(
                    (batch["separation"] < 0.5)
                    | (batch["evidence_complete"] < 0.5),
                    torch.ones_like(batch["target_index"]),
                    torch.full_like(batch["target_index"], 2),
                ),
            )
            labels.append(label[valid].cpu().numpy())
            probabilities.append(output.uad_probabilities[valid].cpu().numpy())
    return classification_metrics(
        np.concatenate(labels), np.concatenate(probabilities)
    )


def online_target_accuracy(model, loader, device) -> float:
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = move(cpu, device)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
            )
            valid = batch["step_mask"] & (batch["target_index"] >= 0)
            correct += int(
                (output.target_logits.argmax(-1)[valid]
                 == batch["target_index"][valid]).sum()
            )
            total += int(valid.sum())
    if not total:
        raise RuntimeError("online calibration has no target labels")
    return correct / total


def train(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError("unsealed MF3B seed")
    if json.loads((OUT / "MF3B_UAD_TRAINING_PROTOCOL.json").read_text()) != protocol():
        raise RuntimeError("MF3B training protocol drift")
    run_dir = OUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)

    legacy_train = LegacyAdapter(RevealFeatureDataset(LEGACY, "train"))
    expiry_train = LegacyAdapter(RevealExpiryFeatureDataset(EXPIRY, "train"))
    online_fit = OnlineUADFeatureDataset(ONLINE, "fit")
    legacy_development = LegacyAdapter(
        RevealFeatureDataset(LEGACY, "development")
    )
    online_calibration = OnlineUADFeatureDataset(ONLINE, "calibration")
    datasets = [legacy_train, expiry_train, online_fit]
    combined = ConcatDataset(datasets)
    masses = (0.4, 0.2, 0.4)
    weights = torch.cat([
        torch.full((len(dataset),), mass / len(dataset), dtype=torch.double)
        for dataset, mass in zip(datasets, masses)
    ])
    sampler = WeightedRandomSampler(
        weights, num_samples=sum(len(row) for row in datasets), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        combined, batch_size=8, sampler=sampler, collate_fn=collate_online_uad
    )
    semantic_loader = DataLoader(
        legacy_development, batch_size=8, shuffle=False,
        collate_fn=collate_online_uad,
    )
    calibration_loader = DataLoader(
        online_calibration, batch_size=2, shuffle=False,
        collate_fn=collate_online_uad,
    )
    positive, classes = label_weights(datasets)
    model = StructuredUADHeads(768, 128).to(device)
    objective = StructuredUADLoss(StructuredUADLossConfig(
        class_weights=classes, positive_weights=positive,
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_key = best_state = None
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = count = 0
        for cpu in train_loader:
            batch = move(cpu, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
            )
            losses = objective(output, batch)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite MF3B training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(batch["step_mask"].sum())
            total += float(losses["total"].detach()) * size
            count += size
        semantic = semantic_metrics(model, semantic_loader, device)
        target_accuracy = online_target_accuracy(
            model, calibration_loader, device
        )
        mean_loss = total / count
        key = (
            min(target_accuracy, semantic["macro_f1"]),
            (target_accuracy + semantic["macro_f1"]) / 2.0,
            -mean_loss,
        )
        history.append({
            "epoch": epoch, "train_total": mean_loss,
            "online_target_accuracy": target_accuracy,
            "semantic_macro_f1": semantic["macro_f1"],
            "semantic_false_ready_rate": semantic["false_ready_rate"],
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
    semantic = semantic_metrics(model, semantic_loader, device)
    target_accuracy = online_target_accuracy(model, calibration_loader, device)
    checkpoint = run_dir / "uad_mf3.pt"
    torch.save({
        "schema_version": "revealnav-mf3b-uad-checkpoint/1",
        "seed": seed, "hidden_dim": 128,
        "model_state_dict": model.state_dict(),
        "protocol_sha256": sha256_file(OUT / "MF3B_UAD_TRAINING_PROTOCOL.json"),
        **MF3B_SCOPE,
    }, checkpoint)
    atomic_json(run_dir / "RESULT.json", {
        "status": "TRAINING_COMPLETE",
        "seed": seed,
        "online_calibration_target_accuracy": target_accuracy,
        "legacy_semantic_development": semantic,
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
