#!/usr/bin/env python3
"""Train causal representation baselines and compare them on development."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path("/mnt/daiyang/vla").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2 import (  # noqa: E402
    RevealFeatureDataset,
    RevealOptionHeads,
    collate_reveal_examples,
)


V2 = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2"
MANIFEST = V2 / "RXR_MULTIBRANCH_FEATURE_MANIFEST_V2_AUTHORIZED.json"
AUTHORIZATION = V2 / "RXR_MULTIBRANCH_TRAINING_AUTHORIZATION_V2.json"
TRAINING = ROOT / "artifacts/training/mf2_multibranch_v2"
OUTPUT = ROOT / "artifacts/evaluation/mf2_representation_v2"
SEEDS = (20260826, 20260827, 20260828)
MODEL_NAMES = (
    "majority",
    "target_visible",
    "current_direct_uad",
    "history_direct_uad",
    "full_ree",
)
CLASS_NAMES = ("U", "A", "D")


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


def uad_labels(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.where(
        batch["target_in_set"] < 0.5,
        torch.zeros_like(batch["target_in_set"], dtype=torch.long),
        torch.where(
            (batch["separation"] < 0.5)
            | (batch["evidence_complete"] < 0.5),
            torch.ones_like(batch["target_in_set"], dtype=torch.long),
            torch.full_like(batch["target_in_set"], 2, dtype=torch.long),
        ),
    )


class DirectBaseline(nn.Module):
    """Current-only or causal-history classifier over the shared frozen inputs."""

    def __init__(
        self, *, history_aware: bool, output_dim: int, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        feature_dim = 768
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.history_aware = history_aware
        self.history_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.instruction_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        if history_aware:
            self.fusion = nn.GRU(hidden_dim * 3, hidden_dim, batch_first=True)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history = self.history_projection(batch["history_embeddings"])
        candidates = self.candidate_projection(batch["candidate_embeddings"])
        weights = batch["candidate_mask"].unsqueeze(-1).to(candidates.dtype)
        pooled = (candidates * weights).sum(2) / weights.sum(2).clamp_min(1.0)
        instruction = self.instruction_projection(
            batch["instruction_embedding"]
        ).unsqueeze(1).expand_as(history)
        fused_input = torch.cat((history, pooled, instruction), dim=-1)
        if self.history_aware:
            fused, _ = self.fusion(fused_input)
        else:
            fused = self.fusion(fused_input)
        return self.head(fused)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def train_baseline(
    name: str,
    seed: int,
    train_set: RevealFeatureDataset,
    development_set: RevealFeatureDataset,
    device: torch.device,
) -> tuple[DirectBaseline, list[dict]]:
    target_visible = name == "target_visible"
    model = DirectBaseline(
        history_aware=name == "history_direct_uad",
        output_dim=1 if target_visible else 3,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=8, shuffle=True, generator=generator,
        collate_fn=collate_reveal_examples,
    )
    development_loader = DataLoader(
        development_set, batch_size=16, shuffle=False,
        collate_fn=collate_reveal_examples,
    )

    def loss_on(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = model(batch)
        mask = batch["step_mask"]
        if target_visible:
            return nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1)[mask], batch["target_in_set"][mask]
            )
        return nn.functional.cross_entropy(logits[mask], uad_labels(batch)[mask])

    history = []
    best_loss = math.inf
    best_state = None
    for epoch in range(1, 21):
        model.train()
        total, count = 0.0, 0
        for cpu_batch in train_loader:
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(batch["step_mask"].sum())
            total += float(loss.detach()) * size
            count += size
        model.eval()
        development_total, development_count = 0.0, 0
        with torch.no_grad():
            for cpu_batch in development_loader:
                batch = move_batch(cpu_batch, device)
                loss = loss_on(batch)
                size = int(batch["step_mask"].sum())
                development_total += float(loss) * size
                development_count += size
        development_loss = development_total / development_count
        history.append({
            "epoch": epoch,
            "train_loss": total / count,
            "development_loss": development_loss,
        })
        if development_loss < best_loss:
            best_loss = development_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError(f"{name}: no best checkpoint")
    model.load_state_dict(best_state, strict=True)
    return model, history


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        confusion[label, prediction] += 1
    per_class = {}
    f1_values = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) \
            if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) \
            if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) \
            if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[index, :].sum()),
        }
    clipped = np.clip(probabilities, 1e-8, 1.0)
    nll = -np.log(clipped[np.arange(len(labels)), labels]).mean()
    one_hot = np.eye(3)[labels]
    brier = np.square(probabilities - one_hot).sum(1).mean()
    confidence = probabilities.max(1)
    correct = predictions == labels
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (confidence >= lower) & (
            confidence <= upper if upper >= 1.0 else confidence < upper
        )
        if selected.any():
            ece += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    not_ready = labels != 2
    ready = labels == 2
    return {
        "prefixes": int(len(labels)),
        "accuracy": float(correct.mean()),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "nll": float(nll),
        "brier": float(brier),
        "ece_10bin": float(ece),
        "false_ready_rate": float((predictions[not_ready] == 2).mean()),
        "missed_ready_rate": float((predictions[ready] != 2).mean()),
        "confusion_true_rows_pred_columns": confusion.tolist(),
    }


def collect_probabilities(
    model_name: str,
    model: nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    full_checkpoint: dict | None = None,
    majority_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels, probabilities = [], []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            mask = batch["step_mask"]
            labels.append(uad_labels(batch)[mask].cpu().numpy())
            if model_name == "majority":
                probability = torch.as_tensor(
                    majority_prior, device=device, dtype=torch.float32
                ).view(1, 3).expand(int(mask.sum()), 3)
            elif model_name == "target_visible":
                visible = torch.sigmoid(model(batch).squeeze(-1)[mask])
                probability = torch.stack(
                    (1.0 - visible, torch.zeros_like(visible), visible), dim=-1
                )
            elif model_name in ("current_direct_uad", "history_direct_uad"):
                probability = torch.softmax(model(batch)[mask], dim=-1)
            elif model_name == "full_ree":
                batch_size, steps = batch["step_mask"].shape
                budgets = torch.tensor(
                    full_checkpoint["normalized_budgets"], device=device
                ).view(1, 1, -1).expand(batch_size, steps, -1)
                output = model(
                    batch["history_embeddings"],
                    batch["candidate_embeddings"],
                    batch["candidate_mask"],
                    budgets,
                    batch["instruction_embedding"],
                )
                target_set = torch.sigmoid(output.target_in_set_logit[mask])
                separated = torch.sigmoid(output.separation_logit[mask])
                evidence = torch.sigmoid(output.evidence_logit[mask])
                decisive = separated * evidence
                probability = torch.stack((
                    1.0 - target_set,
                    target_set * (1.0 - decisive),
                    target_set * decisive,
                ), dim=-1)
            else:
                raise ValueError(model_name)
            probabilities.append(probability.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def run_seed(seed: int, device: torch.device) -> int:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    authorization = json.loads(AUTHORIZATION.read_text())
    reference = authorization.get("training_manifest", {})
    if not (
        authorization.get("status") == "TRAINING_AUTHORIZATION_PASS"
        and reference.get("path") == str(MANIFEST.relative_to(ROOT))
        and sha256_file(MANIFEST) == reference.get("sha256")
    ):
        raise RuntimeError("training authorization binding failed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    train_set = RevealFeatureDataset(MANIFEST, "train")
    development_set = RevealFeatureDataset(MANIFEST, "development")
    development_loader = DataLoader(
        development_set, batch_size=16, shuffle=False,
        collate_fn=collate_reveal_examples,
    )
    train_counts = np.zeros(3, dtype=np.float64)
    for example in train_set:
        values = {
            key: value.unsqueeze(0) for key, value in example.items()
        }
        labels = uad_labels(values).numpy().reshape(-1)
        train_counts += np.bincount(labels, minlength=3)
    majority_prior = train_counts / train_counts.sum()

    run_dir = OUTPUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    results = {}
    labels, probability = collect_probabilities(
        "majority", None, development_loader, device,
        majority_prior=majority_prior,
    )
    results["majority"] = classification_metrics(labels, probability)
    baseline_histories = {}
    checkpoint_sources = {}
    for name in ("target_visible", "current_direct_uad", "history_direct_uad"):
        model, history = train_baseline(
            name, seed, train_set, development_set, device
        )
        checkpoint_path = run_dir / f"{name}.pt"
        torch.save({
            "schema_version": "revealnav-mf2-representation-baseline/2",
            "name": name,
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "manifest_sha256": sha256_file(MANIFEST),
            "best_epoch": min(
                history, key=lambda row: row["development_loss"]
            )["epoch"],
        }, checkpoint_path)
        labels, probability = collect_probabilities(
            name, model, development_loader, device
        )
        results[name] = classification_metrics(labels, probability)
        baseline_histories[name] = history
        checkpoint_sources[name] = {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        }

    full_path = TRAINING / f"seed_{seed}/best_heads.pt"
    full_checkpoint = torch.load(
        full_path, map_location=device, weights_only=True
    )
    if full_checkpoint.get("manifest_sha256") != sha256_file(MANIFEST):
        raise RuntimeError("full REE checkpoint manifest mismatch")
    full = RevealOptionHeads(
        feature_dim=full_checkpoint["feature_dim"],
        hidden_dim=full_checkpoint["hidden_dim"],
        budget_count=full_checkpoint["budget_count"],
    ).to(device)
    full.load_state_dict(full_checkpoint["model_state_dict"], strict=True)
    labels, probability = collect_probabilities(
        "full_ree", full, development_loader, device,
        full_checkpoint=full_checkpoint,
    )
    results["full_ree"] = classification_metrics(labels, probability)
    value = {
        "schema_version": "revealnav-mf2-development-comparison/2",
        "status": "DEVELOPMENT_ENGINEERING_COMPARISON_COMPLETE",
        "seed": seed,
        "scope": (
            "development-set engineering diagnostic; development also selected "
            "checkpoints, so this is not an unbiased paper result"
        ),
        "sources": {
            "authorization": {
                "path": str(AUTHORIZATION.relative_to(ROOT)),
                "sha256": sha256_file(AUTHORIZATION),
            },
            "manifest": {
                "path": str(MANIFEST.relative_to(ROOT)),
                "sha256": sha256_file(MANIFEST),
            },
            "full_ree_checkpoint": {
                "path": str(full_path.relative_to(ROOT)),
                "sha256": sha256_file(full_path),
            },
            "baseline_checkpoints": checkpoint_sources,
        },
        "train_events": len(train_set),
        "development_events": len(development_set),
        "train_uad_prior": dict(zip(CLASS_NAMES, majority_prior.tolist())),
        "results": results,
        "baseline_training_history": baseline_histories,
        "decision_rule": (
            "argmax over U/A/D probabilities; Full REE probabilities are "
            "derived from sigmoid set, separation, and evidence outputs"
        ),
        "gold_read": False,
        "action_entropy_evaluated": False,
        "action_entropy_reason": (
            "authorized frozen feature shards do not contain frozen policy "
            "action probabilities"
        ),
        "paper_result": False,
    }
    output = run_dir / "comparison.json"
    atomic_json(output, value)
    print(json.dumps({
        "seed": seed,
        "status": value["status"],
        "macro_f1": {
            name: results[name]["macro_f1"] for name in MODEL_NAMES
        },
        "output": str(output.relative_to(ROOT)),
    }, indent=2))
    return 0


def aggregate() -> int:
    seed_results = []
    for seed in SEEDS:
        path = OUTPUT / f"seed_{seed}/comparison.json"
        value = json.loads(path.read_text())
        if value.get("status") != "DEVELOPMENT_ENGINEERING_COMPARISON_COMPLETE":
            raise RuntimeError(f"seed {seed} comparison is incomplete")
        seed_results.append(value)
    metrics = (
        "accuracy", "macro_f1", "nll", "brier", "ece_10bin",
        "false_ready_rate", "missed_ready_rate",
    )
    table = {}
    for name in MODEL_NAMES:
        table[name] = {}
        for metric in metrics:
            values = [row["results"][name][metric] for row in seed_results]
            table[name][metric] = {
                "mean": statistics.mean(values),
                "population_std": statistics.pstdev(values),
                "values": values,
            }
        for class_name in CLASS_NAMES:
            values = [
                row["results"][name]["per_class"][class_name]["f1"]
                for row in seed_results
            ]
            table[name][f"{class_name}_f1"] = {
                "mean": statistics.mean(values),
                "population_std": statistics.pstdev(values),
                "values": values,
            }
    winner = max(MODEL_NAMES, key=lambda name: table[name]["macro_f1"]["mean"])
    full = table["full_ree"]["macro_f1"]["mean"]
    best_baseline = max(
        (name for name in MODEL_NAMES if name != "full_ree"),
        key=lambda name: table[name]["macro_f1"]["mean"],
    )
    output = {
        "schema_version": "revealnav-mf2-development-comparison-aggregate/2",
        "status": "THREE_SEED_DEVELOPMENT_COMPARISON_COMPLETE",
        "scope": (
            "development engineering diagnostic only; no gold access, navigation "
            "claim, confidence interval, or paper result"
        ),
        "seeds": list(SEEDS),
        "models": list(MODEL_NAMES),
        "aggregate": table,
        "macro_f1_winner": winner,
        "strongest_non_full_baseline": best_baseline,
        "full_minus_strongest_baseline_macro_f1": (
            full - table[best_baseline]["macro_f1"]["mean"]
        ),
        "seed_sources": {
            str(row["seed"]): {
                "path": str((OUTPUT / f"seed_{row['seed']}/comparison.json").relative_to(ROOT)),
                "sha256": sha256_file(
                    OUTPUT / f"seed_{row['seed']}/comparison.json"
                ),
            }
            for row in seed_results
        },
        "gold_read": False,
        "action_entropy_evaluated": False,
        "paper_result": False,
    }
    path = OUTPUT / "RXR_MULTIBRANCH_REPRESENTATION_COMPARISON_V2.json"
    atomic_json(path, output)
    print(json.dumps({
        "status": output["status"],
        "macro_f1_winner": winner,
        "strongest_non_full_baseline": best_baseline,
        "full_minus_strongest_baseline_macro_f1": output[
            "full_minus_strongest_baseline_macro_f1"
        ],
        "macro_f1": {
            name: table[name]["macro_f1"] for name in MODEL_NAMES
        },
        "output": str(path.relative_to(ROOT)),
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int)
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.aggregate:
        return aggregate()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    return run_seed(args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
