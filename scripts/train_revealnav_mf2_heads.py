#!/usr/bin/env python3
"""Train MF2.1 causal heads from checksummed frozen feature shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/daiyang/vla").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2 import (  # noqa: E402
    RevealFeatureDataset,
    RevealOptionHeads,
    RevealOptionLoss,
    collate_reveal_examples,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(value: str, *, must_exist: bool) -> Path:
    path = Path(value).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("path resolves outside the project")
    if must_exist and (not path.is_file() or path.is_symlink()):
        raise ValueError("input path is missing or unsafe")
    return path


def manifest_gate(path: Path, smoke_test: bool) -> tuple[dict, list[float]]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != "revealnav-mf2-feature-manifest/1":
        raise ValueError("unsupported manifest schema")
    metadata = value.get("metadata", {})
    if smoke_test:
        if metadata.get("synthetic") is not True:
            raise ValueError("--smoke-test accepts synthetic manifests only")
    elif not (
        metadata.get("training_authorized") is True
        and metadata.get("causal_prefix_verified") is True
        and metadata.get("future_frames_used") == 0
        and metadata.get("full_candidate_sets") is True
    ):
        raise ValueError("real training manifest has not passed all authorization gates")
    budgets = metadata.get("normalized_budgets")
    if (
        not isinstance(budgets, list)
        or not budgets
        or not all(isinstance(item, (int, float)) for item in budgets)
    ):
        raise ValueError("manifest lacks normalized_budgets")
    scenes_by_split: dict[str, set[str]] = {}
    event_ids = set()
    for record in value.get("records", []):
        event_id = record.get("event_id")
        scene_id = record.get("scene_id")
        split = record.get("split")
        if not all(isinstance(item, str) and item for item in (
            event_id, scene_id, split
        )):
            raise ValueError("manifest identity fields are invalid")
        if event_id in event_ids:
            raise ValueError("duplicate event_id in feature manifest")
        event_ids.add(event_id)
        scenes_by_split.setdefault(split, set()).add(scene_id)
    splits = sorted(scenes_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            if scenes_by_split[left] & scenes_by_split[right]:
                raise ValueError("scene leakage across feature splits")
    return value, [float(item) for item in budgets]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def forward_loss(model, objective, batch, budgets):
    batch = move_batch(batch, next(model.parameters()).device)
    budget_tensor = budgets.view(1, 1, -1).expand(
        batch["history_embeddings"].shape[0],
        batch["history_embeddings"].shape[1],
        -1,
    )
    output = model(
        batch["history_embeddings"],
        batch["candidate_embeddings"],
        batch["candidate_mask"],
        budget_tensor,
        batch["instruction_embedding"],
    )
    return objective(output, batch)


def evaluate(model, objective, loader, budgets):
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            losses = forward_loss(model, objective, batch, budgets)
            batch_size = batch["history_embeddings"].shape[0]
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
    return {key: value / count for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    manifest_path = checked_path(args.manifest, must_exist=True)
    output_dir = checked_path(args.output_dir, must_exist=False)
    manifest, normalized_budgets = manifest_gate(
        manifest_path, args.smoke_test
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise ValueError("output directory must not be a symlink")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    train_set = RevealFeatureDataset(manifest_path, "train")
    development_set = RevealFeatureDataset(manifest_path, "development")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_reveal_examples,
    )
    development_loader = DataLoader(
        development_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_reveal_examples,
    )
    first = train_set[0]
    feature_dim = first["history_embeddings"].shape[-1]
    budget_count = first["current_feasibility"].shape[-1]
    if budget_count != len(normalized_budgets):
        raise ValueError("budget metadata does not match the training shards")
    model = RevealOptionHeads(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        budget_count=budget_count,
    ).to(device)
    objective = RevealOptionLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    budgets = torch.tensor(normalized_budgets, device=device)
    history = []
    best_total = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            losses = forward_loss(model, objective, batch, budgets)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = batch["history_embeddings"].shape[0]
            train_sum += float(losses["total"].detach()) * batch_size
            train_count += batch_size
        development = evaluate(
            model, objective, development_loader, budgets
        )
        epoch_record = {
            "epoch": epoch,
            "train_total": train_sum / train_count,
            "development": development,
        }
        history.append(epoch_record)
        if development["total"] < best_total:
            best_total = development["total"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    checkpoint_path = output_dir / "best_heads.pt"
    torch.save({
        "schema_version": "revealnav-mf2-head-checkpoint/1",
        "model_state_dict": best_state,
        "feature_dim": feature_dim,
        "hidden_dim": args.hidden_dim,
        "budget_count": budget_count,
        "normalized_budgets": normalized_budgets,
        "manifest_sha256": sha256_file(manifest_path),
    }, checkpoint_path)
    summary = {
        "schema_version": "revealnav-mf2-training-run/1",
        "status": "SMOKE_PASS" if args.smoke_test else "TRAINING_COMPLETE",
        "synthetic": bool(args.smoke_test),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "train_examples": len(train_set),
        "development_examples": len(development_set),
        "feature_dim": feature_dim,
        "budget_count": budget_count,
        "epochs": args.epochs,
        "seed": args.seed,
        "best_development_total": best_total,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history": history,
        "backbone_loaded": False,
        "future_frames_used": 0,
        "paper_result": False,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "summary": str(summary_path),
        "best_development_total": best_total,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
