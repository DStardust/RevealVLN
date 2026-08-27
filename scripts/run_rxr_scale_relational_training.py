#!/usr/bin/env python3
"""Train the scale REE and matched history baseline on Gold-free data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2 import (  # noqa: E402
    RevealFeatureDataset, RevealOptionLossConfig, collate_reveal_examples,
)
from revealnav_mf2r2 import (  # noqa: E402
    BalancedStructuredUADLoss, RelationalRevealOptionHeads,
)
from run_rxr_balanced_tuning_v2 import (  # noqa: E402
    STATE_KEYS, selection_key, training_weights,
)
from run_rxr_representation_comparison_v2 import (  # noqa: E402
    DirectBaseline, classification_metrics, collect_probabilities, move_batch,
    uad_labels,
)
from train_revealnav_mf2_heads import forward_loss  # noqa: E402


EXPANSION_ROOT = ROOT / "artifacts/phase1/rxr_train_expansion"
BASE = EXPANSION_ROOT / "scale_v2/model_training"
MANIFEST = EXPANSION_ROOT / "RXR_SCALE_AUTOMATIC_TRAINING_MANIFEST.json"
AUTHORIZATION = BASE / "RXR_SCALE_AUTOMATIC_TRAINING_AUTHORIZATION.json"
REFERENCE_ROOT = ROOT / "artifacts/evaluation/mf2_relational_augmented_v2"
REFERENCE = REFERENCE_ROOT / "RXR_RELATIONAL_AUGMENTED_COMPARISON_V2.json"
OUT = ROOT / "artifacts/evaluation/mf2_scale_relational_v1"
PROTOCOL = OUT / "RXR_SCALE_RELATIONAL_PROTOCOL_V1.json"
COMPARISON = OUT / "RXR_SCALE_RELATIONAL_COMPARISON_V1.json"
SEEDS = (20260826, 20260827, 20260828)
HIDDEN_DIM = 128
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


def protocol_value() -> dict:
    auth = json.loads(AUTHORIZATION.read_text())
    reference = json.loads(REFERENCE.read_text())
    if not (
        auth.get("status") == "AUTOMATIC_SCALE_TRAINING_AUTHORIZED"
        and auth.get("training_authorized") is True
        and auth["manifest"]["sha256"] == sha256_file(MANIFEST)
        and reference.get("status") == "RELATIONAL_AUGMENTATION_GATE_PASS"
        and reference.get("gold_payload_read") is False
    ):
        raise RuntimeError("scale relational protocol preconditions failed")
    return {
        "schema_version": "revealnav-mf2-scale-relational-protocol/1",
        "status": "SEALED_BEFORE_SCALE_MODEL_TRAINING",
        "seeds": list(SEEDS),
        "models": ["relational_ree", "history_direct_uad"],
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "source_sampling": {
            "human_audited_probability": 2 / 3,
            "automatic_probability": 1 / 3,
            "reason": "preserve the provenance mix of the previously successful augmentation while adding scale diversity",
            "shared_by_both_models": True,
        },
        "checkpoint_selection": "development macro-F1, then lower false-ready, then lower native loss",
        "development": "unchanged 68 human-audited scene-heldout events",
        "success_gates": {
            "relational_macro_f1_at_least_matched_history": True,
            "relational_false_ready_no_higher_than_matched_history": True,
            "scale_macro_f1_at_least_prior_relational": True,
            "scale_improves_prior_in_at_least_two_seeds": True,
            "scale_false_ready_degradation_vs_prior_at_most_0_02": True,
        },
        "sources": {
            str(MANIFEST.relative_to(ROOT)): sha256_file(MANIFEST),
            str(AUTHORIZATION.relative_to(ROOT)): sha256_file(AUTHORIZATION),
            str(REFERENCE.relative_to(ROOT)): sha256_file(REFERENCE),
        },
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed scale training protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def train_loader(dataset: RevealFeatureDataset, seed: int) -> DataLoader:
    human = [
        index for index, row in enumerate(dataset.records)
        if row.get("label_source") == "primary_human_audited"
    ]
    human_set = set(human)
    automatic = [index for index in range(len(dataset)) if index not in human_set]
    if not human or not automatic:
        raise RuntimeError("source-balanced training requires both provenance groups")
    weights = torch.empty(len(dataset), dtype=torch.double)
    weights[human] = (2 / 3) / len(human)
    weights[automatic] = (1 / 3) / len(automatic)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        dataset, batch_size=8, sampler=sampler,
        collate_fn=collate_reveal_examples,
    )


def development_loader(dataset: RevealFeatureDataset) -> DataLoader:
    return DataLoader(
        dataset, batch_size=16, shuffle=False,
        collate_fn=collate_reveal_examples,
    )


def evaluate_relational(model, loader, device):
    return classification_metrics(*collect_probabilities(
        "full_ree", model, loader, device,
        full_checkpoint={"normalized_budgets": [1.5, 2.0, 3.0, 4.0]},
    ))


def evaluate_history(model, loader, device):
    return classification_metrics(*collect_probabilities(
        "history_direct_uad", model, loader, device,
    ))


def train_relational(
    train_set, development_set, seed, state_weights, class_weights, device,
    candidate_count_encoding="batch_fraction",
):
    loader = train_loader(train_set, seed)
    heldout = development_loader(development_set)
    model = RelationalRevealOptionHeads(
        768, HIDDEN_DIM, 4, candidate_count_encoding
    ).to(device)
    objective = BalancedStructuredUADLoss(
        RevealOptionLossConfig(state_pos_weights=tuple(float(x) for x in state_weights)),
        tuple(float(x) for x in class_weights),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    budgets = torch.tensor([1.5, 2.0, 3.0, 4.0], device=device)
    best_key = best_state = best_metrics = None
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = count = 0
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = forward_loss(model, objective, batch, budgets)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite relational training loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(batch["step_mask"].sum())
            total += float(losses["total"].detach()) * size; count += size
        model.eval(); development_total = development_count = 0
        with torch.no_grad():
            for cpu_batch in heldout:
                batch = move_batch(cpu_batch, device)
                losses = forward_loss(model, objective, batch, budgets)
                size = int(batch["step_mask"].sum())
                development_total += float(losses["total"]) * size
                development_count += size
        metrics = evaluate_relational(model, heldout, device)
        native = development_total / development_count
        key = selection_key(metrics, native)
        history.append({"epoch": epoch, "train_total": total / count, "development_total": native, **{
            "development_macro_f1": metrics["macro_f1"],
            "development_false_ready_rate": metrics["false_ready_rate"],
            "development_missed_ready_rate": metrics["missed_ready_rate"],
        }})
        if best_key is None or key > best_key:
            best_key, best_metrics, stale = key, metrics, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state, strict=True)
    return model, best_metrics, history


def train_history(train_set, development_set, seed, class_weights, device):
    loader = train_loader(train_set, seed)
    heldout = development_loader(development_set)
    model = DirectBaseline(history_aware=True, output_dim=3, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    best_key = best_state = best_metrics = None
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = count = 0
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            mask = batch["step_mask"]
            loss = torch.nn.functional.cross_entropy(
                model(batch)[mask], uad_labels(batch)[mask], weight=weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite history training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(mask.sum()); total += float(loss.detach()) * size; count += size
        model.eval(); development_total = development_count = 0
        with torch.no_grad():
            for cpu_batch in heldout:
                batch = move_batch(cpu_batch, device)
                mask = batch["step_mask"]
                loss = torch.nn.functional.cross_entropy(
                    model(batch)[mask], uad_labels(batch)[mask], weight=weights,
                )
                size = int(mask.sum())
                development_total += float(loss) * size
                development_count += size
        metrics = evaluate_history(model, heldout, device)
        native = development_total / development_count
        key = selection_key(metrics, native)
        history.append({"epoch": epoch, "train_total": total / count, "development_total": native, **{
            "development_macro_f1": metrics["macro_f1"],
            "development_false_ready_rate": metrics["false_ready_rate"],
            "development_missed_ready_rate": metrics["missed_ready_rate"],
        }})
        if best_key is None or key > best_key:
            best_key, best_metrics, stale = key, metrics, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state, strict=True)
    return model, best_metrics, history


def train(seed: int, device: torch.device) -> int:
    if seed not in SEEDS or not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("training request is outside the sealed protocol")
    run_dir = OUT / f"seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    train_set = RevealFeatureDataset(MANIFEST, "train")
    development_set = RevealFeatureDataset(MANIFEST, "development")
    state_weights, class_weights, binary_counts, uad_counts = training_weights(train_set)
    relational, relational_metrics, relational_history = train_relational(
        train_set, development_set, seed, state_weights, class_weights, device,
    )
    history, history_metrics, history_trace = train_history(
        train_set, development_set, seed, class_weights, device,
    )
    checkpoints = {}
    for name, model in (("relational_ree", relational), ("history_direct_uad", history)):
        path = run_dir / f"{name}.pt"
        torch.save({
            "schema_version": "revealnav-mf2-scale-model-checkpoint/1",
            "model_name": name, "seed": seed, "hidden_dim": HIDDEN_DIM,
            "model_state_dict": model.state_dict(),
            "manifest_sha256": sha256_file(MANIFEST),
            "protocol_sha256": sha256_file(PROTOCOL),
        }, path)
        checkpoints[name] = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    result = {
        "schema_version": "revealnav-mf2-scale-model-run/1",
        "status": "SCALE_MODEL_RUN_COMPLETE",
        "seed": seed,
        "results": {"relational_ree": relational_metrics, "history_direct_uad": history_metrics},
        "training_history": {"relational_ree": relational_history, "history_direct_uad": history_trace},
        "train_counts": {"events": len(train_set), "binary": binary_counts, "uad": dict(zip(("U", "A", "D"), uad_counts.tolist()))},
        "checkpoints": checkpoints,
        "gold_feature_payload_read": False,
        "paper_result": False,
    }
    atomic_json(run_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "seed": seed, "relational_macro_f1": relational_metrics["macro_f1"], "history_macro_f1": history_metrics["macro_f1"]}, indent=2))
    return 0


def summary(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "population_std": statistics.pstdev(values), "values": values}


def aggregate() -> int:
    protocol_value()
    runs = [json.loads((OUT / f"seed_{seed}/result.json").read_text()) for seed in SEEDS]
    if any(row.get("status") != "SCALE_MODEL_RUN_COMPLETE" for row in runs):
        raise RuntimeError("one or more scale model runs are incomplete")
    metrics = ("accuracy", "macro_f1", "nll", "brier", "ece_10bin", "false_ready_rate", "missed_ready_rate")
    results = {
        model: {metric: summary([row["results"][model][metric] for row in runs]) for metric in metrics}
        for model in ("relational_ree", "history_direct_uad")
    }
    reference = json.loads(REFERENCE.read_text())["results"]["relational_augmented"]
    relational = results["relational_ree"]
    history = results["history_direct_uad"]
    deltas = [value - prior for value, prior in zip(relational["macro_f1"]["values"], reference["macro_f1"]["values"])]
    gates = {
        "relational_macro_f1_at_least_matched_history": relational["macro_f1"]["mean"] >= history["macro_f1"]["mean"],
        "relational_false_ready_no_higher_than_matched_history": relational["false_ready_rate"]["mean"] <= history["false_ready_rate"]["mean"],
        "scale_macro_f1_at_least_prior_relational": relational["macro_f1"]["mean"] >= reference["macro_f1"]["mean"],
        "scale_improves_prior_in_at_least_two_seeds": sum(delta > 0 for delta in deltas) >= 2,
        "scale_false_ready_degradation_vs_prior_at_most_0_02": relational["false_ready_rate"]["mean"] - reference["false_ready_rate"]["mean"] <= 0.02,
    }
    value = {
        "schema_version": "revealnav-mf2-scale-relational-comparison/1",
        "status": "SCALE_RELATIONAL_SCORE_GATE_PASS" if all(gates.values()) else "SCALE_RELATIONAL_SCORE_GATE_FAIL_RETAIN_PRIOR",
        "results": results,
        "prior_relational_reference": {metric: reference[metric] for metric in metrics},
        "scale_minus_prior_macro_f1": summary(deltas),
        "gates": gates,
        "selected_model": "scale_relational" if all(gates.values()) else "prior_relational_augmented",
        "gold_feature_payload_read": False,
        "paper_result": False,
    }
    atomic_json(COMPARISON, value)
    print(json.dumps({"status": value["status"], "gates": gates, "scale_minus_prior_macro_f1": value["scale_minus_prior_macro_f1"], "selected_model": value["selected_model"]}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--seed", type=int)
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if args.aggregate:
        return aggregate()
    return train(args.seed, torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
