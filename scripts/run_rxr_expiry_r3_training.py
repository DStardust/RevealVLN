#!/usr/bin/env python3
"""Train and gate the additive R3.1 expiry/Q adapters on development only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2 import (  # noqa: E402
    RevealFeatureDataset,
    collate_reveal_examples,
)
from revealnav_mf2r3 import (  # noqa: E402
    ExpiryQAdapterLoss,
    RelationalRevealExpiryHeads,
    RevealExpiryQFeatureDataset,
    collate_reveal_expiry_q_examples,
)
from run_rxr_balanced_tuning_v2 import STATE_KEYS, training_weights  # noqa: E402
from run_rxr_representation_comparison_v2 import (  # noqa: E402
    classification_metrics,
    move_batch,
    uad_labels,
)
SEEDS = (20260826, 20260827, 20260828)
CONDITIONS = ("primary", "augmented")
HIDDEN_DIM = 128
FEATURE_ROOT = ROOT / "artifacts/phase1/rxr_train_expansion/expiry_r3_qpair"
MANIFEST = FEATURE_ROOT / "RXR_EXPIRY_R3_Q_FEATURE_MANIFEST.json"
FEATURE_GATE = FEATURE_ROOT / "RXR_EXPIRY_R3_Q_FEATURE_GATE.json"
R2_COMPARISON = ROOT / (
    "artifacts/evaluation/mf2_relational_augmented_v2/"
    "RXR_RELATIONAL_AUGMENTED_COMPARISON_V2.json"
)
R2_FEATURE_MANIFEST = ROOT / (
    "artifacts/phase1/rxr_train_expansion/"
    "RXR_SECONDARY_AUGMENTED_FEATURE_MANIFEST_V1.json"
)
R2_PRIMARY_RUNS = ROOT / "artifacts/evaluation/mf2_relational_v2"
R2_AUGMENTED_RUNS = ROOT / "artifacts/evaluation/mf2_relational_augmented_v2"
REVISION = ROOT / "artifacts/design/MF2_IMPLEMENTATION_CORRECTNESS_REVISION_R3.md"
OPV_REVISION = ROOT / "artifacts/design/MF2_OPV_CONTRACT_CORRECTION_R3Q.md"
ADAPTER_REVISION = ROOT / (
    "artifacts/design/MF2_EXPIRY_ADDITIVE_ADAPTER_CORRECTION_R3_1.md"
)
FAILED_R3 = ROOT / (
    "artifacts/evaluation/mf2_expiry_r3/RXR_EXPIRY_R3_COMPARISON.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_expiry_r3_1"
PROTOCOL = OUT / "RXR_EXPIRY_R3_TRAINING_PROTOCOL.json"
COMPARISON = OUT / "RXR_EXPIRY_R3_COMPARISON.json"


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


def build_protocol() -> dict:
    gate = json.loads(FEATURE_GATE.read_text())
    r2 = json.loads(R2_COMPARISON.read_text())
    failed_r3 = json.loads(FAILED_R3.read_text())
    if not (
        gate.get("status") == "EXPIRY_R3_Q_FEATURE_GATE_PASS"
        and gate.get("training_authorized") is True
        and gate["manifest"]["sha256"] == sha256_file(MANIFEST)
        and r2.get("status") == "RELATIONAL_AUGMENTATION_GATE_PASS"
        and r2.get("selected_training_condition") == "relational_augmented"
        and failed_r3.get("status") == "EXPIRY_R3_GATE_FAIL"
        and failed_r3.get("gold_payload_read") is False
    ):
        raise RuntimeError("R3 training protocol preconditions failed")
    r2_augmented = r2["results"]["relational_augmented"]
    return {
        "schema_version": "revealnav-mf2-expiry-training-protocol/3.1",
        "status": "SEALED_BEFORE_EXPIRY_R3_1_ADAPTER_TRAINING",
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "hidden_dim": HIDDEN_DIM,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "expiry_loss_weight": 1.0,
        "epoch_limit": 20,
        "early_stopping_patience": 4,
        "checkpoint_selection": "minimum development adapter loss",
        "inherited_r2_parameters": "loaded exactly and frozen",
        "trainable_modules": [
            "expiry_temporal", "expiry_head", "no_checkpoint_delta_head"
        ],
        "decision_evaluation_horizon": (
            "original R2 feature horizon; expiry uses the extended R3 horizon"
        ),
        "sources": {
            str(MANIFEST.relative_to(ROOT)): sha256_file(MANIFEST),
            str(FEATURE_GATE.relative_to(ROOT)): sha256_file(FEATURE_GATE),
            str(R2_COMPARISON.relative_to(ROOT)): sha256_file(R2_COMPARISON),
            str(REVISION.relative_to(ROOT)): sha256_file(REVISION),
            str(OPV_REVISION.relative_to(ROOT)): sha256_file(OPV_REVISION),
            str(ADAPTER_REVISION.relative_to(ROOT)): sha256_file(ADAPTER_REVISION),
            str(FAILED_R3.relative_to(ROOT)): sha256_file(FAILED_R3),
            str(R2_FEATURE_MANIFEST.relative_to(ROOT)): sha256_file(
                R2_FEATURE_MANIFEST
            ),
        },
        "r2_reference": {
            "augmented_macro_f1_mean": r2_augmented["macro_f1"]["mean"],
            "augmented_false_ready_mean": r2_augmented[
                "false_ready_rate"
            ]["mean"],
        },
        "success_gates": {
            "expiry_mae_beats_train_median_in_at_least_two_seeds": True,
            "expiry_hazard_auc_above_0_5_in_at_least_two_seeds": True,
            "uad_macro_f1_max_degradation": 0.02,
            "false_ready_max_degradation": 0.02,
            "augmented_macro_f1_at_least_primary": True,
        },
        "gold_access_allowed": False,
        "additional_hyperparameter_search_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = build_protocol()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != value:
            raise RuntimeError("sealed R3 training protocol drift")
    else:
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "conditions": value["conditions"],
        "seeds": value["seeds"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def datasets(condition: str):
    train_all = RevealExpiryQFeatureDataset(MANIFEST, "train")
    development = RevealExpiryQFeatureDataset(MANIFEST, "development")
    decision_development = RevealFeatureDataset(
        R2_FEATURE_MANIFEST, "development"
    )
    if condition == "primary":
        indices = [
            index for index, row in enumerate(train_all.records)
            if row["label_source"] == "primary_human_audited"
        ]
        train = Subset(train_all, indices)
    elif condition == "augmented":
        train = train_all
    else:
        raise ValueError("unknown R3 condition")
    return train, development, decision_development


def loaders(train, development, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            train, batch_size=8, shuffle=True, generator=generator,
            collate_fn=collate_reveal_expiry_q_examples,
        ),
        DataLoader(
            development, batch_size=16, shuffle=False,
            collate_fn=collate_reveal_expiry_q_examples,
        ),
    )


def factorized_probabilities(output, mask):
    target_set = torch.sigmoid(output.target_in_set_logit[mask])
    decisive = (
        torch.sigmoid(output.separation_logit[mask])
        * torch.sigmoid(output.evidence_logit[mask])
    )
    return torch.stack((
        1.0 - target_set,
        target_set * (1.0 - decisive),
        target_set * decisive,
    ), dim=-1)


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
        start = stop
    positive = labels == 1
    positive_count = int(positive.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise RuntimeError("expiry AUROC requires both classes")
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def train_median_offset(dataset) -> float:
    offsets = []
    for example in dataset:
        matches = torch.where(example["expiry_hazard"] == 1)[0]
        if len(matches):
            offsets.append(int(matches[0]))
    if not offsets:
        raise RuntimeError("training split has no observed expiry events")
    return float(statistics.median(offsets))


def evaluate(
    model, expiry_dataset, decision_dataset, device: torch.device,
    median_offset: float,
) -> dict:
    decision_loader = DataLoader(
        decision_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_reveal_examples,
    )
    all_labels, all_probabilities = [], []
    model.eval()
    with torch.no_grad():
        for cpu_batch in decision_loader:
            batch = move_batch(cpu_batch, device)
            mask = batch["step_mask"]
            steps = int(mask.sum())
            budgets = torch.tensor(
                [1.5, 2.0, 3.0, 4.0], device=device
            ).view(1, 1, -1).expand(1, steps, -1)
            output = model(
                batch["history_embeddings"],
                batch["candidate_embeddings"],
                batch["candidate_mask"],
                budgets,
                batch["instruction_embedding"],
            )
            all_labels.append(uad_labels(batch)[mask].cpu().numpy())
            all_probabilities.append(
                factorized_probabilities(output, mask).cpu().numpy()
            )
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    metrics = classification_metrics(labels, probabilities)

    expiry_loader = DataLoader(
        expiry_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_reveal_expiry_q_examples,
    )
    hazard_labels, hazard_scores = [], []
    onset_errors, baseline_errors = [], []
    observed = 0
    censored = 0
    model.eval()
    with torch.no_grad():
        for cpu_batch in expiry_loader:
            batch = move_batch(cpu_batch, device)
            mask = batch["step_mask"]
            steps = int(mask.sum())
            budgets = torch.tensor(
                [1.5, 2.0, 3.0, 4.0], device=device
            ).view(1, 1, -1).expand(1, steps, -1)
            output = model(
                batch["history_embeddings"],
                batch["candidate_embeddings"],
                batch["candidate_mask"],
                budgets,
                batch["instruction_embedding"],
            )
            expiry = batch["expiry_hazard"][0, :steps]
            valid = expiry >= 0
            hazards = torch.sigmoid(output.expiry_hazard_logit[0, :steps])
            hazard_labels.append(expiry[valid].cpu().numpy().astype(np.int64))
            hazard_scores.append(hazards[valid].cpu().numpy())
            event = torch.where(expiry == 1)[0]
            if len(event):
                observed += 1
                valid_hazards = hazards[valid]
                survival_before = torch.cumprod(torch.cat((
                    torch.ones(1, device=device),
                    1.0 - valid_hazards[:-1],
                )), dim=0)
                predicted = int(torch.argmax(survival_before * valid_hazards))
                truth = int(event[0])
                onset_errors.append(abs(predicted - truth))
                baseline_errors.append(abs(
                    min(int(round(median_offset)), steps - 1) - truth
                ))
            else:
                censored += 1
    expiry_labels = np.concatenate(hazard_labels)
    expiry_scores = np.concatenate(hazard_scores)
    clipped = np.clip(expiry_scores, 1e-8, 1.0 - 1e-8)
    metrics["expiry"] = {
        "hazard_auc": rank_auc(expiry_labels, expiry_scores),
        "hazard_nll": float(-(
            expiry_labels * np.log(clipped)
            + (1 - expiry_labels) * np.log(1 - clipped)
        ).mean()),
        "event_time_mae": float(statistics.mean(onset_errors)),
        "train_median_baseline_mae": float(statistics.mean(baseline_errors)),
        "observed_events": observed,
        "right_censored_events": censored,
        "valid_prefixes": int(len(expiry_labels)),
    }
    return metrics


ADAPTER_PREFIXES = (
    "expiry_temporal.", "expiry_head.", "no_checkpoint_delta_head."
)


def inherited_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(ADAPTER_PREFIXES):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialize_additive_model(
    condition: str, seed: int, device: torch.device,
) -> tuple[RelationalRevealExpiryHeads, Path, str]:
    base = R2_PRIMARY_RUNS if condition == "primary" else R2_AUGMENTED_RUNS
    checkpoint = base / f"seed_{seed}/relational_full_ree.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = RelationalRevealExpiryHeads(768, HIDDEN_DIM, 4)
    incompat = model.load_state_dict(payload["model_state_dict"], strict=False)
    expected_missing = {
        name for name in model.state_dict() if name.startswith(ADAPTER_PREFIXES)
    }
    if set(incompat.missing_keys) != expected_missing or incompat.unexpected_keys:
        raise RuntimeError("R2-to-R3.1 exact load coverage failed")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(ADAPTER_PREFIXES))
    if not all(
        parameter.requires_grad == name.startswith(ADAPTER_PREFIXES)
        for name, parameter in model.named_parameters()
    ):
        raise RuntimeError("R3.1 trainable-parameter boundary failed")
    model.to(device)
    return model, checkpoint, inherited_state_sha256(model)


def adapter_forward_loss(model, objective, batch, budgets):
    batch_size, steps = batch["history_embeddings"].shape[:2]
    normalized_budgets = budgets.view(1, 1, -1).expand(
        batch_size, steps, -1
    )
    output = model(
        batch["history_embeddings"], batch["candidate_embeddings"],
        batch["candidate_mask"], normalized_budgets,
        batch["instruction_embedding"],
    )
    return objective(output, batch)


def run(condition: str, seed: int, device: torch.device) -> int:
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError("R3 run is outside the frozen protocol")
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != build_protocol():
        raise RuntimeError("R3 training protocol must be sealed without drift")
    run_dir = OUT / f"{condition}_seed_{seed}"
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    train_set, development_set, decision_development_set = datasets(condition)
    median_offset = train_median_offset(train_set)
    state_weights, class_weights, binary_counts, uad_counts = training_weights(
        train_set
    )
    train_loader, development_loader = loaders(train_set, development_set, seed)
    model, r2_checkpoint, inherited_sha_before = initialize_additive_model(
        condition, seed, device
    )
    objective = ExpiryQAdapterLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-4, weight_decay=1e-4,
    )
    budgets = torch.tensor([1.5, 2.0, 3.0, 4.0], device=device)
    history = []
    best_key = None
    best_state = None
    stale_epochs = 0
    for epoch in range(1, 21):
        model.train()
        train_total = 0.0
        train_count = 0
        for cpu_batch in train_loader:
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = adapter_forward_loss(model, objective, batch, budgets)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite R3 loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(batch["step_mask"].sum())
            train_total += float(losses["total"].detach()) * size
            train_count += size
        model.eval()
        development_total = 0.0
        development_count = 0
        with torch.no_grad():
            for cpu_batch in development_loader:
                batch = move_batch(cpu_batch, device)
                losses = adapter_forward_loss(model, objective, batch, budgets)
                size = int(batch["step_mask"].sum())
                development_total += float(losses["total"]) * size
                development_count += size
        metrics = evaluate(
            model, development_set, decision_development_set, device,
            median_offset,
        )
        native_loss = development_total / development_count
        key = -native_loss
        history.append({
            "epoch": epoch,
            "train_total": train_total / train_count,
            "development_total": native_loss,
            "development_macro_f1": metrics["macro_f1"],
            "development_false_ready_rate": metrics["false_ready_rate"],
            "development_expiry_auc": metrics["expiry"]["hazard_auc"],
            "development_expiry_event_time_mae": metrics["expiry"][
                "event_time_mae"
            ],
        })
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 4:
            break
    model.load_state_dict(best_state, strict=True)
    final_metrics = evaluate(
        model, development_set, decision_development_set, device,
        median_offset,
    )
    inherited_sha_after = inherited_state_sha256(model)
    if inherited_sha_after != inherited_sha_before:
        raise RuntimeError("frozen R2 decision state changed during adapter training")
    checkpoint = run_dir / "relational_expiry_ree.pt"
    torch.save({
        "schema_version": "revealnav-mf2-expiry-checkpoint/3.1",
        "condition": condition,
        "seed": seed,
        "hidden_dim": HIDDEN_DIM,
        "model_state_dict": model.state_dict(),
        "manifest_sha256": sha256_file(MANIFEST),
        "protocol_sha256": sha256_file(PROTOCOL),
        "train_median_expiry_offset": median_offset,
        "inherited_r2_checkpoint_sha256": sha256_file(r2_checkpoint),
        "inherited_state_sha256": inherited_sha_after,
    }, checkpoint)
    value = {
        "schema_version": "revealnav-mf2-expiry-run/3.1",
        "status": "EXPIRY_R3_1_RUN_COMPLETE",
        "condition": condition,
        "seed": seed,
        "hidden_dim": HIDDEN_DIM,
        "train_events": len(train_set),
        "development_events": len(development_set),
        "decision_development_events": len(decision_development_set),
        "train_median_expiry_offset": median_offset,
        "metrics": final_metrics,
        "training_history": history,
        "train_counts": {
            "binary": binary_counts,
            "uad": dict(zip(("U", "A", "D"), uad_counts.tolist())),
        },
        "weights": {
            "state": dict(zip(STATE_KEYS, state_weights)),
            "uad": dict(zip(("U", "A", "D"), class_weights.tolist())),
            "expiry": 1.0,
        },
        "adapter_boundary": {
            "inherited_r2_checkpoint": str(r2_checkpoint.relative_to(ROOT)),
            "inherited_r2_checkpoint_sha256": sha256_file(r2_checkpoint),
            "inherited_state_sha256_before": inherited_sha_before,
            "inherited_state_sha256_after": inherited_sha_after,
            "inherited_state_unchanged": inherited_sha_before == inherited_sha_after,
            "trainable_prefixes": list(ADAPTER_PREFIXES),
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "gold_payload_read": False,
        "paper_result": False,
    }
    atomic_json(run_dir / "result.json", value)
    print(json.dumps({
        "status": value["status"],
        "condition": condition,
        "seed": seed,
        "macro_f1": final_metrics["macro_f1"],
        "false_ready": final_metrics["false_ready_rate"],
        "expiry": final_metrics["expiry"],
    }, indent=2))
    return 0


def summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "values": values,
    }


def aggregate() -> int:
    protocol = build_protocol()
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol:
        raise RuntimeError("R3 training protocol drift")
    runs = {}
    for condition in CONDITIONS:
        runs[condition] = []
        for seed in SEEDS:
            path = OUT / f"{condition}_seed_{seed}/result.json"
            value = json.loads(path.read_text())
            if value.get("status") != "EXPIRY_R3_1_RUN_COMPLETE":
                raise RuntimeError("incomplete R3 run")
            runs[condition].append(value)
    metric_paths = {
        "macro_f1": ("metrics", "macro_f1"),
        "accuracy": ("metrics", "accuracy"),
        "false_ready_rate": ("metrics", "false_ready_rate"),
        "missed_ready_rate": ("metrics", "missed_ready_rate"),
        "expiry_hazard_auc": ("metrics", "expiry", "hazard_auc"),
        "expiry_event_time_mae": ("metrics", "expiry", "event_time_mae"),
        "expiry_train_median_baseline_mae": (
            "metrics", "expiry", "train_median_baseline_mae"
        ),
    }

    def read(row: dict, path: tuple[str, ...]):
        value = row
        for key in path:
            value = value[key]
        return value

    results = {
        condition: {
            name: summary([read(row, path) for row in rows])
            for name, path in metric_paths.items()
        }
        for condition, rows in runs.items()
    }
    augmented = results["augmented"]
    primary = results["primary"]
    reference = protocol["r2_reference"]
    gates = {
        "expiry_mae_beats_train_median_in_at_least_two_seeds": sum(
            learned < baseline for learned, baseline in zip(
                augmented["expiry_event_time_mae"]["values"],
                augmented["expiry_train_median_baseline_mae"]["values"],
            )
        ) >= 2,
        "expiry_hazard_auc_above_0_5_in_at_least_two_seeds": sum(
            value > 0.5 for value in augmented["expiry_hazard_auc"]["values"]
        ) >= 2,
        "uad_macro_f1_degradation_at_most_0_02": (
            augmented["macro_f1"]["mean"]
            >= reference["augmented_macro_f1_mean"] - 0.02
        ),
        "false_ready_degradation_at_most_0_02": (
            augmented["false_ready_rate"]["mean"]
            <= reference["augmented_false_ready_mean"] + 0.02
        ),
        "augmented_macro_f1_at_least_primary": (
            augmented["macro_f1"]["mean"] >= primary["macro_f1"]["mean"]
        ),
        "inherited_r2_state_bit_exact_in_all_runs": all(
            row["adapter_boundary"]["inherited_state_unchanged"]
            for rows in runs.values() for row in rows
        ),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-expiry-comparison/3.1",
        "status": "EXPIRY_R3_1_GATE_PASS" if passed else "EXPIRY_R3_1_GATE_FAIL",
        "scope": "fixed train/development implementation-correctness revision",
        "results": results,
        "gates": gates,
        "selected_condition": "augmented" if passed else None,
        "sources": {
            "protocol_sha256": sha256_file(PROTOCOL),
            "manifest_sha256": sha256_file(MANIFEST),
        },
        "gold_payload_read": False,
        "paper_result": False,
        "next_step": (
            "learned ECOG/OPP integration" if passed
            else "versioned expiry correctness diagnosis without Gold access"
        ),
    }
    atomic_json(COMPARISON, value)
    print(json.dumps({
        "status": value["status"],
        "results": results,
        "gates": gates,
        "selected": value["selected_condition"],
        "output": str(COMPARISON.relative_to(ROOT)),
    }, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if args.aggregate:
        return aggregate()
    if args.condition is None or args.seed is None:
        parser.error("--condition and --seed are required with --run")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return run(args.condition, args.seed, device)


if __name__ == "__main__":
    raise SystemExit(main())
