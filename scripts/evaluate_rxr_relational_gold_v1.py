#!/usr/bin/env python3
"""Seal and run the one-shot held-out Gold pilot for frozen REE models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2 import (  # noqa: E402
    RevealFeatureDataset,
    RevealOptionHeads,
    collate_reveal_examples,
)
from revealnav_mf2r2 import RelationalRevealOptionHeads  # noqa: E402
from run_rxr_representation_comparison_v2 import (  # noqa: E402
    DirectBaseline,
    classification_metrics,
    move_batch,
    uad_labels,
)


SEEDS = (20260826, 20260827, 20260828)
CONDITIONS = (
    "frozen_full_primary",
    "frozen_full_augmented",
    "history_primary",
    "history_augmented",
    "relational_primary",
    "relational_augmented",
)
V2 = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2"
MANIFEST = V2 / "RXR_MULTIBRANCH_FEATURE_MANIFEST_V2_AUTHORIZED.json"
AUTHORIZATION = V2 / "RXR_MULTIBRANCH_TRAINING_AUTHORIZATION_V2.json"
RELATIONAL_PRIMARY = ROOT / "artifacts/evaluation/mf2_relational_v2"
RELATIONAL_AUGMENTED = ROOT / "artifacts/evaluation/mf2_relational_augmented_v2"
FROZEN_PRIMARY = ROOT / "artifacts/evaluation/mf2_balanced_tuning_v2"
FROZEN_AUGMENTED = ROOT / "artifacts/evaluation/mf2_secondary_augmentation_v1"
SELECTION = RELATIONAL_AUGMENTED / "RXR_RELATIONAL_AUGMENTED_COMPARISON_V2.json"
OUT = ROOT / "artifacts/evaluation/mf2_relational_gold_v1"
PROTOCOL = OUT / "RXR_RELATIONAL_GOLD_PROTOCOL_V1.json"
RESULT = OUT / "RXR_RELATIONAL_GOLD_RESULT_V1.json"
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260826
SCALAR_METRICS = (
    "accuracy",
    "macro_f1",
    "nll",
    "brier",
    "ece_10bin",
    "false_ready_rate",
    "missed_ready_rate",
)


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


def source_for(condition: str, seed: int) -> dict:
    if condition == "relational_primary":
        result_path = RELATIONAL_PRIMARY / f"seed_{seed}/result.json"
        expected_status = "RELATIONAL_PRIMARY_RUN_COMPLETE"
        checkpoint_key = None
    elif condition == "relational_augmented":
        result_path = RELATIONAL_AUGMENTED / f"seed_{seed}/result.json"
        expected_status = "RELATIONAL_AUGMENTED_RUN_COMPLETE"
        checkpoint_key = None
    elif condition in ("frozen_full_primary", "history_primary"):
        result_path = FROZEN_PRIMARY / f"h128_seed_{seed}/result.json"
        expected_status = "BALANCED_TUNING_RUN_COMPLETE"
        checkpoint_key = (
            "balanced_full_ree"
            if condition == "frozen_full_primary"
            else "balanced_history_direct_uad"
        )
    else:
        result_path = FROZEN_AUGMENTED / f"seed_{seed}/result.json"
        expected_status = "AUGMENTED_DEVELOPMENT_RUN_COMPLETE"
        checkpoint_key = (
            "balanced_full_ree"
            if condition == "frozen_full_augmented"
            else "balanced_history_direct_uad"
        )
    value = json.loads(result_path.read_text())
    if value.get("status") != expected_status or value.get("seed") != seed:
        raise RuntimeError(f"invalid frozen run source: {condition} seed {seed}")
    reference = (
        value["checkpoint"] if checkpoint_key is None
        else value["checkpoints"][checkpoint_key]
    )
    checkpoint = (ROOT / reference["path"]).resolve()
    if (
        ROOT not in checkpoint.parents
        or checkpoint.is_symlink()
        or not checkpoint.is_file()
        or checkpoint.stat().st_size != reference["bytes"]
        or sha256_file(checkpoint) != reference["sha256"]
    ):
        raise RuntimeError(f"checkpoint provenance failure: {condition} seed {seed}")
    return {
        "condition": condition,
        "seed": seed,
        "path": str(checkpoint.relative_to(ROOT)),
        "bytes": checkpoint.stat().st_size,
        "sha256": reference["sha256"],
        "run_result": {
            "path": str(result_path.relative_to(ROOT)),
            "sha256": sha256_file(result_path),
        },
    }


def build_protocol() -> dict:
    authorization = json.loads(AUTHORIZATION.read_text())
    selection = json.loads(SELECTION.read_text())
    manifest = json.loads(MANIFEST.read_text())
    gold_count = sum(row.get("split") == "gold" for row in manifest["records"])
    if not (
        authorization.get("status") == "TRAINING_AUTHORIZATION_PASS"
        and authorization["training_manifest"]["sha256"] == sha256_file(MANIFEST)
        and selection.get("status") == "RELATIONAL_AUGMENTATION_GATE_PASS"
        and selection.get("selected_training_condition") == "relational_augmented"
        and gold_count == 107
    ):
        raise RuntimeError("Gold protocol preconditions failed")
    checkpoints = {
        f"{condition}/seed_{seed}": source_for(condition, seed)
        for condition in CONDITIONS
        for seed in SEEDS
    }
    return {
        "schema_version": "revealnav-mf2-relational-gold-protocol/1",
        "status": "SEALED_BEFORE_GOLD_MODEL_EVALUATION",
        "scope": "one-shot held-out pilot; not the final 600-event paper Gold test",
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)),
            "sha256": sha256_file(MANIFEST),
            "split": "gold",
            "events": gold_count,
        },
        "selection_source": {
            "path": str(SELECTION.relative_to(ROOT)),
            "sha256": sha256_file(SELECTION),
            "selected_condition": "relational_augmented",
        },
        "authorization_source": {
            "path": str(AUTHORIZATION.relative_to(ROOT)),
            "sha256": sha256_file(AUTHORIZATION),
            "three_reviewer_agreement_measured": False,
            "full_submission_gate_satisfied": False,
        },
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "checkpoints": checkpoints,
        "decision_rule": "argmax of causally derived U/A/D probabilities",
        "primary_comparison": "relational_augmented minus relational_primary",
        "selection_gates": {
            "macro_f1_mean_nonnegative_delta": True,
            "macro_f1_improves_in_at_least_two_seeds": True,
            "false_ready_mean_degradation_max": 0.02,
        },
        "statistical_strength_flag": "event-paired macro-F1 95% CI lower bound > 0",
        "bootstrap": {
            "unit": "event",
            "paired": True,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "gold_access_allowed_after_seal": True,
        "retraining_or_threshold_tuning_allowed_after_gold": False,
        "output_overwrite_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = build_protocol()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != value:
            raise RuntimeError("existing Gold protocol drift")
    else:
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "gold_events": value["manifest"]["events"],
        "conditions": len(value["conditions"]),
        "checkpoints": len(value["checkpoints"]),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def load_model(condition: str, seed: int, device: torch.device):
    source = source_for(condition, seed)
    checkpoint = torch.load(
        ROOT / source["path"], map_location=device, weights_only=True
    )
    if checkpoint.get("seed") != seed or checkpoint.get("hidden_dim") != 128:
        raise RuntimeError(f"checkpoint metadata mismatch: {condition} {seed}")
    if condition.startswith("relational_"):
        if checkpoint.get("schema_version") != "revealnav-mf2-relational-checkpoint/2":
            raise RuntimeError("unexpected relational checkpoint schema")
        model = RelationalRevealOptionHeads(768, 128, 4)
        model_kind = "full"
    elif condition.startswith("frozen_full_"):
        if checkpoint.get("model_name") != "balanced_full_ree":
            raise RuntimeError("unexpected frozen Full checkpoint")
        model = RevealOptionHeads(768, 128, 4)
        model_kind = "full"
    else:
        if checkpoint.get("model_name") != "balanced_history_direct_uad":
            raise RuntimeError("unexpected history checkpoint")
        model = DirectBaseline(history_aware=True, output_dim=3, hidden_dim=128)
        model_kind = "history"
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, model_kind, source


def predict_batch(model, model_kind: str, cpu_batch: dict, device: torch.device):
    batch = move_batch(cpu_batch, device)
    mask = batch["step_mask"]
    labels = uad_labels(batch)[mask]
    with torch.no_grad():
        if model_kind == "history":
            probability = torch.softmax(model(batch)[mask], dim=-1)
        else:
            batch_size, steps = mask.shape
            budgets = torch.tensor(
                [1.5, 2.0, 3.0, 4.0], device=device
            ).view(1, 1, -1).expand(batch_size, steps, -1)
            output = model(
                batch["history_embeddings"],
                batch["candidate_embeddings"],
                batch["candidate_mask"],
                budgets,
                batch["instruction_embedding"],
            )
            target_set = torch.sigmoid(output.target_in_set_logit[mask])
            decisive = (
                torch.sigmoid(output.separation_logit[mask])
                * torch.sigmoid(output.evidence_logit[mask])
            )
            probability = torch.stack((
                1.0 - target_set,
                target_set * (1.0 - decisive),
                target_set * decisive,
            ), dim=-1)
    return labels.cpu().numpy(), probability.cpu().numpy()


def confusion(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    predictions = probabilities.argmax(1)
    return np.bincount(
        labels * 3 + predictions, minlength=9
    ).reshape(3, 3)


def confusion_rates(matrix: np.ndarray) -> tuple[float, float, float]:
    f1_values = []
    for index in range(3):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    not_ready = matrix[:2, :].sum()
    ready = matrix[2, :].sum()
    return (
        float(np.mean(f1_values)),
        float(matrix[:2, 2].sum() / not_ready),
        float(matrix[2, :2].sum() / ready),
    )


def summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "values": values,
    }


def bootstrap_deltas(event_confusions: dict) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    event_count = event_confusions["relational_primary"][SEEDS[0]].shape[0]
    macro_deltas = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    false_deltas = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    missed_deltas = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for offset in range(BOOTSTRAP_RESAMPLES):
        sampled = rng.integers(0, event_count, size=event_count)
        per_seed = []
        for seed in SEEDS:
            primary = confusion_rates(
                event_confusions["relational_primary"][seed][sampled].sum(0)
            )
            augmented = confusion_rates(
                event_confusions["relational_augmented"][seed][sampled].sum(0)
            )
            per_seed.append(tuple(a - p for a, p in zip(augmented, primary)))
        values = np.asarray(per_seed).mean(0)
        macro_deltas[offset], false_deltas[offset], missed_deltas[offset] = values

    def interval(values: np.ndarray) -> dict:
        lower, upper = np.quantile(values, (0.025, 0.975))
        return {"lower": float(lower), "upper": float(upper)}

    return {
        "macro_f1_delta": interval(macro_deltas),
        "false_ready_rate_delta": interval(false_deltas),
        "missed_ready_rate_delta": interval(missed_deltas),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "event",
        "paired": True,
    }


def evaluate(device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != build_protocol():
        raise RuntimeError("Gold protocol must be sealed without drift before evaluation")
    if RESULT.exists():
        raise RuntimeError("refusing to overwrite the one-shot Gold result")
    dataset = RevealFeatureDataset(MANIFEST, "gold")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_reveal_examples,
    )
    event_confusions = {condition: {} for condition in CONDITIONS}
    per_seed_metrics = {condition: {} for condition in CONDITIONS}
    checkpoint_sources = {}
    reference_labels = None
    for condition in CONDITIONS:
        for seed in SEEDS:
            model, model_kind, source = load_model(condition, seed, device)
            event_rows = [predict_batch(model, model_kind, batch, device)
                          for batch in loader]
            labels = np.concatenate([row[0] for row in event_rows])
            probabilities = np.concatenate([row[1] for row in event_rows])
            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(labels, reference_labels):
                raise RuntimeError("Gold label order changed between conditions")
            event_confusions[condition][seed] = np.stack([
                confusion(row_labels, row_probabilities)
                for row_labels, row_probabilities in event_rows
            ])
            per_seed_metrics[condition][str(seed)] = classification_metrics(
                labels, probabilities
            )
            checkpoint_sources[f"{condition}/seed_{seed}"] = source
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    aggregate = {
        condition: {
            metric: summary([
                per_seed_metrics[condition][str(seed)][metric]
                for seed in SEEDS
            ])
            for metric in SCALAR_METRICS
        }
        for condition in CONDITIONS
    }
    primary = aggregate["relational_primary"]
    augmented = aggregate["relational_augmented"]
    macro_delta_values = [
        augmented["macro_f1"]["values"][index]
        - primary["macro_f1"]["values"][index]
        for index in range(len(SEEDS))
    ]
    false_delta = (
        augmented["false_ready_rate"]["mean"]
        - primary["false_ready_rate"]["mean"]
    )
    bootstrap = bootstrap_deltas(event_confusions)
    gates = {
        "all_18_frozen_checkpoints_evaluated": len(checkpoint_sources) == 18,
        "macro_f1_mean_nonnegative_delta": statistics.mean(macro_delta_values) >= 0,
        "macro_f1_improves_in_at_least_two_seeds": sum(
            value > 0 for value in macro_delta_values
        ) >= 2,
        "false_ready_mean_degradation_at_most_0_02": false_delta <= 0.02,
    }
    directional_pass = all(gates.values())
    strong = bootstrap["macro_f1_delta"]["lower"] > 0
    if directional_pass and strong:
        status = "GOLD_PILOT_STRONG_PASS"
    elif directional_pass:
        status = "GOLD_PILOT_DIRECTIONAL_PASS"
    else:
        status = "GOLD_PILOT_FAIL_RETAIN_PRIMARY_RELATIONAL"
    event_order = [row["event_id"] for row in dataset.records]
    value = {
        "schema_version": "revealnav-mf2-relational-gold-result/1",
        "status": status,
        "scope": "one-shot 107-event held-out Gold pilot",
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": sha256_file(PROTOCOL),
        },
        "counts": {
            "events": len(dataset),
            "prefixes": int(len(reference_labels)),
            "conditions": len(CONDITIONS),
            "seeds": len(SEEDS),
        },
        "aggregate": aggregate,
        "per_seed_metrics": per_seed_metrics,
        "selected_comparison": {
            "augmented_minus_primary_macro_f1": summary(macro_delta_values),
            "augmented_minus_primary_false_ready_rate_mean": false_delta,
            "augmented_minus_primary_missed_ready_rate_mean": (
                augmented["missed_ready_rate"]["mean"]
                - primary["missed_ready_rate"]["mean"]
            ),
        },
        "event_paired_bootstrap_95pct": bootstrap,
        "selection_gates": gates,
        "statistical_strength_flag": strong,
        "selected_condition_after_gold": (
            "relational_augmented" if directional_pass else "relational_primary"
        ),
        "event_order_sha256": hashlib.sha256(
            "\n".join(event_order).encode()
        ).hexdigest(),
        "checkpoint_sources": checkpoint_sources,
        "gold_payload_read": True,
        "secondary_evaluation_labels_used": False,
        "retraining_or_threshold_tuning_after_gold": False,
        "three_reviewer_agreement_measured": False,
        "full_submission_gate_satisfied": False,
        "paper_result": False,
        "next_step": (
            "implement frozen ECOG/OPP online integration"
            if directional_pass
            else "retain primary relational model and diagnose without Gold tuning"
        ),
    }
    atomic_json(RESULT, value)
    print(json.dumps({
        "status": status,
        "gold_events": len(dataset),
        "gold_prefixes": len(reference_labels),
        "relational_primary_macro_f1": primary["macro_f1"],
        "relational_augmented_macro_f1": augmented["macro_f1"],
        "macro_f1_delta": value["selected_comparison"][
            "augmented_minus_primary_macro_f1"
        ],
        "macro_f1_delta_bootstrap_95pct": bootstrap["macro_f1_delta"],
        "false_ready_delta": false_delta,
        "gates": gates,
        "selected": value["selected_condition_after_gold"],
        "output": str(RESULT.relative_to(ROOT)),
    }, indent=2, sort_keys=True))
    return 0 if directional_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.seal:
        return seal()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    torch.use_deterministic_algorithms(True)
    return evaluate(device)


if __name__ == "__main__":
    raise SystemExit(main())
