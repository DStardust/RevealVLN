#!/usr/bin/env python3
"""Test whether scale REE gains use candidate relations above frozen features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2 import RevealFeatureDataset, collate_reveal_examples  # noqa: E402
from revealnav_mf2r2 import RelationalRevealOptionHeads  # noqa: E402
from run_rxr_representation_comparison_v2 import (  # noqa: E402
    DirectBaseline, classification_metrics, uad_labels,
)


EXPANSION = ROOT / "artifacts/phase1/rxr_train_expansion"
MANIFEST = EXPANSION / "RXR_SCALE_AUTOMATIC_TRAINING_MANIFEST.json"
AUTHORIZATION = (
    EXPANSION / "scale_v2/model_training/RXR_SCALE_AUTOMATIC_TRAINING_AUTHORIZATION.json"
)
TRAINING_PROTOCOL = (
    ROOT / "artifacts/evaluation/mf2_scale_relational_v1/RXR_SCALE_RELATIONAL_PROTOCOL_V1.json"
)
TRAINING_ROOT = ROOT / "artifacts/evaluation/mf2_scale_relational_v1"
COMPARISON = TRAINING_ROOT / "RXR_SCALE_RELATIONAL_COMPARISON_V1.json"
OUT = ROOT / "artifacts/evaluation/mf2_scale_frontend_absorption_v1"
PROTOCOL = OUT / "RXR_SCALE_FRONTEND_ABSORPTION_PROTOCOL_V1.json"
RESULT = OUT / "RXR_SCALE_FRONTEND_ABSORPTION_RESULT_V1.json"
SEEDS = (20260826, 20260827, 20260828)
METRICS = (
    "accuracy", "macro_f1", "nll", "brier", "ece_10bin",
    "false_ready_rate", "missed_ready_rate",
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


def checkpoint_sources() -> dict:
    sources = {}
    for seed in SEEDS:
        run = json.loads((TRAINING_ROOT / f"seed_{seed}/result.json").read_text())
        if run.get("status") != "SCALE_MODEL_RUN_COMPLETE":
            raise RuntimeError(f"incomplete scale checkpoint source: {seed}")
        sources[str(seed)] = run["checkpoints"]
        for record in sources[str(seed)].values():
            path = (ROOT / record["path"]).resolve()
            if ROOT not in path.parents or path.is_symlink() or not path.is_file():
                raise RuntimeError(f"unsafe checkpoint source: {path}")
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"checkpoint provenance drift: {path}")
    return sources


def protocol_value() -> dict:
    authorization = json.loads(AUTHORIZATION.read_text())
    comparison = json.loads(COMPARISON.read_text())
    if not (
        authorization.get("status") == "AUTOMATIC_SCALE_TRAINING_AUTHORIZED"
        and authorization.get("training_authorized") is True
        and authorization["manifest"]["sha256"] == sha256_file(MANIFEST)
        and comparison.get("status") == "SCALE_RELATIONAL_SCORE_GATE_PASS"
        and comparison.get("selected_model") == "scale_relational"
        and comparison.get("gold_feature_payload_read") is False
    ):
        raise RuntimeError("frontend absorption preconditions failed")
    return {
        "schema_version": "revealnav-mf2-scale-frontend-absorption-protocol/1",
        "status": "SEALED_BEFORE_FRONTEND_ABSORPTION_PROBE",
        "scope": "68-event human-audited development engineering diagnostic",
        "seeds": list(SEEDS),
        "conditions": {
            "intact": "unaltered causal frozen features",
            "candidate_order_reversed": "candidate-axis reversal with masks reversed",
            "candidate_relation_collapsed": (
                "each valid candidate replaced by the within-prefix candidate mean"
            ),
            "cross_event_candidate_swap": (
                "candidate sequences replaced by a deterministic different-event donor "
                "with relative-time alignment"
            ),
            "history_direct_uad": "matched trained history-aware baseline",
        },
        "success_thresholds": {
            "checkpoint_metric_reproduction_max_abs": 1e-5,
            "candidate_order_probability_max_abs": 1e-6,
            "intact_minus_history_macro_f1_mean_min": 0.10,
            "intact_minus_history_false_ready_mean_max": 0.0,
            "cross_event_swap_probability_tv_mean_min": 0.05,
            "cross_event_swap_macro_f1_drop_mean_min": 0.03,
            "cross_event_swap_drop_positive_seed_min": 2,
            "relation_collapse_probability_tv_mean_min": 0.02,
            "relation_collapse_macro_f1_drop_mean_min": 0.01,
            "relation_collapse_drop_positive_seed_min": 2,
        },
        "sources": {
            "manifest": {
                "path": str(MANIFEST.relative_to(ROOT)),
                "sha256": sha256_file(MANIFEST),
            },
            "authorization": {
                "path": str(AUTHORIZATION.relative_to(ROOT)),
                "sha256": sha256_file(AUTHORIZATION),
            },
            "training_protocol": {
                "path": str(TRAINING_PROTOCOL.relative_to(ROOT)),
                "sha256": sha256_file(TRAINING_PROTOCOL),
            },
            "comparison": {
                "path": str(COMPARISON.relative_to(ROOT)),
                "sha256": sha256_file(COMPARISON),
            },
            "checkpoints": checkpoint_sources(),
        },
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed frontend absorption protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def reverse_candidates(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    changed = dict(batch)
    changed["candidate_embeddings"] = batch["candidate_embeddings"].flip(2)
    changed["candidate_mask"] = batch["candidate_mask"].flip(2)
    return changed


def collapse_candidate_relations(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    changed = dict(batch)
    mask = batch["candidate_mask"].unsqueeze(-1)
    weight = mask.to(batch["candidate_embeddings"].dtype)
    mean = (batch["candidate_embeddings"] * weight).sum(2, keepdim=True) / (
        weight.sum(2, keepdim=True).clamp_min(1.0)
    )
    changed["candidate_embeddings"] = mean.expand_as(
        batch["candidate_embeddings"]
    ) * weight
    return changed


def swap_cross_event_candidates(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if batch["candidate_embeddings"].shape[0] < 2:
        raise ValueError("cross-event swap requires at least two events")
    changed = dict(batch)
    source_embeddings = batch["candidate_embeddings"]
    source_mask = batch["candidate_mask"]
    target_embeddings = torch.zeros_like(source_embeddings)
    target_mask = torch.zeros_like(source_mask)
    step_counts = batch["step_mask"].sum(1).to(torch.long)
    donor_order = torch.arange(
        len(step_counts), device=step_counts.device
    ).roll(1)
    for target, donor in enumerate(donor_order.tolist()):
        target_steps = int(step_counts[target])
        donor_steps = int(step_counts[donor])
        if target_steps < 1 or donor_steps < 1:
            raise RuntimeError("development event has no valid prefix")
        donor_indices = torch.linspace(
            0, donor_steps - 1, target_steps, device=step_counts.device
        ).round().to(torch.long)
        target_embeddings[target, :target_steps] = source_embeddings[
            donor, donor_indices
        ]
        target_mask[target, :target_steps] = source_mask[donor, donor_indices]
    changed["candidate_embeddings"] = target_embeddings
    changed["candidate_mask"] = target_mask
    return changed


def relational_probabilities(
    model: RelationalRevealOptionHeads,
    batch: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mask = batch["step_mask"]
    budgets = torch.tensor(
        [1.5, 2.0, 3.0, 4.0], device=mask.device,
        dtype=batch["history_embeddings"].dtype,
    ).view(1, 1, -1).expand(mask.shape[0], mask.shape[1], -1)
    with torch.no_grad():
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], budgets, batch["instruction_embedding"],
        )
        target_set = torch.sigmoid(output.target_in_set_logit[mask])
        decisive = torch.sigmoid(output.separation_logit[mask]) * torch.sigmoid(
            output.evidence_logit[mask]
        )
        probabilities = torch.stack((
            1.0 - target_set,
            target_set * (1.0 - decisive),
            target_set * decisive,
        ), dim=-1)
        labels = uad_labels(batch)[mask]
    return labels.cpu().numpy(), probabilities.cpu().numpy()


def history_probabilities(
    model: DirectBaseline, batch: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mask = batch["step_mask"]
    with torch.no_grad():
        probabilities = torch.softmax(model(batch)[mask], dim=-1)
        labels = uad_labels(batch)[mask]
    return labels.cpu().numpy(), probabilities.cpu().numpy()


def load_model(path: Path, name: str, seed: int, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not (
        payload.get("schema_version") == "revealnav-mf2-scale-model-checkpoint/1"
        and payload.get("model_name") == name
        and payload.get("seed") == seed
        and payload.get("manifest_sha256") == sha256_file(MANIFEST)
        and payload.get("protocol_sha256") == sha256_file(TRAINING_PROTOCOL)
    ):
        raise RuntimeError(f"checkpoint binding failed: {path}")
    if name == "relational_ree":
        model = RelationalRevealOptionHeads(768, payload["hidden_dim"], 4)
    elif name == "history_direct_uad":
        model = DirectBaseline(
            history_aware=True, output_dim=3, hidden_dim=payload["hidden_dim"]
        )
    else:
        raise ValueError(name)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


def probability_tv(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum(1).mean())


def summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "values": values,
    }


def run(device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("frontend absorption probe is not sealed")
    if RESULT.exists():
        raise RuntimeError(f"refusing to overwrite {RESULT}")
    dataset = RevealFeatureDataset(MANIFEST, "development")
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False,
        collate_fn=collate_reveal_examples,
    )
    cpu_batch = next(iter(loader))
    batch = {key: value.to(device) for key, value in cpu_batch.items()}
    comparison = json.loads(COMPARISON.read_text())
    per_seed = []
    for seed in SEEDS:
        run_result = json.loads((TRAINING_ROOT / f"seed_{seed}/result.json").read_text())
        relational_path = ROOT / run_result["checkpoints"]["relational_ree"]["path"]
        history_path = ROOT / run_result["checkpoints"]["history_direct_uad"]["path"]
        relational = load_model(relational_path, "relational_ree", seed, device)
        history = load_model(history_path, "history_direct_uad", seed, device)
        labels, intact_probability = relational_probabilities(relational, batch)
        reversed_labels, reversed_probability = relational_probabilities(
            relational, reverse_candidates(batch)
        )
        collapsed_labels, collapsed_probability = relational_probabilities(
            relational, collapse_candidate_relations(batch)
        )
        swapped_labels, swapped_probability = relational_probabilities(
            relational, swap_cross_event_candidates(batch)
        )
        history_labels, history_probability = history_probabilities(history, batch)
        if not all(np.array_equal(labels, other) for other in (
            reversed_labels, collapsed_labels, swapped_labels, history_labels,
        )):
            raise RuntimeError("intervention changed development labels")
        metrics = {
            "intact": classification_metrics(labels, intact_probability),
            "candidate_order_reversed": classification_metrics(
                labels, reversed_probability
            ),
            "candidate_relation_collapsed": classification_metrics(
                labels, collapsed_probability
            ),
            "cross_event_candidate_swap": classification_metrics(
                labels, swapped_probability
            ),
            "history_direct_uad": classification_metrics(
                labels, history_probability
            ),
        }
        stored = run_result["results"]
        reproduction_error = max(
            abs(metrics[condition][metric] - stored[source][metric])
            for condition, source in (
                ("intact", "relational_ree"),
                ("history_direct_uad", "history_direct_uad"),
            )
            for metric in METRICS
        )
        per_seed.append({
            "seed": seed,
            "metrics": metrics,
            "probability_interventions": {
                "candidate_order_reversed_max_abs": float(
                    np.abs(reversed_probability - intact_probability).max()
                ),
                "candidate_relation_collapsed_tv": probability_tv(
                    intact_probability, collapsed_probability
                ),
                "cross_event_candidate_swap_tv": probability_tv(
                    intact_probability, swapped_probability
                ),
            },
            "checkpoint_metric_reproduction_max_abs": reproduction_error,
        })
        del relational, history
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = {}
    for condition in (
        "intact", "candidate_relation_collapsed",
        "cross_event_candidate_swap", "history_direct_uad",
    ):
        aggregate[condition] = {
            metric: summary([
                row["metrics"][condition][metric] for row in per_seed
            ]) for metric in METRICS
        }
    thresholds = json.loads(PROTOCOL.read_text())["success_thresholds"]
    intact_f1 = aggregate["intact"]["macro_f1"]["values"]
    history_f1 = aggregate["history_direct_uad"]["macro_f1"]["values"]
    collapsed_f1 = aggregate["candidate_relation_collapsed"]["macro_f1"]["values"]
    swapped_f1 = aggregate["cross_event_candidate_swap"]["macro_f1"]["values"]
    order_error = max(
        row["probability_interventions"]["candidate_order_reversed_max_abs"]
        for row in per_seed
    )
    collapsed_tv = statistics.mean(
        row["probability_interventions"]["candidate_relation_collapsed_tv"]
        for row in per_seed
    )
    swapped_tv = statistics.mean(
        row["probability_interventions"]["cross_event_candidate_swap_tv"]
        for row in per_seed
    )
    intact_minus_history_f1 = statistics.mean(intact_f1) - statistics.mean(history_f1)
    intact_minus_history_false_ready = (
        aggregate["intact"]["false_ready_rate"]["mean"]
        - aggregate["history_direct_uad"]["false_ready_rate"]["mean"]
    )
    collapsed_drops = [a - b for a, b in zip(intact_f1, collapsed_f1)]
    swapped_drops = [a - b for a, b in zip(intact_f1, swapped_f1)]
    gates = {
        "checkpoint_metrics_reproduced": max(
            row["checkpoint_metric_reproduction_max_abs"] for row in per_seed
        ) <= thresholds["checkpoint_metric_reproduction_max_abs"],
        "candidate_order_is_invariant": (
            order_error <= thresholds["candidate_order_probability_max_abs"]
        ),
        "intact_relational_beats_history": (
            intact_minus_history_f1
            >= thresholds["intact_minus_history_macro_f1_mean_min"]
            and intact_minus_history_false_ready
            <= thresholds["intact_minus_history_false_ready_mean_max"]
        ),
        "cross_event_candidate_alignment_is_used": (
            swapped_tv >= thresholds["cross_event_swap_probability_tv_mean_min"]
            and statistics.mean(swapped_drops)
            >= thresholds["cross_event_swap_macro_f1_drop_mean_min"]
            and sum(drop > 0 for drop in swapped_drops)
            >= thresholds["cross_event_swap_drop_positive_seed_min"]
        ),
        "within_prefix_candidate_relations_are_used": (
            collapsed_tv >= thresholds["relation_collapse_probability_tv_mean_min"]
            and statistics.mean(collapsed_drops)
            >= thresholds["relation_collapse_macro_f1_drop_mean_min"]
            and sum(drop > 0 for drop in collapsed_drops)
            >= thresholds["relation_collapse_drop_positive_seed_min"]
        ),
    }
    value = {
        "schema_version": "revealnav-mf2-scale-frontend-absorption-result/1",
        "status": (
            "FRONTEND_ABSORPTION_GATE_PASS"
            if all(gates.values()) else "FRONTEND_ABSORPTION_GATE_FAIL"
        ),
        "counts": {"development_events": len(dataset), "prefixes": len(labels)},
        "aggregate": aggregate,
        "diagnostics": {
            "intact_minus_history_macro_f1_mean": intact_minus_history_f1,
            "intact_minus_history_false_ready_mean": intact_minus_history_false_ready,
            "candidate_order_probability_max_abs": order_error,
            "candidate_relation_collapsed_probability_tv_mean": collapsed_tv,
            "candidate_relation_collapsed_macro_f1_drop": summary(collapsed_drops),
            "cross_event_candidate_swap_probability_tv_mean": swapped_tv,
            "cross_event_candidate_swap_macro_f1_drop": summary(swapped_drops),
        },
        "gates": gates,
        "per_seed": per_seed,
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": sha256_file(PROTOCOL),
        },
        "comparison_binding": {
            "selected_model": comparison["selected_model"],
            "sha256": sha256_file(COMPARISON),
        },
        "gold_feature_payload_read": False,
        "paper_result": False,
        "next_step": (
            "sealed controller witness before Action-Conditional OPP implementation"
            if all(gates.values()) else "diagnose failed candidate-relation gate"
        ),
    }
    atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "diagnostics": value["diagnostics"],
    }, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seal:
        return seal()
    return run(torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
