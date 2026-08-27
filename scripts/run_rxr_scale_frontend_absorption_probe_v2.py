#!/usr/bin/env python3
"""Re-run the frontend absorption probe on batch-invariant scale REE."""

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
from run_rxr_representation_comparison_v2 import classification_metrics  # noqa: E402
from run_rxr_scale_frontend_absorption_probe import (  # noqa: E402
    collapse_candidate_relations, probability_tv, relational_probabilities,
    reverse_candidates, sha256_file, summary, swap_cross_event_candidates,
)


EXPANSION = ROOT / "artifacts/phase1/rxr_train_expansion"
MANIFEST = EXPANSION / "RXR_SCALE_AUTOMATIC_TRAINING_MANIFEST.json"
STABLE_ROOT = ROOT / "artifacts/evaluation/mf2_scale_relational_v2_count_stable"
STABLE_COMPARISON = (
    STABLE_ROOT / "RXR_SCALE_RELATIONAL_COUNT_STABLE_COMPARISON_V2.json"
)
STABLE_PROTOCOL = STABLE_ROOT / "RXR_SCALE_RELATIONAL_COUNT_STABLE_PROTOCOL_V2.json"
LEGACY_COMPARISON = (
    ROOT / "artifacts/evaluation/mf2_scale_relational_v1/"
    "RXR_SCALE_RELATIONAL_COMPARISON_V1.json"
)
FAILED_PROBE = (
    ROOT / "artifacts/evaluation/mf2_scale_frontend_absorption_v1/"
    "RXR_SCALE_FRONTEND_ABSORPTION_RESULT_V1.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_scale_frontend_absorption_v2"
PROTOCOL = OUT / "RXR_SCALE_FRONTEND_ABSORPTION_PROTOCOL_V2.json"
RESULT = OUT / "RXR_SCALE_FRONTEND_ABSORPTION_RESULT_V2.json"
SEEDS = (20260826, 20260827, 20260828)
METRICS = (
    "accuracy", "macro_f1", "nll", "brier", "ece_10bin",
    "false_ready_rate", "missed_ready_rate",
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def stable_checkpoint_sources() -> dict:
    sources = {}
    for seed in SEEDS:
        result_path = STABLE_ROOT / f"seed_{seed}/result.json"
        result = json.loads(result_path.read_text())
        if result.get("status") != "COUNT_STABLE_MODEL_RUN_COMPLETE":
            raise RuntimeError(f"incomplete count-stable checkpoint: {seed}")
        checkpoint = result["checkpoint"]
        path = (ROOT / checkpoint["path"]).resolve()
        if not (
            ROOT in path.parents and path.is_file() and not path.is_symlink()
            and path.stat().st_size == checkpoint["bytes"]
            and sha256_file(path) == checkpoint["sha256"]
        ):
            raise RuntimeError(f"count-stable checkpoint provenance failed: {seed}")
        sources[str(seed)] = {
            "result": {
                "path": str(result_path.relative_to(ROOT)),
                "sha256": sha256_file(result_path),
            },
            "checkpoint": checkpoint,
        }
    return sources


def protocol_value() -> dict:
    stable = json.loads(STABLE_COMPARISON.read_text())
    failed = json.loads(FAILED_PROBE.read_text())
    if not (
        stable.get("status") == "COUNT_STABLE_SCORE_GATE_PASS"
        and stable.get("selected_model") == "relational_ree_count_stable"
        and stable.get("gold_feature_payload_read") is False
        and failed.get("status") == "FRONTEND_ABSORPTION_GATE_FAIL"
        and failed.get("gates", {}).get("checkpoint_metrics_reproduced") is False
        and all(failed.get("gates", {}).get(name) is True for name in (
            "candidate_order_is_invariant",
            "cross_event_candidate_alignment_is_used",
            "intact_relational_beats_history",
            "within_prefix_candidate_relations_are_used",
        ))
    ):
        raise RuntimeError("corrected absorption preconditions failed")
    return {
        "schema_version": "revealnav-mf2-scale-frontend-absorption-protocol/2",
        "status": "SEALED_BEFORE_COUNT_STABLE_FRONTEND_ABSORPTION_PROBE",
        "scope": "68-event human-audited development engineering diagnostic",
        "seeds": list(SEEDS),
        "candidate_count_encoding": "count/(count+1)",
        "conditions": {
            "intact_batch16": "unaltered features using the training evaluator batch size",
            "intact_batch68": "same ordered events in a single batch",
            "candidate_order_reversed": "candidate-axis reversal with masks reversed",
            "candidate_relation_collapsed": (
                "valid candidates replaced by the within-prefix candidate mean"
            ),
            "cross_event_candidate_swap": (
                "deterministic different-event donor within each evaluator batch"
            ),
            "history_direct_uad": "fixed matched legacy history reference",
        },
        "success_thresholds": {
            "checkpoint_metric_reproduction_max_abs": 1e-5,
            "batch_partition_probability_max_abs": 1e-6,
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
            "count_stable_protocol": {
                "path": str(STABLE_PROTOCOL.relative_to(ROOT)),
                "sha256": sha256_file(STABLE_PROTOCOL),
            },
            "count_stable_comparison": {
                "path": str(STABLE_COMPARISON.relative_to(ROOT)),
                "sha256": sha256_file(STABLE_COMPARISON),
            },
            "legacy_comparison": {
                "path": str(LEGACY_COMPARISON.relative_to(ROOT)),
                "sha256": sha256_file(LEGACY_COMPARISON),
            },
            "failed_probe_retained": {
                "path": str(FAILED_PROBE.relative_to(ROOT)),
                "sha256": sha256_file(FAILED_PROBE),
            },
            "checkpoints": stable_checkpoint_sources(),
        },
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed corrected absorption protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def load_stable_model(seed: int, device: torch.device):
    result = json.loads((STABLE_ROOT / f"seed_{seed}/result.json").read_text())
    path = ROOT / result["checkpoint"]["path"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not (
        payload.get("schema_version") == "revealnav-mf2-scale-model-checkpoint/2"
        and payload.get("model_name") == "relational_ree_count_stable"
        and payload.get("seed") == seed
        and payload.get("candidate_count_encoding") == "saturating"
        and payload.get("manifest_sha256") == sha256_file(MANIFEST)
        and payload.get("protocol_sha256") == sha256_file(STABLE_PROTOCOL)
    ):
        raise RuntimeError(f"count-stable checkpoint binding failed: {seed}")
    model = RelationalRevealOptionHeads(
        768, payload["hidden_dim"], 4, candidate_count_encoding="saturating"
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def evaluate(
    model, dataset, batch_size: int, device: torch.device, intervention=None,
):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_reveal_examples,
    )
    labels, probabilities = [], []
    for cpu_batch in loader:
        batch = {key: value.to(device) for key, value in cpu_batch.items()}
        if intervention is not None:
            batch = intervention(batch)
        batch_labels, batch_probabilities = relational_probabilities(model, batch)
        labels.append(batch_labels)
        probabilities.append(batch_probabilities)
    return np.concatenate(labels), np.concatenate(probabilities)


def run(device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("corrected absorption probe is not sealed")
    if RESULT.exists():
        raise RuntimeError(f"refusing to overwrite {RESULT}")
    dataset = RevealFeatureDataset(MANIFEST, "development")
    stable_results = {
        seed: json.loads((STABLE_ROOT / f"seed_{seed}/result.json").read_text())
        for seed in SEEDS
    }
    legacy = json.loads(LEGACY_COMPARISON.read_text())
    per_seed = []
    for seed in SEEDS:
        model = load_stable_model(seed, device)
        labels, intact = evaluate(model, dataset, 16, device)
        labels68, intact68 = evaluate(model, dataset, len(dataset), device)
        reversed_labels, reversed_probability = evaluate(
            model, dataset, 16, device, reverse_candidates
        )
        collapsed_labels, collapsed_probability = evaluate(
            model, dataset, 16, device, collapse_candidate_relations
        )
        swapped_labels, swapped_probability = evaluate(
            model, dataset, 16, device, swap_cross_event_candidates
        )
        if not all(np.array_equal(labels, other) for other in (
            labels68, reversed_labels, collapsed_labels, swapped_labels,
        )):
            raise RuntimeError("corrected intervention changed development labels")
        metrics = {
            "intact": classification_metrics(labels, intact),
            "candidate_relation_collapsed": classification_metrics(
                labels, collapsed_probability
            ),
            "cross_event_candidate_swap": classification_metrics(
                labels, swapped_probability
            ),
            "history_direct_uad": {
                metric: legacy["results"]["history_direct_uad"][metric]["values"][
                    SEEDS.index(seed)
                ] for metric in METRICS
            },
        }
        reproduction_error = max(
            abs(metrics["intact"][metric] - stable_results[seed]["results"][metric])
            for metric in METRICS
        )
        per_seed.append({
            "seed": seed,
            "metrics": metrics,
            "probability_interventions": {
                "batch16_vs_batch68_max_abs": float(np.abs(intact - intact68).max()),
                "candidate_order_reversed_max_abs": float(
                    np.abs(intact - reversed_probability).max()
                ),
                "candidate_relation_collapsed_tv": probability_tv(
                    intact, collapsed_probability
                ),
                "cross_event_candidate_swap_tv": probability_tv(
                    intact, swapped_probability
                ),
            },
            "checkpoint_metric_reproduction_max_abs": reproduction_error,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = {}
    for condition in (
        "intact", "candidate_relation_collapsed",
        "cross_event_candidate_swap", "history_direct_uad",
    ):
        aggregate[condition] = {
            metric: summary([row["metrics"][condition][metric] for row in per_seed])
            for metric in METRICS
        }
    thresholds = json.loads(PROTOCOL.read_text())["success_thresholds"]
    intact_f1 = aggregate["intact"]["macro_f1"]["values"]
    history_f1 = aggregate["history_direct_uad"]["macro_f1"]["values"]
    collapsed_f1 = aggregate["candidate_relation_collapsed"]["macro_f1"]["values"]
    swapped_f1 = aggregate["cross_event_candidate_swap"]["macro_f1"]["values"]
    collapsed_drops = [a - b for a, b in zip(intact_f1, collapsed_f1)]
    swapped_drops = [a - b for a, b in zip(intact_f1, swapped_f1)]
    batch_error = max(
        row["probability_interventions"]["batch16_vs_batch68_max_abs"]
        for row in per_seed
    )
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
    f1_margin = statistics.mean(intact_f1) - statistics.mean(history_f1)
    false_ready_margin = (
        aggregate["intact"]["false_ready_rate"]["mean"]
        - aggregate["history_direct_uad"]["false_ready_rate"]["mean"]
    )
    gates = {
        "checkpoint_metrics_reproduced": max(
            row["checkpoint_metric_reproduction_max_abs"] for row in per_seed
        ) <= thresholds["checkpoint_metric_reproduction_max_abs"],
        "batch_partition_is_invariant": (
            batch_error <= thresholds["batch_partition_probability_max_abs"]
        ),
        "candidate_order_is_invariant": (
            order_error <= thresholds["candidate_order_probability_max_abs"]
        ),
        "intact_relational_beats_history": (
            f1_margin >= thresholds["intact_minus_history_macro_f1_mean_min"]
            and false_ready_margin
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
    diagnostics = {
        "batch16_vs_batch68_probability_max_abs": batch_error,
        "candidate_order_probability_max_abs": order_error,
        "intact_minus_history_macro_f1_mean": f1_margin,
        "intact_minus_history_false_ready_mean": false_ready_margin,
        "candidate_relation_collapsed_probability_tv_mean": collapsed_tv,
        "candidate_relation_collapsed_macro_f1_drop": summary(collapsed_drops),
        "cross_event_candidate_swap_probability_tv_mean": swapped_tv,
        "cross_event_candidate_swap_macro_f1_drop": summary(swapped_drops),
    }
    value = {
        "schema_version": "revealnav-mf2-scale-frontend-absorption-result/2",
        "status": (
            "FRONTEND_ABSORPTION_COUNT_STABLE_GATE_PASS"
            if all(gates.values()) else "FRONTEND_ABSORPTION_COUNT_STABLE_GATE_FAIL"
        ),
        "counts": {"development_events": len(dataset), "prefixes": len(labels)},
        "aggregate": aggregate,
        "diagnostics": diagnostics,
        "gates": gates,
        "per_seed": per_seed,
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": sha256_file(PROTOCOL),
        },
        "gold_feature_payload_read": False,
        "paper_result": False,
        "next_step": (
            "sealed controller witness before Action-Conditional OPP implementation"
            if all(gates.values()) else "stop and diagnose corrected absorption failure"
        ),
    }
    atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"], "gates": gates, "diagnostics": diagnostics,
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
