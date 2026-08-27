#!/usr/bin/env python3
"""Adjudicate structural versus floating-point batch partition effects."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2 import RevealFeatureDataset, collate_reveal_examples  # noqa: E402
from run_rxr_representation_comparison_v2 import classification_metrics  # noqa: E402
from run_rxr_scale_frontend_absorption_probe import (  # noqa: E402
    probability_tv, relational_probabilities, sha256_file,
)
from run_rxr_scale_frontend_absorption_probe_v2 import (  # noqa: E402
    MANIFEST, SEEDS, STABLE_ROOT, load_stable_model,
)


FAILED_RESULT = (
    ROOT / "artifacts/evaluation/mf2_scale_frontend_absorption_v2/"
    "RXR_SCALE_FRONTEND_ABSORPTION_RESULT_V2.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_scale_batch_invariance_adjudication_v1"
PROTOCOL = OUT / "RXR_SCALE_BATCH_INVARIANCE_ADJUDICATION_PROTOCOL_V1.json"
RESULT = OUT / "RXR_SCALE_BATCH_INVARIANCE_ADJUDICATION_RESULT_V1.json"
DISCRETE_METRICS = (
    "accuracy", "macro_f1", "false_ready_rate", "missed_ready_rate",
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def protocol_value() -> dict:
    failed = json.loads(FAILED_RESULT.read_text())
    if not (
        failed.get("status") == "FRONTEND_ABSORPTION_COUNT_STABLE_GATE_FAIL"
        and failed.get("gates", {}).get("batch_partition_is_invariant") is False
        and all(failed.get("gates", {}).get(name) is True for name in (
            "candidate_order_is_invariant", "checkpoint_metrics_reproduced",
            "cross_event_candidate_alignment_is_used",
            "intact_relational_beats_history",
            "within_prefix_candidate_relations_are_used",
        ))
    ):
        raise RuntimeError("batch invariance adjudication preconditions failed")
    return {
        "schema_version": "revealnav-mf2-scale-batch-invariance-adjudication-protocol/1",
        "status": "SEALED_BEFORE_BATCH_INVARIANCE_ADJUDICATION",
        "reason": (
            "the corrected structural probe passed every scientific gate but "
            "exceeded an exact-style 1e-6 GPU float32 tolerance"
        ),
        "seeds": list(SEEDS),
        "conditions": {
            "gpu_float32": "batch16 versus batch68",
            "cpu_float64": "batch1 versus batch68",
        },
        "success_thresholds": {
            "gpu_float32_probability_max_abs": 1e-4,
            "gpu_float32_probability_tv_mean": 1e-5,
            "gpu_float32_argmax_agreement": 1.0,
            "gpu_float32_discrete_metric_max_abs": 0.0,
            "cpu_float64_probability_max_abs": 1e-10,
            "cpu_float64_argmax_agreement": 1.0,
        },
        "sources": {
            "failed_result": {
                "path": str(FAILED_RESULT.relative_to(ROOT)),
                "sha256": sha256_file(FAILED_RESULT),
            },
            "manifest": {
                "path": str(MANIFEST.relative_to(ROOT)),
                "sha256": sha256_file(MANIFEST),
            },
            "checkpoints": {
                str(seed): {
                    "path": json.loads(
                        (STABLE_ROOT / f"seed_{seed}/result.json").read_text()
                    )["checkpoint"]["path"],
                    "sha256": json.loads(
                        (STABLE_ROOT / f"seed_{seed}/result.json").read_text()
                    )["checkpoint"]["sha256"],
                } for seed in SEEDS
            },
        },
        "failed_gate_reclassified": False,
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed batch invariance adjudication protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"], "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def evaluate(model, dataset, batch_size, device, dtype):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_reveal_examples,
    )
    labels, probabilities = [], []
    for cpu_batch in loader:
        batch = {
            key: (
                value.to(device=device, dtype=dtype)
                if value.is_floating_point() else value.to(device=device)
            ) for key, value in cpu_batch.items()
        }
        batch_labels, batch_probability = relational_probabilities(model, batch)
        labels.append(batch_labels)
        probabilities.append(batch_probability)
    return np.concatenate(labels), np.concatenate(probabilities)


def comparison(left_labels, left_probability, right_labels, right_probability):
    if not np.array_equal(left_labels, right_labels):
        raise RuntimeError("batch partition changed labels")
    left_metrics = classification_metrics(left_labels, left_probability)
    right_metrics = classification_metrics(right_labels, right_probability)
    return {
        "probability_max_abs": float(np.abs(left_probability - right_probability).max()),
        "probability_tv_mean": probability_tv(left_probability, right_probability),
        "argmax_agreement": float(np.mean(
            left_probability.argmax(1) == right_probability.argmax(1)
        )),
        "discrete_metric_max_abs": max(
            abs(left_metrics[name] - right_metrics[name])
            for name in DISCRETE_METRICS
        ),
        "left_metrics": {name: left_metrics[name] for name in DISCRETE_METRICS},
        "right_metrics": {name: right_metrics[name] for name in DISCRETE_METRICS},
    }


def run(gpu_device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("batch invariance adjudication is not sealed")
    if RESULT.exists():
        raise RuntimeError(f"refusing to overwrite {RESULT}")
    dataset = RevealFeatureDataset(MANIFEST, "development")
    rows = []
    for seed in SEEDS:
        gpu_model = load_stable_model(seed, gpu_device).float()
        gpu16 = evaluate(gpu_model, dataset, 16, gpu_device, torch.float32)
        gpu68 = evaluate(gpu_model, dataset, len(dataset), gpu_device, torch.float32)
        gpu = comparison(*gpu16, *gpu68)
        del gpu_model
        torch.cuda.empty_cache()

        cpu_model = load_stable_model(seed, torch.device("cpu")).double()
        cpu1 = evaluate(cpu_model, dataset, 1, torch.device("cpu"), torch.float64)
        cpu68 = evaluate(
            cpu_model, dataset, len(dataset), torch.device("cpu"), torch.float64
        )
        cpu = comparison(*cpu1, *cpu68)
        rows.append({"seed": seed, "gpu_float32": gpu, "cpu_float64": cpu})

    thresholds = json.loads(PROTOCOL.read_text())["success_thresholds"]
    maxima = {
        "gpu_float32_probability_max_abs": max(
            row["gpu_float32"]["probability_max_abs"] for row in rows
        ),
        "gpu_float32_probability_tv_mean": max(
            row["gpu_float32"]["probability_tv_mean"] for row in rows
        ),
        "gpu_float32_argmax_agreement": min(
            row["gpu_float32"]["argmax_agreement"] for row in rows
        ),
        "gpu_float32_discrete_metric_max_abs": max(
            row["gpu_float32"]["discrete_metric_max_abs"] for row in rows
        ),
        "cpu_float64_probability_max_abs": max(
            row["cpu_float64"]["probability_max_abs"] for row in rows
        ),
        "cpu_float64_argmax_agreement": min(
            row["cpu_float64"]["argmax_agreement"] for row in rows
        ),
    }
    gates = {
        name: (
            value >= thresholds[name] if name.endswith("argmax_agreement")
            else value <= thresholds[name]
        ) for name, value in maxima.items()
    }
    value = {
        "schema_version": "revealnav-mf2-scale-batch-invariance-adjudication-result/1",
        "status": (
            "BATCH_INVARIANCE_NUMERICAL_ADJUDICATION_PASS"
            if all(gates.values()) else "BATCH_INVARIANCE_NUMERICAL_ADJUDICATION_FAIL"
        ),
        "counts": {"development_events": len(dataset)},
        "aggregate": maxima,
        "gates": gates,
        "per_seed": rows,
        "failed_gate_reclassified": False,
        "interpretation": (
            "structural batch invariance holds; retained GPU differences are "
            "bounded floating-point GEMM effects with identical decisions"
            if all(gates.values()) else "batch partition effect remains unresolved"
        ),
        "gold_feature_payload_read": False,
        "paper_result": False,
    }
    atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"], "aggregate": maxima, "gates": gates,
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
