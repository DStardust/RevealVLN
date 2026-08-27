#!/usr/bin/env python3
"""Train relational REE on augmented data and aggregate the 2x2 ablation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_rxr_relational_v2 as runner


PHASE = ROOT / "artifacts/phase1/rxr_train_expansion"
MANIFEST = PHASE / "RXR_SECONDARY_AUGMENTED_FEATURE_MANIFEST_V1.json"
AUTHORIZATION = PHASE / (
    "RXR_SECONDARY_AUGMENTED_TRAINING_AUTHORIZATION_V1.json"
)
PRIMARY_RELATIONAL = ROOT / "artifacts/evaluation/mf2_relational_v2"
PRIMARY_COMPARISON = PRIMARY_RELATIONAL / "RXR_RELATIONAL_COMPARISON_V2.json"
FROZEN_PRIMARY = ROOT / "artifacts/evaluation/mf2_balanced_tuning_v2"
FROZEN_AUGMENTED = ROOT / "artifacts/evaluation/mf2_secondary_augmentation_v1"
FROZEN_AUGMENTED_COMPARISON = FROZEN_AUGMENTED / (
    "RXR_SECONDARY_AUGMENTATION_COMPARISON_V1.json"
)
OUT = ROOT / "artifacts/evaluation/mf2_relational_augmented_v2"
PROTOCOL = OUT / "RXR_RELATIONAL_AUGMENTED_PROTOCOL_V2.json"
AGGREGATE = OUT / "RXR_RELATIONAL_AUGMENTED_COMPARISON_V2.json"
SEEDS = runner.SEEDS


def preconditions() -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    authorization = json.loads(AUTHORIZATION.read_text())
    primary = json.loads(PRIMARY_COMPARISON.read_text())
    frozen_augmented = json.loads(FROZEN_AUGMENTED_COMPARISON.read_text())
    if not (
        protocol.get("status") == "FROZEN_BEFORE_RELATIONAL_AUGMENTED_RUNS"
        and protocol.get("seeds") == list(SEEDS)
        and protocol.get("gold_access_allowed") is False
        and authorization.get("status")
        == "AUGMENTED_DEVELOPMENT_EXPERIMENT_AUTHORIZED"
        and authorization.get("training_authorized") is True
        and authorization["manifest"]["sha256"] == runner.sha256_file(MANIFEST)
        and primary.get("status") == "RELATIONAL_GATE_PASS"
        and frozen_augmented.get("status") in (
            "POSITIVE_DEVELOPMENT_AUGMENTATION_SIGNAL",
            "NO_CLEAR_DEVELOPMENT_AUGMENTATION_SIGNAL",
        )
    ):
        raise RuntimeError("relational augmented precondition failed")
    return protocol


def configure_runner() -> None:
    runner.MANIFEST = MANIFEST
    runner.AUTHORIZATION = AUTHORIZATION
    runner.OUT = OUT
    runner.PROTOCOL = PROTOCOL
    runner.RUN_STATUS = "RELATIONAL_AUGMENTED_RUN_COMPLETE"
    runner.USES_SECONDARY_EXPANSION = True
    runner.preconditions = preconditions


def summary(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "values": values,
    }


def load_condition(condition: str, seed: int) -> dict:
    if condition == "relational_primary":
        path = PRIMARY_RELATIONAL / f"seed_{seed}/result.json"
        key = "relational_full_ree"
    elif condition == "relational_augmented":
        path = OUT / f"seed_{seed}/result.json"
        key = "relational_full_ree"
    elif condition in ("frozen_full_primary", "history_primary"):
        path = FROZEN_PRIMARY / f"h128_seed_{seed}/result.json"
        key = (
            "balanced_full_ree" if condition == "frozen_full_primary"
            else "balanced_history_direct_uad"
        )
    else:
        path = FROZEN_AUGMENTED / f"seed_{seed}/result.json"
        key = (
            "balanced_full_ree" if condition == "frozen_full_augmented"
            else "balanced_history_direct_uad"
        )
    return json.loads(path.read_text())["results"][key]


def aggregate() -> int:
    preconditions()
    conditions = (
        "frozen_full_primary", "frozen_full_augmented",
        "relational_primary", "relational_augmented",
        "history_primary", "history_augmented",
    )
    metrics = (
        "accuracy", "macro_f1", "nll", "brier", "ece_10bin",
        "false_ready_rate", "missed_ready_rate",
    )
    rows = {
        condition: [load_condition(condition, seed) for seed in SEEDS]
        for condition in conditions
    }
    results = {
        condition: {
            metric: summary([row[metric] for row in condition_rows])
            for metric in metrics
        }
        for condition, condition_rows in rows.items()
    }
    primary = results["relational_primary"]
    augmented = results["relational_augmented"]
    macro_delta_values = [
        revised - original for revised, original in zip(
            augmented["macro_f1"]["values"], primary["macro_f1"]["values"]
        )
    ]
    false_delta = (
        augmented["false_ready_rate"]["mean"]
        - primary["false_ready_rate"]["mean"]
    )
    gates = {
        "augmented_macro_f1_mean_at_least_primary_relational": (
            augmented["macro_f1"]["mean"] >= primary["macro_f1"]["mean"]
        ),
        "augmented_macro_f1_improves_in_at_least_two_seeds": sum(
            value > 0 for value in macro_delta_values
        ) >= 2,
        "augmented_false_ready_degradation_at_most_0_02": false_delta <= 0.02,
    }
    accepted = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-relational-augmentation-comparison/2",
        "status": (
            "RELATIONAL_AUGMENTATION_GATE_PASS"
            if accepted else "PRIMARY_RELATIONAL_RETAINED"
        ),
        "scope": "fixed development-only model-by-data 2x2 ablation",
        "results": results,
        "effects": {
            "relational_data_macro_f1": summary(macro_delta_values),
            "relational_data_false_ready_mean": false_delta,
            "primary_architecture_macro_f1": (
                results["relational_primary"]["macro_f1"]["mean"]
                - results["frozen_full_primary"]["macro_f1"]["mean"]
            ),
            "augmented_architecture_macro_f1": (
                results["relational_augmented"]["macro_f1"]["mean"]
                - results["frozen_full_augmented"]["macro_f1"]["mean"]
            ),
            "frozen_full_data_macro_f1": (
                results["frozen_full_augmented"]["macro_f1"]["mean"]
                - results["frozen_full_primary"]["macro_f1"]["mean"]
            ),
            "history_data_macro_f1": (
                results["history_augmented"]["macro_f1"]["mean"]
                - results["history_primary"]["macro_f1"]["mean"]
            ),
        },
        "predeclared_success_gates": gates,
        "selected_training_condition": (
            "relational_augmented" if accepted else "relational_primary"
        ),
        "sources": {
            "protocol_sha256": runner.sha256_file(PROTOCOL),
            "manifest_sha256": runner.sha256_file(MANIFEST),
            "primary_relational_sha256": runner.sha256_file(PRIMARY_COMPARISON),
            "frozen_augmented_sha256": runner.sha256_file(
                FROZEN_AUGMENTED_COMPARISON
            ),
        },
        "secondary_evaluation_labels_used": False,
        "topology_only_training_included": False,
        "gold_payload_read": False,
        "paper_result": False,
    }
    runner.atomic_json(AGGREGATE, value)
    print(json.dumps({
        "status": value["status"], "effects": value["effects"],
        "gates": gates,
        "selected": value["selected_training_condition"],
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int)
    group.add_argument("--aggregate", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure_runner()
    if args.aggregate:
        return aggregate()
    return runner.train(args.seed, runner.torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
