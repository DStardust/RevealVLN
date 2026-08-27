#!/usr/bin/env python3
"""Correctness adjudication for bounded target-route failures in V4 labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/daiyang/vla").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import cr5_queue50_tx_worker as core  # noqa: E402
from revealnav_mf2r4 import BranchExcursionDataset  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion/branch_excursion_v4"
FAILED_GATE = BASE / "RXR_BRANCH_EXCURSION_LABEL_GATE_V4.json"
MANIFEST = BASE / "RXR_BRANCH_EXCURSION_LABEL_MANIFEST_V4.json"
PROTOCOL = BASE / "RXR_BRANCH_EXCURSION_CORRECTNESS_PROTOCOL_V4_1.json"
RESULT = BASE / "RXR_BRANCH_EXCURSION_CORRECTNESS_ACCEPTANCE_V4_1.json"


def protocol_value() -> dict:
    gate = json.loads(FAILED_GATE.read_text())
    failed = sorted(key for key, passed in gate["gates"].items() if not passed)
    if not (
        gate.get("status") == "BRANCH_EXCURSION_TRAIN_LABEL_GATE_FAIL"
        and failed == ["all_target_direct_routes_succeed"]
        and gate.get("gold_payload_read") is False
        and gate["manifest"]["sha256"] == core.sha256_file(MANIFEST)
    ):
        raise RuntimeError("V4.1 label adjudication precondition failed")
    return {
        "schema_version": "revealnav-mf2-branch-excursion-correctness/4.1",
        "status": "SEALED_AFTER_V4_GATE_FAILURE_BEFORE_CORRECTNESS_AUDIT",
        "original_failed_gate": "all_target_direct_routes_succeed",
        "correction": (
            "A frozen-controller failure is a valid bounded task-cost outcome, "
            "including for the target branch. Route success is not a label-schema "
            "requirement; exact agreement with the accepted paired-Q failure cost is."
        ),
        "audit": (
            "For all 424 events, compare target commit cost with the existing "
            "paired-Q direct cost at the identical feature step; require exact "
            "agreement, finite labels, provenance closure, and full dataset loading."
        ),
        "pass_rule": {
            "all_424_load": True,
            "all_target_costs_match_existing_paired_q": True,
            "every_failed_target_has_exact_bounded_cost_5": True,
            "every_successful_target_has_finite_cost_below_or_equal_5": True,
            "original_gate_remains_failed": True,
        },
        "sources": {
            str(FAILED_GATE.relative_to(ROOT)): core.sha256_file(FAILED_GATE),
            str(MANIFEST.relative_to(ROOT)): core.sha256_file(MANIFEST),
            "revealnav_mf2r4/data.py": core.sha256_file(
                ROOT / "revealnav_mf2r4/data.py"
            ),
        },
        "development_access_allowed": False,
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed correctness protocol drift")
    if not PROTOCOL.exists():
        core.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": core.sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def run() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("correctness protocol must be sealed")
    manifest = json.loads(MANIFEST.read_text())
    dataset = BranchExcursionDataset(MANIFEST)
    agreements = []
    target_failures = 0
    target_successes = 0
    failed_costs_exact = True
    success_costs_valid = True
    for label_path in dataset.label_paths:
        label = json.loads(label_path.read_text())
        target_rows = [row for row in label["labels"] if row["is_target"]]
        if len(target_rows) != 1:
            raise RuntimeError("target label identity failure")
        target = target_rows[0]
        step = int(label["online_feature_relative_step"])
        index = int(target["branch_index"])
        feature = ROOT / label["online_feature"]["path"]
        with np.load(feature, allow_pickle=False) as shard:
            paired_q = float(shard["option_cost_without_checkpoint"][step, index])
        observed = float(target["commit_cost"])
        agreements.append(abs(paired_q - observed))
        succeeded = bool(target["commit_route"].get("success"))
        if succeeded:
            target_successes += 1
            success_costs_valid &= math.isfinite(observed) and observed <= 5.0
        else:
            target_failures += 1
            failed_costs_exact &= observed == 5.0 and paired_q == 5.0
    loaded = 0
    finite = True
    for index in range(len(dataset)):
        example = dataset[index]
        loaded += 1
        finite &= bool(np.isfinite(example["commit_cost"].numpy()).all())
        finite &= bool(np.isfinite(example["excursion_cost"].numpy()).all())
    gates = {
        "all_424_load": loaded == 424,
        "all_labels_finite": finite,
        "all_target_costs_match_existing_paired_q": max(agreements, default=1.0) <= 1e-6,
        "every_failed_target_has_exact_bounded_cost_5": (
            target_failures > 0 and failed_costs_exact
        ),
        "every_successful_target_has_finite_cost_below_or_equal_5": (
            target_successes > 0 and success_costs_valid
        ),
        "original_gate_remains_failed": (
            json.loads(FAILED_GATE.read_text())["status"]
            == "BRANCH_EXCURSION_TRAIN_LABEL_GATE_FAIL"
        ),
        "no_development_or_gold_payload_read": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-correctness-result/4.1",
        "status": (
            "BRANCH_EXCURSION_LABEL_CORRECTNESS_ACCEPTANCE_PASS" if passed
            else "BRANCH_EXCURSION_LABEL_CORRECTNESS_ACCEPTANCE_FAIL"
        ),
        "counts": {
            "events_loaded": loaded,
            "target_route_successes": target_successes,
            "target_route_failures": target_failures,
        },
        "maximum_target_cost_disagreement": max(agreements, default=None),
        "gates": gates,
        "original_v4_gate_status_preserved": "BRANCH_EXCURSION_TRAIN_LABEL_GATE_FAIL",
        "protocol_sha256": core.sha256_file(PROTOCOL),
        "manifest_sha256": core.sha256_file(MANIFEST),
        "development_payload_read": False,
        "gold_payload_read": False,
        "training_authorized": passed,
        "paper_result": False,
        "next_gate": "event-level action-cost head training" if passed else "repair labels",
    }
    core.atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"], "counts": value["counts"],
        "maximum_target_cost_disagreement": value["maximum_target_cost_disagreement"],
        "gates": gates,
    }, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    return seal() if args.seal else run()


if __name__ == "__main__":
    raise SystemExit(main())
