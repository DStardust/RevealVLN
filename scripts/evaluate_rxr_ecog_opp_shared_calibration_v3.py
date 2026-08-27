#!/usr/bin/env python3
"""Select one shared ECOG/OPP operating point across training seeds.

This is a post-diagnostic development repair.  The development heldout rows
were used by earlier controller experiments, so this script cannot create an
independent paper result; it only nominates one fixed controller for the next
unseen evaluation gate.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_rxr_ecog_opp_development as legacy  # noqa: E402
from revealnav_mf2r3 import LearnedOPPConfig  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


OUT = ROOT / "artifacts/evaluation/mf2_ecog_opp_shared_calibration_v3"
PROTOCOL = OUT / "RXR_ECOG_OPP_SHARED_CALIBRATION_PROTOCOL_V3.json"
RESULT = OUT / "RXR_ECOG_OPP_SHARED_CALIBRATION_RESULT_V3.json"
EVENTS = OUT / "RXR_ECOG_OPP_SHARED_CALIBRATION_EVENTS_V3.jsonl"
V2 = ROOT / (
    "artifacts/evaluation/mf2_ecog_opp_development_v2/"
    "RXR_ECOG_OPP_DEVELOPMENT_COMPARISON.json"
)
GRID = {
    "discriminable_threshold": (0.4, 0.5, 0.6, 0.7),
    "target_threshold": (0.3, 0.4, 0.5),
    "expiry_threshold": (0.3, 0.5, 0.7),
    "opv_threshold": (0.0, 0.025, 0.05, 0.1, 0.2, 0.4),
}


def protocol_value() -> dict:
    previous = json.loads(V2.read_text())
    if not (
        previous.get("status") == "ECOG_OPP_DEVELOPMENT_GATE_PASS"
        and previous.get("gold_payload_read") is False
    ):
        raise RuntimeError("shared-calibration precondition failed")
    return {
        "schema_version": "revealnav-mf2-ecog-opp-shared-calibration/3",
        "status": "SEALED_BEFORE_SHARED_CALIBRATION_ENGINEERING_RUN",
        "seeds": list(legacy.SEEDS),
        "conditions": list(legacy.CONDITIONS),
        "event_counts_per_seed": {"calibration": 40, "heldout": 28},
        "threshold_grid": {key: list(values) for key, values in GRID.items()},
        "fixed_parameters": {
            "persistence_k": 3,
            "evidence_threshold": 0.5,
            "reveal_threshold": 0.5,
            "active_width": 2,
            "retrieval_limit": 8,
            "wrong_commitment_weight": 5.0,
        },
        "selection": (
            "one shared configuration minimizing pooled calibration mean task "
            "loss; tie-break worst-seed loss, pooled premature+missed rate, "
            "checkpoint count, then lexicographic thresholds"
        ),
        "success_gates": {
            "full_task_loss_strictly_below_branch_memory": True,
            "full_success_not_below_branch_memory": True,
            "full_checkpoint_count_below_branch_memory": True,
            "full_checkpoint_precision_not_below_branch_memory": True,
            "full_task_loss_below_no_ecog": True,
        },
        "sources": {
            str(V2.relative_to(ROOT)): sha256_file(V2),
            str(legacy.MANIFEST.relative_to(ROOT)): sha256_file(legacy.MANIFEST),
            str(legacy.Q_ADJUDICATION.relative_to(ROOT)): sha256_file(
                legacy.Q_ADJUDICATION
            ),
        },
        "heldout_reused_after_prior_diagnostics": True,
        "unbiased_claim_allowed": False,
        "gold_access_allowed": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed shared-calibration protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def configurations():
    for discriminable, target, expiry, opv in product(
        GRID["discriminable_threshold"],
        GRID["target_threshold"],
        GRID["expiry_threshold"],
        GRID["opv_threshold"],
    ):
        yield LearnedOPPConfig(
            persistence_k=3,
            opv_threshold=opv,
            discriminable_threshold=discriminable,
            evidence_threshold=0.5,
            target_threshold=target,
            expiry_threshold=expiry,
            reveal_threshold=0.5,
            active_width=2,
            retrieval_limit=8,
            wrong_commitment_weight=5.0,
        )


def choose_shared(events_by_seed: dict[int, list[dict]]):
    choices = []
    for config in configurations():
        per_seed = []
        for seed in legacy.SEEDS:
            rows = [
                legacy.simulate(event, "full_ree_ecog_opp", config)
                for event in events_by_seed[seed]
                if event["partition"] == "calibration"
            ]
            per_seed.append(legacy.metrics(rows))
        key = (
            statistics.mean(row["mean_task_loss"] for row in per_seed),
            max(row["mean_task_loss"] for row in per_seed),
            statistics.mean(
                row["premature_commitment_rate"]
                + row["missed_opportunity_rate"]
                for row in per_seed
            ),
            statistics.mean(row["mean_checkpoint_count"] for row in per_seed),
            config.discriminable_threshold,
            config.target_threshold,
            config.expiry_threshold,
            config.opv_threshold,
        )
        choices.append((key, config, per_seed))
    return min(choices, key=lambda row: row[0])


def summarize(per_seed: dict[str, dict[str, dict]]) -> dict:
    output = {}
    for condition in legacy.CONDITIONS:
        output[condition] = {}
        keys = [
            key for key, value in per_seed[str(legacy.SEEDS[0])][condition].items()
            if key != "events" and isinstance(value, (int, float))
        ]
        for key in keys:
            values = [per_seed[str(seed)][condition][key] for seed in legacy.SEEDS]
            output[condition][key] = {
                "mean": statistics.mean(values),
                "population_std": statistics.pstdev(values),
                "values": values,
            }
    return output


def run() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("shared-calibration protocol must be sealed")
    events_by_seed = {seed: legacy.precompute(seed) for seed in legacy.SEEDS}
    selection_key, config, calibration = choose_shared(events_by_seed)
    rows = []
    per_seed = {}
    for seed in legacy.SEEDS:
        heldout = [
            event for event in events_by_seed[seed]
            if event["partition"] == "heldout"
        ]
        per_seed[str(seed)] = {}
        for condition in legacy.CONDITIONS:
            condition_rows = [
                legacy.simulate(event, condition, config) for event in heldout
            ]
            rows.extend({**row, "seed": seed} for row in condition_rows)
            per_seed[str(seed)][condition] = legacy.metrics(condition_rows)
    part = EVENTS.with_name(EVENTS.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    with part.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(part, EVENTS)
    aggregate = summarize(per_seed)
    full = aggregate["full_ree_ecog_opp"]
    memory = aggregate["branch_memory_without_expiry"]
    no_ecog = aggregate["ree_without_ecog"]
    gates = {
        "all_runs_complete": len(rows) == 3 * 28 * len(legacy.CONDITIONS),
        "full_task_loss_strictly_below_branch_memory": (
            full["mean_task_loss"]["mean"] < memory["mean_task_loss"]["mean"]
        ),
        "full_success_not_below_branch_memory": (
            full["target_success_rate"]["mean"]
            >= memory["target_success_rate"]["mean"]
        ),
        "full_checkpoint_count_below_branch_memory": (
            full["mean_checkpoint_count"]["mean"]
            < memory["mean_checkpoint_count"]["mean"]
        ),
        "full_checkpoint_precision_not_below_branch_memory": (
            full["checkpoint_positive_rate"]["mean"]
            >= memory["checkpoint_positive_rate"]["mean"]
        ),
        "full_task_loss_below_no_ecog": (
            full["mean_task_loss"]["mean"] < no_ecog["mean_task_loss"]["mean"]
        ),
        "no_gold_payload_read": True,
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-ecog-opp-shared-calibration-result/3",
        "status": (
            "CONTROLLER_SHARED_CALIBRATION_ENGINEERING_PASS" if passed
            else "CONTROLLER_SHARED_CALIBRATION_ENGINEERING_FAIL"
        ),
        "selected_shared_config": asdict(config),
        "selection_key": list(selection_key),
        "calibration_per_seed": {
            str(seed): row for seed, row in zip(legacy.SEEDS, calibration)
        },
        "heldout_per_seed": per_seed,
        "heldout_aggregate": aggregate,
        "gates": gates,
        "events": {
            "path": str(EVENTS.relative_to(ROOT)),
            "rows": len(rows),
            "bytes": EVENTS.stat().st_size,
            "sha256": sha256_file(EVENTS),
        },
        "protocol_sha256": sha256_file(PROTOCOL),
        "heldout_reused_after_prior_diagnostics": True,
        "unbiased_claim_allowed": False,
        "gold_payload_read": False,
        "paper_result": False,
        "next_gate": "unseen controller evaluation or branch-excursion Q labels",
    }
    atomic_json(RESULT, value)
    print(json.dumps({
        "status": value["status"],
        "selected_shared_config": value["selected_shared_config"],
        "gates": gates,
        "full": {
            key: full[key]["mean"] for key in (
                "mean_task_loss", "target_success_rate",
                "mean_checkpoint_count", "checkpoint_positive_rate",
            )
        },
        "branch_memory": {
            key: memory[key]["mean"] for key in (
                "mean_task_loss", "target_success_rate",
                "mean_checkpoint_count", "checkpoint_positive_rate",
            )
        },
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
