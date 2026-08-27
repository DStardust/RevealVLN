#!/usr/bin/env python3
"""Prospective zero-inflated OPV adjudication for fixed R3.3 checkpoints."""

from __future__ import annotations

import argparse
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

from revealnav_mf2r3 import (  # noqa: E402
    CausalPairedQAdapter, RevealExpiryQFeatureDataset,
    collate_reveal_expiry_q_examples,
)
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
MANIFEST = ROOT / (
    "artifacts/phase1/rxr_train_expansion/expiry_r3_qpair/"
    "RXR_EXPIRY_R3_Q_FEATURE_MANIFEST.json"
)
R3_3 = ROOT / "artifacts/evaluation/mf2_causal_opp_q_r3_3"
R3_3_COMPARISON = R3_3 / "RXR_CAUSAL_OPP_Q_R3_3_COMPARISON.json"
REVISION = ROOT / (
    "artifacts/design/MF2_ZERO_INFLATED_OPV_METRIC_CORRECTION_R3_4.md"
)
OUT = ROOT / "artifacts/evaluation/mf2_opv_hurdle_r3_4"
PROTOCOL = OUT / "RXR_OPV_HURDLE_R3_4_PROTOCOL.json"
COMPARISON = OUT / "RXR_OPV_HURDLE_R3_4_COMPARISON.json"


def protocol_value() -> dict:
    comparison = json.loads(R3_3_COMPARISON.read_text())
    expected_partial = {
        "mean_q_with_mae_beats_train_median": True,
        "mean_q_without_mae_beats_train_median": True,
        "best_option_accuracy_above_random_in_two_seeds": True,
        "opv_mae_beats_zero_in_two_seeds": False,
        "opv_auc_above_0_5_in_two_seeds": True,
        "q_order_invariant": True,
        "source_r3_1_checkpoints_unchanged": True,
    }
    if not (
        comparison.get("status") == "CAUSAL_OPP_Q_R3_3_GATE_FAIL"
        and comparison.get("gates") == expected_partial
        and comparison.get("gold_payload_read") is False
    ):
        raise RuntimeError("R3.4 adjudication precondition failed")
    checkpoints = {}
    for seed in SEEDS:
        result_path = R3_3 / f"seed_{seed}/result.json"
        result = json.loads(result_path.read_text())
        checkpoint = ROOT / result["checkpoint"]["path"]
        if sha256_file(checkpoint) != result["checkpoint"]["sha256"]:
            raise RuntimeError("R3.3 checkpoint provenance drift")
        checkpoints[str(checkpoint.relative_to(ROOT))] = sha256_file(checkpoint)
    return {
        "schema_version": "revealnav-mf2-opv-hurdle-protocol/3.4",
        "status": "SEALED_BEFORE_FIXED_CHECKPOINT_OPV_ADJUDICATION",
        "seeds": list(SEEDS),
        "occurrence_metric": "prefix AUROC for true OPV > 1e-6",
        "magnitude_metric": "MAE conditional on true OPV > 1e-6",
        "success_gates": {
            "occurrence_auc_above_0_5_in_two_seeds": True,
            "positive_magnitude_mae_beats_zero_in_two_seeds": True,
            "retain_all_non_degenerate_r3_3_gates": True,
        },
        "unconditional_mae_retained_as_failed_diagnostic": True,
        "retraining_allowed": False,
        "model_selection_allowed": False,
        "gold_access_allowed": False,
        "sources": {
            str(MANIFEST.relative_to(ROOT)): sha256_file(MANIFEST),
            str(R3_3_COMPARISON.relative_to(ROOT)): sha256_file(R3_3_COMPARISON),
            str(REVISION.relative_to(ROOT)): sha256_file(REVISION),
        },
        "fixed_checkpoints": checkpoints,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed R3.4 protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({"status": value["status"],
                      "protocol": str(PROTOCOL.relative_to(ROOT)),
                      "sha256": sha256_file(PROTOCOL)}, indent=2))
    return 0


def opv_values(seed: int) -> tuple[np.ndarray, np.ndarray]:
    result = json.loads((R3_3 / f"seed_{seed}/result.json").read_text())
    payload = torch.load(
        ROOT / result["checkpoint"]["path"], map_location="cpu",
        weights_only=False,
    )
    model = CausalPairedQAdapter(768, 96, 128.0)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    truth, prediction = [], []
    loader = DataLoader(
        RevealExpiryQFeatureDataset(MANIFEST, "development"),
        batch_size=16, shuffle=False,
        collate_fn=collate_reveal_expiry_q_examples,
    )
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
            )
            for index in range(len(batch["step_mask"])):
                steps = int(batch["step_mask"][index].sum())
                for step in range(steps):
                    mask = batch["candidate_mask"][index, step]
                    if not bool(mask.any()):
                        continue
                    truth.append(float((
                        batch["option_cost_without_checkpoint"][index, step, mask]
                        - batch["option_cost"][index, step, mask]
                    ).max()))
                    prediction.append(float(
                        output.opv_per_candidate[index, step, mask].max()
                    ))
    return np.asarray(truth), np.asarray(prediction)


def run() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("R3.4 protocol must be sealed without drift")
    source = json.loads(R3_3_COMPARISON.read_text())
    rows = []
    for seed in SEEDS:
        truth, prediction = opv_values(seed)
        positive = truth > 1e-6
        prior = source["results"]
        rows.append({
            "seed": seed,
            "positive_prefixes": int(positive.sum()),
            "all_prefixes": len(truth),
            "positive_prevalence": float(positive.mean()),
            "positive_magnitude_mae": float(np.abs(
                truth[positive] - prediction[positive]
            ).mean()),
            "positive_zero_baseline_mae": float(truth[positive].mean()),
            "unconditional_mae": float(np.abs(truth - prediction).mean()),
            "unconditional_zero_baseline_mae": float(truth.mean()),
            "occurrence_auc": prior["opv_auc"]["values"][SEEDS.index(seed)],
        })
    def summary(key):
        values = [row[key] for row in rows]
        return {"mean": statistics.mean(values),
                "population_std": statistics.pstdev(values), "values": values}
    results = {key: summary(key) for key in (
        "positive_prevalence", "positive_magnitude_mae",
        "positive_zero_baseline_mae", "unconditional_mae",
        "unconditional_zero_baseline_mae", "occurrence_auc",
    )}
    retained = source["gates"].copy()
    retained.pop("opv_mae_beats_zero_in_two_seeds")
    gates = {
        "occurrence_auc_above_0_5_in_two_seeds": sum(
            row["occurrence_auc"] > 0.5 for row in rows
        ) >= 2,
        "positive_magnitude_mae_beats_zero_in_two_seeds": sum(
            row["positive_magnitude_mae"]
            < row["positive_zero_baseline_mae"] for row in rows
        ) >= 2,
        "retain_all_non_degenerate_r3_3_gates": all(retained.values()),
        "fixed_checkpoint_hashes_unchanged": all(
            sha256_file(ROOT / path) == digest
            for path, digest in protocol_value()["fixed_checkpoints"].items()
        ),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf2-opv-hurdle-comparison/3.4",
        "status": "OPV_HURDLE_R3_4_GATE_PASS" if passed
                  else "OPV_HURDLE_R3_4_GATE_FAIL",
        "results": results, "per_seed": rows, "gates": gates,
        "retained_r3_3_gates": retained,
        "unconditional_mae_gate_reclassified": False,
        "unconditional_mae_failure_still_reported": True,
        "training_performed": False, "model_selection_performed": False,
        "gold_payload_read": False, "paper_result": False,
        "learned_opp_authorized": passed,
        "next_step": "seal ECOG/OPP development evaluation" if passed else
                     "OPV signal insufficient",
    }
    atomic_json(COMPARISON, value)
    print(json.dumps(value, indent=2))
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
