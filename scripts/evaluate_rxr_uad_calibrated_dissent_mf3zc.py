#!/usr/bin/env python3
"""Seal the train-shadow gate for MF3ZC cold-start calibrated dissent."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluate_rxr_uad_horizon_mf3v import DATA, collect, load_models, manifest_path, score
from select_rxr_uad_policy_risk_mf3s import exact_control
from select_rxr_uad_rescue_harm_mf3p import wilson
from train_rxr_uad_correction_mf3e import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf3zc_calibrated_dissent_shadow_gate_v1"
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZC_CALIBRATED_DISSENT.md"
LOW_QUANTILE = 0.985
UPPER_QUANTILE = 0.995
COLD_START_STEPS = 3
FLOOR_RATIO_QUANTILE = 0.95
RELATIVE_MAD_QUANTILE = 0.75


def consensus_features(row: dict) -> tuple[float, float]:
    values = torch.tensor(row["member_logits"], dtype=torch.float32)
    median = float(values.median())
    denominator = max(abs(median), 1e-6)
    floor_ratio = float(values.min()) / denominator
    relative_mad = float((values - median).abs().median()) / denominator
    return floor_ratio, relative_mad


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(device)
    fit = collect(models, "fit", manifest_path("final"), device)
    shadow = collect(models, "shadow", DATA, device)
    fit_scores = [score(row) for sequence in fit for row in sequence if row is not None]
    lower = float(torch.quantile(torch.tensor(fit_scores), LOW_QUANTILE))
    upper = float(torch.quantile(torch.tensor(fit_scores), UPPER_QUANTILE))
    calibration = [
        row for sequence in shadow for row in sequence
        if row is not None and lower < score(row) <= upper
    ]
    floor_ratios, relative_mads = zip(*(consensus_features(row) for row in calibration))
    floor_threshold = float(torch.quantile(torch.tensor(floor_ratios), FLOOR_RATIO_QUANTILE))
    mad_threshold = float(torch.quantile(torch.tensor(relative_mads), RELATIVE_MAD_QUANTILE))
    selected = []
    for sequence in shadow:
        for step, row in enumerate(sequence):
            if row is None or not lower < score(row) <= upper:
                continue
            floor_ratio, relative_mad = consensus_features(row)
            if step >= COLD_START_STEPS or (
                floor_ratio <= floor_threshold and relative_mad <= mad_threshold
            ):
                selected.append(row)
                break
    rescues = sum(row["outcome"] == "RESCUE" for row in selected)
    harms = sum(row["outcome"] == "HARM" for row in selected)
    summary = {
        "interventions": len(selected), "rescues": rescues, "harms": harms,
        "neither": len(selected) - rescues - harms, "net_rescues": rescues - harms,
        "rescue_vs_harm_wilson95_lower": wilson(rescues, harms),
    }
    control = exact_control(shadow, len(selected))
    gates = {
        "train_shadow_has_ten_interventions": len(selected) >= 10,
        "train_shadow_net_rescue_positive": summary["net_rescues"] > 0,
        "train_shadow_beats_exact_budget_control": summary["net_rescues"] > control["net_rescues"],
        "rescue_vs_harm_wilson_lower_above_half": summary["rescue_vs_harm_wilson95_lower"] > 0.5,
    }
    passed = all(gates.values())
    rule = {
        "hidden_dim": 128, "horizon": 3, "mad_weight": 0.5,
        "policy_risk_beta": 0.25, "training_score_quantile": LOW_QUANTILE,
        "final_training_threshold": lower, "score_upper_quantile": UPPER_QUANTILE,
        "score_upper_threshold": upper, "persistence_steps": 1,
        "cold_start_steps": COLD_START_STEPS,
        "cold_start_floor_ratio_quantile": FLOOR_RATIO_QUANTILE,
        "cold_start_floor_ratio_threshold": floor_threshold,
        "cold_start_relative_mad_quantile": RELATIVE_MAD_QUANTILE,
        "cold_start_relative_mad_threshold": mad_threshold,
    }
    atomic_json(OUT / "MF3ZC_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3zc-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "selected_architecture": {"hidden_dim": 128}, "selected_rule": rule,
        "calibration_rows": len(calibration), "shadow": summary,
        "exact_budget_control": control,
        "uncertainty_rule": {"native_margin_max": control["native_margin_max"]},
        "gates": gates, "task_metric_run_authorized": passed,
        "public_unseen_authorized": False,
        "prior_unseen_development_result": {
            "path": "artifacts/evaluation/mf3za_uad_rxr_val_unseen_independent_v1/MF3ZA_RXR_VAL_UNSEEN_RESULT.json",
            "used_to_identify_cold_start_failure_mode": True,
        },
        "fresh_data": {
            "path": str(DATA.relative_to(ROOT)), "bytes": DATA.stat().st_size,
            "sha256": sha256_file(DATA), "shadow_episodes": len(shadow),
        },
        "checkpoints": checkpoints, "design_sha256": sha256_file(DESIGN),
    })
    print(json.dumps({"status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
                      "rule": rule, "shadow": summary, "control": control,
                      "gates": gates}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
