#!/usr/bin/env python3
"""Fresh RxR-train shadow gate for MF3Y consensus-gated MF3V tail recovery."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from evaluate_rxr_uad_horizon_mf3v import DATA, collect, load_models, manifest_path, score
from select_rxr_uad_policy_risk_mf3s import exact_control
from train_rxr_uad_correction_mf3e import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "artifacts/training/mf3v_horizon_ranker_v1"
OUT = ROOT / "artifacts/evaluation/mf3y_consensus_tail_shadow_gate_v1"
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3Y_CONSENSUS_TAIL_GATE.md"
SEEDS = (20260826, 20260827, 20260828)
HORIZON = 3
HIDDEN = 128
MAD_WEIGHT = 0.5
BETA = 0.25
LOW_QUANTILE = 0.985
UPPER_QUANTILE = 0.995
CONSENSUS_MAD_QUANTILE = 0.75
CONSENSUS_MARGIN_QUANTILE = 0.25


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(device)
    fit = collect(models, "fit", manifest_path("final"), device)
    shadow = collect(models, "shadow", DATA, device)
    fit_rows = []
    for sequence in fit:
        for row in sequence:
            if row is None:
                continue
            median = sorted(row["member_logits"])[1]
            mad = sorted(abs(value - median) for value in row["member_logits"])[1]
            fit_rows.append((score(row), mad, row["native_margin"]))
    lower = float(torch.quantile(torch.tensor([row[0] for row in fit_rows]), LOW_QUANTILE))
    upper = float(torch.quantile(torch.tensor([row[0] for row in fit_rows]), UPPER_QUANTILE))
    mad_threshold = float(torch.quantile(torch.tensor([row[1] for row in fit_rows]), CONSENSUS_MAD_QUANTILE))
    margin_threshold = float(torch.quantile(torch.tensor([row[2] for row in fit_rows]), CONSENSUS_MARGIN_QUANTILE))
    selected = []
    for sequence in shadow:
        for row in sequence:
            if row is None:
                continue
            median = sorted(row["member_logits"])[1]
            mad = sorted(abs(value - median) for value in row["member_logits"])[1]
            value = score(row)
            upper_ok = value <= upper or (
                mad <= mad_threshold and row["native_margin"] <= margin_threshold
            )
            if value > lower and upper_ok:
                selected.append(row)
                break
    candidate = {
        "interventions": len(selected),
        "rescues": sum(row["outcome"] == "RESCUE" for row in selected),
        "harms": sum(row["outcome"] == "HARM" for row in selected),
        "neither": sum(row["outcome"] == "NEITHER" for row in selected),
    }
    candidate["net_rescues"] = candidate["rescues"] - candidate["harms"]
    control = exact_control(shadow, candidate["interventions"])
    gates = {
        "fresh_shadow_has_twenty_interventions": candidate["interventions"] >= 20,
        "fresh_shadow_net_rescue_positive": candidate["net_rescues"] > 0,
        "fresh_shadow_beats_exact_budget_control": candidate["net_rescues"] > control["net_rescues"] or (
            candidate["net_rescues"] == control["net_rescues"] and candidate["harms"] < control["harms"]
        ),
    }
    passed = all(gates.values())
    protocol = TRAIN / "MF3V_TRAINING_PROTOCOL.json"
    atomic_json(OUT / "MF3Y_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3y-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "selected_architecture": {"hidden_dim": HIDDEN},
        "selected_rule": {
            "hidden_dim": HIDDEN, "horizon": HORIZON, "mad_weight": MAD_WEIGHT,
            "policy_risk_beta": BETA, "training_score_quantile": LOW_QUANTILE,
            "final_training_threshold": lower, "score_upper_quantile": UPPER_QUANTILE,
            "score_upper_threshold": upper, "persistence_steps": 1,
            "consensus_mad_quantile": CONSENSUS_MAD_QUANTILE,
            "consensus_mad_threshold": mad_threshold,
            "consensus_margin_quantile": CONSENSUS_MARGIN_QUANTILE,
            "consensus_margin_threshold": margin_threshold,
        },
        "shadow": candidate, "exact_budget_control": control,
        "uncertainty_rule": {"native_margin_max": control["native_margin_max"]},
        "gates": gates, "task_metric_run_authorized": passed,
        "fresh_data": {"path": str(DATA.relative_to(ROOT)), "bytes": DATA.stat().st_size, "sha256": sha256_file(DATA), "shadow_episodes": 336},
        "checkpoints": checkpoints,
        "training_protocol_sha256": sha256_file(protocol),
        "design_sha256": sha256_file(DESIGN),
        "public_unseen_authorized": False,
        "not_authorized_for_unseen": True,
    })
    print(json.dumps({"status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL", "thresholds": {"lower": lower, "upper": upper, "consensus_mad": mad_threshold, "consensus_margin": margin_threshold}, "shadow": candidate, "control": control, "gates": gates}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
