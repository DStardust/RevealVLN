#!/usr/bin/env python3
"""Train-only diagnostic for a conservative MF3V widened score band."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluate_rxr_uad_horizon_mf3v import DATA, MAD_WEIGHT, BETA, collect, load_models, manifest_path, score
from select_rxr_uad_policy_risk_mf3s import exact_control
from train_rxr_uad_correction_mf3e import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/evaluation/mf3v_widened_band_shadow_diagnostic_v1"
LOW_QUANTILE = 0.985
UPPER_QUANTILE = 0.9975


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(device)
    fit = collect(models, "fit", manifest_path("final"), device)
    shadow = collect(models, "shadow", DATA, device)
    values = torch.tensor([score(row) for sequence in fit for row in sequence if row is not None])
    lower = float(torch.quantile(values, LOW_QUANTILE))
    upper = float(torch.quantile(values, UPPER_QUANTILE))
    selected = []
    for sequence in shadow:
        for row in sequence:
            if row is not None and lower < score(row) <= upper:
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
        "twenty_interventions": candidate["interventions"] >= 20,
        "positive_net_rescue": candidate["net_rescues"] > 0,
        "beats_exact_budget": candidate["net_rescues"] > control["net_rescues"] or (
            candidate["net_rescues"] == control["net_rescues"] and candidate["harms"] < control["harms"]
        ),
    }
    atomic_json(OUT / "MF3V_WIDENED_BAND_SHADOW_DIAGNOSTIC.json", {
        "schema_version": "revealnav-mf3v-widened-band-diagnostic/1",
        "status": "TRAIN_ONLY_CANDIDATE_PASS" if all(gates.values()) else "TRAIN_ONLY_CANDIDATE_FAIL",
        "scope": "diagnostic only; train fit and shadow labels only",
        "candidate": "MF3X widened band: q98.5 < score <= q99.75",
        "training_score_quantile": LOW_QUANTILE,
        "upper_score_quantile": UPPER_QUANTILE,
        "fit_lower_threshold": lower,
        "fit_upper_threshold": upper,
        "mad_weight": MAD_WEIGHT,
        "policy_risk_beta": BETA,
        "shadow": candidate,
        "exact_budget_control": control,
        "gates": gates,
        "checkpoints": checkpoints,
        "shadow_manifest": {"path": str(DATA.relative_to(ROOT)), "bytes": DATA.stat().st_size, "sha256": sha256_file(DATA)},
        "not_authorized_for_unseen": True,
    })
    print(json.dumps({"status": "TRAIN_ONLY_CANDIDATE_PASS" if all(gates.values()) else "TRAIN_ONLY_CANDIDATE_FAIL", "lower": lower, "upper": upper, "candidate": candidate, "control": control, "gates": gates}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
