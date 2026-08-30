#!/usr/bin/env python3
"""Seal the pre-registered MF3ZA consensus-band train-shadow rule."""

from __future__ import annotations

import json
from pathlib import Path

from train_rxr_uad_correction_mf3e import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "artifacts/evaluation/mf3z_adaptive_tail_development_v1/MF3Z_ADAPTIVE_TAIL_DEVELOPMENT.json"
OUT_DIR = ROOT / "artifacts/evaluation/mf3za_consensus_band_shadow_gate_v1"
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3ZA_CONSENSUS_BAND.md"


def main() -> int:
    development = json.loads(DEV.read_text())
    candidates = development.get("candidates", [])
    selected = next(
        (
            row for row in candidates
            if row.get("mad_floor_quantile") == 0.25
            and row.get("mad_quantile") == 0.75
            and row.get("ratio_quantile") == 0.75
        ),
        None,
    )
    if development.get("status") != "DEVELOPMENT_PASS" or selected is None:
        raise RuntimeError("MF3ZA pre-registered train-shadow rule is unavailable")
    shadow = selected["shadow"]
    control = selected["exact_budget_control"]
    gates = {
        "train_shadow_has_twenty_interventions": shadow["interventions"] >= 20,
        "train_shadow_net_rescue_positive": shadow["net_rescues"] > 0,
        "train_shadow_beats_exact_budget_control": (
            shadow["net_rescues"] > control["net_rescues"]
            or (
                shadow["net_rescues"] == control["net_rescues"]
                and shadow["harms"] < control["harms"]
            )
        ),
        "train_shadow_harm_rate_at_most_twenty_percent": (
            shadow["harms"] / shadow["interventions"] <= 0.20
        ),
    }
    rule = {
        "hidden_dim": 128,
        "horizon": 3,
        "mad_weight": 0.5,
        "policy_risk_beta": 0.25,
        "training_score_quantile": development["fit_only_thresholds"]["lower_quantile"],
        "final_training_threshold": development["fit_only_thresholds"]["lower_threshold"],
        "score_upper_quantile": development["fit_only_thresholds"]["upper_quantile"],
        "score_upper_threshold": development["fit_only_thresholds"]["upper_threshold"],
        "persistence_steps": 1,
        "consensus_mad_floor_quantile": selected["mad_floor_quantile"],
        "consensus_mad_floor_threshold": selected["mad_floor_threshold"],
        "consensus_mad_quantile": selected["mad_quantile"],
        "consensus_mad_threshold": selected["mad_threshold"],
        "consensus_relative_margin_quantile": selected["ratio_quantile"],
        "consensus_relative_margin_threshold": selected["ratio_threshold"],
    }
    passed = all(gates.values())
    atomic_json(OUT_DIR / "MF3ZA_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3za-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "selected_architecture": {"hidden_dim": 128},
        "selected_rule": rule,
        "shadow": shadow,
        "exact_budget_control": control,
        "gates": gates,
        "task_metric_run_authorized": passed,
        "public_unseen_authorized": False,
        "not_authorized_for_unseen": True,
        "development_sha256": sha256_file(DEV),
        "design_sha256": sha256_file(DESIGN),
        "train_manifest": development["train_manifest"],
        "checkpoints": development["checkpoints"],
    })
    print(json.dumps({"status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL", "rule": rule, "gates": gates}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
