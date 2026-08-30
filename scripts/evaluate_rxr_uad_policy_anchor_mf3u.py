#!/usr/bin/env python3
"""Evaluate the sealed MF3T ranker with its policy-anchor trust region."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import MF3B_SCOPE
from scripts.evaluate_rxr_uad_coverage_shadow_mf3t import DATA as DATA36_41
from scripts.select_rxr_uad_coverage_mf3t import load_models
from scripts.select_rxr_uad_policy_risk_mf3s import exact_control, hybrid
from scripts.select_rxr_uad_rescue_harm_mf3p import collect
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file
from scripts.train_rxr_uad_crossfit_mf3q import manifest_path

BASE_GATE = ROOT / (
    "artifacts/evaluation/mf3t_coverage_shadow_gate_v2/"
    "MF3T_SHADOW_GATE.json"
)
OUT = ROOT / "artifacts/evaluation/mf3u_policy_anchor_shadow_gate_v1"
DESIGN = ROOT / (
    "artifacts/design/METHOD_FREEZE_3U_POLICY_ANCHORED_COVERAGE_GUARD.md"
)


def guarded_summary(episodes, weight, beta, cutoff, persistence, upper):
    selected = []
    for sequence in episodes:
        run = 0
        for row in sequence:
            eligible = (
                row is not None
                and hybrid(row, weight, beta) > cutoff
                and hybrid(row, weight, beta) <= upper
            )
            run = run + 1 if eligible else 0
            if run >= persistence:
                selected.append(row)
                break
    rescues = sum(row["outcome"] == "RESCUE" for row in selected)
    harms = sum(row["outcome"] == "HARM" for row in selected)
    return {
        "interventions": len(selected), "rescues": rescues,
        "harms": harms, "neither": len(selected) - rescues - harms,
        "net_rescues": rescues - harms,
    }


def main() -> int:
    base = json.loads(BASE_GATE.read_text())
    if not (
        base.get("status") == "SHADOW_GATE_PASS"
        and base.get("task_metric_run_authorized") is True
    ):
        raise RuntimeError("MF3T prerequisite gate is not PASS")
    rule = base["selected_rule"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(int(rule["hidden_dim"]), "final", device)
    fit_episodes = collect(models, "fit", device, manifest_path("final"))
    fit_scores = [
        hybrid(row, float(rule["mad_weight"]), float(rule["policy_risk_beta"]))
        for sequence in fit_episodes for row in sequence if row is not None
    ]
    upper_quantile = 0.995
    score_upper = float(torch.quantile(torch.tensor(fit_scores), upper_quantile))
    episodes = collect(models, "shadow", device, DATA36_41)
    shadow = guarded_summary(
        episodes, float(rule["mad_weight"]), float(rule["policy_risk_beta"]),
        float(rule["final_training_threshold"]), int(rule["persistence_steps"]),
        score_upper,
    )
    control = exact_control(episodes, shadow["interventions"])
    gates = {
        "fresh_shadow_has_twenty_interventions": shadow["interventions"] >= 20,
        "fresh_shadow_net_rescue_positive": shadow["net_rescues"] > 0,
        "fresh_shadow_beats_exact_budget_control": (
            shadow["net_rescues"] > control["net_rescues"]
            or (
                shadow["net_rescues"] == control["net_rescues"]
                and shadow["harms"] < control["harms"]
            )
        ),
    }
    passed = all(gates.values())
    atomic_json(OUT / "MF3U_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3u-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "selected_architecture": {"hidden_dim": int(rule["hidden_dim"])},
        "selected_rule": {
            **{key: rule[key] for key in (
                "hidden_dim", "mad_weight", "policy_risk_beta",
                "training_score_quantile", "final_training_threshold",
                "persistence_steps",
            )},
            "score_upper_quantile": upper_quantile,
            "score_upper_threshold": score_upper,
            "source": "MF3T final-fit score distribution",
        },
        "shadow": shadow, "exact_budget_control": control, "gates": gates,
        "task_metric_run_authorized": passed,
        "mf3t_prerequisite_gate_sha256": sha256_file(BASE_GATE),
        "fresh_data": {
            "path": str(DATA36_41.relative_to(ROOT)),
            "bytes": DATA36_41.stat().st_size,
            "sha256": sha256_file(DATA36_41), "shadow_episodes": 336,
        },
        "ranks36_41_payload_read": True,
        "design_sha256": sha256_file(DESIGN),
        **MF3B_SCOPE,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
