#!/usr/bin/env python3
"""Run the once-only rank-14 gate for the sealed MF3J switch rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import MF3B_SCOPE  # noqa: E402
from scripts.select_rxr_uad_switch_rule_mf3j import (  # noqa: E402
    DATA,
    OUT as DEVELOPMENT_OUT,
    collect,
    counts,
    load_models,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file  # noqa: E402

OUT = ROOT / "artifacts/evaluation/mf3j_switch_utility_shadow_gate_v1"


def uncertainty_control(rows: list[dict], budget: int) -> dict:
    selected = sorted(rows, key=lambda row: row["native_margin"])[:budget]
    rescues = sum(row["runner_up_outcome"] == "RESCUE" for row in selected)
    harms = sum(row["runner_up_outcome"] == "HARM" for row in selected)
    neither = sum(row["runner_up_outcome"] == "NEITHER" for row in selected)
    return {
        "matched_intervention_budget": budget,
        "rescues": rescues,
        "harms": harms,
        "neither": neither,
        "net_rescues": rescues - harms,
    }


def main() -> int:
    selection_path = DEVELOPMENT_OUT / "MF3J_DEVELOPMENT_SELECTION.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "DEVELOPMENT_PASS":
        raise RuntimeError("MF3J development does not authorize shadow")
    prior = json.loads((ROOT / (
        "artifacts/evaluation/mf3i_contextual_uad_shadow_gate_v1/"
        "MF3I_UAD_SHADOW_GATE.json"
    )).read_text())
    if prior.get("shadow") != {}:
        raise RuntimeError("rank-14 was already opened")
    hidden = int(selection["selected_architecture"]["hidden_dim"])
    rule = selection["selected_rule"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(hidden, device)
    shadow_rows = collect(models, "shadow", device)
    shadow = counts(shadow_rows, rule["agreement"], float(rule["threshold"]))
    uncertainty = uncertainty_control(shadow_rows, shadow["interventions"])
    gates = {
        "fresh_shadow_has_five_interventions": shadow["interventions"] >= 5,
        "fresh_shadow_net_rescue_positive": shadow["net_rescues"] > 0,
        "fresh_shadow_exceeds_uncertainty": (
            shadow["net_rescues"] > uncertainty["net_rescues"]
        ),
    }
    passed = all(gates.values())
    atomic_json(OUT / "MF3J_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3j-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "selected_architecture": selection["selected_architecture"],
        "selected_rule": rule,
        "shadow": shadow,
        "uncertainty_matched_shadow": uncertainty,
        "uncertainty_calibration_budget_match": selection[
            "uncertainty_calibration_budget_match"
        ],
        "gates": gates,
        "checkpoints": checkpoints,
        "development_selection_sha256": sha256_file(selection_path),
        "data_sha256": sha256_file(DATA),
        "rank14_payload_read": True,
        "execution_recovery": {
            "attempts": 2,
            "first_attempt_result_observed": False,
            "first_attempt_failure": (
                "post-evaluation KeyError while serializing a missing "
                "uncertainty_calibration_budget_match field"
            ),
            "old_selection_sha256": (
                "9667f1907472056de3e41efccd9421df3b34dc464b33f16d71507a8bf0771b05"
            ),
            "model_rule_or_gate_changed": False,
        },
        "task_metric_run_authorized": passed,
        **MF3B_SCOPE,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
