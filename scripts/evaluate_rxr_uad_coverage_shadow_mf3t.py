#!/usr/bin/env python3
"""Open MF3T ranks 36--41 once and apply its predeclared shadow gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import MF3B_SCOPE
from scripts.select_rxr_uad_rescue_harm_mf3p import collect
from scripts.select_rxr_uad_coverage_mf3t import load_models
from scripts.select_rxr_uad_policy_risk_mf3s import exact_control, hybrid, summarize
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file


DATA = ROOT / (
    "artifacts/phase1/mf3t_coverage_gate_rank41/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
SELECTION = ROOT / (
    "artifacts/evaluation/mf3t_coverage_development_v2/"
    "MF3T_DEVELOPMENT_SELECTION.json"
)
OUT = ROOT / "artifacts/evaluation/mf3t_coverage_shadow_gate_v2"


def main() -> int:
    selection = json.loads(SELECTION.read_text())
    if not (
        selection.get("status") == "DEVELOPMENT_PASS"
        and selection.get("ranks36_41_payload_read") is False
    ):
        raise RuntimeError("MF3T development does not authorize shadow")
    manifest = json.loads(DATA.read_text())
    if not (
        manifest.get("status") == "PASS"
        and manifest.get("counts") == {"shadow": 336}
    ):
        raise RuntimeError("MF3T fresh manifest drift")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rule = selection["selected_rule"]
    models, checkpoints = load_models(int(rule["hidden_dim"]), "final", device)
    episodes = collect(models, "shadow", device, DATA)
    shadow = summarize(
        episodes,
        float(rule["mad_weight"]),
        float(rule["policy_risk_beta"]),
        float(rule["final_training_threshold"]),
        int(rule["persistence_steps"]),
    )
    control = exact_control(episodes, shadow["interventions"])
    gates = {
        "fresh_shadow_has_thirty_interventions": shadow["interventions"] >= 30,
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
    atomic_json(OUT / "MF3T_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3t-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "selected_rule": rule,
        "shadow": shadow,
        "exact_budget_control": control,
        "gates": gates,
        "task_metric_run_authorized": passed,
        "ranks36_41_payload_read": True,
        "fresh_data": {
            "path": str(DATA.relative_to(ROOT)),
            "bytes": DATA.stat().st_size,
            "sha256": sha256_file(DATA),
            "shadow_episodes": 336,
        },
        "checkpoints": checkpoints,
        **MF3B_SCOPE,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
