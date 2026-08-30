#!/usr/bin/env python3
"""Open the sealed MF3K ranks-15--17 shadow exactly once."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import MF3B_SCOPE  # noqa: E402
from scripts.select_rxr_uad_policy_top2_mf3k import (  # noqa: E402
    OUT as DEVELOPMENT,
    collect,
    load_models,
    summarize,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file  # noqa: E402

DATA = ROOT / (
    "artifacts/phase1/mf3k_policy_top2_rank17/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
SELECTION = DEVELOPMENT / "MF3K_DEVELOPMENT_SELECTION.json"
OUT = ROOT / "artifacts/evaluation/mf3k_policy_top2_shadow_gate_v1"


def uncertainty_control(rows: list[dict], budget: int) -> dict:
    selected = sorted(rows, key=lambda row: row["native_margin"])[:budget]
    rescues = sum(row["outcome"] == "RESCUE" for row in selected)
    harms = sum(row["outcome"] == "HARM" for row in selected)
    neither = sum(row["outcome"] == "NEITHER" for row in selected)
    return {
        "matched_intervention_budget": budget,
        "interventions": len(selected),
        "rescues": rescues,
        "harms": harms,
        "neither": neither,
        "net_rescues": rescues - harms,
    }


def main() -> int:
    selection = json.loads(SELECTION.read_text())
    if not (
        selection.get("status") == "DEVELOPMENT_PASS"
        and selection.get("ranks15_17_payload_read") is False
    ):
        raise RuntimeError("MF3K development does not authorize fresh shadow")
    manifest = json.loads(DATA.read_text())
    if not (
        manifest.get("status") == "PASS"
        and manifest.get("counts", {}).get("fit") == 519
        and manifest.get("counts", {}).get("calibration") == 112
        and manifest.get("counts", {}).get("diagnostic") == 168
        and manifest.get("counts", {}).get("shadow", 0) >= 160
        and all(
            row.get("observation_frontend")
            == "frozen_etp_r1_policy_fusion_token"
            and int(row.get("candidate_feature_dim")) == 1536
            for row in manifest["records"]
        )
    ):
        raise RuntimeError("MF3K ranks15-17 manifest gate failed")
    architecture = selection["selected_architecture"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(
        int(architecture["hidden_dim"]),
        float(architecture["correction_bound"]), device,
    )
    rows = collect(models, "shadow", device, DATA)
    rule = selection["selected_rule"]
    shadow = summarize(
        rows, float(rule["utility_threshold"]),
        float(rule["native_margin_max"]),
    )
    uncertainty = uncertainty_control(rows, shadow["interventions"])
    gates = {
        "fresh_shadow_has_fifteen_interventions": (
            shadow["interventions"] >= 15
        ),
        "fresh_shadow_net_rescue_positive": shadow["net_rescues"] > 0,
        "fresh_shadow_beats_uncertainty": (
            shadow["net_rescues"] > uncertainty["net_rescues"]
            or (
                shadow["net_rescues"] == uncertainty["net_rescues"]
                and shadow["harms"] < uncertainty["harms"]
            )
        ),
    }
    passed = all(gates.values())
    atomic_json(OUT / "MF3K_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3k-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "selected_architecture": architecture,
        "selected_rule": rule,
        "shadow": shadow,
        "uncertainty_matched_shadow": uncertainty,
        "uncertainty_calibration_budget_match": selection[
            "uncertainty_development_budget_match"
        ],
        "gates": gates,
        "task_metric_run_authorized": passed,
        "ranks15_17_payload_read": True,
        "rank14_role": "consumed_development_after_mf3j_failure",
        "fresh_data": {
            "path": str(DATA.relative_to(ROOT)),
            "bytes": DATA.stat().st_size,
            "sha256": sha256_file(DATA),
            "shadow_episodes": manifest["counts"]["shadow"],
        },
        "checkpoints": checkpoints,
        **MF3B_SCOPE,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
