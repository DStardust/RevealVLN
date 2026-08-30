#!/usr/bin/env python3
"""One-shot MF3J development and gate monitor."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "artifacts/training/mf3j_switch_utility_v1"

complete = []
for hidden in (64, 128):
    for seed in (20260826, 20260827, 20260828):
        path = TRAIN / f"hidden_{hidden}/seed_{seed}/RESULT.json"
        if path.is_file():
            value = json.loads(path.read_text())
            metric = value["calibration"]
            complete.append((hidden, seed))
            print(
                f"h{hidden} seed {seed}: nll={metric['pairwise_nll']:.4f} "
                f"proposal R/H/N={metric['ungated_proposal_rescues']}/"
                f"{metric['ungated_proposal_harms']}/"
                f"{metric['ungated_proposal_neither']}"
            )
print(f"training complete: {len(complete)}/6")

selection_path = ROOT / (
    "artifacts/evaluation/mf3j_switch_utility_development_v1/"
    "MF3J_DEVELOPMENT_SELECTION.json"
)
if selection_path.is_file():
    value = json.loads(selection_path.read_text())
    print(f"development: {value['status']}")
    if value.get("selected_rule"):
        print(
            f"selected h={value['selected_architecture']['hidden_dim']} "
            f"agreement={value['selected_rule']['agreement']} "
            f"threshold={value['selected_rule']['threshold']:.2f} "
            f"R/H/N={value['selected_rule']['rescues']}/"
            f"{value['selected_rule']['harms']}/"
            f"{value['selected_rule']['neither']}"
        )
else:
    print("development: pending")

gate_path = ROOT / (
    "artifacts/evaluation/mf3j_switch_utility_shadow_gate_v1/"
    "MF3J_SHADOW_GATE.json"
)
if gate_path.is_file():
    value = json.loads(gate_path.read_text())
    print(f"shadow: {value['status']}")
else:
    print("shadow: unopened")
