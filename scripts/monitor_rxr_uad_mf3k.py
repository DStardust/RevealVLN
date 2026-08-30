#!/usr/bin/env python3
"""One-shot monitor for the persistent MF3K development-to-metrics chain."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    value = ROOT / path
    return json.loads(value.read_text()) if value.is_file() else None


results = list((ROOT / "artifacts/training/mf3k_policy_top2_v1").glob(
    "hidden_*_bound_*/seed_*/RESULT.json"
))
print(f"MF3K training: {len(results)}/12 models")
development = load(
    "artifacts/evaluation/mf3k_policy_top2_development_v1/"
    "MF3K_DEVELOPMENT_SELECTION.json"
)
print("development:", development.get("status") if development else "WAITING")
if development and development.get("selected_rule"):
    rule = development["selected_rule"]
    print("  architecture:", development["selected_architecture"])
    print("  rule:", {
        "utility_threshold": rule["utility_threshold"],
        "native_margin_max": rule["native_margin_max"],
        "pooled": rule["pooled"],
        "wilson95_lower": rule["rescue_precision_wilson95_lower"],
    })
progress = load(
    "artifacts/phase1/mf3k_policy_top2_rank17/dataset_v1/"
    "MF3B_ONLINE_DATA_PROGRESS.json"
)
print("fresh ranks15-17:", (
    {key: progress.get(key) for key in (
        "status", "completed", "total", "new_completed", "new_total",
        "remaining", "failed", "eta_s",
    )} if progress else "NOT_STARTED"
))
shadow = load(
    "artifacts/evaluation/mf3k_policy_top2_shadow_gate_v1/"
    "MF3K_SHADOW_GATE.json"
)
print("shadow gate:", shadow.get("status") if shadow else "UNOPENED")
if shadow:
    print("  learned:", shadow["shadow"])
    print("  uncertainty:", shadow["uncertainty_matched_shadow"])
metrics = load(
    "artifacts/evaluation/mf3k_uad_rxr_val_seen_v1/"
    "MF3K_RXR_VAL_SEEN_PROGRESS.json"
)
print("RxR val_seen:", metrics if metrics else "NOT_AUTHORIZED_OR_NOT_STARTED")
result = load(
    "artifacts/evaluation/mf3k_uad_rxr_val_seen_v1/"
    "MF3K_RXR_VAL_SEEN_RESULT.json"
)
print("task metric result:", result.get("status") if result else "PENDING")
