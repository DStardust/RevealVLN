#!/usr/bin/env python3
"""One-shot MF3M fresh-confirmation and task-metric monitor."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    value = ROOT / path
    return json.loads(value.read_text()) if value.is_file() else None


development = load(
    "artifacts/evaluation/mf3m_robust_top2_development_v1/"
    "MF3M_DEVELOPMENT_SELECTION.json"
)
print("MF3M development:", development.get("status") if development else "MISSING")
if development and development.get("selected_rule"):
    rule = development["selected_rule"]
    print("  rule:", {
        "mad_weight": rule["mad_weight"],
        "threshold": rule["robust_advantage_threshold"],
        "pooled": rule["pooled"],
        "strata": rule["strata"],
        "coverage_ratio": rule["stratum_intervention_ratio"],
        "wilson95_lower": rule["rescue_precision_wilson95_lower"],
    })
progress = load(
    "artifacts/phase1/mf3m_robust_top2_rank23/dataset_v1/"
    "MF3B_ONLINE_DATA_PROGRESS.json"
)
print("fresh ranks18-23:", (
    {key: progress.get(key) for key in (
        "status", "completed", "total", "new_completed", "new_total",
        "remaining", "failed", "eta_s",
    )} if progress else "NOT_STARTED"
))
shadow = load(
    "artifacts/evaluation/mf3m_robust_top2_shadow_gate_v1/"
    "MF3M_SHADOW_GATE.json"
)
print("shadow gate:", shadow.get("status") if shadow else "UNOPENED")
if shadow:
    print("  learned:", shadow["shadow"])
    print("  uncertainty:", shadow["uncertainty_matched_shadow"])
metrics = load(
    "artifacts/evaluation/mf3m_uad_rxr_val_seen_v1/"
    "MF3M_RXR_VAL_SEEN_PROGRESS.json"
)
print("RxR val_seen:", metrics if metrics else "NOT_AUTHORIZED_OR_NOT_STARTED")
result = load(
    "artifacts/evaluation/mf3m_uad_rxr_val_seen_v1/"
    "MF3M_RXR_VAL_SEEN_RESULT.json"
)
print("task metric result:", result.get("status") if result else "PENDING")
