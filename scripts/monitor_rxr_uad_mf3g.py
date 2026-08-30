#!/usr/bin/env python3
"""One-shot monitor for the current MF3I UAD mainline."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


progress = read(ROOT / (
    "artifacts/phase1/mf3i_policy_token_uad/dataset_v1/"
    "MF3B_ONLINE_DATA_PROGRESS.json"
))
if progress:
    eta = progress.get("eta_s")
    eta_text = "pending" if eta is None else f"{eta / 60:.1f} min"
    print(
        f"data {progress['status']}: {progress['completed']}/{progress['total']} "
        f"active={len(progress['active'])} failed={progress['failed']} eta={eta_text}"
    )
else:
    print("data: not started")

for seed in (20260826, 20260827, 20260828):
    result = read(ROOT / (
        f"artifacts/training/mf3i_policy_token_uad_v1/seed_{seed}/RESULT.json"
    ))
    if result:
        metrics = result["calibration"]
        print(
            f"seed {seed}: complete posterior_acc={metrics['posterior_accuracy']:.4f} "
            f"native_acc={metrics['native_accuracy']:.4f} "
            f"rescue/harm={metrics['posterior_rescues']}/"
            f"{metrics['posterior_harms']}"
        )
    else:
        print(f"seed {seed}: pending")

gate = read(ROOT / (
    "artifacts/evaluation/mf3i_contextual_uad_shadow_gate_v1/"
    "MF3I_UAD_SHADOW_GATE.json"
))
print(f"shadow: {gate['status']}" if gate else "shadow: pending")
metric_progress = read(ROOT / (
    "artifacts/evaluation/mf3i_uad_rxr_val_seen_v1/"
    "MF3I_RXR_VAL_SEEN_PROGRESS.json"
))
if metric_progress:
    print(
        f"paired metrics {metric_progress['status']}: "
        f"{metric_progress['completed']}/{metric_progress['total']} "
        f"active={len(metric_progress['active'])} failed={metric_progress['failed']}"
    )
else:
    print("paired metrics: pending")
result = read(ROOT / (
    "artifacts/evaluation/mf3i_uad_rxr_val_seen_v1/"
    "MF3I_RXR_VAL_SEEN_RESULT.json"
))
if result:
    aggregate = result["aggregate_uad_ensemble_minus_baseline"]
    print(
        f"result {result['status']}: dSR={aggregate['success']['mean']:.4f} "
        f"dSPL={aggregate['spl']['mean']:.4f} "
        f"dnDTW={aggregate['ndtw']['mean']:.4f} "
        f"dUtility={aggregate['utility']['mean']:.4f}"
    )
