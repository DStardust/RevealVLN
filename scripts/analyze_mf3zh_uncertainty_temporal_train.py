#!/usr/bin/env python3
"""Analyze first uncertainty crossings using only frozen RxR-train proxy labels."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_rxr_uad_horizon_mf3v import collect, load_models, manifest_path, score
from train_rxr_uad_correction_mf3e import atomic_json, sha256_file


MF3V_GATE = ROOT / (
    "artifacts/evaluation/mf3v_horizon_shadow_gate_v1/MF3V_SHADOW_GATE.json"
)
OUT = ROOT / (
    "artifacts/analysis/mf3zh_uncertainty_temporal_train_v1/"
    "MF3ZH_UNCERTAINTY_TEMPORAL_TRAIN.json"
)


def main() -> int:
    gate = json.loads(MF3V_GATE.read_text())
    margin_max = float(gate["exact_budget_control"]["native_margin_max"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(device)
    source = manifest_path("final")
    sequences = collect(models, "fit", source, device)
    records = json.loads(source.read_text())["records"]
    fit_records = [row for row in records if row["split"] == "fit"]
    if len(sequences) != len(fit_records) or len(sequences) != 1303:
        raise RuntimeError("MF3ZH train alignment drift")
    selected = []
    for metadata, sequence in zip(fit_records, sequences):
        for step, row in enumerate(sequence):
            if row is not None and float(row["native_margin"]) <= margin_max:
                selected.append({
                    "episode_id": str(metadata["episode_id"]),
                    "scene_id": str(metadata["scene_id"]),
                    "step": step,
                    "native_margin": float(row["native_margin"]),
                    "mf3v_score": float(score(row)),
                    "proxy_outcome": row["outcome"],
                })
                break
    candidates = []
    for maximum_step in range(0, 31):
        rows = [row for row in selected if row["step"] <= maximum_step]
        outcomes = Counter(row["proxy_outcome"] for row in rows)
        candidates.append({
            "maximum_step": maximum_step,
            "interventions": len(rows),
            "rescues": outcomes["RESCUE"],
            "harms": outcomes["HARM"],
            "neither": outcomes["NEITHER"],
            "net_rescues": outcomes["RESCUE"] - outcomes["HARM"],
        })
    payload = {
        "schema_version": "revealnav-mf3zh-uncertainty-temporal-train/1",
        "status": "TRAIN_ONLY_ANALYSIS_READY",
        "rows": selected,
        "counts": {
            "fit_episodes": len(sequences),
            "first_crossings": len(selected),
            "scenes": len({row["scene_id"] for row in selected}),
        },
        "candidates": candidates,
        "margin_max": margin_max,
        "sources": {
            "manifest_sha256": sha256_file(source),
            "mf3v_gate_sha256": sha256_file(MF3V_GATE),
            "checkpoints": checkpoints,
        },
        "unseen_or_test_read": False,
    }
    atomic_json(OUT, payload)
    print(json.dumps({
        "counts": payload["counts"],
        "candidates": candidates,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
