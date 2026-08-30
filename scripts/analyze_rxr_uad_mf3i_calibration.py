#!/usr/bin/env python3
"""Calibration-only analysis for the rejected MF3I posterior gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import (  # noqa: E402
    NativeConditionedUAD,
    OnlineUADFeatureDataset,
    collate_online_uad,
    native_alternative_posterior_gain,
    native_residual_logits,
)

DATA = ROOT / (
    "artifacts/phase1/mf3i_policy_token_uad/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
TRAIN = ROOT / "artifacts/training/mf3i_policy_token_uad_v1"
SEEDS = (20260826, 20260827, 20260828)


def load(device):
    result = []
    for seed in SEEDS:
        payload = torch.load(
            TRAIN / f"seed_{seed}/uad_contextual_mf3i.pt",
            map_location="cpu", weights_only=True,
        )
        model = NativeConditionedUAD(
            768, int(payload["hidden_dim"]), candidate_feature_dim=1536
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        result.append(model.to(device).eval())
    return tuple(result)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = load(device)
    loader = DataLoader(
        OnlineUADFeatureDataset(DATA, "calibration"), batch_size=1,
        shuffle=False, collate_fn=collate_online_uad,
    )
    rows = []
    with torch.no_grad():
        for cpu in loader:
            batch = {key: value.to(device) for key, value in cpu.items()}
            outputs = tuple(model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            ) for model in models)
            fused = tuple(native_residual_logits(
                output, batch["native_scores"], batch["candidate_mask"],
                correction_bound=1.0,
            )[0] for output in outputs)
            choices = tuple(value.argmax(-1) for value in fused)
            gains = tuple(native_alternative_posterior_gain(
                output, choice
            ) for output, choice in zip(outputs, choices))
            for step in range(int(batch["step_mask"][0].sum())):
                mask = batch["candidate_mask"][0, step]
                native = int(batch["native_index"][0, step])
                target = int(batch["target_index"][0, step])
                if native < 0 or target < 0 or int(mask.sum()) < 2:
                    continue
                member_choices = tuple(int(choice[0, step]) for choice in choices)
                member_gains = tuple(float(gain[0, step]) for gain in gains)
                if len(set(member_choices)) != 1 or member_choices[0] == native:
                    continue
                adapted = member_choices[0]
                values = batch["native_scores"][0, step, mask]
                order = torch.argsort(values, descending=True)
                outcome = (
                    "RESCUE" if adapted == target and native != target else
                    "HARM" if native == target and adapted != target else
                    "NEITHER"
                )
                rows.append({
                    "outcome": outcome,
                    "minimum_gain": min(member_gains),
                    "median_gain": sorted(member_gains)[1],
                    "maximum_gain": max(member_gains),
                    "native_margin": float(values[order[0]] - values[order[1]]),
                    "member_choices": member_choices,
                    "member_gains": member_gains,
                })
    output = {
        "role": "calibration-only method development; fresh shadow untouched",
        "interventions": len(rows), "rows": rows,
        "threshold_scan": {
            str(threshold): {
                "interventions": sum(row["minimum_gain"] > threshold for row in rows),
                "rescues": sum(
                    row["minimum_gain"] > threshold and row["outcome"] == "RESCUE"
                    for row in rows
                ),
                "harms": sum(
                    row["minimum_gain"] > threshold and row["outcome"] == "HARM"
                    for row in rows
                ),
            }
            for threshold in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        },
    }
    path = ROOT / "artifacts/diagnostics/MF3I_CALIBRATION_GATE_ANALYSIS.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
