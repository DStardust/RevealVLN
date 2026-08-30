#!/usr/bin/env python3
"""Describe fixed MF3G ensemble interventions without selecting a policy."""

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
    median_native_conditioned_outputs,
    native_residual_logits,
)


SEEDS = (20260826, 20260827, 20260828)
DATA = ROOT / (
    "artifacts/phase1/mf3h_uad_online_rank12/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
TRAIN = ROOT / "artifacts/training/mf3g_uad_residual_v1"
OUT = ROOT / "artifacts/diagnostics/MF3H_CONSENSUS_INTERVENTIONS.json"


def models(device: torch.device):
    result = []
    for seed in SEEDS:
        value = torch.load(
            TRAIN / f"seed_{seed}/uad_residual_mf3g.pt",
            map_location="cpu", weights_only=True,
        )
        model = NativeConditionedUAD(768, int(value["hidden_dim"]))
        model.load_state_dict(value["model_state_dict"], strict=True)
        result.append(model.to(device).eval())
    return tuple(result)


def analyze(split: str, ensemble, device: torch.device) -> dict:
    loader = DataLoader(
        OnlineUADFeatureDataset(DATA, split), batch_size=1, shuffle=False,
        collate_fn=collate_online_uad,
    )
    interventions = []
    eligible = 0
    with torch.no_grad():
        for cpu in loader:
            batch = {key: value.to(device) for key, value in cpu.items()}
            outputs = tuple(model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            ) for model in ensemble)
            median = median_native_conditioned_outputs(outputs)
            fused_members = tuple(native_residual_logits(
                output, batch["native_scores"], batch["candidate_mask"],
                correction_bound=2.0,
            )[0] for output in outputs)
            fused_median = native_residual_logits(
                median, batch["native_scores"], batch["candidate_mask"],
                correction_bound=2.0,
            )[0]
            steps = int(batch["step_mask"][0].sum())
            for step in range(steps):
                mask = batch["candidate_mask"][0, step]
                native = int(batch["native_index"][0, step])
                target = int(batch["target_index"][0, step])
                if native < 0 or target < 0 or int(mask.sum()) < 2:
                    continue
                eligible += 1
                choices = [int(value[0, step].argmax()) for value in fused_members]
                if len(set(choices)) != 1 or choices[0] == native:
                    continue
                adapted = choices[0]
                native_values = batch["native_scores"][0, step, mask]
                native_order = torch.argsort(native_values, descending=True)
                median_values = fused_median[0, step, mask]
                median_order = torch.argsort(median_values, descending=True)
                indices = torch.nonzero(mask, as_tuple=False).flatten()
                error_probabilities = [
                    float(torch.sigmoid(output.native_error_logit[0, step]))
                    for output in outputs
                ]
                alternative_probabilities = [
                    float(torch.softmax(
                        output.alternative_logits[0, step, mask], dim=0
                    )[torch.nonzero(indices == adapted, as_tuple=False)[0, 0]])
                    for output in outputs
                ]
                native_correct = native == target
                adapted_correct = adapted == target
                outcome = (
                    "RESCUE" if adapted_correct and not native_correct else
                    "HARM" if native_correct and not adapted_correct else
                    "NEITHER"
                )
                interventions.append({
                    "outcome": outcome,
                    "native_margin": float(
                        native_values[native_order[0]]
                        - native_values[native_order[1]]
                    ),
                    "fused_margin": float(
                        median_values[median_order[0]]
                        - median_values[median_order[1]]
                    ),
                    "error_probability_min": min(error_probabilities),
                    "error_probability_median": sorted(error_probabilities)[1],
                    "alternative_probability_min": min(alternative_probabilities),
                    "alternative_probability_median": sorted(
                        alternative_probabilities
                    )[1],
                    "posterior_gain_min": min(
                        error * alternative - (1.0 - error)
                        for error, alternative in zip(
                            error_probabilities, alternative_probabilities
                        )
                    ),
                })
    return {
        "eligible": eligible,
        "interventions": len(interventions),
        "rows": interventions,
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble = models(device)
    value = {
        "role": "post-failure diagnostics only; not a policy selection artifact",
        "splits": {
            split: analyze(split, ensemble, device)
            for split in ("fit", "calibration", "diagnostic", "shadow")
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
