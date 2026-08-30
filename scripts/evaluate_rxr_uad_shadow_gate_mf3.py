#!/usr/bin/env python3
"""Calibrate on RxR-train scenes and evaluate the locked MF3 shadow gate."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import (  # noqa: E402
    MF3B_SCOPE,
    OnlineUADFeatureDataset,
    StructuredUADHeads,
    collate_online_uad,
)


SEEDS = (20260826, 20260827, 20260828)
DATA = ROOT / (
    "artifacts/phase1/mf3b_uad_online/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
TRAIN = ROOT / "artifacts/training/mf3b_uad_online_v1"
OUT = ROOT / "artifacts/evaluation/mf3b_uad_shadow_gate_v1"
ALPHAS = (0.25, 0.5, 1.0)
DECISIVE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
MARGIN_THRESHOLDS = (0.0, 0.05, 0.1, 0.2)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def load_models(device: torch.device):
    result = []
    for seed in SEEDS:
        path = TRAIN / f"seed_{seed}/uad_mf3.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not (
            payload.get("schema_version") == "revealnav-mf3b-uad-checkpoint/1"
            and payload.get("seed") == seed
            and payload.get("method_scope") == "uad_readiness_residual_adapter"
        ):
            raise RuntimeError("MF3B checkpoint schema drift")
        model = StructuredUADHeads(768, int(payload["hidden_dim"]))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        result.append((seed, model.to(device).eval()))
    return result


def evaluate(model, loader, device, parameters) -> dict:
    alpha, decisive_threshold, margin_threshold = parameters
    counts = Counter()
    eligible = authorized = interventions = 0
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = {name: value.to(device) for name, value in cpu.items()}
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
            )
            for step in range(int(batch["step_mask"][0].sum())):
                current = torch.nonzero(
                    batch["candidate_mask"][0, step], as_tuple=False
                ).flatten()
                native = int(batch["native_index"][0, step])
                teacher = int(batch["target_index"][0, step])
                if len(current) < 2 or native < 0 or teacher < 0:
                    continue
                eligible += 1
                target_logits = output.target_logits[0, step, current]
                top_two = torch.topk(target_logits, 2).values
                margin = float(top_two[0] - top_two[1])
                decisive = float(output.uad_probabilities[0, step, 2])
                adapted = native
                if decisive >= decisive_threshold and margin >= margin_threshold:
                    authorized += 1
                    centered = target_logits - target_logits.mean()
                    adjusted = batch["native_scores"][0, step, current] + alpha * centered
                    best = int(torch.argmax(adjusted))
                    if float(adjusted[best]) > float(batch["outside_score"][0, step]):
                        adapted = int(current[best])
                if adapted != native:
                    interventions += 1
                native_correct = native == teacher
                adapted_correct = adapted == teacher
                if adapted_correct and not native_correct:
                    counts["RESCUE"] += 1
                elif native_correct and not adapted_correct:
                    counts["HARM"] += 1
                elif native_correct:
                    counts["AGREE_CORRECT"] += 1
                elif adapted == native:
                    counts["AGREE_INCORRECT"] += 1
                else:
                    counts["DISAGREE_NEITHER"] += 1
    return {
        "eligible": eligible,
        "authorized": authorized,
        "interventions": interventions,
        "outcomes": dict(sorted(counts.items())),
        "rescues": counts["RESCUE"],
        "harms": counts["HARM"],
        "net_rescues": counts["RESCUE"] - counts["HARM"],
    }


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = load_models(device)
    calibration = DataLoader(
        OnlineUADFeatureDataset(DATA, "calibration"), batch_size=1,
        shuffle=False, collate_fn=collate_online_uad,
    )
    shadow = DataLoader(
        OnlineUADFeatureDataset(DATA, "shadow"), batch_size=1,
        shuffle=False, collate_fn=collate_online_uad,
    )
    grid = []
    for parameters in (
        (alpha, decisive, margin)
        for alpha in ALPHAS
        for decisive in DECISIVE_THRESHOLDS
        for margin in MARGIN_THRESHOLDS
    ):
        members = {
            str(seed): evaluate(model, calibration, device, parameters)
            for seed, model in models
        }
        rescue = sum(row["rescues"] for row in members.values())
        harm = sum(row["harms"] for row in members.values())
        interventions = sum(row["interventions"] for row in members.values())
        grid.append({
            "alpha": parameters[0],
            "decisive_threshold": parameters[1],
            "margin_threshold": parameters[2],
            "rescues": rescue,
            "harms": harm,
            "net_rescues": rescue - harm,
            "interventions": interventions,
            "members": members,
        })
    viable = [
        row for row in grid
        if row["interventions"] >= 3
        and row["rescues"] > row["harms"]
    ]
    selected = max(
        viable,
        key=lambda row: (
            row["net_rescues"], -row["harms"], row["rescues"],
            -row["interventions"], -row["alpha"],
            row["decisive_threshold"], row["margin_threshold"],
        ),
        default=None,
    )
    shadow_members = {}
    if selected is not None:
        parameters = (
            selected["alpha"], selected["decisive_threshold"],
            selected["margin_threshold"],
        )
        shadow_members = {
            str(seed): evaluate(model, shadow, device, parameters)
            for seed, model in models
        }
    gates = {
        "calibration_positive_net_rescue": selected is not None,
        "shadow_rescue_exceeds_harm_every_seed": (
            bool(shadow_members)
            and all(row["rescues"] > row["harms"]
                    for row in shadow_members.values())
        ),
        "shadow_nonzero_intervention_every_seed": (
            bool(shadow_members)
            and all(row["interventions"] > 0
                    for row in shadow_members.values())
        ),
    }
    passed = all(gates.values())
    value = {
        "schema_version": "revealnav-mf3b-uad-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "parameter_grid": {
            "alpha": list(ALPHAS),
            "decisive_threshold": list(DECISIVE_THRESHOLDS),
            "margin_threshold": list(MARGIN_THRESHOLDS),
            "calibration_rows": grid,
        },
        "selected": selected,
        "shadow_members": shadow_members,
        "gates": gates,
        "task_metric_run_authorized": passed,
        **MF3B_SCOPE,
    }
    atomic_json(OUT / "MF3B_UAD_SHADOW_GATE.json", value)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
