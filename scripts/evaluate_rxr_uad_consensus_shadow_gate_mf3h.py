#!/usr/bin/env python3
"""Evaluate the frozen unanimous UAD rule on fresh RxR-train rank-12."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import (  # noqa: E402
    MF3B_SCOPE,
    NativeConditionedUAD,
    OnlineUADFeatureDataset,
    collate_online_uad,
    native_residual_logits,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
DATA = ROOT / (
    "artifacts/phase1/mf3h_uad_online_rank12/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
TRAIN = ROOT / "artifacts/training/mf3g_uad_residual_v1"
OUT = ROOT / "artifacts/evaluation/mf3h_uad_consensus_shadow_gate_v1"


def load_models(device: torch.device):
    models = []
    for seed in SEEDS:
        payload = torch.load(
            TRAIN / f"seed_{seed}/uad_residual_mf3g.pt",
            map_location="cpu", weights_only=True,
        )
        if not (
            payload.get("schema_version") == "revealnav-mf3g-uad-checkpoint/1"
            and payload.get("seed") == seed
            and float(payload.get("correction_bound")) == 2.0
        ):
            raise RuntimeError("MF3G checkpoint schema drift")
        model = NativeConditionedUAD(768, int(payload["hidden_dim"]))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append(model.to(device).eval())
    return tuple(models)


def collect(models, split: str, device: torch.device) -> list[dict]:
    loader = DataLoader(
        OnlineUADFeatureDataset(DATA, split), batch_size=1, shuffle=False,
        collate_fn=collate_online_uad,
    )
    rows = []
    with torch.no_grad():
        for cpu in loader:
            batch = {name: value.to(device) for name, value in cpu.items()}
            outputs = tuple(model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            ) for model in models)
            fused = tuple(native_residual_logits(
                output, batch["native_scores"], batch["candidate_mask"],
                correction_bound=2.0,
            )[0] for output in outputs)
            for step in range(int(batch["step_mask"][0].sum())):
                mask = batch["candidate_mask"][0, step]
                native = int(batch["native_index"][0, step])
                teacher = int(batch["target_index"][0, step])
                if native < 0 or teacher < 0 or int(mask.sum()) < 2:
                    continue
                members = tuple(int(value[0, step].argmax()) for value in fused)
                adapted = (
                    members[0]
                    if len(set(members)) == 1 and members[0] != native
                    else native
                )
                native_values = batch["native_scores"][0, step, mask]
                indices = torch.nonzero(mask, as_tuple=False).flatten()
                order = torch.argsort(native_values, descending=True)
                rows.append({
                    "native": native, "teacher": teacher, "adapted": adapted,
                    "native_margin": float(
                        native_values[order[0]] - native_values[order[1]]
                    ),
                    "native_runner_up": int(indices[order[1]]),
                })
    return rows


def evaluate(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        native_correct = row["native"] == row["teacher"]
        adapted_correct = row["adapted"] == row["teacher"]
        if adapted_correct and not native_correct:
            counts["RESCUE"] += 1
        elif native_correct and not adapted_correct:
            counts["HARM"] += 1
        elif native_correct:
            counts["AGREE_CORRECT"] += 1
        elif row["adapted"] == row["native"]:
            counts["AGREE_INCORRECT"] += 1
        else:
            counts["DISAGREE_NEITHER"] += 1
    interventions = sum(row["adapted"] != row["native"] for row in rows)
    return {
        "eligible": len(rows), "interventions": interventions,
        "rescues": counts["RESCUE"], "harms": counts["HARM"],
        "net_rescues": counts["RESCUE"] - counts["HARM"],
        "outcomes": dict(sorted(counts.items())),
    }


def uncertainty_control(rows: list[dict], budget: int) -> dict:
    selected = {
        index for index, _ in sorted(
            enumerate(rows), key=lambda pair: pair[1]["native_margin"]
        )[:budget]
    }
    rescues = harms = 0
    for index, row in enumerate(rows):
        if index not in selected:
            continue
        adapted = row["native_runner_up"]
        rescues += row["native"] != row["teacher"] and adapted == row["teacher"]
        harms += row["native"] == row["teacher"] and adapted != row["teacher"]
    return {
        "matched_intervention_budget": budget,
        "rescues": rescues, "harms": harms, "net_rescues": rescues - harms,
    }


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = load_models(device)
    calibration_rows = collect(models, "calibration", device)
    calibration = evaluate(calibration_rows)
    ordered_margins = sorted(row["native_margin"] for row in calibration_rows)
    uncertainty_margin_max = ordered_margins[
        min(calibration["interventions"], len(ordered_margins)) - 1
    ]
    calibration_pass = (
        calibration["interventions"] >= 5
        and calibration["rescues"] > calibration["harms"]
    )
    shadow = {}
    uncertainty = {}
    if calibration_pass:
        shadow_rows = collect(models, "shadow", device)
        shadow = evaluate(shadow_rows)
        uncertainty = uncertainty_control(
            shadow_rows, shadow["interventions"]
        )
    gates = {
        "calibration_frozen_rule_positive": calibration_pass,
        "fresh_shadow_positive_with_two_interventions": (
            bool(shadow) and shadow["interventions"] >= 2
            and shadow["rescues"] > shadow["harms"]
        ),
        "fresh_shadow_exceeds_uncertainty_control": (
            bool(shadow) and shadow["net_rescues"] > uncertainty["net_rescues"]
        ),
    }
    passed = all(gates.values())
    atomic_json(OUT / "MF3H_UAD_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3h-uad-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "selected": {
            "error_threshold": 0.0,
            "fused_advantage_threshold": 0.0,
            "unanimous_required": True,
            "selection_tuned_on_rank12": False,
        },
        "calibration": calibration, "shadow": shadow,
        "uncertainty_calibration_budget_match": {
            "native_margin_max": uncertainty_margin_max,
            "matched_intervention_budget": calibration["interventions"],
            "role": "task-metric uncertainty-only control selected by exact "
                    "calibration intervention budget; shadow comparison uses "
                    "an exact matched budget",
        },
        "uncertainty_matched_shadow": uncertainty,
        "gates": gates, "task_metric_run_authorized": passed,
        **MF3B_SCOPE,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
