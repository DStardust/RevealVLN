#!/usr/bin/env python3
"""Calibrate and evaluate the fresh MF3G train-shadow residual gate."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from itertools import product
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
    median_native_conditioned_outputs,
    native_residual_logits,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json  # noqa: E402


SEEDS = (20260826, 20260827, 20260828)
DATA = ROOT / (
    "artifacts/phase1/mf3g_uad_online_expanded/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
TRAIN = ROOT / "artifacts/training/mf3g_uad_residual_v1"
OUT = ROOT / "artifacts/evaluation/mf3g_uad_residual_shadow_gate_v1"
ERROR_THRESHOLDS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8)
FUSED_ADVANTAGE_THRESHOLDS = (0.0, 0.05, 0.1, 0.25, 0.5)
UNCERTAINTY_MARGIN_MAXIMA = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def load_models(device: torch.device):
    models = []
    for seed in SEEDS:
        path = TRAIN / f"seed_{seed}/uad_residual_mf3g.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not (
            payload.get("schema_version") == "revealnav-mf3g-uad-checkpoint/1"
            and payload.get("seed") == seed
            and float(payload.get("correction_bound")) == 2.0
        ):
            raise RuntimeError("MF3G checkpoint schema drift")
        model = NativeConditionedUAD(768, int(payload["hidden_dim"]))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append((seed, model.to(device).eval()))
    return models


def collect(model_or_models, loader, device) -> list[dict]:
    rows = []
    models = (
        model_or_models if isinstance(model_or_models, tuple)
        else (model_or_models,)
    )
    for model in models:
        model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = {name: value.to(device) for name, value in cpu.items()}
            output = median_native_conditioned_outputs(tuple(
                model(
                    batch["history_embeddings"],
                    batch["candidate_embeddings"],
                    batch["candidate_mask"],
                    batch["instruction_embedding"],
                    batch["native_scores"], batch["native_index"],
                )
                for model in models
            ))
            fused, _ = native_residual_logits(
                output, batch["native_scores"], batch["candidate_mask"],
                correction_bound=2.0,
            )
            for step in range(int(batch["step_mask"][0].sum())):
                mask = batch["candidate_mask"][0, step]
                native = int(batch["native_index"][0, step])
                teacher = int(batch["target_index"][0, step])
                if native < 0 or teacher < 0 or int(mask.sum()) < 2:
                    continue
                adapted = int(torch.argmax(fused[0, step]))
                native_values = batch["native_scores"][0, step, mask]
                native_top = torch.topk(native_values, 2).values
                rows.append({
                    "native": native, "teacher": teacher,
                    "adapted": adapted,
                    "error_probability": float(torch.sigmoid(
                        output.native_error_logit[0, step]
                    )),
                    "fused_advantage": max(
                        0.0,
                        float(fused[0, step, adapted] - fused[0, step, native]),
                    ),
                    "native_margin": float(native_top[0] - native_top[1]),
                    "native_runner_up": int(
                        torch.nonzero(mask, as_tuple=False).flatten()[
                            torch.argsort(native_values, descending=True)[1]
                        ]
                    ),
                })
    return rows


def evaluate(rows: list[dict], parameters: tuple[float, float]) -> dict:
    error_threshold, advantage_threshold = parameters
    counts = Counter()
    interventions = 0
    for row in rows:
        adapted = row["native"]
        if (
            row["adapted"] != row["native"]
            and row["error_probability"] >= error_threshold
            and row["fused_advantage"] >= advantage_threshold
        ):
            adapted = row["adapted"]
        interventions += adapted != row["native"]
        native_correct = row["native"] == row["teacher"]
        adapted_correct = adapted == row["teacher"]
        if adapted_correct and not native_correct:
            counts["RESCUE"] += 1
        elif native_correct and not adapted_correct:
            counts["HARM"] += 1
        elif native_correct:
            counts["AGREE_CORRECT"] += 1
        elif adapted == row["native"]:
            counts["AGREE_INCORRECT"] += 1
        else:
            counts["DISAGREE_NEITHER"] += 1
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


def evaluate_uncertainty_threshold(rows: list[dict], margin_max: float) -> dict:
    selected = [row for row in rows if row["native_margin"] <= margin_max]
    rescues = sum(
        row["native"] != row["teacher"]
        and row["native_runner_up"] == row["teacher"]
        for row in selected
    )
    harms = sum(row["native"] == row["teacher"] for row in selected)
    return {
        "native_margin_max": margin_max, "interventions": len(selected),
        "rescues": rescues, "harms": harms, "net_rescues": rescues - harms,
    }


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = load_models(device)
    calibration_loader = DataLoader(
        OnlineUADFeatureDataset(DATA, "calibration"), batch_size=1,
        shuffle=False, collate_fn=collate_online_uad,
    )
    calibration_rows = {
        str(seed): collect(model, calibration_loader, device)
        for seed, model in models
    }
    grid = []
    for parameters in product(ERROR_THRESHOLDS, FUSED_ADVANTAGE_THRESHOLDS):
        members = {
            str(seed): evaluate(calibration_rows[str(seed)], parameters)
            for seed, _ in models
        }
        grid.append({
            "error_threshold": parameters[0],
            "fused_advantage_threshold": parameters[1],
            "members": members,
            "rescues": sum(row["rescues"] for row in members.values()),
            "harms": sum(row["harms"] for row in members.values()),
        })
    viable = [
        row for row in grid
        if all(
            member["interventions"] >= 10
            and member["rescues"] > member["harms"]
            for member in row["members"].values()
        )
    ]
    selected = max(
        viable,
        key=lambda row: (
            min(member["net_rescues"] for member in row["members"].values()),
            row["rescues"] - row["harms"], -row["harms"],
            row["error_threshold"], row["fused_advantage_threshold"],
        ),
        default=None,
    )

    shadow_members = {}
    uncertainty_members = {}
    uncertainty_selected = None
    ensemble_shadow = {}
    ensemble_uncertainty = {}
    if selected is not None:
        parameters = (
            selected["error_threshold"],
            selected["fused_advantage_threshold"],
        )
        shadow_loader = DataLoader(
            OnlineUADFeatureDataset(DATA, "shadow"), batch_size=1,
            shuffle=False, collate_fn=collate_online_uad,
        )
        target_budget = round(sum(
            row["interventions"] for row in selected["members"].values()
        ) / len(SEEDS))
        uncertainty_candidates = [
            evaluate_uncertainty_threshold(
                calibration_rows[str(SEEDS[0])], margin
            )
            for margin in UNCERTAINTY_MARGIN_MAXIMA
        ]
        uncertainty_selected = min(
            uncertainty_candidates,
            key=lambda row: (
                abs(row["interventions"] - target_budget),
                row["native_margin_max"],
            ),
        )
        for seed, model in models:
            rows = collect(model, shadow_loader, device)
            result = evaluate(rows, parameters)
            shadow_members[str(seed)] = result
            uncertainty_members[str(seed)] = uncertainty_control(
                rows, result["interventions"]
            )
        ensemble_rows = collect(
            tuple(model for _, model in models), shadow_loader, device
        )
        ensemble_shadow = evaluate(ensemble_rows, parameters)
        ensemble_uncertainty = uncertainty_control(
            ensemble_rows, ensemble_shadow["interventions"]
        )
    gates = {
        "calibration_positive_every_seed": selected is not None,
        "shadow_positive_with_five_interventions_every_seed": (
            bool(shadow_members) and all(
                row["interventions"] >= 5 and row["rescues"] > row["harms"]
                for row in shadow_members.values()
            )
        ),
        "learned_exceeds_uncertainty_net_rescue_every_seed": (
            bool(shadow_members) and all(
                shadow_members[str(seed)]["net_rescues"]
                > uncertainty_members[str(seed)]["net_rescues"]
                for seed in SEEDS
            )
        ),
        "ensemble_positive_and_exceeds_uncertainty": (
            bool(ensemble_shadow)
            and ensemble_shadow["interventions"] >= 5
            and ensemble_shadow["rescues"] > ensemble_shadow["harms"]
            and ensemble_shadow["net_rescues"]
            > ensemble_uncertainty["net_rescues"]
        ),
    }
    passed = all(gates.values())
    atomic_json(OUT / "MF3G_UAD_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3g-uad-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "parameter_grid": {
            "error_threshold": list(ERROR_THRESHOLDS),
            "fused_advantage_threshold": list(FUSED_ADVANTAGE_THRESHOLDS),
        },
        "calibration_grid": grid, "selected": selected,
        "shadow_members": shadow_members,
        "uncertainty_matched_shadow_members": uncertainty_members,
        "uncertainty_calibration_budget_match": uncertainty_selected,
        "ensemble_shadow": ensemble_shadow,
        "ensemble_uncertainty_matched_shadow": ensemble_uncertainty,
        "gates": gates, "task_metric_run_authorized": passed,
        **MF3B_SCOPE,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
