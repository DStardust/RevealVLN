#!/usr/bin/env python3
"""Calibrate MF3F on RxR-train and open shadow only after a positive gate."""

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
)


SEEDS = (20260826, 20260827, 20260828)
DATA = ROOT / (
    "artifacts/phase1/mf3b_uad_online/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
TRAIN = ROOT / "artifacts/training/mf3e_uad_correction_v1"
OUT = ROOT / "artifacts/evaluation/mf3f_uad_correction_shadow_gate_v1"
ERROR_THRESHOLDS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8)
NATIVE_MARGIN_MAXIMA = (0.25, 0.5, 0.75, 1.0)
ALTERNATIVE_MARGIN_MINIMA = (0.0, 0.25, 0.5, 1.0)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def load_models(device: torch.device):
    models = []
    for seed in SEEDS:
        path = TRAIN / f"seed_{seed}/uad_correction_mf3e.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not (
            payload.get("schema_version") == "revealnav-mf3e-uad-checkpoint/1"
            and payload.get("seed") == seed
            and payload.get("method_scope") == "uad_readiness_residual_adapter"
        ):
            raise RuntimeError("MF3E checkpoint schema drift")
        model = NativeConditionedUAD(768, int(payload["hidden_dim"]))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append((seed, model.to(device).eval()))
    return models


def collect(model, loader, device) -> list[dict]:
    rows = []
    model.eval()
    with torch.no_grad():
        for cpu in loader:
            batch = {name: value.to(device) for name, value in cpu.items()}
            output = model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            )
            for step in range(int(batch["step_mask"][0].sum())):
                mask = batch["candidate_mask"][0, step]
                native = int(batch["native_index"][0, step])
                teacher = int(batch["target_index"][0, step])
                if native < 0 or teacher < 0 or int(mask.sum()) < 2:
                    continue
                native_values = batch["native_scores"][0, step, mask]
                native_top = torch.topk(native_values, 2).values
                logits = output.alternative_logits[0, step]
                finite = logits[torch.isfinite(logits)]
                alternative = int(torch.argmax(logits))
                alternative_margin = (
                    float("inf") if finite.numel() == 1
                    else float(torch.topk(finite, 2).values[0]
                               - torch.topk(finite, 2).values[1])
                )
                rows.append({
                    "native": native,
                    "teacher": teacher,
                    "alternative": alternative,
                    "error_probability": float(torch.sigmoid(
                        output.native_error_logit[0, step]
                    )),
                    "native_margin": float(native_top[0] - native_top[1]),
                    "alternative_margin": alternative_margin,
                })
    return rows


def evaluate(rows: list[dict], parameters: tuple[float, float, float]) -> dict:
    error_threshold, native_margin_max, alternative_margin_min = parameters
    counts = Counter()
    interventions = 0
    for row in rows:
        adapted = row["native"]
        if (
            row["error_probability"] >= error_threshold
            and row["native_margin"] <= native_margin_max
            and row["alternative_margin"] >= alternative_margin_min
        ):
            adapted = row["alternative"]
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
    for parameters in product(
        ERROR_THRESHOLDS, NATIVE_MARGIN_MAXIMA, ALTERNATIVE_MARGIN_MINIMA
    ):
        members = {
            str(seed): evaluate(calibration_rows[str(seed)], parameters)
            for seed, _ in models
        }
        grid.append({
            "error_threshold": parameters[0],
            "native_margin_max": parameters[1],
            "alternative_margin_min": parameters[2],
            "members": members,
            "rescues": sum(row["rescues"] for row in members.values()),
            "harms": sum(row["harms"] for row in members.values()),
            "interventions": sum(
                row["interventions"] for row in members.values()
            ),
        })
    viable = [
        row for row in grid
        if all(
            member["interventions"] >= 3
            and member["rescues"] > member["harms"]
            for member in row["members"].values()
        )
    ]
    selected = max(
        viable,
        key=lambda row: (
            min(member["net_rescues"] for member in row["members"].values()),
            row["rescues"] - row["harms"], -row["harms"],
            row["error_threshold"], -row["native_margin_max"],
            row["alternative_margin_min"],
        ),
        default=None,
    )

    shadow_members = {}
    if selected is not None:
        parameters = (
            selected["error_threshold"], selected["native_margin_max"],
            selected["alternative_margin_min"],
        )
        shadow_loader = DataLoader(
            OnlineUADFeatureDataset(DATA, "shadow"), batch_size=1,
            shuffle=False, collate_fn=collate_online_uad,
        )
        shadow_members = {
            str(seed): evaluate(collect(model, shadow_loader, device), parameters)
            for seed, model in models
        }
    gates = {
        "calibration_positive_every_seed": selected is not None,
        "shadow_rescue_exceeds_harm_every_seed": (
            bool(shadow_members) and all(
                row["rescues"] > row["harms"]
                for row in shadow_members.values()
            )
        ),
        "shadow_nonzero_intervention_every_seed": (
            bool(shadow_members) and all(
                row["interventions"] > 0 for row in shadow_members.values()
            )
        ),
    }
    passed = all(gates.values())
    atomic_json(OUT / "MF3F_UAD_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3f-uad-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "parameter_grid": {
            "error_threshold": list(ERROR_THRESHOLDS),
            "native_margin_max": list(NATIVE_MARGIN_MAXIMA),
            "alternative_margin_min": list(ALTERNATIVE_MARGIN_MINIMA),
        },
        "calibration_grid": grid,
        "selected": selected,
        "shadow_members": shadow_members,
        "gates": gates,
        "task_metric_run_authorized": passed,
        **MF3B_SCOPE,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
