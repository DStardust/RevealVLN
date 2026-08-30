#!/usr/bin/env python3
"""Select the sealed MF3J architecture and switch rule on development data."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import (  # noqa: E402
    MF3B_SCOPE,
    OnlineUADFeatureDataset,
    PairwiseSwitchUtility,
    collate_online_uad,
    pairwise_expected_utility,
    pairwise_switch_targets,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file  # noqa: E402

SEEDS = (20260826, 20260827, 20260828)
HIDDEN_DIMS = (64, 128)
THRESHOLDS = (-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
DATA = ROOT / (
    "artifacts/phase1/mf3i_policy_token_uad/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
TRAIN = ROOT / "artifacts/training/mf3j_switch_utility_v1"
OUT = ROOT / "artifacts/evaluation/mf3j_switch_utility_development_v1"


def load_models(hidden: int, device: torch.device):
    result = []
    evidence = []
    for seed in SEEDS:
        path = TRAIN / f"hidden_{hidden}/seed_{seed}/switch_utility_mf3j.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not (
            payload.get("schema_version") == "revealnav-mf3j-checkpoint/1"
            and payload.get("hidden_dim") == hidden
            and payload.get("seed") == seed
            and payload.get("candidate_feature_dim") == 1536
        ):
            raise RuntimeError("MF3J checkpoint schema drift")
        model = PairwiseSwitchUtility(768, 1536, hidden)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        result.append(model.to(device).eval())
        evidence.append({
            "seed": seed,
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return tuple(result), evidence


def collect(models, split: str, device: torch.device) -> list[dict]:
    loader = DataLoader(
        OnlineUADFeatureDataset(DATA, split), batch_size=1, shuffle=False,
        collate_fn=collate_online_uad,
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
            utilities = tuple(pairwise_expected_utility(output) for output in outputs)
            _, pair_mask = pairwise_switch_targets(batch)
            member_proposals = tuple(
                value.masked_fill(~pair_mask, -torch.inf).argmax(-1)
                for value in utilities
            )
            median = torch.stack(utilities).median(0).values.masked_fill(
                ~pair_mask, -torch.inf
            )
            proposal = median.argmax(-1)
            valid_steps = pair_mask.any(-1)
            for step in range(valid_steps.shape[1]):
                if not bool(valid_steps[0, step]):
                    continue
                adapted = int(proposal[0, step])
                choices = tuple(int(value[0, step]) for value in member_proposals)
                scores = tuple(float(
                    value[0, step, adapted]
                ) for value in utilities)
                native = int(batch["native_index"][0, step])
                target = int(batch["target_index"][0, step])
                candidate_mask = batch["candidate_mask"][0, step]
                indices = torch.nonzero(
                    candidate_mask, as_tuple=False
                ).flatten()
                native_values = batch["native_scores"][0, step, candidate_mask]
                order = torch.argsort(native_values, descending=True)
                runner_up = int(indices[order[1]])
                outcome = (
                    "RESCUE" if native != target and adapted == target else
                    "HARM" if native == target else "NEITHER"
                )
                rows.append({
                    "outcome": outcome,
                    "minimum_utility": min(scores),
                    "median_utility": sorted(scores)[1],
                    "unanimous": len(set(choices)) == 1 and choices[0] == adapted,
                    "native_margin": float(
                        native_values[order[0]] - native_values[order[1]]
                    ),
                    "runner_up_outcome": (
                        "RESCUE" if native != target and runner_up == target else
                        "HARM" if native == target else "NEITHER"
                    ),
                })
    return rows


def counts(rows: list[dict], agreement: str, threshold: float) -> dict:
    selected = [
        row for row in rows
        if row["minimum_utility"] > threshold
        and (agreement == "median" or row["unanimous"])
    ]
    rescues = sum(row["outcome"] == "RESCUE" for row in selected)
    harms = sum(row["outcome"] == "HARM" for row in selected)
    neither = sum(row["outcome"] == "NEITHER" for row in selected)
    return {
        "interventions": len(selected),
        "rescues": rescues,
        "harms": harms,
        "neither": neither,
        "net_rescues": rescues - harms,
    }


def main() -> int:
    prior = json.loads((ROOT / (
        "artifacts/evaluation/mf3i_contextual_uad_shadow_gate_v1/"
        "MF3I_UAD_SHADOW_GATE.json"
    )).read_text())
    if prior.get("shadow") != {}:
        raise RuntimeError("rank-14 was already opened")
    architecture_rows = []
    for hidden in HIDDEN_DIMS:
        nlls = [json.loads((TRAIN / (
            f"hidden_{hidden}/seed_{seed}/RESULT.json"
        )).read_text())["calibration"]["pairwise_nll"] for seed in SEEDS]
        architecture_rows.append({
            "hidden_dim": hidden,
            "member_calibration_nll": nlls,
            "median_calibration_nll": statistics.median(nlls),
        })
    selected_architecture = min(
        architecture_rows,
        key=lambda row: (row["median_calibration_nll"], row["hidden_dim"]),
    )
    hidden = selected_architecture["hidden_dim"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(hidden, device)
    diagnostic_rows = collect(models, "diagnostic", device)
    candidates = []
    for agreement in ("unanimous", "median"):
        for threshold in THRESHOLDS:
            result = counts(diagnostic_rows, agreement, threshold)
            candidates.append({
                "agreement": agreement,
                "threshold": threshold,
                **result,
                "qualifies": (
                    result["interventions"] >= 10
                    and result["net_rescues"] > 0
                ),
            })
    qualifying = [row for row in candidates if row["qualifies"]]
    selected_rule = None
    if qualifying:
        selected_rule = max(qualifying, key=lambda row: (
            row["net_rescues"], -row["harms"], row["interventions"],
            row["agreement"] == "unanimous", row["threshold"],
        ))
    passed = selected_rule is not None
    uncertainty_control = None
    if passed:
        margins = sorted(row["native_margin"] for row in diagnostic_rows)
        budget = selected_rule["interventions"]
        uncertainty_control = {
            "matched_intervention_budget": budget,
            "native_margin_max": margins[min(budget, len(margins)) - 1],
        }
    atomic_json(OUT / "MF3J_DEVELOPMENT_SELECTION.json", {
        "schema_version": "revealnav-mf3j-development-selection/1",
        "status": "DEVELOPMENT_PASS" if passed else "DEVELOPMENT_FAIL",
        "architecture_candidates": architecture_rows,
        "selected_architecture": selected_architecture,
        "rule_candidates": candidates,
        "selected_rule": selected_rule,
        "uncertainty_calibration_budget_match": uncertainty_control,
        "diagnostic_eligible_steps": len(diagnostic_rows),
        "checkpoints": checkpoints,
        "rank14_payload_read": False,
        "data_sha256": sha256_file(DATA),
        **MF3B_SCOPE,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
