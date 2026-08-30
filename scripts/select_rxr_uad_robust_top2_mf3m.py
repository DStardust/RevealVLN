#!/usr/bin/env python3
"""Select the MF3M median-MAD robust conditional-advantage rule."""

from __future__ import annotations

import json
import math
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
    collate_online_uad,
    median_mad_lower_confidence,
    top2_conditional_advantage,
    top2_switch_targets,
)
from scripts.select_rxr_uad_conditional_top2_mf3l import load_models  # noqa: E402
from scripts.train_rxr_uad_conditional_top2_mf3l import DATA  # noqa: E402
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file  # noqa: E402

L_SELECTION = ROOT / (
    "artifacts/evaluation/mf3l_conditional_top2_development_v1/"
    "MF3L_DEVELOPMENT_SELECTION.json"
)
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3M_ROBUST_CONDITIONAL_TOP2.md"
MAD_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
THRESHOLDS = (0.00, 0.01, 0.025, 0.05, 0.075, 0.10)
OUT = ROOT / "artifacts/evaluation/mf3m_robust_top2_development_v1"


def collect(models, split: str, device: torch.device, data: Path = DATA):
    loader = DataLoader(
        OnlineUADFeatureDataset(data, split), batch_size=1, shuffle=False,
        collate_fn=collate_online_uad,
    )
    rows = []
    with torch.no_grad():
        for cpu in loader:
            batch = {key: value.to(device) for key, value in cpu.items()}
            member = []
            for model in models:
                output = model(
                    batch["history_embeddings"], batch["candidate_embeddings"],
                    batch["candidate_mask"], batch["instruction_embedding"],
                    batch["native_scores"], batch["native_index"],
                )
                member.append(top2_conditional_advantage(
                    output, batch["native_scores"], batch["candidate_mask"],
                    batch["native_index"],
                )[0])
            labels, runner, valid = top2_switch_targets(batch)
            for step in range(valid.shape[1]):
                if not bool(valid[0, step]):
                    continue
                native = int(batch["native_index"][0, step])
                rows.append({
                    "outcome": ("RESCUE" if int(labels[0, step]) == 1 else
                                "HARM" if int(labels[0, step]) == 2 else
                                "NEITHER"),
                    "member_advantages": [
                        float(value[0, step]) for value in member
                    ],
                    "native_margin": float(
                        batch["native_scores"][0, step, native]
                        - batch["native_scores"][0, step, runner[0, step]]
                    ),
                })
    return rows


def robust_score(row: dict, mad_weight: float) -> float:
    values = torch.tensor(row["member_advantages"])
    return float(median_mad_lower_confidence(
        values, mad_weight=mad_weight
    ))


def summarize(rows: list[dict], mad_weight: float, threshold: float) -> dict:
    selected = [
        row for row in rows if robust_score(row, mad_weight) > threshold
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


def wilson_lower(rescues: int, harms: int, z: float = 1.96) -> float:
    count = rescues + harms
    if count == 0:
        return 0.0
    proportion = rescues / count
    denominator = 1.0 + z * z / count
    center = proportion + z * z / (2.0 * count)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / count
        + z * z / (4.0 * count * count)
    )
    return (center - radius) / denominator


def main() -> int:
    prior = json.loads(L_SELECTION.read_text())
    if not (
        prior.get("status") == "DEVELOPMENT_FAIL"
        and prior.get("ranks18_23_payload_read") is False
    ):
        raise RuntimeError("MF3L development boundary drift")
    architecture = prior["selected_architecture"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(
        int(architecture["hidden_dim"]),
        float(architecture["correction_bound"]), device,
    )
    strata = {
        "consumed_ranks_12_14": collect(models, "diagnostic", device),
        "consumed_ranks_15_17": collect(models, "shadow", device),
    }
    candidates = []
    for mad_weight in MAD_WEIGHTS:
        for threshold in THRESHOLDS:
            summaries = {
                name: summarize(rows, mad_weight, threshold)
                for name, rows in strata.items()
            }
            pooled = {
                key: sum(value[key] for value in summaries.values())
                for key in ("interventions", "rescues", "harms", "neither",
                            "net_rescues")
            }
            counts = [value["interventions"] for value in summaries.values()]
            ratio = min(counts) / max(counts) if max(counts) else 0.0
            lower = wilson_lower(pooled["rescues"], pooled["harms"])
            qualifies = (
                all(value["interventions"] >= 10
                    and value["net_rescues"] > 0
                    for value in summaries.values())
                and pooled["interventions"] >= 25
                and ratio >= 0.5
                and lower > 0.5
            )
            candidates.append({
                "mad_weight": mad_weight,
                "robust_advantage_threshold": threshold,
                "strata": summaries,
                "pooled": pooled,
                "stratum_intervention_ratio": ratio,
                "rescue_precision_wilson95_lower": lower,
                "qualifies": qualifies,
            })
    qualifying = [row for row in candidates if row["qualifies"]]
    selected = max(qualifying, key=lambda row: (
        min(value["net_rescues"] for value in row["strata"].values()),
        row["rescue_precision_wilson95_lower"],
        row["pooled"]["net_rescues"],
        -row["pooled"]["harms"],
        row["robust_advantage_threshold"], row["mad_weight"],
    )) if qualifying else None
    uncertainty = None
    if selected is not None:
        margins = sorted(
            row["native_margin"] for rows in strata.values() for row in rows
        )
        budget = selected["pooled"]["interventions"]
        uncertainty = {
            "matched_intervention_budget": budget,
            "native_margin_max": margins[min(budget, len(margins)) - 1],
            "selection_source": "pooled consumed development ranks 12-17",
        }
    status = "DEVELOPMENT_PASS" if selected else "DEVELOPMENT_FAIL"
    atomic_json(OUT / "MF3M_DEVELOPMENT_SELECTION.json", {
        "schema_version": "revealnav-mf3m-development-selection/1",
        "status": status,
        "selected_architecture": architecture,
        "rule_candidates": candidates,
        "selected_rule": selected,
        "uncertainty_development_budget_match": uncertainty,
        "eligible_steps_by_stratum": {
            name: len(rows) for name, rows in strata.items()
        },
        "checkpoints": checkpoints,
        "checkpoint_source": "fixed MF3L calibration-selected ensemble",
        "ranks18_23_payload_read": False,
        "design_sha256": sha256_file(DESIGN),
        "data_sha256": sha256_file(DATA),
        **MF3B_SCOPE,
    })
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
