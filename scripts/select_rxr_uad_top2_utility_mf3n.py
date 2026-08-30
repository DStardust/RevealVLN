#!/usr/bin/env python3
"""Select the sealed MF3N architecture and robust top-2 utility rule."""

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
    PairwiseSwitchUtility,
    collate_online_uad,
    median_mad_lower_confidence,
    top2_expected_switch_utility,
    top2_switch_targets,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file  # noqa: E402
from scripts.train_rxr_uad_top2_utility_mf3n import (  # noqa: E402
    DATA,
    DESIGN,
    HIDDEN_DIMS,
    OUT as TRAIN,
    SEEDS,
)

MAD_WEIGHTS = (0.0, 0.5, 1.0, 1.5, 2.0)
THRESHOLDS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
OUT = ROOT / "artifacts/evaluation/mf3n_top2_utility_development_v1"


def load_models(hidden: int, device: torch.device):
    models = []
    checkpoints = []
    for seed in SEEDS:
        path = TRAIN / f"hidden_{hidden}/seed_{seed}/top2_utility_mf3n.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not (
            payload.get("schema_version") == "revealnav-mf3n-checkpoint/1"
            and payload.get("hidden_dim") == hidden
            and payload.get("seed") == seed
            and payload.get("candidate_feature_dim") == 1536
        ):
            raise RuntimeError("MF3N checkpoint schema drift")
        model = PairwiseSwitchUtility(768, 1536, hidden)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append(model.to(device).eval())
        checkpoints.append({
            "seed": seed,
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return tuple(models), checkpoints


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
                    batch["history_embeddings"],
                    batch["candidate_embeddings"],
                    batch["candidate_mask"],
                    batch["instruction_embedding"],
                    batch["native_scores"], batch["native_index"],
                )
                member.append(top2_expected_switch_utility(output, batch)[0])
            labels, runner, valid = top2_switch_targets(batch)
            for step in range(valid.shape[1]):
                if not bool(valid[0, step]):
                    continue
                native = int(batch["native_index"][0, step])
                rows.append({
                    "outcome": (
                        "RESCUE" if int(labels[0, step]) == 1 else
                        "HARM" if int(labels[0, step]) == 2 else "NEITHER"
                    ),
                    "member_utilities": [
                        float(value[0, step]) for value in member
                    ],
                    "native_margin": float(
                        batch["native_scores"][0, step, native]
                        - batch["native_scores"][0, step, runner[0, step]]
                    ),
                })
    return rows


def robust_score(row: dict, mad_weight: float) -> float:
    return float(median_mad_lower_confidence(
        torch.tensor(row["member_utilities"]), mad_weight=mad_weight
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
    architecture_rows = []
    for hidden in HIDDEN_DIMS:
        nlls = [json.loads((TRAIN / (
            f"hidden_{hidden}/seed_{seed}/RESULT.json"
        )).read_text())["calibration"]["top2_nll"] for seed in SEEDS]
        architecture_rows.append({
            "hidden_dim": hidden,
            "member_calibration_top2_nll": nlls,
            "median_calibration_top2_nll": statistics.median(nlls),
        })
    architecture = min(architecture_rows, key=lambda row: (
        row["median_calibration_top2_nll"], row["hidden_dim"]
    ))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(int(architecture["hidden_dim"]), device)
    strata = {
        "calibration_ranks_18_20": collect(
            models, "calibration", device
        ),
        "development_ranks_21_23": collect(
            models, "diagnostic", device
        ),
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
                "utility_threshold": threshold,
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
        row["pooled"]["net_rescues"], -row["pooled"]["harms"],
        row["utility_threshold"], row["mad_weight"],
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
            "selection_source": "pooled consumed ranks 18-23",
        }
    status = "DEVELOPMENT_PASS" if selected else "DEVELOPMENT_FAIL"
    atomic_json(OUT / "MF3N_DEVELOPMENT_SELECTION.json", {
        "schema_version": "revealnav-mf3n-development-selection/1",
        "status": status,
        "architecture_candidates": architecture_rows,
        "selected_architecture": architecture,
        "rule_candidates": candidates,
        "selected_rule": selected,
        "uncertainty_development_budget_match": uncertainty,
        "eligible_steps_by_stratum": {
            name: len(rows) for name, rows in strata.items()
        },
        "checkpoints": checkpoints,
        "ranks24_29_payload_read": False,
        "data_sha256": sha256_file(DATA),
        "design_sha256": sha256_file(DESIGN),
        **MF3B_SCOPE,
    })
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
