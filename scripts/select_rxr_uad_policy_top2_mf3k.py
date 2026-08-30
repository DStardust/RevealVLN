#!/usr/bin/env python3
"""Select the sealed MF3K architecture and conservative top-2 rule."""

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
    PolicyAnchoredTop2UAD,
    collate_online_uad,
    top2_posterior_advantage,
    top2_switch_targets,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file  # noqa: E402
from scripts.train_rxr_uad_policy_top2_mf3k import (  # noqa: E402
    ARCHITECTURES,
    DATA,
    OUT as TRAIN,
    SEEDS,
    architecture_name,
)

UTILITY_THRESHOLDS = (0.00, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
MARGIN_MAXIMA = (0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 2.00)
OUT = ROOT / "artifacts/evaluation/mf3k_policy_top2_development_v1"


def load_models(hidden: int, bound: float, device: torch.device):
    models = []
    evidence = []
    for seed in SEEDS:
        path = (TRAIN / architecture_name(hidden, bound) / f"seed_{seed}"
                / "policy_top2_mf3k.pt")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not (
            payload.get("schema_version") == "revealnav-mf3k-checkpoint/1"
            and payload.get("hidden_dim") == hidden
            and float(payload.get("correction_bound")) == bound
            and payload.get("seed") == seed
            and payload.get("candidate_feature_dim") == 1536
        ):
            raise RuntimeError("MF3K checkpoint schema drift")
        model = PolicyAnchoredTop2UAD(768, 1536, hidden, bound)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append(model.to(device).eval())
        evidence.append({
            "seed": seed,
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return tuple(models), evidence


def collect(
    models, split: str, device: torch.device, data: Path = DATA
) -> list[dict]:
    loader = DataLoader(
        OnlineUADFeatureDataset(data, split), batch_size=1, shuffle=False,
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
            member_advantages = tuple(top2_posterior_advantage(
                output, batch["native_scores"], batch["candidate_mask"],
                batch["native_index"],
            )[0] for output in outputs)
            labels, runner, valid = top2_switch_targets(batch)
            for step in range(valid.shape[1]):
                if not bool(valid[0, step]):
                    continue
                native = int(batch["native_index"][0, step])
                candidate_mask = batch["candidate_mask"][0, step]
                native_score = float(batch["native_scores"][0, step, native])
                runner_score = float(batch["native_scores"][0, step,
                                                          runner[0, step]])
                rows.append({
                    "outcome": ("RESCUE" if int(labels[0, step]) == 1 else
                                "HARM" if int(labels[0, step]) == 2 else
                                "NEITHER"),
                    "minimum_advantage": min(
                        float(value[0, step]) for value in member_advantages
                    ),
                    "median_advantage": statistics.median(
                        float(value[0, step]) for value in member_advantages
                    ),
                    "native_margin": native_score - runner_score,
                    "candidate_count": int(candidate_mask.sum()),
                })
    return rows


def summarize(rows: list[dict], utility: float, margin: float) -> dict:
    selected = [
        row for row in rows
        if row["minimum_advantage"] > utility
        and row["native_margin"] <= margin
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
    for hidden, bound in ARCHITECTURES:
        nlls = [json.loads((
            TRAIN / architecture_name(hidden, bound) / f"seed_{seed}"
            / "RESULT.json"
        ).read_text())["calibration"]["target_nll"] for seed in SEEDS]
        architecture_rows.append({
            "hidden_dim": hidden,
            "correction_bound": bound,
            "member_calibration_nll": nlls,
            "median_calibration_nll": statistics.median(nlls),
        })
    selected_architecture = min(architecture_rows, key=lambda row: (
        row["median_calibration_nll"], row["hidden_dim"],
        row["correction_bound"],
    ))
    hidden = int(selected_architecture["hidden_dim"])
    bound = float(selected_architecture["correction_bound"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(hidden, bound, device)
    strata = {
        "ranks_12_13": collect(models, "diagnostic", device),
        "consumed_rank_14": collect(models, "shadow", device),
    }
    candidates = []
    for utility in UTILITY_THRESHOLDS:
        for margin in MARGIN_MAXIMA:
            summaries = {
                name: summarize(rows, utility, margin)
                for name, rows in strata.items()
            }
            pooled = {
                key: sum(value[key] for value in summaries.values())
                for key in ("interventions", "rescues", "harms", "neither",
                            "net_rescues")
            }
            lower = wilson_lower(pooled["rescues"], pooled["harms"])
            qualifies = (
                all(value["interventions"] >= 5
                    and value["net_rescues"] > 0
                    for value in summaries.values())
                and pooled["interventions"] >= 15
                and lower > 0.5
            )
            candidates.append({
                "utility_threshold": utility,
                "native_margin_max": margin,
                "strata": summaries,
                "pooled": pooled,
                "rescue_precision_wilson95_lower": lower,
                "qualifies": qualifies,
            })
    qualifying = [row for row in candidates if row["qualifies"]]
    selected_rule = max(qualifying, key=lambda row: (
        row["rescue_precision_wilson95_lower"],
        row["pooled"]["net_rescues"],
        -row["pooled"]["harms"],
        row["utility_threshold"],
        -row["native_margin_max"],
    )) if qualifying else None
    uncertainty_control = None
    if selected_rule is not None:
        margins = sorted(
            row["native_margin"] for rows in strata.values() for row in rows
        )
        budget = selected_rule["pooled"]["interventions"]
        uncertainty_control = {
            "matched_intervention_budget": budget,
            "native_margin_max": margins[min(budget, len(margins)) - 1],
            "selection_source": "pooled consumed development ranks 12-14",
        }
    status = "DEVELOPMENT_PASS" if selected_rule else "DEVELOPMENT_FAIL"
    atomic_json(OUT / "MF3K_DEVELOPMENT_SELECTION.json", {
        "schema_version": "revealnav-mf3k-development-selection/1",
        "status": status,
        "architecture_candidates": architecture_rows,
        "selected_architecture": selected_architecture,
        "rule_candidates": candidates,
        "selected_rule": selected_rule,
        "uncertainty_development_budget_match": uncertainty_control,
        "eligible_steps_by_stratum": {
            key: len(value) for key, value in strata.items()
        },
        "checkpoints": checkpoints,
        "rank14_role": "consumed_development_after_mf3j_failure",
        "ranks15_17_payload_read": False,
        "data_sha256": sha256_file(DATA),
        **MF3B_SCOPE,
    })
    return 0 if selected_rule else 2


if __name__ == "__main__":
    raise SystemExit(main())
