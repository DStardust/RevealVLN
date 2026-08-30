#!/usr/bin/env python3
"""Train-only development selection for an adaptive MF3V tail gate.

MF3Z keeps MF3V's frozen ranker and lower crossing threshold.  In the
upper-score tail it uses a fit-derived relative native-margin cap, so the
admission rule is scale-aware instead of using a fixed absolute margin.
No val_seen or unseen payload is read by this script.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from evaluate_rxr_uad_horizon_mf3v import DATA, collect, load_models, manifest_path
from select_rxr_uad_policy_risk_mf3s import exact_control
from train_rxr_uad_correction_mf3e import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "artifacts/training/mf3v_horizon_ranker_v1"
OUT = ROOT / "artifacts/evaluation/mf3z_adaptive_tail_development_v1"
DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3Y_CONSENSUS_TAIL_GATE.md"
LOW_QUANTILE = 0.985
UPPER_QUANTILE = 0.995
MAD_QUANTILES = (0.5, 0.75, 0.9)
MAD_FLOOR_QUANTILES = (0.0, 0.1, 0.25)
MARGIN_QUANTILES = (0.05, 0.1, 0.15, 0.25)
RATIO_QUANTILES = (0.5, 0.75, 0.9, 0.95)


def terms(row: dict) -> tuple[float, float, float]:
    median = sorted(row["member_logits"])[1]
    mad = sorted(abs(value - median) for value in row["member_logits"])[1]
    score = median - 0.5 * mad - 0.25 * math.log1p(
        max(0.0, row["native_margin"])
    )
    return score, mad, row["native_margin"]


def quantile(values: list[float], level: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), level))


def summarize(episodes, low, upper, mad_floor, mad_cap, margin_cap, ratio_cap):
    selected = []
    for sequence in episodes:
        for row in sequence:
            if row is None:
                continue
            score, mad, margin = terms(row)
            relative_margin = margin / max(score, 1e-6)
            eligible = score > low and (
                score <= upper
                or (
                    mad_floor <= mad <= mad_cap
                    and relative_margin <= ratio_cap
                )
            )
            if eligible:
                selected.append(row)
                break
    rescues = sum(row["outcome"] == "RESCUE" for row in selected)
    harms = sum(row["outcome"] == "HARM" for row in selected)
    return {
        "interventions": len(selected),
        "rescues": rescues,
        "harms": harms,
        "neither": len(selected) - rescues - harms,
        "net_rescues": rescues - harms,
    }


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(device)
    fit = collect(models, "fit", manifest_path("final"), device)
    shadow = collect(models, "shadow", DATA, device)
    fit_terms = [terms(row) for sequence in fit for row in sequence if row is not None]
    fit_scores = [row[0] for row in fit_terms]
    fit_mads = [row[1] for row in fit_terms]
    fit_margins = [row[2] for row in fit_terms]
    low = quantile(fit_scores, LOW_QUANTILE)
    upper = quantile(fit_scores, UPPER_QUANTILE)
    tail_ratios = [
        margin / max(score, 1e-6)
        for score, _, margin in fit_terms
        if score > upper
    ]
    ratio_caps = {q: quantile(tail_ratios, q) for q in RATIO_QUANTILES}
    mad_caps = {q: quantile(fit_mads, q) for q in MAD_QUANTILES}
    mad_floors = {q: quantile(fit_mads, q) for q in MAD_FLOOR_QUANTILES}
    margin_caps = {q: quantile(fit_margins, q) for q in MARGIN_QUANTILES}
    controls = {}
    candidates = []
    for mad_floor_q, mad_floor in mad_floors.items():
      for mad_q, mad_cap in mad_caps.items():
        for ratio_q, ratio_cap in ratio_caps.items():
            result = summarize(
                shadow, low, upper, mad_floor, mad_cap, float("inf"), ratio_cap
            )
            budget = result["interventions"]
            controls.setdefault(budget, None)
            candidates.append({
                "rule": "relative_margin_tail",
                "mad_quantile": mad_q,
                "mad_floor_quantile": mad_floor_q,
                "mad_floor_threshold": mad_floor,
                "mad_threshold": mad_cap,
                "ratio_quantile": ratio_q,
                "ratio_threshold": ratio_cap,
                "shadow": result,
            })
    for budget in controls:
        controls[budget] = exact_control(shadow, budget)
    for candidate in candidates:
        control = controls[candidate["shadow"]["interventions"]]
        candidate["exact_budget_control"] = control
        result = candidate["shadow"]
        candidate["qualifies"] = (
            result["interventions"] >= 20
            and result["net_rescues"] > 0
            and (
                result["net_rescues"] > control["net_rescues"]
                or (
                    result["net_rescues"] == control["net_rescues"]
                    and result["harms"] < control["harms"]
                )
            )
        )
    # MF3Z is a coverage-constrained tail rule: among train-shadow rules that
    # beat the exact-budget control, retain at least 20 interventions and keep
    # the observed harm fraction at or below 20%.  This fixed objective is
    # selected without reading val_seen or unseen.
    for candidate in candidates:
        result = candidate["shadow"]
        candidate["harm_rate"] = (
            result["harms"] / result["interventions"]
            if result["interventions"] else 1.0
        )
        candidate["qualifies"] = candidate["qualifies"] and (
            candidate["harm_rate"] <= 0.20
        )
    qualified = [row for row in candidates if row["qualifies"]]
    selected = max(
        qualified,
        key=lambda row: (
            row["shadow"]["interventions"],
            row["shadow"]["net_rescues"]
            - row["exact_budget_control"]["net_rescues"],
            -row["shadow"]["harms"],
            -row["ratio_quantile"],
            row["mad_quantile"],
        ),
    ) if qualified else None
    atomic_json(OUT / "MF3Z_ADAPTIVE_TAIL_DEVELOPMENT.json", {
        "schema_version": "revealnav-mf3z-adaptive-tail-development/1",
        "status": "DEVELOPMENT_PASS" if selected else "DEVELOPMENT_FAIL",
        "fit_only_thresholds": {
            "lower_quantile": LOW_QUANTILE,
            "lower_threshold": low,
            "upper_quantile": UPPER_QUANTILE,
            "upper_threshold": upper,
            "mad_thresholds": mad_caps,
            "mad_floor_thresholds": mad_floors,
            "tail_relative_margin_thresholds": ratio_caps,
            "tail_fit_rows": len(tail_ratios),
        },
        "selected_rule": selected,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "qualified_count": len(qualified),
        "checkpoints": checkpoints,
        "train_manifest": {
            "path": str(DATA.relative_to(ROOT)),
            "bytes": DATA.stat().st_size,
            "sha256": sha256_file(DATA),
            "shadow_episodes": len(shadow),
        },
        "design_sha256": sha256_file(DESIGN),
        "public_unseen_authorized": False,
        "task_metric_run_authorized": False,
    })
    print(json.dumps({
        "status": "DEVELOPMENT_PASS" if selected else "DEVELOPMENT_FAIL",
        "fit_only_thresholds": {
            "lower": low, "upper": upper,
            "mad": mad_caps, "ratio": ratio_caps,
        },
        "selected": selected,
        "qualified_count": len(qualified),
    }, indent=2))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
