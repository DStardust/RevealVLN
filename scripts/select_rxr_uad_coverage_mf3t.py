#!/usr/bin/env python3
"""Select the frozen MF3T coverage-constrained operating point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import MF3B_SCOPE, PairwiseSwitchUtility
from scripts.select_rxr_uad_policy_risk_mf3s import (
    exact_control,
    summarize,
    threshold,
)
from scripts.select_rxr_uad_rescue_harm_mf3p import collect, wilson
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file
from scripts.train_rxr_uad_coverage_mf3t import (
    HIDDEN_DIMS,
    OUT as TRAIN,
    RANKING_WEIGHT,
    SEEDS,
    STEPS,
)
from scripts.train_rxr_uad_crossfit_mf3q import FOLDS, manifest_path


DESIGN = ROOT / "artifacts/design/METHOD_FREEZE_3T_COVERAGE_CONSTRAINED_UAD.md"
DATA29 = ROOT / (
    "artifacts/phase1/mf3n_top2_utility_rank29/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
DATA35 = ROOT / (
    "artifacts/phase1/mf3s_joint_gate_rank35/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
OUT = ROOT / "artifacts/evaluation/mf3t_coverage_development_v2"
MAD_WEIGHTS = (0.0, 0.5, 1.0)
BETAS = (0.25, 0.5, 1.0)
QUANTILES = (0.975, 0.985, 0.99, 0.995)


def load_models(hidden: int, fold: int | str, device: torch.device):
    models = []
    evidence = []
    for seed in SEEDS:
        checkpoint = TRAIN / (
            f"hidden_{hidden}/fold_{fold}/seed_{seed}/coverage_ranker_mf3t.pt"
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not (
            payload.get("schema_version") == "revealnav-mf3t-checkpoint/2"
            and payload.get("fold") == fold
            and payload.get("seed") == seed
            and payload.get("hidden_dim") == hidden
            and payload.get("optimizer_steps") == STEPS
            and float(payload.get("ranking_weight")) == RANKING_WEIGHT
        ):
            raise RuntimeError("MF3T checkpoint drift")
        model = PairwiseSwitchUtility(768, 1536, hidden)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append(model.to(device).eval())
        evidence.append({
            "fold": fold,
            "seed": seed,
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        })
    return tuple(models), evidence


def beats(summary: dict, control: dict) -> bool:
    return summary["net_rescues"] > control["net_rescues"] or (
        summary["net_rescues"] == control["net_rescues"]
        and summary["harms"] < control["harms"]
    )


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    candidates = []
    checkpoint_evidence = []
    for hidden in HIDDEN_DIMS:
        train = {}
        strata = {}
        for fold in FOLDS:
            models, evidence = load_models(hidden, fold, device)
            checkpoint_evidence.extend(evidence)
            train[fold] = collect(models, "fit", device, manifest_path(fold))
            strata[fold] = collect(
                models, "calibration", device, manifest_path(fold)
            )
        final_models, evidence = load_models(hidden, "final", device)
        checkpoint_evidence.extend(evidence)
        train["final"] = collect(
            final_models, "fit", device, manifest_path("final")
        )
        strata["ranks24_29"] = collect(final_models, "shadow", device, DATA29)
        strata["ranks30_35"] = collect(final_models, "shadow", device, DATA35)
        controls = {"ranks24_29": {}, "ranks30_35": {}}
        for mad_weight in MAD_WEIGHTS:
            for beta in BETAS:
                for quantile in QUANTILES:
                    cutoffs = {
                        fold: threshold(
                            train[fold], mad_weight, beta, quantile
                        )
                        for fold in FOLDS
                    }
                    final_cutoff = threshold(
                        train["final"], mad_weight, beta, quantile
                    )
                    summaries = {
                        f"oof_fold_{fold}": summarize(
                            strata[fold], mad_weight, beta, cutoffs[fold], 1
                        )
                        for fold in FOLDS
                    }
                    for name in ("ranks24_29", "ranks30_35"):
                        summaries[name] = summarize(
                            strata[name], mad_weight, beta, final_cutoff, 1
                        )
                    controls_for_rule = {}
                    for name in ("ranks24_29", "ranks30_35"):
                        budget = summaries[name]["interventions"]
                        if budget not in controls[name]:
                            controls[name][budget] = exact_control(
                                strata[name], budget
                            )
                        controls_for_rule[name] = controls[name][budget]
                    pooled = {
                        key: sum(row[key] for row in summaries.values())
                        for key in (
                            "interventions", "rescues", "harms", "neither",
                            "net_rescues",
                        )
                    }
                    lower = wilson(pooled["rescues"], pooled["harms"])
                    oof_ok = all(
                        summaries[f"oof_fold_{fold}"]["interventions"] >= 10
                        and summaries[f"oof_fold_{fold}"]["net_rescues"] > 0
                        for fold in FOLDS
                    )
                    development_ok = all(
                        summaries[name]["interventions"] >= 30
                        and summaries[name]["net_rescues"] > 0
                        and beats(summaries[name], controls_for_rule[name])
                        for name in ("ranks24_29", "ranks30_35")
                    )
                    candidates.append({
                        "hidden_dim": hidden,
                        "mad_weight": mad_weight,
                        "policy_risk_beta": beta,
                        "training_score_quantile": quantile,
                        "persistence_steps": 1,
                        "fold_thresholds": cutoffs,
                        "final_training_threshold": final_cutoff,
                        "strata": summaries,
                        "development_exact_budget_uncertainty": controls_for_rule,
                        "pooled": pooled,
                        "rescue_precision_wilson95_lower": lower,
                        "qualifies": oof_ok and development_ok and lower > 0.5,
                    })
    qualifying = [row for row in candidates if row["qualifies"]]
    selected = max(
        qualifying,
        key=lambda row: (
            min(
                row["strata"][name]["net_rescues"]
                for name in ("ranks24_29", "ranks30_35")
            ),
            row["rescue_precision_wilson95_lower"],
            row["pooled"]["net_rescues"],
            row["pooled"]["interventions"],
            -row["pooled"]["harms"],
            row["training_score_quantile"],
            -row["hidden_dim"],
        ),
    ) if qualifying else None
    atomic_json(OUT / "MF3T_DEVELOPMENT_SELECTION.json", {
        "schema_version": "revealnav-mf3t-development-selection/1",
        "status": "DEVELOPMENT_PASS" if selected else "DEVELOPMENT_FAIL",
        "selected_rule": selected,
        "candidate_count": len(candidates),
        "qualifying_count": len(qualifying),
        "rule_candidates": candidates,
        "checkpoint_evidence": checkpoint_evidence,
        "ranks36_41_payload_read": False,
        "design_sha256": sha256_file(DESIGN),
        **MF3B_SCOPE,
    })
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
