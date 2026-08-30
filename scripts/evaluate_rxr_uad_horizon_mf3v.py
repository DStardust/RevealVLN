#!/usr/bin/env python3
"""Fresh RxR-train shadow gate for MF3V horizon-consistent UAD."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import (  # noqa: E402
    MF3B_SCOPE, OnlineUADFeatureDataset, PairwiseSwitchUtility,
    collate_online_uad, median_mad_lower_confidence,
    top2_horizon_switch_targets, top2_rescue_harm_logit,
)
from scripts.select_rxr_uad_policy_risk_mf3s import exact_control  # noqa: E402
from scripts.train_rxr_uad_crossfit_mf3q import manifest_path  # noqa: E402
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file  # noqa: E402


TRAIN = ROOT / "artifacts/training/mf3v_horizon_ranker_v1"
DATA = ROOT / (
    "artifacts/phase1/mf3t_coverage_gate_rank41/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
OUT = ROOT / "artifacts/evaluation/mf3v_horizon_shadow_gate_v1"
SEEDS = (20260826, 20260827, 20260828)
HIDDEN = 128
HORIZON = 3
MAD_WEIGHT = 0.5
BETA = 0.25
LOW_QUANTILE = 0.985
UPPER_QUANTILE = 0.995


def load_models(device):
    models = []; evidence = []
    for seed in SEEDS:
        path = TRAIN / f"fold_final/seed_{seed}/horizon_ranker_mf3v.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not (
            payload.get("schema_version") == "revealnav-mf3v-checkpoint/1"
            and payload.get("fold") == "final"
            and payload.get("seed") == seed
            and payload.get("hidden_dim") == HIDDEN
            and payload.get("horizon") == HORIZON
        ):
            raise RuntimeError("MF3V checkpoint drift")
        model = PairwiseSwitchUtility(768, 1536, HIDDEN)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append(model.to(device).eval())
        evidence.append({"seed": seed, "path": str(path.relative_to(ROOT)),
                         "bytes": path.stat().st_size,
                         "sha256": sha256_file(path), "strict_load": True})
    return tuple(models), evidence


def collect(models, split, data, device):
    loader = DataLoader(
        OnlineUADFeatureDataset(data, split), batch_size=1, shuffle=False,
        collate_fn=collate_online_uad,
    )
    episodes = []
    with torch.no_grad():
        for cpu in loader:
            batch = {key: value.to(device) for key, value in cpu.items()}
            outputs = [model(
                batch["history_embeddings"], batch["candidate_embeddings"],
                batch["candidate_mask"], batch["instruction_embedding"],
                batch["native_scores"], batch["native_index"],
            ) for model in models]
            labels, runner, valid = top2_horizon_switch_targets(
                batch, horizon=HORIZON
            )
            sequence = []
            for step in range(valid.shape[1]):
                if not bool(valid[0, step]):
                    sequence.append(None); continue
                native = int(batch["native_index"][0, step])
                runner_index = int(runner[0, step])
                margin = float(
                    batch["native_scores"][0, step, native]
                    - batch["native_scores"][0, step, runner_index]
                )
                member_logits = [
                    float(top2_rescue_harm_logit(output, batch)[0][0, step])
                    for output in outputs
                ]
                outcome = {1: "RESCUE", 2: "HARM"}.get(
                    int(labels[0, step]), "NEITHER"
                )
                sequence.append({
                    "outcome": outcome, "member_logits": member_logits,
                    "native_margin": margin,
                })
            episodes.append(sequence)
    return episodes


def score(row):
    median = sorted(row["member_logits"])[1]
    mad = sorted(abs(value - median) for value in row["member_logits"])[1]
    return median - MAD_WEIGHT * mad - BETA * math.log1p(
        max(0.0, row["native_margin"])
    )


def summarize(episodes, low, high):
    selected = []
    for sequence in episodes:
        for row in sequence:
            if row is not None and low < score(row) <= high:
                selected.append(row); break
    rescues = sum(row["outcome"] == "RESCUE" for row in selected)
    harms = sum(row["outcome"] == "HARM" for row in selected)
    return {"interventions": len(selected), "rescues": rescues,
            "harms": harms, "neither": len(selected) - rescues - harms,
            "net_rescues": rescues - harms}


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(device)
    fit = collect(models, "fit", manifest_path("final"), device)
    fit_scores = [score(row) for sequence in fit for row in sequence
                  if row is not None]
    low = float(torch.quantile(torch.tensor(fit_scores), LOW_QUANTILE))
    high = float(torch.quantile(torch.tensor(fit_scores), UPPER_QUANTILE))
    shadow_episodes = collect(models, "shadow", DATA, device)
    shadow = summarize(shadow_episodes, low, high)
    # Reuse the already computed selection budget on a fresh pass; this keeps
    # the exact-control definition independent of model scores.
    control = exact_control(shadow_episodes, shadow["interventions"])
    gates = {
        "fresh_shadow_has_twenty_interventions": shadow["interventions"] >= 20,
        "fresh_shadow_net_rescue_positive": shadow["net_rescues"] > 0,
        "fresh_shadow_beats_exact_budget_control": (
            shadow["net_rescues"] > control["net_rescues"] or (
                shadow["net_rescues"] == control["net_rescues"]
                and shadow["harms"] < control["harms"]
            )
        ),
    }
    passed = all(gates.values())
    protocol = TRAIN / "MF3V_TRAINING_PROTOCOL.json"
    atomic_json(OUT / "MF3V_SHADOW_GATE.json", {
        "schema_version": "revealnav-mf3v-shadow-gate/1",
        "status": "SHADOW_GATE_PASS" if passed else "SHADOW_GATE_FAIL",
        "selected_architecture": {"hidden_dim": HIDDEN},
        "selected_rule": {
            "hidden_dim": HIDDEN, "horizon": HORIZON,
            "mad_weight": MAD_WEIGHT, "policy_risk_beta": BETA,
            "training_score_quantile": LOW_QUANTILE,
            "final_training_threshold": low,
            "score_upper_quantile": UPPER_QUANTILE,
            "score_upper_threshold": high, "persistence_steps": 1,
        },
        "shadow": shadow, "exact_budget_control": control, "gates": gates,
        "task_metric_run_authorized": passed,
        "fresh_data": {"path": str(DATA.relative_to(ROOT)),
                        "bytes": DATA.stat().st_size,
                        "sha256": sha256_file(DATA), "shadow_episodes": 336},
        "checkpoints": checkpoints,
        "training_protocol_sha256": sha256_file(protocol),
        "design_sha256": sha256_file(
            ROOT / "artifacts/design/METHOD_FREEZE_3V_HORIZON_CONSISTENT_UAD.md"
        ),
        **MF3B_SCOPE,
    })
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
