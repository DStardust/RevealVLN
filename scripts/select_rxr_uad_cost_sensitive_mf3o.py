#!/usr/bin/env python3
"""Select MF3O's cost-sensitive rule on consumed ranks only."""

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
    MF3B_SCOPE, OnlineUADFeatureDataset, PairwiseSwitchUtility,
    collate_online_uad, median_mad_lower_confidence,
    top2_expected_switch_utility, top2_switch_targets,
)
from scripts.train_rxr_uad_correction_mf3e import atomic_json, sha256_file  # noqa: E402
from scripts.train_rxr_uad_cost_sensitive_mf3o import (  # noqa: E402
    DATA, DESIGN, HIDDEN_DIMS, OUT as TRAIN, SEEDS,
)

MAD_WEIGHTS = (0.0, 0.5, 1.0, 1.5, 2.0)
THRESHOLDS = (-0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.025, 0.05,
              0.075, 0.10, 0.15, 0.20, 0.30)
OUT = ROOT / "artifacts/evaluation/mf3o_cost_sensitive_top2_development_v1"


def load_models(hidden: int, device: torch.device):
    models, checkpoints = [], []
    for seed in SEEDS:
        path = TRAIN / f"hidden_{hidden}/seed_{seed}/cost_sensitive_mf3o.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not (
            payload.get("schema_version") == "revealnav-mf3o-checkpoint/1"
            and payload.get("hidden_dim") == hidden
            and payload.get("seed") == seed
            and tuple(payload.get("class_weights", ())) == (1.0, 2.0, 0.5)
        ):
            raise RuntimeError("MF3O checkpoint schema drift")
        model = PairwiseSwitchUtility(768, 1536, hidden)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append(model.to(device).eval())
        checkpoints.append({"seed": seed, "path": str(path.relative_to(ROOT)),
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    return tuple(models), checkpoints


def collect(models, split: str, device: torch.device):
    loader = DataLoader(OnlineUADFeatureDataset(DATA, split), batch_size=1,
                        shuffle=False, collate_fn=collate_online_uad)
    rows = []
    with torch.no_grad():
        for cpu in loader:
            b = {k: v.to(device) for k, v in cpu.items()}
            values = []
            for model in models:
                o = model(b["history_embeddings"], b["candidate_embeddings"],
                          b["candidate_mask"], b["instruction_embedding"],
                          b["native_scores"], b["native_index"])
                values.append(top2_expected_switch_utility(o, b)[0])
            labels, runner, valid = top2_switch_targets(b)
            for step in range(valid.shape[1]):
                if not bool(valid[0, step]):
                    continue
                native = int(b["native_index"][0, step])
                rows.append({
                    "outcome": ("RESCUE" if int(labels[0, step]) == 1
                                else "HARM" if int(labels[0, step]) == 2
                                else "NEITHER"),
                    "member_utilities": [float(x[0, step]) for x in values],
                    "native_margin": float(
                        b["native_scores"][0, step, native]
                        - b["native_scores"][0, step, runner[0, step]]),
                })
    return rows


def score(row: dict, weight: float) -> float:
    return float(median_mad_lower_confidence(
        torch.tensor(row["member_utilities"]), mad_weight=weight
    ))


def summarize(rows: list[dict], weight: float, threshold: float) -> dict:
    selected = [r for r in rows if score(r, weight) > threshold]
    rescue = sum(r["outcome"] == "RESCUE" for r in selected)
    harm = sum(r["outcome"] == "HARM" for r in selected)
    return {"interventions": len(selected), "rescues": rescue,
            "harms": harm,
            "neither": len(selected) - rescue - harm,
            "net_rescues": rescue - harm}


def wilson_lower(rescue: int, harm: int) -> float:
    n = rescue + harm
    if not n: return 0.0
    z = 1.96; p = rescue / n; den = 1 + z*z/n
    return (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / den


def main() -> int:
    arch = []
    for hidden in HIDDEN_DIMS:
        nll = [json.loads((TRAIN / f"hidden_{hidden}/seed_{seed}/RESULT.json").read_text())["calibration"]["top2_nll"] for seed in SEEDS]
        arch.append({"hidden_dim": hidden, "member_calibration_top2_nll": nll,
                     "median_calibration_top2_nll": statistics.median(nll)})
    selected_arch = min(arch, key=lambda r: (r["median_calibration_top2_nll"], r["hidden_dim"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, checkpoints = load_models(int(selected_arch["hidden_dim"]), device)
    strata = {"calibration_ranks_18_20": collect(models, "calibration", device),
              "development_ranks_21_23": collect(models, "diagnostic", device)}
    candidates=[]
    for weight in MAD_WEIGHTS:
        for threshold in THRESHOLDS:
            s={k:summarize(v,weight,threshold) for k,v in strata.items()}
            pooled={key:sum(x[key] for x in s.values()) for key in ("interventions","rescues","harms","neither","net_rescues")}
            counts=[x["interventions"] for x in s.values()]
            ratio=min(counts)/max(counts) if max(counts) else 0.0
            lower=wilson_lower(pooled["rescues"],pooled["harms"])
            qualifies=(all(x["interventions"]>=10 and x["net_rescues"]>0 for x in s.values()) and pooled["interventions"]>=25 and ratio>=.5 and lower>.5)
            candidates.append({"mad_weight":weight,"utility_threshold":threshold,"strata":s,"pooled":pooled,"stratum_intervention_ratio":ratio,"rescue_precision_wilson95_lower":lower,"qualifies":qualifies})
    qualifying=[x for x in candidates if x["qualifies"]]
    rule=max(qualifying,key=lambda x:(min(v["net_rescues"] for v in x["strata"].values()),x["rescue_precision_wilson95_lower"],x["pooled"]["net_rescues"],-x["pooled"]["harms"],x["utility_threshold"],x["mad_weight"])) if qualifying else None
    atomic_json(OUT/"MF3O_DEVELOPMENT_SELECTION.json",{
        "schema_version":"revealnav-mf3o-development-selection/1",
        "status":"DEVELOPMENT_PASS" if rule else "DEVELOPMENT_FAIL",
        "architecture_candidates":arch,"selected_architecture":selected_arch,
        "rule_candidates":candidates,"selected_rule":rule,
        "eligible_steps_by_stratum":{k:len(v) for k,v in strata.items()},
        "checkpoints":checkpoints,"ranks24_29_payload_read":False,
        "data_sha256":sha256_file(DATA),"design_sha256":sha256_file(DESIGN),**MF3B_SCOPE})
    return 0 if rule else 2


if __name__ == "__main__": raise SystemExit(main())
