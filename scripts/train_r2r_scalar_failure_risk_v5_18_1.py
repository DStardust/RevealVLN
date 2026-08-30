#!/usr/bin/env python3
"""Train the low-capacity V5.18.1 frozen-ETP failure-risk gate."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_r2r_etp_failure_risk_v5_18 as base  # noqa: E402
from revealnav_scalar_failure_risk import (  # noqa: E402
    FEATURE_NAMES, ScalarETPFailureRiskHead, causal_scalar_features,
)


SEEDS = base.SEEDS
DEFAULT_OUT = ROOT / "artifacts/phase1/r2r_scalar_failure_risk_v5_18_1"


def _raw_batch(arrays, indices, device):
    return base._batch(arrays, indices, device)


def _forward(model, batch):
    return model(
        batch["instruction"], batch["current_history"],
        batch["temporal_history"], batch["native"], batch["alternative"],
        batch["immediate_costs"],
    )


def train(records, arrays, output: Path, device: torch.device) -> dict:
    indices = {
        split: np.asarray([
            row["row_index"] for row in records if row["partition"] == split
        ], dtype=np.int64)
        for split in ("train", "calibration", "dev")
    }
    batches = {
        split: _raw_batch(arrays, local, device)
        for split, local in indices.items()
    }
    train = batches["train"]
    with torch.no_grad():
        raw = causal_scalar_features(
            train["instruction"], train["current_history"],
            train["temporal_history"], train["native"], train["alternative"],
            train["immediate_costs"],
        )
        weights = train["group_weight"].unsqueeze(-1)
        mean = (raw * weights).sum(0) / weights.sum()
        variance = (((raw - mean) ** 2) * weights).sum(0) / weights.sum()
        scale = variance.sqrt().clamp_min(1e-4)
    states = []
    members = []
    labels = train["failure"]
    group_weight = train["group_weight"]
    weighted_positive = float((group_weight * labels).sum())
    weighted_negative = float((group_weight * (1.0 - labels)).sum())
    pos_weight = torch.tensor(
        weighted_negative / max(weighted_positive, 1e-6), device=device
    )
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = ScalarETPFailureRiskHead(mean, scale).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-3, weight_decay=5e-2
        )
        for _ in range(400):
            model.train()
            logits = _forward(model, train)
            losses = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight, reduction="none"
            )
            loss = (losses * group_weight).sum() / group_weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        member = {"seed": seed}
        with torch.no_grad():
            for split in indices:
                probability = torch.sigmoid(_forward(model, batches[split])).cpu().numpy()
                labels_g, scores_g, _ = base._group_scores(
                    [records[index] for index in indices[split]], probability
                )
                member[f"{split}_group_auc"] = base.auc(labels_g, scores_g)
        members.append(member)
    models = []
    for state in states:
        model = ScalarETPFailureRiskHead(mean, scale).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
    group_values = {}
    for split in indices:
        with torch.no_grad():
            probability = torch.stack([
                torch.sigmoid(_forward(model, batches[split])) for model in models
            ]).mean(0).cpu().numpy()
        group_values[split] = base._group_scores(
            [records[index] for index in indices[split]], probability
        )
    threshold, calibration_policy = base._threshold(
        *group_values["calibration"][:2]
    )
    policies = {
        split: base._policy(values[0], values[1], threshold)
        for split, values in group_values.items()
    }
    group_auc = {
        split: base.auc(values[0], values[1])
        for split, values in group_values.items()
    }
    checkpoint = output / "scalar_etp_failure_risk_ensemble_v5_18_1.pt"
    payload = {
        "schema_version": "revealnav-scalar-etp-failure-risk-ensemble/1",
        "member_seeds": list(SEEDS),
        "model_state_dicts": states,
        "aggregation": "mean_failure_probability",
        "threshold": threshold,
        "threshold_rule": (
            "maximize calibration group failure recall subject to frozen-ETP "
            "successful-group false-positive rate <= 0.10; tie-break by "
            "precision, sparsity, then higher threshold"
        ),
        "feature_names": list(FEATURE_NAMES),
        "input_contract": "causal cosine alignments and online branch distances",
        "immediate_cost_scale_m": 10.0,
        "label": "1 - final frozen ETP-R1 train episode success",
    }
    part = checkpoint.with_name(checkpoint.name + ".part")
    torch.save(payload, part)
    os.replace(part, checkpoint)
    gates = {
        "low_capacity_logistic_model": sum(value.numel() for value in models[0].parameters()) == 17,
        "calibration_fpr_at_most_0_10": calibration_policy["fpr"] <= 0.10,
        "calibration_detects_failure": calibration_policy["tp"] > 0,
        "dev_auc_above_random": group_auc["dev"] > 0.5,
        "dev_detects_failure": policies["dev"]["tp"] > 0,
        "dev_fpr_at_most_0_20": policies["dev"]["fpr"] <= 0.20,
    }
    result = {
        "schema_version": "revealnav-r2r-scalar-etp-failure-risk-training/1",
        "status": "R2R_SCALAR_FAILURE_RISK_PASS" if all(gates.values()) else "R2R_SCALAR_FAILURE_RISK_FAIL",
        "revision_reason": "raw-embedding V5.18 overfit across scenes",
        "fixed_epochs": 400,
        "members": members,
        "ensemble_group_auc": group_auc,
        "threshold": threshold,
        "policies": policies,
        "gates": gates,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": base.sha256_file(checkpoint),
        },
        "calibration_selected_threshold": True,
        "dev_used_for_threshold_or_training": False,
        "unseen_or_test_read": False,
        "paper_result": False,
    }
    base.atomic_json(output / "R2R_SCALAR_ETP_FAILURE_RISK_TRAINING.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT not in output.parents or output.exists():
        raise SystemExit("output directory must be new and inside the project")
    output.mkdir(parents=True)
    records, arrays, dataset = base.build_dataset(output)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = train(records, arrays, output, device)
    print(json.dumps({
        "dataset": dataset["counts"],
        "status": result["status"],
        "auc": result["ensemble_group_auc"],
        "threshold": result["threshold"],
        "policies": result["policies"],
        "gates": result["gates"],
    }, indent=2, sort_keys=True))
    return 0 if all(result["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
