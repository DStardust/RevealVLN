#!/usr/bin/env python3
"""Diagnose V4 branch/action instability on the train-only development split."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from revealnav_mf2r4 import (  # noqa: E402
    BranchExcursionQHead, collate_branch_excursion_examples,
)
import run_rxr_branch_excursion_q_v4 as training  # noqa: E402
from run_rxr_opp_q_adapter_r3_2 import atomic_json, sha256_file  # noqa: E402


LOCK = ROOT / "locks/RXR_UNSEEN_CHECKPOINT_LOCK_V4_2.json"
OUT = ROOT / "artifacts/evaluation/mf2_branch_excursion_q_v4_3"
PROTOCOL = OUT / "RXR_BRANCH_EXCURSION_Q_STABILITY_PROTOCOL_V4_3.json"
RESULT = OUT / "RXR_BRANCH_EXCURSION_Q_STABILITY_DIAGNOSIS_V4_3.json"
TOLERANCES = (0.0, 0.1, 0.25, 0.5)


def checkpoint_rows() -> list[dict]:
    lock = json.loads(LOCK.read_text())
    if lock.get("status") != "LOCKED_BEFORE_UNSEEN_EVALUATION":
        raise RuntimeError("checkpoint lock status drift")
    rows = lock.get("checkpoints", [])
    if [row.get("seed") for row in rows] != list(training.SEEDS):
        raise RuntimeError("checkpoint cohort drift")
    for row in rows:
        path = (ROOT / row["path"]).resolve()
        if (
            ROOT not in path.parents or path.is_symlink() or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise RuntimeError(f"checkpoint provenance drift: {path}")
    return rows


def protocol_value() -> dict:
    _, development_ids, counts = training.partitions()
    rows = checkpoint_rows()
    return {
        "schema_version": "revealnav-mf2-branch-excursion-stability-protocol/4.3",
        "status": "SEALED_BEFORE_TRAIN_ONLY_DEVELOPMENT_STABILITY_DIAGNOSIS",
        "scope": "RxR train-only scene-disjoint internal development partition",
        "development_events": len(development_ids),
        "partition_counts": counts,
        "checkpoint_cohort": rows,
        "predeclared_selectors": [
            "individual_seed_argmin",
            "mean_predicted_cost_argmin",
            "median_predicted_cost_argmin",
            "mean_within_event_rank_argmin",
        ],
        "predeclared_metrics": [
            "macro_action_agreement",
            "branch_index_agreement",
            "joint_action_branch_agreement",
            "teacher_cost_regret",
            "oracle_equivalence",
            "predicted_top1_top2_margin",
            "teacher_best_to_next_distinct_margin",
            "candidate_count_stratification",
        ],
        "oracle_equivalence_tolerances": list(TOLERANCES),
        "selection_policy": (
            "Diagnosis only. No selector, threshold, checkpoint, or training "
            "revision is accepted automatically from this artifact."
        ),
        "forbidden_inputs": ["val_unseen", "test", "test_challenge", "Gold"],
        "sources": {
            str(LOCK.relative_to(ROOT)): sha256_file(LOCK),
            str(training.MANIFEST.relative_to(ROOT)): sha256_file(training.MANIFEST),
            "revealnav_mf2r4/model.py": sha256_file(ROOT / "revealnav_mf2r4/model.py"),
            "revealnav_mf2r4/data.py": sha256_file(ROOT / "revealnav_mf2r4/data.py"),
            "revealnav_mf2r4/losses.py": sha256_file(ROOT / "revealnav_mf2r4/losses.py"),
        },
        "gold_payload_read": False,
        "unseen_payload_read": False,
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed stability protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "sha256": sha256_file(PROTOCOL),
    }, indent=2))
    return 0


def load_models(device: torch.device) -> list[torch.nn.Module]:
    models = []
    for row in checkpoint_rows():
        path = ROOT / row["path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not (
            payload.get("seed") == row["seed"]
            and payload.get("protocol_sha256") == sha256_file(training.PROTOCOL)
            and payload.get("manifest_sha256") == sha256_file(training.MANIFEST)
        ):
            raise RuntimeError("checkpoint payload provenance drift")
        model = BranchExcursionQHead(768, 96, 128.0).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models.append(model.eval())
    return models


def quantiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def decode(index: int, candidates: int) -> tuple[str, int]:
    return ("commit", index) if index < candidates else (
        "checkpointed_excursion", index - candidates
    )


def within_event_ranks(costs: np.ndarray) -> np.ndarray:
    order = np.argsort(costs, kind="stable")
    ranks = np.empty_like(costs, dtype=np.float64)
    ranks[order] = np.arange(len(costs), dtype=np.float64)
    return ranks


def summarize_selector(indices: list[int], teachers: list[np.ndarray]) -> dict:
    regrets = []
    equivalent = {str(tolerance): 0 for tolerance in TOLERANCES}
    for index, teacher in zip(indices, teachers):
        regret = float(teacher[index] - teacher.min())
        regrets.append(regret)
        for tolerance in TOLERANCES:
            equivalent[str(tolerance)] += int(regret <= tolerance + 1e-9)
    return {
        "mean_teacher_cost_regret": statistics.mean(regrets),
        "regret_quantiles": quantiles(regrets),
        "oracle_equivalence_rate": {
            key: count / len(regrets) for key, count in equivalent.items()
        },
    }


def run(device: torch.device) -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("stability protocol must be sealed before diagnosis")
    models = load_models(device)
    _, development = training.datasets()
    loader = DataLoader(
        development, batch_size=1, shuffle=False,
        collate_fn=collate_branch_excursion_examples,
    )
    teachers: list[np.ndarray] = []
    predictions: list[list[np.ndarray]] = [[] for _ in models]
    candidate_counts = []
    teacher_margins = []
    predicted_margins = [[] for _ in models]
    with torch.no_grad():
        for cpu in loader:
            batch = training.move(cpu, device)
            valid = torch.isfinite(batch["commit_cost"][0])
            count = int(valid.sum())
            candidate_counts.append(count)
            teacher = torch.cat((
                batch["commit_cost"][0, valid],
                batch["excursion_cost"][0, valid],
            )).cpu().numpy().astype(np.float64)
            teachers.append(teacher)
            distinct = np.unique(teacher)
            teacher_margins.append(
                float(distinct[1] - distinct[0]) if len(distinct) > 1 else 0.0
            )
            for seed_index, model in enumerate(models):
                output = training.forward(model, batch)
                prediction = torch.cat((
                    output.commit_cost[0, valid],
                    output.excursion_cost[0, valid],
                )).cpu().numpy().astype(np.float64)
                predictions[seed_index].append(prediction)
                ordered = np.sort(prediction)
                predicted_margins[seed_index].append(float(ordered[1] - ordered[0]))

    individual_indices = [
        [int(row.argmin()) for row in seed_predictions]
        for seed_predictions in predictions
    ]
    mean_indices, median_indices, rank_indices = [], [], []
    for event_index in range(len(teachers)):
        stack = np.stack([
            seed_predictions[event_index] for seed_predictions in predictions
        ])
        mean_indices.append(int(stack.mean(0).argmin()))
        median_indices.append(int(np.median(stack, axis=0).argmin()))
        rank_indices.append(int(np.stack([
            within_event_ranks(row) for row in stack
        ]).mean(0).argmin()))

    decoded = [
        [decode(index, candidate_counts[event]) for event, index in enumerate(indices)]
        for indices in individual_indices
    ]
    action_agree = []
    branch_agree = []
    joint_agree = []
    per_candidate_count: dict[str, dict[str, int]] = {}
    for event in range(len(teachers)):
        values = [row[event] for row in decoded]
        same_action = len({value[0] for value in values}) == 1
        same_branch = len({value[1] for value in values}) == 1
        action_agree.append(same_action)
        branch_agree.append(same_branch)
        joint_agree.append(same_action and same_branch)
        key = str(candidate_counts[event])
        group = per_candidate_count.setdefault(key, {
            "events": 0, "action_agree": 0, "branch_agree": 0, "joint_agree": 0,
        })
        group["events"] += 1
        group["action_agree"] += int(same_action)
        group["branch_agree"] += int(same_branch)
        group["joint_agree"] += int(same_action and same_branch)

    pairwise = {}
    seeds = list(training.SEEDS)
    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            pairs = list(zip(decoded[left], decoded[right]))
            pairwise[f"{seeds[left]}_{seeds[right]}"] = {
                "macro_action_agreement": statistics.mean(
                    a[0] == b[0] for a, b in pairs
                ),
                "branch_index_agreement": statistics.mean(
                    a[1] == b[1] for a, b in pairs
                ),
                "joint_action_branch_agreement": statistics.mean(a == b for a, b in pairs),
            }

    for group in per_candidate_count.values():
        events = group["events"]
        group["macro_action_agreement_rate"] = group.pop("action_agree") / events
        group["branch_index_agreement_rate"] = group.pop("branch_agree") / events
        group["joint_action_branch_agreement_rate"] = group.pop("joint_agree") / events

    selectors = {
        str(seed): summarize_selector(indices, teachers)
        for seed, indices in zip(seeds, individual_indices)
    }
    selectors.update({
        "mean_predicted_cost": summarize_selector(mean_indices, teachers),
        "median_predicted_cost": summarize_selector(median_indices, teachers),
        "mean_within_event_rank": summarize_selector(rank_indices, teachers),
    })
    value = {
        "schema_version": "revealnav-mf2-branch-excursion-stability-diagnosis/4.3",
        "status": "TRAIN_ONLY_DEVELOPMENT_STABILITY_DIAGNOSIS_COMPLETE",
        "events": len(teachers),
        "all_three": {
            "macro_action_agreement": statistics.mean(action_agree),
            "branch_index_agreement": statistics.mean(branch_agree),
            "joint_action_branch_agreement": statistics.mean(joint_agree),
        },
        "pairwise": pairwise,
        "per_candidate_count": per_candidate_count,
        "teacher_best_to_next_distinct_margin": quantiles(teacher_margins),
        "predicted_top1_top2_margin": {
            str(seed): quantiles(values)
            for seed, values in zip(seeds, predicted_margins)
        },
        "selectors": selectors,
        "protocol_sha256": sha256_file(PROTOCOL),
        "checkpoint_lock_sha256": sha256_file(LOCK),
        "gold_payload_read": False,
        "unseen_payload_read": False,
        "paper_result": False,
        "next_gate": "main-agent causal diagnosis before any method revision",
    }
    atomic_json(RESULT, value)
    print(json.dumps(value, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    return seal() if args.seal else run(torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
