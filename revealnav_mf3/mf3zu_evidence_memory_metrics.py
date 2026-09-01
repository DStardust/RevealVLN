"""Ranking metrics, scene bootstrap, and fixed RxR-only gates for MF3ZU."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


ARM_CURRENT = "ETP_CURRENT"
ARM_MEMORY = "ETP_PLUS_EVIDENCE_MEMORY"
ARM_SHUFFLED = "ETP_PLUS_SHUFFLED_MEMORY"
ARMS = (ARM_CURRENT, ARM_MEMORY, ARM_SHUFFLED)
SUBGROUP_ALL = "ALL"
SUBGROUP_REQUIRED = "MEMORY_REQUIRED"
SUBGROUP_NOT_REQUIRED = "MEMORY_NOT_REQUIRED"
SUBGROUPS = (SUBGROUP_ALL, SUBGROUP_REQUIRED, SUBGROUP_NOT_REQUIRED)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_901
MIN_REQUIRED_DECISIONS = 50
MIN_REQUIRED_SCENES = 10


class EvidenceMemoryMetricError(ValueError):
    """Raised for incomplete OOF predictions or invalid scientific metrics."""


def _validated_ranks(
    scores: np.ndarray,
    target_index: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target_index, dtype=np.int64)
    mask = np.asarray(candidate_mask, dtype=bool)
    if score.ndim != 2 or mask.shape != score.shape or target.shape != (len(score),):
        raise EvidenceMemoryMetricError("ranking arrays have incompatible shapes")
    if len(score) == 0 or np.any(mask.sum(axis=1) < 2):
        raise EvidenceMemoryMetricError("ranking evaluation needs non-empty candidate sets")
    if np.any((target < 0) | (target >= score.shape[1])):
        raise EvidenceMemoryMetricError("target index is out of bounds")
    if np.any(~mask[np.arange(len(score)), target]):
        raise EvidenceMemoryMetricError("target candidate is not executable")
    if not np.isfinite(score[mask]).all():
        raise EvidenceMemoryMetricError("active OOF scores contain non-finite values")

    ranks = np.empty(len(score), dtype=np.int64)
    top1 = np.empty(len(score), dtype=np.float64)
    pair_wins = np.empty(len(score), dtype=np.float64)
    pair_totals = mask.sum(axis=1).astype(np.int64) - 1
    for row in range(len(score)):
        active = np.flatnonzero(mask[row])
        # Stable sorting makes score ties deterministic in frozen candidate
        # order instead of introducing another random decision rule.
        order = active[np.argsort(-score[row, active], kind="stable")]
        rank = int(np.flatnonzero(order == target[row])[0]) + 1
        ranks[row] = rank
        top1[row] = float(order[0] == target[row])
        target_score = score[row, target[row]]
        competitor = score[row, active[active != target[row]]]
        pair_wins[row] = float(
            np.sum(target_score > competitor)
            + 0.5 * np.sum(target_score == competitor)
        )
    return ranks, top1, pair_wins, pair_totals


def ranking_contributions(
    scores: np.ndarray,
    target_index: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return paired event contributions used by metrics and bootstrap."""

    ranks, top1, pair_wins, pair_totals = _validated_ranks(
        scores, target_index, candidate_mask
    )
    return {
        "Acc@1": top1,
        "MRR": 1.0 / ranks.astype(np.float64),
        "TargetRank": ranks.astype(np.float64),
        "PairWins": pair_wins,
        "PairTotals": pair_totals.astype(np.float64),
    }


def summarize_ranking(
    scores: np.ndarray,
    target_index: np.ndarray,
    candidate_mask: np.ndarray,
    selection_mask: np.ndarray | None = None,
) -> dict[str, object]:
    contributions = ranking_contributions(scores, target_index, candidate_mask)
    selected = (
        np.ones(len(target_index), dtype=bool)
        if selection_mask is None
        else np.asarray(selection_mask, dtype=bool)
    )
    if selected.shape != (len(target_index),):
        raise EvidenceMemoryMetricError("selection mask has the wrong shape")
    count = int(selected.sum())
    if count == 0:
        return {
            "decisions": 0,
            "Acc@1": None,
            "MRR": None,
            "MeanRank": None,
            "pairwise_accuracy": None,
            "unsupported": True,
        }
    pair_total = float(contributions["PairTotals"][selected].sum())
    return {
        "decisions": count,
        "Acc@1": float(contributions["Acc@1"][selected].mean()),
        "MRR": float(contributions["MRR"][selected].mean()),
        "MeanRank": float(contributions["TargetRank"][selected].mean()),
        "pairwise_accuracy": float(
            contributions["PairWins"][selected].sum() / pair_total
        ),
        "unsupported": False,
    }


def _metric_delta(
    memory: Mapping[str, object], control: Mapping[str, object]
) -> dict[str, object]:
    if bool(memory.get("unsupported")) or bool(control.get("unsupported")):
        return {
            "Acc@1": None,
            "MRR": None,
            "MeanRank": None,
            "pairwise_accuracy": None,
            "unsupported": True,
        }
    return {
        "Acc@1": float(memory["Acc@1"]) - float(control["Acc@1"]),
        "MRR": float(memory["MRR"]) - float(control["MRR"]),
        "MeanRank": float(memory["MeanRank"]) - float(control["MeanRank"]),
        "pairwise_accuracy": float(memory["pairwise_accuracy"])
        - float(control["pairwise_accuracy"]),
        "unsupported": False,
    }


def scene_cluster_bootstrap_paired_delta(
    memory_values: Sequence[float],
    control_values: Sequence[float],
    scene_ids: Sequence[object],
    selection_mask: Sequence[bool],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Bootstrap a paired row-mean delta by raw MP3D scene clusters."""

    memory = np.asarray(memory_values, dtype=np.float64)
    control = np.asarray(control_values, dtype=np.float64)
    scenes = np.asarray([str(value) for value in scene_ids])
    selected = np.asarray(selection_mask, dtype=bool)
    if (
        memory.ndim != 1
        or control.shape != memory.shape
        or scenes.shape != memory.shape
        or selected.shape != memory.shape
    ):
        raise EvidenceMemoryMetricError("bootstrap arrays have incompatible shapes")
    if isinstance(replicates, bool) or int(replicates) < 1:
        raise EvidenceMemoryMetricError("bootstrap replicate count must be positive")
    if not np.isfinite(memory[selected]).all() or not np.isfinite(control[selected]).all():
        raise EvidenceMemoryMetricError("bootstrap values contain non-finite entries")
    selected_scenes = sorted(set(scenes[selected].tolist()))
    if int(selected.sum()) == 0 or len(selected_scenes) < 2:
        return {
            "observed": None,
            "lower_95": None,
            "upper_95": None,
            "scene_count": len(selected_scenes),
            "decision_count": int(selected.sum()),
            "cluster": "raw_mp3d_scene",
            "replicates": int(replicates),
            "seed": int(seed),
            "unsupported": True,
        }
    delta = memory - control
    scene_sum = np.asarray(
        [delta[selected & (scenes == scene)].sum() for scene in selected_scenes],
        dtype=np.float64,
    )
    scene_count = np.asarray(
        [int(np.sum(selected & (scenes == scene))) for scene in selected_scenes],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    # Chunking avoids allocating replicates x scenes for very large protocols.
    chunk = 1_000
    for start in range(0, int(replicates), chunk):
        stop = min(start + chunk, int(replicates))
        draw = rng.integers(
            0, len(selected_scenes), size=(stop - start, len(selected_scenes))
        )
        samples[start:stop] = scene_sum[draw].sum(axis=1) / scene_count[draw].sum(axis=1)
    return {
        "observed": float(delta[selected].mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "scene_count": len(selected_scenes),
        "decision_count": int(selected.sum()),
        "cluster": "raw_mp3d_scene",
        "replicates": int(replicates),
        "seed": int(seed),
        "unsupported": False,
    }


def evaluate_three_arm_probe(
    scores_by_arm: Mapping[str, np.ndarray],
    target_index: np.ndarray,
    candidate_mask: np.ndarray,
    scene_ids: Sequence[object],
    memory_required: Sequence[bool],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Compute every fixed metric and paired B-A/B-C bootstrap."""

    if set(scores_by_arm) != set(ARMS):
        raise EvidenceMemoryMetricError("the fixed three arms are required")
    target = np.asarray(target_index, dtype=np.int64)
    mask = np.asarray(candidate_mask, dtype=bool)
    scenes = np.asarray([str(value) for value in scene_ids])
    required = np.asarray(memory_required, dtype=bool)
    if scenes.shape != target.shape or required.shape != target.shape:
        raise EvidenceMemoryMetricError("scene/subgroup arrays have the wrong shape")
    subgroup_masks = {
        SUBGROUP_ALL: np.ones(len(target), dtype=bool),
        SUBGROUP_REQUIRED: required,
        SUBGROUP_NOT_REQUIRED: ~required,
    }
    contributions = {
        arm: ranking_contributions(np.asarray(scores_by_arm[arm]), target, mask)
        for arm in ARMS
    }
    metrics: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        metrics[arm] = {
            subgroup: summarize_ranking(
                np.asarray(scores_by_arm[arm]), target, mask, subgroup_mask
            )
            for subgroup, subgroup_mask in subgroup_masks.items()
        }

    comparisons = {
        "B_minus_A": (ARM_MEMORY, ARM_CURRENT),
        "B_minus_C": (ARM_MEMORY, ARM_SHUFFLED),
    }
    deltas: dict[str, dict[str, object]] = defaultdict(dict)
    bootstrap: dict[str, dict[str, object]] = defaultdict(dict)
    for label, (left, right) in comparisons.items():
        for subgroup, subgroup_mask in subgroup_masks.items():
            deltas[label][subgroup] = _metric_delta(
                metrics[left][subgroup], metrics[right][subgroup]
            )
            bootstrap[label][subgroup] = {
                metric: scene_cluster_bootstrap_paired_delta(
                    contributions[left][metric],
                    contributions[right][metric],
                    scenes,
                    subgroup_mask,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                )
                for metric in ("Acc@1", "MRR")
            }
    return {
        "domain": "RxR",
        "metrics": metrics,
        "pairwise_deltas": dict(deltas),
        "scene_bootstrap_CI": dict(bootstrap),
        "subgroup_support": {
            subgroup: {
                "decisions": int(subgroup_mask.sum()),
                "raw_scenes": len(set(scenes[subgroup_mask].tolist())),
            }
            for subgroup, subgroup_mask in subgroup_masks.items()
        },
    }


def apply_fixed_rxr_gates(evaluation: Mapping[str, object]) -> dict[str, object]:
    """Apply the sealed single-domain diagnostic gates without rescue knobs."""

    support = evaluation["subgroup_support"][SUBGROUP_REQUIRED]
    delta_ba = evaluation["pairwise_deltas"]["B_minus_A"]
    delta_bc = evaluation["pairwise_deltas"]["B_minus_C"]
    boot_ba = evaluation["scene_bootstrap_CI"]["B_minus_A"]
    boot_bc = evaluation["scene_bootstrap_CI"]["B_minus_C"]
    checks = {
        "memory_required_decisions_at_least_50": int(support["decisions"])
        >= MIN_REQUIRED_DECISIONS,
        "memory_required_raw_scenes_at_least_10": int(support["raw_scenes"])
        >= MIN_REQUIRED_SCENES,
        "memory_required_B_minus_A_Acc_positive": (
            delta_ba[SUBGROUP_REQUIRED]["Acc@1"] is not None
            and float(delta_ba[SUBGROUP_REQUIRED]["Acc@1"]) > 0.0
        ),
        "memory_required_B_minus_A_Acc_lower95_positive": (
            boot_ba[SUBGROUP_REQUIRED]["Acc@1"]["lower_95"] is not None
            and float(boot_ba[SUBGROUP_REQUIRED]["Acc@1"]["lower_95"]) > 0.0
        ),
        "memory_required_B_minus_A_MRR_positive": (
            delta_ba[SUBGROUP_REQUIRED]["MRR"] is not None
            and float(delta_ba[SUBGROUP_REQUIRED]["MRR"]) > 0.0
        ),
        "memory_required_B_minus_C_Acc_positive": (
            delta_bc[SUBGROUP_REQUIRED]["Acc@1"] is not None
            and float(delta_bc[SUBGROUP_REQUIRED]["Acc@1"]) > 0.0
        ),
        "memory_required_B_minus_C_Acc_lower95_positive": (
            boot_bc[SUBGROUP_REQUIRED]["Acc@1"]["lower_95"] is not None
            and float(boot_bc[SUBGROUP_REQUIRED]["Acc@1"]["lower_95"]) > 0.0
        ),
        "memory_not_required_B_minus_A_Acc_not_below_minus_0_01": (
            delta_ba[SUBGROUP_NOT_REQUIRED]["Acc@1"] is not None
            and float(delta_ba[SUBGROUP_NOT_REQUIRED]["Acc@1"]) >= -0.01
        ),
        "all_B_minus_A_Acc_nonnegative": (
            delta_ba[SUBGROUP_ALL]["Acc@1"] is not None
            and float(delta_ba[SUBGROUP_ALL]["Acc@1"]) >= 0.0
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    support_failed = not (
        checks["memory_required_decisions_at_least_50"]
        and checks["memory_required_raw_scenes_at_least_10"]
    )
    specificity_failed = any(
        name.startswith("memory_required_B_minus_C") for name in failures
    )
    other_scientific_failure = any(
        not name.startswith("memory_required_B_minus_C")
        and name not in {
            "memory_required_decisions_at_least_50",
            "memory_required_raw_scenes_at_least_10",
        }
        for name in failures
    )
    status = (
            "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_PASS"
            if not failures
            else "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL"
            if support_failed
            else "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_FAIL"
            if other_scientific_failure
            else "MF3ZU_RXR_EVIDENCE_SPECIFICITY_FAIL"
            if specificity_failed
            else "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_FAIL"
        )
    return {
        "status": status,
        "final_PASS_FAIL": (
            "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_PASS"
            if not failures
            else "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_FAIL"
        ),
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "rescue_authorized": False,
    }


__all__ = [
    "ARM_CURRENT",
    "ARM_MEMORY",
    "ARM_SHUFFLED",
    "ARMS",
    "SUBGROUP_ALL",
    "SUBGROUP_REQUIRED",
    "SUBGROUP_NOT_REQUIRED",
    "SUBGROUPS",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "EvidenceMemoryMetricError",
    "ranking_contributions",
    "summarize_ranking",
    "scene_cluster_bootstrap_paired_delta",
    "evaluate_three_arm_probe",
    "apply_fixed_rxr_gates",
]
