#!/usr/bin/env python3
"""Fit the versioned MF3ZK nested, pooled return/harm gate.

This is a new development revision.  The original MF3ZK outputs are never
overwritten.  Core and expansion rows share one estimator; the frozen MF3ZG
proposal hierarchy remains an inference-time routing rule.  Every L2 and
operating-point choice is made inside an inner scene cross-fit, while the
outer scene predictions are reserved for an unbiased development report.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from collections import Counter
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These loaders are the read-only, already-audited MF3ZK source readers.  The
# new revision changes selection/model fitting, not the source manifest.
from train_mf3zk_joint_action_aligned_gate import (  # noqa: E402
    _load_r2r,
    _load_rxr,
    _vector,
    provenance_path,
    sha256_file,
)
from revealnav_mf2r6.protocol import scene_fold  # noqa: E402
from revealnav_mf3.nested_selection import (  # noqa: E402
    CATASTROPHIC_THRESHOLD,
    HARM_LABEL_THRESHOLD,
    NestedSelectionError,
    canonicalize_exact_counterfactual_rows,
    coverage_funnel,
    domain_evidence,
    equal_budget_baselines,
    nested_scene_fit,
    outcome_evidence,
    risk_coverage_curve,
)
from revealnav_mf3.action_aligned import FEATURE_NAMES  # noqa: E402


REVISION = "mf3zk_nested_pooled_v9"
PROTOCOL = ROOT / "artifacts/training/mf3zk_joint_v1/MF3ZK_JOINT_PROTOCOL.json"
R2R_MANIFEST = ROOT / (
    "artifacts/training/mf3zk_joint_v1/r2r_collection/"
    "MF3ZK_R2R_DIRECT_SWITCH_MANIFEST.json"
)
RXR_CORE = ROOT / (
    "artifacts/phase1/mf3zd_direct_switch_returns_v1/"
    "MF3ZD_DIRECT_SWITCH_MANIFEST.json"
)
RXR_EXPANSION = ROOT / (
    "artifacts/phase1/mf3zf_expanded_direct_switch_returns_v1/"
    "MF3ZF_DIRECT_SWITCH_MANIFEST.json"
)
HIERARCHY_GATE = ROOT / (
    "artifacts/training/mf3zg_hierarchical_core_preserving_gate_v1/"
    "MF3ZG_SHADOW_GATE.json"
)
OUT = ROOT / "artifacts/training/mf3zk_nested_pooled_v9"
GATES = OUT / "gates"
RESULT = OUT / "MF3ZK_NESTED_POOLED_TRAINING_RESULT.json"
TRAINING_PROTOCOL = OUT / "MF3ZK_NESTED_POOLED_PROTOCOL.json"
SEED = 20260830
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAPS = 24
L2_GRID = (0.1, 1.0, 10.0, 100.0)
INNER_FOLD_SALT = "mf3zk-nested-pooled-inner-scenes/1"
MIN_AUTHORIZED_PER_DOMAIN = 8
MIN_AUTHORIZED_SINGLE_DOMAIN = 12


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def _load_hierarchy() -> dict:
    if HIERARCHY_GATE.is_symlink() or not HIERARCHY_GATE.is_file():
        raise RuntimeError("frozen MF3ZG hierarchy gate is unavailable")
    value = json.loads(HIERARCHY_GATE.read_text())
    if value.get("status") != "SHADOW_GATE_PASS":
        raise RuntimeError("frozen MF3ZG hierarchy is not accepted")
    rule = value.get("selected_rule", {})
    hierarchy = value.get("hierarchy", {})
    required = (
        "expansion_score_threshold",
        "core_score_threshold",
        "score_upper_threshold",
    )
    if any(key not in hierarchy for key in required):
        raise RuntimeError("MF3ZG hierarchy threshold schema drift")
    if not all(np.isfinite(float(hierarchy[key])) for key in required):
        raise RuntimeError("MF3ZG hierarchy threshold is non-finite")
    return {
        "expansion_score_threshold": float(hierarchy["expansion_score_threshold"]),
        "core_score_threshold": float(hierarchy["core_score_threshold"]),
        "score_upper_threshold": float(hierarchy["score_upper_threshold"]),
        "source": str(HIERARCHY_GATE.relative_to(ROOT)),
        "source_sha256": sha256_file(HIERARCHY_GATE),
        "source_rule": {
            key: rule[key] for key in (
                "final_training_threshold",
                "score_upper_threshold",
                "policy_risk_beta",
                "mad_weight",
            ) if key in rule
        },
    }


def _validate_manifest_rows(path: Path, allowed_indices: set[int] | None = None) -> None:
    """Check causal/split fields before handing rows to the audited loaders."""

    value = json.loads(path.read_text())
    if value.get("unseen_or_test_read") is not False:
        raise RuntimeError(f"source manifest crossed a public split: {path}")
    for raw in value.get("records", []):
        if allowed_indices is not None and int(raw["row_index"]) not in allowed_indices:
            continue
        if raw.get("split", "train") != "train":
            raise RuntimeError(f"non-train row entered pooled fit: {path}")
        future = raw.get("future_frames_used", 0)
        if future not in (None, False, 0, "0"):
            raise RuntimeError(f"future-frame field entered pooled fit: {path}")


def load_rows(protocol: dict, hierarchy: dict) -> tuple[list[dict], dict]:
    holdout = set(protocol["strict_scene_holdout"]["confirmation_scenes"])
    rxr_allowed = {
        (str(row["tier"]), int(row["source_row_index"]))
        for row in protocol["rxr_sources"]["fit_rows"]
    }
    _validate_manifest_rows(
        RXR_CORE, {index for tier, index in rxr_allowed if tier == "core"}
    )
    _validate_manifest_rows(
        RXR_EXPANSION,
        {index for tier, index in rxr_allowed if tier == "expansion"},
    )
    rows = []
    rows.extend(_load_rxr(
        RXR_CORE, "core", {index for tier, index in rxr_allowed if tier == "core"}
    ))
    rows.extend(_load_rxr(
        RXR_EXPANSION,
        "expansion",
        {index for tier, index in rxr_allowed if tier == "expansion"},
    ))
    rxr_before_holdout = len(rows)
    rows = [row for row in rows if row["scene_id"] not in holdout]
    rxr_excluded = rxr_before_holdout - len(rows)

    r2r_fit_scenes = set(protocol["r2r_train"]["fit_scenes"]) - holdout
    _validate_manifest_rows(R2R_MANIFEST)
    r2r_rows = _load_r2r(R2R_MANIFEST, r2r_fit_scenes, holdout)
    rows.extend(r2r_rows)
    if not rows:
        raise RuntimeError("pooled fit has no rows")
    if any(row["scene_id"] in holdout for row in rows):
        raise RuntimeError("confirmation scene entered pooled fit")
    rows, canonicalization = canonicalize_exact_counterfactual_rows(
        rows, hierarchy
    )
    keys = [(
        str(row["dataset"]),
        str(row["episode_id"]),
        int(row["decision"].get("step", row.get("decision_step", -1))),
    ) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        raise RuntimeError(f"duplicate exact episode pair: {duplicates[:5]}")
    meta = {
        "old_confirmation_scenes": sorted(holdout),
        "old_confirmation_reused": False,
        "rxr_fit_rows_before_holdout_exclusion": rxr_before_holdout,
        "rxr_rows_excluded_as_old_confirmation": rxr_excluded,
        "r2r_exact_pair_rows": len(r2r_rows),
        "canonicalization": canonicalization,
        "r2r_collection_route_population": int(
            protocol["r2r_train"]["collection_route_limit"]
        ),
        "source_population": {
            "RxR_fit_manifest_rows": len(protocol["rxr_sources"]["fit_rows"]),
            "R2R_collection_routes": int(
                protocol["r2r_train"]["collection_route_limit"]
            ),
        },
    }
    return rows, meta


def _development_gate_pass(
    arm: str,
    evidence: dict,
    domains: dict,
    ungated: dict,
    ungated_domains: dict,
) -> bool:
    minimum = (
        MIN_AUTHORIZED_PER_DOMAIN
        if arm == "joint" else MIN_AUTHORIZED_SINGLE_DOMAIN
    )
    if not (
        evidence["authorized"] >= minimum
        and evidence["total_utility"] > 0.0
        and evidence["catastrophic_rate"] <= ungated["catastrophic_rate"]
        and evidence["minimum_leave_one_selected_scene_out_total"] > 0.0
    ):
        return False
    if arm == "joint":
        return all(
            value["authorized"] >= MIN_AUTHORIZED_PER_DOMAIN
            and value["total_utility"] > 0.0
            and value["catastrophic_rate"]
            <= ungated_domains[domain]["catastrophic_rate"]
            and value["minimum_leave_one_selected_scene_out_total"] > 0.0
            for domain, value in domains.items()
        )
    return True


def _save_models(path: Path, models: list[tuple[np.ndarray, ...]], feature_names: Sequence[str]) -> dict:
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        np.savez(
            stream,
            means=np.stack([model[0] for model in models]),
            scales=np.stack([model[1] for model in models]),
            return_coefficients=np.stack([model[2] for model in models]),
            harm_coefficients=np.stack([model[3] for model in models]),
            feature_names=np.asarray(feature_names),
        )
    os.replace(part, path)
    return {
        "path": provenance_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "members": len(models),
    }


def _arm_rows(rows: list[dict], arm: str) -> list[dict]:
    if arm == "joint":
        selected = list(rows)
    else:
        selected = [row for row in rows if row["dataset"] == arm]
    if not selected:
        raise RuntimeError(f"no rows for pooled arm {arm}")
    if len({row["tier"] for row in selected}) < 2:
        raise RuntimeError(f"pooled arm {arm} lost a proposal tier")
    return selected


def fit_arm(
    rows: list[dict], arm: str, hierarchy: dict, rows_meta: dict,
) -> dict:
    selected = _arm_rows(rows, arm)
    matrix = np.stack([_vector(row) for row in selected])
    target = np.asarray([float(row["target"]) for row in selected], dtype=np.float64)
    scenes = np.asarray([str(row["scene_id"]) for row in selected])
    datasets = np.asarray([str(row["dataset"]) for row in selected])
    if len(set(scenes)) < OUTER_FOLDS:
        raise RuntimeError(f"insufficient scene diversity for pooled arm {arm}")
    if arm == "joint" and set(datasets) != {"RxR", "R2R"}:
        raise RuntimeError("joint pooled arm lost one benchmark")
    outer_folds = np.asarray([scene_fold(scene) for scene in scenes], dtype=np.int64)
    fit = nested_scene_fit(
        matrix,
        target,
        scenes,
        datasets,
        outer_folds,
        outer_fold_count=OUTER_FOLDS,
        inner_fold_count=INNER_FOLDS,
        l2_grid=L2_GRID,
        seed=SEED,
        bootstraps=BOOTSTRAPS,
        minimum_authorized=(
            MIN_AUTHORIZED_PER_DOMAIN
            if arm == "joint" else MIN_AUTHORIZED_SINGLE_DOMAIN
        ),
        minimum_per_domain=MIN_AUTHORIZED_PER_DOMAIN,
        inner_salt=f"{INNER_FOLD_SALT}:{arm}",
    )
    outer = fit["outer_oof"]
    evidence = outer["evidence"]
    domains = outer["domains"]
    ungated = outcome_evidence(np.ones(len(target), dtype=bool), target, scenes)
    ungated_domains = domain_evidence(
        np.ones(len(target), dtype=bool), target, scenes, datasets
    )
    development_pass = _development_gate_pass(
        arm, evidence, domains, ungated, ungated_domains
    )
    rule = fit["final_rule"]
    funnel = coverage_funnel(
        selected,
        outer["expected"],
        outer["upper_harm"],
        rule,
        hierarchy,
        return_safe_mask=outer["return_safe_mask"],
        harm_safe_mask=outer["harm_safe_mask"],
        authorized_mask=outer["authorized_mask"],
        source_population={
            "supervised_exact_one_switch_rows": len(selected),
            "collection_population": rows_meta.get("source_population", {}),
            "dataset_rows": {
                domain: int((datasets == domain).sum())
                for domain in sorted(set(datasets))
            },
        },
    )
    curve = risk_coverage_curve(
        outer["expected"], outer["upper_harm"], target
    )
    equal_budget = equal_budget_baselines(
        selected, target, outer["authorized_mask"], seed=SEED + 7000
    )
    oof_rows = []
    for index, row in enumerate(selected):
        oof_rows.append({
            "row_index": index,
            "dataset": row["dataset"],
            "tier": row["tier"],
            "episode_id": row["episode_id"],
            "scene_id": row["scene_id"],
            "target_utility": float(target[index]),
            "robust_expected_utility": float(outer["expected"][index]),
            "upper_harm_probability": float(outer["upper_harm"][index]),
            "outer_return_threshold": float(
                outer["row_return_threshold"][index]
            ),
            "outer_harm_probability_threshold": float(
                outer["row_harm_threshold"][index]
            ),
            "authorized": bool(outer["authorized_mask"][index]),
        })
    model_path = GATES / f"MF3ZK2_{arm.upper()}_POOLED_GATE_MODELS.npz"
    model = _save_models(model_path, fit["final_models"], FEATURE_NAMES)
    status = "NESTED_TRAIN_GATE_PASS" if development_pass else "NESTED_TRAIN_GATE_FAIL"
    return {
        "schema_version": f"revealnav-{REVISION}-{arm.lower()}-gate/3",
        "status": status,
        "model_fit_status": "PASS",
        "scientific_control_status": (
            "PASS" if development_pass else "FAIL"
        ),
        "confirmation_authorization_status": (
            "NOT_AUTHORIZED_OLD_CONFIRMATION_CONSUMED"
        ),
        "public_eval_authorization_status": "NOT_AUTHORIZED",
        "task_metric_run_authorized": False,
        "public_unseen_authorized": False,
        "fresh_confirmation_required": True,
        "arm": arm,
        "estimator": "one_pooled_return_harm_estimator_without_tier_feature",
        "proposal_hierarchy": "frozen_mf3zg_core_preserving",
        "rows": len(selected),
        "scenes": len(set(scenes)),
        "datasets": sorted(set(datasets)),
        "tier_counts": dict(sorted(Counter(row["tier"] for row in selected).items())),
        "feature_names": list(FEATURE_NAMES),
        "selected_rule": rule,
        "nested_oof_evidence": evidence,
        "nested_oof_domain_evidence": domains,
        "ungated_oof_evidence": ungated,
        "ungated_domain_evidence": ungated_domains,
        "coverage_funnel": funnel,
        "risk_coverage_curve": curve,
        "equal_budget_baselines": equal_budget,
        "oof_rows": oof_rows,
        "model": model,
        "controls": {
            "outer_scene_folds": OUTER_FOLDS,
            "inner_scene_folds": INNER_FOLDS,
            "scene_fold_function": "revealnav_mf2r6.protocol.scene_fold",
            "inner_scene_fold_salt": INNER_FOLD_SALT,
            "scene_overlap_all_outer_folds": 0,
            "bootstraps_per_fit": BOOTSTRAPS,
            "l2_grid": list(L2_GRID),
            "harm_label_threshold": HARM_LABEL_THRESHOLD,
            "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
            "threshold_selection": "inner_scene_oof_only",
            "final_rule_aggregation": "modal_l2_median_outer_inner_thresholds",
            "bootstrap_cluster_unit": "global_mp3d_scene_shared_across_benchmarks",
            "dataset_balanced_effective_weight": arm == "joint",
            "outer_authorization_for_development_report": (
                "fold_specific_inner_selected_rule"
            ),
            "aggregated_rule_evidence_is_diagnostic_only": True,
            "equal_budget_baselines_selection_used": False,
            "old_confirmation_reused": False,
            "unseen_or_test_read": False,
        },
    }


def failed_arm(
    rows: list[dict], arm: str, hierarchy: dict, rows_meta: dict,
    error: Exception,
) -> dict:
    """Keep proposal coverage visible when nested selection fails closed."""

    selected = _arm_rows(rows, arm)
    datasets = np.asarray([str(row["dataset"]) for row in selected])
    inactive = np.zeros(len(selected), dtype=bool)
    funnel = coverage_funnel(
        selected,
        np.zeros(len(selected), dtype=np.float64),
        np.ones(len(selected), dtype=np.float64),
        {"return_threshold": 0.0, "harm_probability_threshold": 0.0},
        hierarchy,
        return_safe_mask=inactive,
        harm_safe_mask=inactive,
        authorized_mask=inactive,
        source_population={
            "supervised_exact_one_switch_rows": len(selected),
            "collection_population": rows_meta.get("source_population", {}),
            "dataset_rows": {
                domain: int((datasets == domain).sum())
                for domain in sorted(set(datasets))
            },
        },
    )
    funnel["authorization_source"] = "not_evaluated_nested_selection_failed"
    funnel["post_proposal_stage_status"] = "NOT_EVALUATED"
    return {
        "schema_version": f"revealnav-{REVISION}-{arm.lower()}-gate/3",
        "status": "NESTED_TRAIN_GATE_FAIL",
        "model_fit_status": "FAIL",
        "scientific_control_status": "FAIL",
        "confirmation_authorization_status": (
            "NOT_AUTHORIZED_OLD_CONFIRMATION_CONSUMED"
        ),
        "public_eval_authorization_status": "NOT_AUTHORIZED",
        "task_metric_run_authorized": False,
        "public_unseen_authorized": False,
        "fresh_confirmation_required": True,
        "arm": arm,
        "estimator": "one_pooled_return_harm_estimator_without_tier_feature",
        "proposal_hierarchy": "frozen_mf3zg_core_preserving",
        "rows": len(selected),
        "scenes": len({str(row["scene_id"]) for row in selected}),
        "datasets": sorted(set(datasets)),
        "tier_counts": dict(sorted(Counter(
            row["tier"] for row in selected
        ).items())),
        "coverage_funnel": funnel,
        "risk_coverage_status": "NOT_EVALUATED",
        "equal_budget_baselines_status": "NOT_EVALUATED",
        "error": f"{type(error).__name__}: {error}",
    }


def write_protocol(protocol: dict, hierarchy: dict, rows_meta: dict) -> None:
    value = {
        "schema_version": "revealnav-mf3zk-nested-pooled-protocol/3",
        "status": "SEALED_BEFORE_MF3ZK_NESTED_POOLED_TRAINING",
        "revision": REVISION,
        "changes_from_mf3zk": [
            "nested outer/inner whole-scene selection",
            "one pooled core+expansion return/harm estimator",
            "byte-identical cross-source counterfactual rows collapsed once",
            "proposal tiers assigned by the frozen hierarchy",
            "coverage funnel and fixed risk-coverage diagnostics",
            "outer-fold authorization masks kept separate from final-rule diagnostics",
            "equal-budget proposal baselines",
            "separate model/control/confirmation/public-evaluation status fields",
        ],
        "supersedes": {
            "path": "artifacts/training/mf3zk_nested_pooled_v8",
            "reason": "v9 constrains catastrophic rate rather than coverage-confounded count",
        },
        "frozen_components": [
            "ETP-R1 frontend",
            "MF3V proposal ranker",
            "MF3ZG core-preserving hierarchy",
            "exact one-switch action-aligned labels",
        ],
        "selection": {
            "outer_scene_folds": OUTER_FOLDS,
            "inner_scene_folds": INNER_FOLDS,
            "inner_scene_fold_salt": INNER_FOLD_SALT,
            "l2_grid": list(L2_GRID),
            "bootstrap_members": BOOTSTRAPS,
            "harm_label_threshold": HARM_LABEL_THRESHOLD,
            "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
            "final_l2_rule": "modal across outer folds, smallest on tie",
            "final_threshold_rule": "median across outer-fold inner selections",
            "outer_predictions_used_for_selection": False,
            "bootstrap_cluster_unit": "global_mp3d_scene_shared_across_benchmarks",
        },
        "confirmation": {
            "previous_confirmation": "consumed_retrospective_failure_only",
            "reuse_allowed": False,
            "fresh_confirmation_required": True,
            "public_eval_authorized": False,
        },
        "source_hashes": {
            "mf3zk_protocol": sha256_file(PROTOCOL),
            "r2r_manifest": sha256_file(R2R_MANIFEST),
            "rxr_core": sha256_file(RXR_CORE),
            "rxr_expansion": sha256_file(RXR_EXPANSION),
            "mf3zg_hierarchy": hierarchy["source_sha256"],
        },
        "rows": rows_meta,
        "public_split_access": {
            "r2r_val_seen": False,
            "r2r_val_unseen": False,
            "rxr_val_seen": False,
            "rxr_val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
    }
    atomic_json(TRAINING_PROTOCOL, value)


def fit() -> int:
    if OUT.exists():
        raise RuntimeError(
            f"refusing to overwrite versioned output directory: {OUT}"
        )
    source_protocol = json.loads(PROTOCOL.read_text())
    if source_protocol.get("status") != "SEALED_BEFORE_MF3ZK_JOINT_TRAINING":
        raise RuntimeError("source MF3ZK protocol is not sealed")
    if source_protocol.get("public_split_access", {}).get("r2r_val_unseen") is not False:
        raise RuntimeError("source protocol public-split boundary drift")
    hierarchy = _load_hierarchy()
    rows, rows_meta = load_rows(source_protocol, hierarchy)
    OUT.mkdir(parents=True)
    GATES.mkdir(parents=True)
    write_protocol(source_protocol, hierarchy, rows_meta)
    arms = {}
    for arm in ("joint", "RxR", "R2R"):
        try:
            arms[arm] = fit_arm(rows, arm, hierarchy, rows_meta)
            atomic_json(GATES / f"MF3ZK2_{arm.upper()}_POOLED_GATE.json", arms[arm])
        except (RuntimeError, ValueError, NestedSelectionError) as error:
            arms[arm] = failed_arm(
                rows, arm, hierarchy, rows_meta, error
            )
            atomic_json(GATES / f"MF3ZK2_{arm.upper()}_POOLED_GATE.json", arms[arm])
    model_fit_status = (
        "PASS" if all(value.get("model_fit_status") == "PASS" for value in arms.values())
        else "FAIL"
    )
    control_status = (
        "PASS" if all(value.get("scientific_control_status") == "PASS" for value in arms.values())
        else "FAIL"
    )
    summary = {
        "schema_version": "revealnav-mf3zk-nested-pooled-training-result/3",
        "status": "TRAIN_DEVELOPMENT_PASS" if (
            model_fit_status == "PASS" and control_status == "PASS"
        ) else "TRAIN_DEVELOPMENT_FAIL",
        "model_fit_status": model_fit_status,
        "scientific_control_status": control_status,
        "confirmation_authorization_status": (
            "NOT_AUTHORIZED_OLD_CONFIRMATION_CONSUMED"
        ),
        "public_eval_authorization_status": "NOT_AUTHORIZED",
        "mainline": "joint",
        "arms": arms,
        "rows": len(rows),
        "scene_count": len({row["scene_id"] for row in rows}),
        "tier_counts": dict(sorted(Counter(row["tier"] for row in rows).items())),
        "old_confirmation_reused": False,
        "fresh_confirmation_required": True,
        "public_split_access": {
            "r2r_val_seen": False,
            "r2r_val_unseen": False,
            "rxr_val_seen": False,
            "rxr_val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
        "source_hashes": {
            "protocol": sha256_file(PROTOCOL),
            "r2r_manifest": sha256_file(R2R_MANIFEST),
            "rxr_core": sha256_file(RXR_CORE),
            "rxr_expansion": sha256_file(RXR_EXPANSION),
            "mf3zg_hierarchy": hierarchy["source_sha256"],
        },
    }
    atomic_json(RESULT, summary)
    print(json.dumps({
        "status": summary["status"],
        "model_fit_status": model_fit_status,
        "scientific_control_status": control_status,
        "arms": {
            arm: {
                "status": value.get("status"),
                "rows": value.get("rows"),
                "nested_oof_authorized": value.get("nested_oof_evidence", {}).get("authorized"),
            }
            for arm, value in arms.items()
        },
    }, indent=2, sort_keys=True))
    return 0 if summary["status"] == "TRAIN_DEVELOPMENT_PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit",))
    args = parser.parse_args()
    return fit()


if __name__ == "__main__":
    raise SystemExit(main())
