#!/usr/bin/env python3
"""Independent nested-scene sensitivity audit for the MF3ZJ train gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2r6.protocol import scene_fold  # noqa: E402
from scripts.train_mf3zj_counterfactual_transfer_gate import (  # noqa: E402
    MIN_AUTHORIZED,
    SEED,
    bootstrap_fit,
    load,
    rule_evidence,
    select_rule,
)
from scripts.train_rxr_uad_action_aligned_gate_mf3ze import (  # noqa: E402
    atomic_json,
    ensemble_predict,
    sha256_file,
)


GATE = ROOT / (
    "artifacts/training/mf3zj_counterfactual_transfer_gate_v1/"
    "MF3ZJ_CROSSFIT_GATE.json"
)
OUT = ROOT / (
    "artifacts/analysis/mf3zj_nested_scene_audit_v1/"
    "MF3ZJ_NESTED_SCENE_AUDIT.json"
)


def main() -> int:
    if OUT.exists():
        raise RuntimeError("refusing to overwrite MF3ZJ nested audit")
    matrix, target, scenes, sources, _ = load()
    folds = np.asarray([scene_fold(scene) for scene in scenes])
    fallback = sources == "native_margin"
    authorized = np.zeros(len(target), dtype=bool)
    outer_evidence = []
    for outer in range(5):
        outer_fit = folds != outer
        inner_expected = np.zeros(len(target), dtype=np.float64)
        inner_harm = np.zeros(len(target), dtype=np.float64)
        for inner in range(5):
            if inner == outer:
                continue
            inner_eval = (folds == inner) & outer_fit
            models = bootstrap_fit(
                matrix[outer_fit & (folds != inner)],
                target[outer_fit & (folds != inner)],
                scenes[outer_fit & (folds != inner)],
                SEED + outer * 10000 + inner * 1000,
            )
            inner_expected[inner_eval], inner_harm[inner_eval] = ensemble_predict(
                models, matrix[inner_eval]
            )
        rule, _ = select_rule(
            inner_expected, inner_harm, target, scenes, fallback & outer_fit
        )
        outer_eval = fallback & (folds == outer)
        if rule is None:
            outer_evidence.append({
                "outer_fold": outer,
                "inner_rule_available": False,
                "evaluation": rule_evidence(authorized & outer_eval, target, scenes),
            })
            continue
        models = bootstrap_fit(
            matrix[outer_fit], target[outer_fit], scenes[outer_fit],
            SEED + outer * 10000 + 9000,
        )
        expected, harm = ensemble_predict(models, matrix[outer_eval])
        local = (
            (expected >= rule["return_threshold"])
            & (harm <= rule["harm_probability_threshold"])
        )
        authorized[np.flatnonzero(outer_eval)] = local
        outer_evidence.append({
            "outer_fold": outer,
            "inner_rule_available": True,
            "inner_selected_rule": rule,
            "evaluation": rule_evidence(
                authorized & outer_eval, target, scenes
            ),
        })
    evidence = rule_evidence(authorized & fallback, target, scenes)
    gates = {
        "authorized_at_least_twelve": evidence["authorized"] >= MIN_AUTHORIZED,
        "total_utility_positive": evidence["total_utility"] > 0.0,
        "catastrophic_zero": evidence["catastrophic"] == 0,
        "leave_one_selected_scene_positive": (
            evidence["minimum_leave_one_selected_scene_out_total"] > 0.0
        ),
        "every_outer_fold_had_an_inner_rule": all(
            row["inner_rule_available"] for row in outer_evidence
        ),
    }
    payload = {
        "schema_version": "revealnav-mf3zj-nested-scene-audit/1",
        "status": "NESTED_AUDIT_PASS" if all(gates.values()) else "NESTED_AUDIT_FAIL",
        "role": (
            "conservative sensitivity analysis; not used to select the sealed "
            "MF3ZJ model or rule"
        ),
        "aggregate_outer_evaluation": evidence,
        "outer_folds": outer_evidence,
        "gates": gates,
        "source_gate_sha256": sha256_file(GATE),
        "unseen_or_test_read": False,
    }
    atomic_json(OUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "aggregate_outer_evaluation": evidence,
        "gates": gates,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
