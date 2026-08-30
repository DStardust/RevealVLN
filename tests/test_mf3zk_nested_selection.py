from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.nested_selection import (
    NestedSelectionError,
    canonicalize_exact_counterfactual_rows,
    coverage_funnel,
    deterministic_scene_folds,
    nested_scene_fit,
    outcome_evidence,
)


class NestedSelectionTest(unittest.TestCase):
    def test_scene_assignment_is_deterministic_and_nonempty(self):
        scenes = ["s3", "s1", "s4", "s2", "s5", "s6"]
        first, mapping = deterministic_scene_folds(
            scenes, 3, salt="unit-test"
        )
        second, repeated = deterministic_scene_folds(
            list(reversed(scenes)), 3, salt="unit-test"
        )
        self.assertEqual(mapping, repeated)
        self.assertEqual(set(first), {0, 1, 2})
        self.assertEqual(set(second), {0, 1, 2})

    def test_nested_fit_uses_outer_predictions_only_for_reporting(self):
        scenes = np.asarray([f"scene-{i // 2}" for i in range(20)])
        datasets = np.asarray(["RxR" if i % 2 else "R2R" for i in range(20)])
        matrix = np.zeros((20, 3), dtype=np.float64)
        matrix[:, 0] = np.arange(20, dtype=np.float64) / 20.0
        matrix[:, 1] = 1.0
        target = np.full(20, 0.2, dtype=np.float64)
        outer = np.asarray([(i // 2) % 5 for i in range(20)], dtype=np.int64)
        result = nested_scene_fit(
            matrix,
            target,
            scenes,
            datasets,
            outer,
            outer_fold_count=5,
            inner_fold_count=4,
            l2_grid=(1.0,),
            seed=11,
            bootstraps=2,
            minimum_authorized=1,
            minimum_per_domain=1,
            inner_salt="unit-test-nested",
        )
        self.assertEqual(result["status"], "NESTED_SELECTION_PASS")
        self.assertEqual(
            result["final_rule"]["selection_source"],
            "inner_scene_oof_only",
        )
        self.assertEqual(len(result["outer_folds"]), 5)
        self.assertEqual(len(result["outer_oof"]["expected"]), 20)
        self.assertFalse(result["outer_oof"]["authorized_mask"].dtype == object)
        self.assertTrue(np.isfinite(
            result["outer_oof"]["row_return_threshold"]
        ).all())
        self.assertTrue(np.array_equal(
            result["outer_oof"]["authorized_mask"],
            result["outer_oof"]["return_safe_mask"]
            & result["outer_oof"]["harm_safe_mask"],
        ))

    def test_exact_source_overlap_is_counted_once(self):
        decision = {
            "step": 2,
            "policy_risk_adjusted_score": 2.5,
        }
        common = {
            "dataset": "RxR",
            "episode_id": "17",
            "scene_id": "scene-a",
            "decision": decision,
            "target": 0.2,
            "feature": {"sha256": "a" * 64},
        }
        rows = [
            {
                **common,
                "tier": "core",
                "source_manifest": "core.json",
                "source_row_index": 3,
            },
            {
                **common,
                "tier": "expansion",
                "source_manifest": "expansion.json",
                "source_row_index": 8,
            },
        ]
        hierarchy = {
            "expansion_score_threshold": 1.0,
            "core_score_threshold": 2.0,
            "score_upper_threshold": 3.0,
        }
        canonical, audit = canonicalize_exact_counterfactual_rows(
            rows, hierarchy
        )
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["tier"], "core")
        self.assertEqual(canonical[0]["source_tiers"], ["core", "expansion"])
        self.assertEqual(audit["duplicate_rows_collapsed"], 1)

        conflicting = [dict(row) for row in rows]
        conflicting[1] = {**conflicting[1], "target": -0.2}
        with self.assertRaises(NestedSelectionError):
            canonicalize_exact_counterfactual_rows(conflicting, hierarchy)

    def test_evidence_and_funnel_use_group_denominators(self):
        mask = np.asarray([True, False, True, False])
        evidence = outcome_evidence(
            mask,
            np.asarray([0.2, -0.2, 0.1, 0.0]),
            np.asarray([1, 1, 2, 2]),
        )
        self.assertEqual(evidence["authorized"], 2)
        self.assertEqual(evidence["eligible"], 4)
        rows = [
            {
                "dataset": "RxR",
                "tier": "core",
                "scene_id": "scene-rxr",
                "target": 0.2,
                "decision": {
                    "current_local_action_ids": ["a", "b"],
                    "policy_risk_adjusted_score": 2.0,
                    "native_margin": 0.1,
                },
            },
            {
                "dataset": "R2R",
                "tier": "expansion",
                "scene_id": "scene-r2r",
                "target": -0.2,
                "decision": {
                    "current_local_action_ids": ["a", "b"],
                    "policy_risk_adjusted_score": 1.8,
                    "native_margin": 0.1,
                },
            },
        ]
        funnel = coverage_funnel(
            rows,
            np.asarray([0.3, -0.1]),
            np.asarray([0.1, 0.7]),
            {"return_threshold": 0.0, "harm_probability_threshold": 0.5},
            {
                "expansion_score_threshold": 1.0,
                "core_score_threshold": 2.0,
                "score_upper_threshold": 3.0,
            },
        )
        self.assertEqual(
            funnel["by_domain_and_tier"]["RxR/core"]["eligible_decisions"]["eligible"],
            1,
        )
        self.assertEqual(
            funnel["by_domain_and_tier"]["R2R/expansion"]["eligible_decisions"]["eligible"],
            1,
        )
        self.assertEqual(
            funnel["stages"]["actually_changed"]["authorized"], 1
        )


if __name__ == "__main__":
    unittest.main()
