from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.dsr_selection import (
    _candidate_feasibility,
    domain_scene_weights,
    nested_distributional_fit,
    proposal_support_audit,
    stratified_equal_budget_baselines,
)


def synthetic_rows() -> tuple[list[dict], np.ndarray]:
    rows = []
    folds = []
    for scene_index in range(20):
        for dataset_index, dataset in enumerate(("RxR", "R2R")):
            rows.append({
                "dataset": dataset,
                "scene_id": f"scene-{scene_index:02d}",
                "episode_id": f"{scene_index}-{dataset_index}",
                "target": 0.20 + 0.01 * (scene_index % 3),
                "decision": {
                    "step": dataset_index,
                    "native_margin": 0.01 * scene_index,
                    "policy_risk_adjusted_score": 2.0 - 0.01 * scene_index,
                },
            })
            folds.append(scene_index % 5)
    return rows, np.asarray(folds, dtype=np.int64)


class DSRSelectionTest(unittest.TestCase):
    def test_domain_scene_weights_balance_both_levels(self):
        scenes = np.asarray(["a", "a", "b", "c", "c", "c"])
        datasets = np.asarray(["RxR", "RxR", "RxR", "R2R", "R2R", "R2R"])
        weights = domain_scene_weights(scenes, datasets)
        self.assertAlmostEqual(float(weights.sum()), 6.0)
        self.assertAlmostEqual(float(weights[datasets == "RxR"].sum()), 3.0)
        self.assertAlmostEqual(float(weights[datasets == "R2R"].sum()), 3.0)
        self.assertAlmostEqual(float(weights[scenes == "a"].sum()), 1.5)
        self.assertAlmostEqual(float(weights[scenes == "b"].sum()), 1.5)

    def test_support_audit_is_diagnostic_and_scene_aware(self):
        rows, folds = synthetic_rows()
        audit = proposal_support_audit(rows, folds)
        self.assertEqual(audit["status"], "PROPOSAL_SUPPORT_AUDIT_PASS")
        self.assertFalse(audit["selection_use"])
        self.assertFalse(audit["architecture_use"])
        self.assertFalse(audit["threshold_use"])
        self.assertEqual(audit["domains"]["RxR"]["scenes"], 20)

        bad = [dict(row) for row in rows]
        for row in bad:
            row["target"] = -0.2
        failed = proposal_support_audit(bad, folds)
        self.assertEqual(failed["status"], "PROPOSAL_SUPPORT_AUDIT_FAIL")
        self.assertTrue(any(
            "oracle_10_and_20_percent_nonpositive" in reason
            for reason in failed["failure_reasons"]
        ))

    @staticmethod
    def _fake_fit(*args, seeds, **kwargs):
        return [object() for _ in seeds], [f"init-{seed}" for seed in seeds], [0.1] * len(seeds)

    @staticmethod
    def _fake_predict(models, matrix):
        return (
            np.full(len(matrix), 0.10),
            np.full(len(matrix), 0.20),
            np.full(len(matrix), 0.30),
        )

    def test_nested_folds_are_whole_scene_and_rng_is_common(self):
        rows, outer_folds = synthetic_rows()
        matrix = np.zeros((len(rows), 28), dtype=np.float64)
        target = np.asarray([row["target"] for row in rows])
        scenes = np.asarray([row["scene_id"] for row in rows])
        datasets = np.asarray([row["dataset"] for row in rows])
        config = {
            "outer_folds": 5,
            "inner_folds": 4,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "seeds": [11, 12, 13],
            "learning_rate": 0.01,
            "training_steps": 1,
            "inner_fold_salt": "dsr-test",
        }
        with patch(
            "revealnav_mf3.dsr_selection._fit_ensemble",
            side_effect=self._fake_fit,
        ), patch(
            "revealnav_mf3.dsr_selection._predict_ensemble",
            side_effect=self._fake_predict,
        ):
            result = nested_distributional_fit(
                matrix, target, scenes, datasets, outer_folds, config
            )
        self.assertEqual(result["status"], "NESTED_DSR_PASS")
        self.assertEqual(result["selected_weight_decay"], 0.0001)
        self.assertEqual(len(result["outer_folds"]), 5)
        for record in result["outer_folds"]:
            self.assertFalse(
                set(record["fit_scenes"]) & set(record["evaluation_scenes"])
            )
            self.assertTrue(record["common_random_numbers_verified"])
            hashes = [
                trial["inner_cv"][0]["initialization_hashes"]
                for trial in record["trials"]
            ]
            self.assertTrue(all(value == hashes[0] for value in hashes[1:]))
        # The two benchmark rows from every raw scene remain in one fold.
        self.assertTrue(all(
            len(set(outer_folds[scenes == scene])) == 1
            for scene in set(scenes)
        ))

    def test_outer_target_does_not_change_that_folds_inner_selection(self):
        rows, outer_folds = synthetic_rows()
        matrix = np.zeros((len(rows), 28), dtype=np.float64)
        target = np.asarray([row["target"] for row in rows])
        scenes = np.asarray([row["scene_id"] for row in rows])
        datasets = np.asarray([row["dataset"] for row in rows])
        config = {
            "outer_folds": 5, "inner_folds": 4,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "seeds": [11, 12, 13], "learning_rate": 0.01,
            "training_steps": 1, "inner_fold_salt": "dsr-target-isolation",
        }
        changed = target.copy()
        changed[outer_folds == 0] = 0.9
        outputs = []
        with patch(
            "revealnav_mf3.dsr_selection._fit_ensemble",
            side_effect=self._fake_fit,
        ), patch(
            "revealnav_mf3.dsr_selection._predict_ensemble",
            side_effect=self._fake_predict,
        ):
            for values in (target, changed):
                outputs.append(nested_distributional_fit(
                    matrix, values, scenes, datasets, outer_folds, config
                ))
        self.assertEqual(
            outputs[0]["outer_folds"][0]["selected_weight_decay"],
            outputs[1]["outer_folds"][0]["selected_weight_decay"],
        )
        self.assertEqual(
            outputs[0]["outer_folds"][0]["trials"],
            outputs[1]["outer_folds"][0]["trials"],
        )

    def test_fold_domain_matched_baselines_preserve_each_budget(self):
        rows, folds = synthetic_rows()
        target = np.asarray([row["target"] for row in rows])
        gate = np.zeros(len(rows), dtype=bool)
        for fold in range(5):
            gate[np.flatnonzero((folds == fold) & np.asarray([
                row["dataset"] == "RxR" for row in rows
            ]))[: fold % 2 + 1]] = True
            gate[np.flatnonzero((folds == fold) & np.asarray([
                row["dataset"] == "R2R" for row in rows
            ]))[: 1]] = True
        result = stratified_equal_budget_baselines(rows, target, gate, folds)
        for mode, mask in result["internal_masks"]["fold_domain_matched"].items():
            self.assertEqual(int(mask.sum()), int(gate.sum()), mode)
            for stratum in result["fold_domain_matched"]["strata"]:
                selected = (
                    (folds == stratum["outer_fold"])
                    & np.asarray([
                        row["dataset"] == stratum["dataset"] for row in rows
                    ])
                    & mask
                )
                self.assertEqual(int(selected.sum()), stratum["budget"])

    def test_zero_coverage_fold_domain_is_a_failure(self):
        rows, folds = synthetic_rows()
        target = np.asarray([row["target"] for row in rows])
        scenes = np.asarray([row["scene_id"] for row in rows])
        datasets = np.asarray([row["dataset"] for row in rows])
        mask = np.ones(len(rows), dtype=bool)
        mask[(folds == 2) & (datasets == "RxR")] = False
        feasible, failures, _ = _candidate_feasibility(
            mask, target, scenes, datasets, folds
        )
        self.assertFalse(feasible)
        self.assertIn("fold_2:RxR:zero_intervention", failures)


if __name__ == "__main__":
    unittest.main()
