"""One-shot scene-OOF protocol tests for MF3ZN-TUAD v1."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.tuad_selection import (
    FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED,
    PolicyOutcomes,
    REQUIRED_POLICIES,
    assemble_development_policies,
    evaluate_tuad_development,
    matched_budget_baselines,
    validate_exact_fold_domain_budgets,
    validate_lattice_fold_integrity,
)


class TUADSelectionTest(unittest.TestCase):
    def population(self):
        scenes = np.asarray([f"scene-{index // 2}" for index in range(10)])
        datasets = np.asarray(["R2R", "RxR"] * 5)
        folds = np.asarray([index // 2 for index in range(10)])
        episodes = np.asarray([f"episode-{index}" for index in range(10)])
        lattices = np.asarray([f"lattice-{index}" for index in range(10)])
        return scenes, datasets, folds, episodes, lattices

    def outcomes(self, value: float, *, selected: bool = True):
        mask = np.full(10, selected, dtype=bool)
        utility = np.full(10, value if selected else 0.0)
        return PolicyOutcomes(utility, mask, np.zeros(10, dtype=bool))

    def test_family_tombstone_is_machine_readable(self):
        self.assertIs(FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED, True)

    def test_matched_baselines_preserve_every_fold_domain_budget(self):
        scenes, datasets, folds, _, _ = self.population()
        selected = np.asarray([True, False] * 5)
        masks = matched_budget_baselines(
            selected,
            np.linspace(0.0, 1.0, 10),
            np.linspace(1.0, 0.0, 10),
            folds,
            datasets,
            [f"event-{index}" for index in range(10)],
        )
        validate_exact_fold_domain_budgets(selected, masks, folds, datasets)
        for fold in range(5):
            for domain in ("R2R", "RxR"):
                stratum = (folds == fold) & (datasets == domain)
                for mask in masks.values():
                    self.assertEqual(int(mask[stratum].sum()), int(selected[stratum].sum()))

    def test_scene_episode_and_lattice_arms_cannot_cross_folds(self):
        scenes, _, folds, episodes, lattices = self.population()
        validate_lattice_fold_integrity(scenes, episodes, lattices, folds)
        split_scenes = scenes.copy()
        split_scenes[-1] = split_scenes[0]
        with self.assertRaises(ValueError):
            validate_lattice_fold_integrity(split_scenes, episodes, lattices, folds)

    def test_complete_positive_fixture_passes_without_model_selection(self):
        scenes, datasets, folds, _, _ = self.population()
        policies = {
            "TUAD-full": self.outcomes(0.50),
            "current-only": self.outcomes(0.20),
            "temporal-no-UAD-supervision": self.outcomes(0.30),
            "oracle-UAD": self.outcomes(0.60),
            "runner-only-support": self.outcomes(0.40),
            "frozen-native": self.outcomes(0.0, selected=False),
            "matched-high-proposal-score": self.outcomes(0.10),
            "matched-low-native-margin": self.outcomes(0.05),
            "matched-random": self.outcomes(0.02),
        }
        self.assertEqual(set(policies), set(REQUIRED_POLICIES))
        result = evaluate_tuad_development(
            policies, scenes, datasets, folds,
            bootstrap_replicates=100,
        )
        self.assertEqual(result["status"], "TUAD_DEVELOPMENT_PASS")
        self.assertFalse(result["selection_performed"])
        self.assertFalse(result["public_authorization"])

    def test_negative_fold_fails_closed(self):
        scenes, datasets, folds, _, _ = self.population()
        full_utility = np.full(10, 0.5)
        full_utility[0] = -1.0
        policies = {
            "TUAD-full": PolicyOutcomes(
                full_utility, np.ones(10, dtype=bool), np.zeros(10, dtype=bool)
            ),
            "current-only": self.outcomes(0.0, selected=False),
            "temporal-no-UAD-supervision": self.outcomes(0.0, selected=False),
            "oracle-UAD": self.outcomes(0.6),
            "runner-only-support": self.outcomes(0.1),
            "frozen-native": self.outcomes(0.0, selected=False),
            "matched-high-proposal-score": self.outcomes(0.05),
            "matched-low-native-margin": self.outcomes(0.04),
            "matched-random": self.outcomes(0.01),
        }
        result = evaluate_tuad_development(
            policies, scenes, datasets, folds, bootstrap_replicates=100
        )
        self.assertEqual(result["status"], "TUAD_DEVELOPMENT_FAIL")
        self.assertIn("R2R:fold_0:utility_negative", result["failures"])

    def test_missing_control_is_rejected_not_silently_dropped(self):
        scenes, datasets, folds, _, _ = self.population()
        policies = {name: self.outcomes(0.1) for name in REQUIRED_POLICIES}
        del policies["current-only"]
        with self.assertRaises(ValueError):
            evaluate_tuad_development(policies, scenes, datasets, folds)

    def test_native_inclusive_choices_materialize_all_fixed_controls(self):
        _, datasets, folds, _, _ = self.population()
        action_mask = np.ones((10, 3), dtype=bool)
        is_native = np.tile(np.asarray([True, False, False]), (10, 1))
        utility = np.tile(np.asarray([0.0, 0.2, -0.1]), (10, 1))
        catastrophic = utility < -0.05
        chosen = {
            "TUAD-full": np.ones(10, dtype=np.int64),
            "current-only": np.zeros(10, dtype=np.int64),
            "temporal-no-UAD-supervision": np.zeros(10, dtype=np.int64),
            "oracle-UAD": np.ones(10, dtype=np.int64),
            "runner-only-support": np.ones(10, dtype=np.int64),
        }
        policies = assemble_development_policies(
            chosen,
            utility,
            catastrophic,
            action_mask,
            is_native,
            np.linspace(0.0, 1.0, 10),
            np.linspace(1.0, 0.0, 10),
            folds,
            datasets,
            [f"event-{index}" for index in range(10)],
        )
        self.assertEqual(set(policies), set(REQUIRED_POLICIES))
        self.assertTrue(np.allclose(policies["TUAD-full"].utility, 0.2))
        self.assertFalse(policies["frozen-native"].selected.any())
        for baseline in (
            "matched-high-proposal-score",
            "matched-low-native-margin",
            "matched-random",
        ):
            self.assertTrue(policies[baseline].selected.all())


if __name__ == "__main__":
    unittest.main()
