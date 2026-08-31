"""Archive-strengthening regressions for the stopped MF3ZM-CAR family.

These tests do not reopen CAR or authorize another scientific run.  They make
the already reviewed execution equivalence and nested-selection invariants
explicit for artifact preservation.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.car import CAR_POLICY_FEATURE_NAMES
from revealnav_mf3.car_fast import fit_car_ensemble_fast
from revealnav_mf3.car_selection import (
    NestedSelectionError,
    fit_car_ensemble,
    nested_car_fit,
)
from revealnav_mf3.dsr_selection import stratified_equal_budget_baselines


def _nested_fixture():
    rows = []
    target = []
    scenes = []
    datasets = []
    episodes = []
    policy = []
    folds = []
    for scene_index in range(20):
        for domain in ("RxR", "R2R"):
            rows.append({
                "dataset": domain,
                "scene_id": f"scene-{scene_index:02d}",
                "episode_id": f"{domain}-{scene_index}",
                "target": 0.2,
                "decision": {
                    "step": 2,
                    "native_margin": scene_index / 100.0,
                    "policy_risk_adjusted_score": 2.0 - scene_index / 100.0,
                },
            })
            target.append(0.2)
            scenes.append(f"scene-{scene_index:02d}")
            datasets.append(domain)
            episodes.append(f"{domain}-{scene_index}")
            policy.append(np.linspace(-0.2, 0.2, len(CAR_POLICY_FEATURE_NAMES)))
            folds.append(scene_index % 5)
    return (
        rows,
        {"policy_only": np.asarray(policy)},
        np.asarray(target),
        np.asarray(scenes),
        np.asarray(datasets),
        np.asarray(episodes),
        np.asarray(folds),
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class CARArchivalRegressionTest(unittest.TestCase):
    @staticmethod
    def _fake_fit(*args, seeds, **kwargs):
        return [object() for _ in seeds], [f"init-{seed}" for seed in seeds], []

    @staticmethod
    def _fake_predict(models, inputs, representation):
        return np.ones(len(inputs["policy_only"]), dtype=np.float64)

    @staticmethod
    def _config(salt: str):
        return {
            "outer_folds": 5,
            "inner_folds": 4,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "seeds": [11, 12, 13],
            "learning_rate": 0.005,
            "dual_learning_rate": 0.05,
            "dual_cap": 100.0,
            "training_steps": 1,
            "inner_fold_salt": salt,
            "use_cuda": False,
        }

    def test_outer_target_mutation_does_not_change_that_folds_inner_trials(self):
        rows, inputs, target, scenes, datasets, episodes, folds = _nested_fixture()
        changed = target.copy()
        changed[folds == 0] = -0.9
        results = []
        with patch(
            "revealnav_mf3.car_selection.fit_car_ensemble",
            side_effect=self._fake_fit,
        ), patch(
            "revealnav_mf3.car_selection.predict_car_ensemble",
            side_effect=self._fake_predict,
        ):
            for values in (target, changed):
                results.append(nested_car_fit(
                    rows, inputs, values, scenes, datasets, episodes, folds,
                    self._config("car-outer-target-isolation"),
                    representation="policy_only",
                ))
        self.assertEqual(
            _plain(results[0]["outer_folds"][0]["trials"]),
            _plain(results[1]["outer_folds"][0]["trials"]),
        )

    def test_common_initialization_drift_fails_closed(self):
        rows, inputs, target, scenes, datasets, episodes, folds = _nested_fixture()

        def drifted_fit(*args, seeds, weight_decay, **kwargs):
            return (
                [object() for _ in seeds],
                [f"init-{seed}-wd-{weight_decay}" for seed in seeds],
                [],
            )

        with patch(
            "revealnav_mf3.car_selection.fit_car_ensemble",
            side_effect=drifted_fit,
        ), patch(
            "revealnav_mf3.car_selection.predict_car_ensemble",
            side_effect=self._fake_predict,
        ), self.assertRaisesRegex(NestedSelectionError, "share initialization"):
            nested_car_fit(
                rows, inputs, target, scenes, datasets, episodes, folds,
                self._config("car-common-initialization"),
                representation="policy_only",
            )

    def test_matched_baseline_budget_is_exact_in_every_fold_domain(self):
        rows, _, target, _, datasets, _, folds = _nested_fixture()
        gate = np.zeros(len(rows), dtype=bool)
        for fold in range(5):
            for domain in ("RxR", "R2R"):
                indices = np.flatnonzero((folds == fold) & (datasets == domain))
                gate[indices[: (fold % 3) + 1]] = True
        baselines = stratified_equal_budget_baselines(
            rows, target, gate, folds, seed=20260830
        )
        for name, mask in baselines["internal_masks"]["fold_domain_matched"].items():
            for fold in range(5):
                for domain in ("RxR", "R2R"):
                    stratum = (folds == fold) & (datasets == domain)
                    self.assertEqual(
                        int(np.asarray(mask)[stratum].sum()),
                        int(gate[stratum].sum()),
                        (name, fold, domain),
                    )

    def test_fast_equivalence_all_arms_small_fixture(self):
        rng = np.random.default_rng(7)
        rows = 24
        all_inputs = {
            "semantic": {
                "policy": rng.normal(size=(rows, len(CAR_POLICY_FEATURE_NAMES))),
                "instruction": rng.normal(size=(rows, 768)),
                "history": rng.normal(size=(rows, 768)),
                "native": rng.normal(size=(rows, 768)),
                "runner": rng.normal(size=(rows, 768)),
            },
            "engineered_28d": {"engineered": rng.normal(size=(rows, 28))},
            "policy_only": {
                "policy_only": rng.normal(size=(rows, len(CAR_POLICY_FEATURE_NAMES))
            )},
        }
        target = rng.normal(scale=0.2, size=rows)
        scenes = np.asarray([f"scene-{index % 6}" for index in range(rows)])
        datasets = np.asarray(["R2R" if index % 2 == 0 else "RxR" for index in range(rows)])
        episodes = np.asarray([f"episode-{index}" for index in range(rows)])
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for representation in ("semantic", "engineered_28d", "policy_only"):
                for risk_mode in ("hard", "soft", "none"):
                    for scene_constraint in (True, False):
                        with self.subTest(
                            representation=representation,
                            risk_mode=risk_mode,
                            scene_constraint=scene_constraint,
                        ):
                            kwargs = {
                                "weight_decay": 0.001,
                                "seeds": (11, 12, 13),
                                "learning_rate": 0.005,
                                "dual_learning_rate": 0.05,
                                "training_steps": 5,
                                "dual_cap": 100.0,
                                "representation": representation,
                                "risk_mode": risk_mode,
                                "scene_constraint": scene_constraint,
                                "use_cuda": False,
                            }
                            reference = fit_car_ensemble(
                                all_inputs[representation], target, scenes,
                                datasets, episodes, **kwargs,
                            )
                            fast = fit_car_ensemble_fast(
                                all_inputs[representation], target, scenes,
                                datasets, episodes, **kwargs,
                            )
                            self.assertEqual(reference[1], fast[1])
                            for left, right in zip(reference[2], fast[2], strict=True):
                                for field in (
                                    "hard_selected_by_domain", "zero_risk_steps",
                                    "zero_scene_steps", "ungated_event_catastrophic_rate",
                                ):
                                    self.assertEqual(left[field], right[field])
                                self.assertAlmostEqual(
                                    left["preference_loss"], right["preference_loss"], places=7
                                )
                                for family in ("risk", "utility"):
                                    for domain, value in left["dual_variables"][family].items():
                                        self.assertAlmostEqual(
                                            value, right["dual_variables"][family][domain], places=7
                                        )
                                self.assertAlmostEqual(
                                    left["dual_variables"]["scene_max"],
                                    right["dual_variables"]["scene_max"],
                                    places=7,
                                )
                            maximum = max(
                                float((value - fast[0][model_index].state_dict()[name]).abs().max())
                                for model_index, model in enumerate(reference[0])
                                for name, value in model.state_dict().items()
                            )
                            self.assertLessEqual(maximum, 1e-7)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
