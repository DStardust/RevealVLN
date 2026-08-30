from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.rcsp_selection import (
    domain_scene_episode_weights,
    nested_rcsp_fit,
    rcsp_equal_budget_baselines,
)


def fixture():
    rows = []
    policy = []
    embeddings = {name: [] for name in ("instruction", "history", "native", "runner")}
    target = []
    scenes = []
    datasets = []
    episodes = []
    folds = []
    for scene_index in range(20):
        for domain_index, dataset in enumerate(("RxR", "R2R")):
            scene = f"scene-{scene_index:02d}"
            episode = f"{dataset}-{scene_index}"
            rows.append({
                "dataset": dataset,
                "scene_id": scene,
                "episode_id": episode,
                "decision": {
                    "step": 2,
                    "native_margin": scene_index / 100,
                    "policy_risk_adjusted_score": 2.0 - scene_index / 100,
                },
            })
            policy.append(np.zeros(10))
            instruction = np.ones(768)
            history = np.ones(768) * 2
            native = np.ones(768) * 0.2
            runner = native.copy()
            runner[scene_index % 768] += 1.0
            embeddings["instruction"].append(instruction)
            embeddings["history"].append(history)
            embeddings["native"].append(native)
            embeddings["runner"].append(runner)
            target.append(0.2 + 0.01 * domain_index)
            scenes.append(scene)
            datasets.append(dataset)
            episodes.append(episode)
            folds.append(scene_index % 5)
    inputs = {"policy": np.asarray(policy)}
    inputs.update({name: np.asarray(value) for name, value in embeddings.items()})
    return (
        rows, inputs, np.asarray(target), np.asarray(scenes),
        np.asarray(datasets), np.asarray(episodes), np.asarray(folds),
    )


class RCSPSelectionTest(unittest.TestCase):
    def test_four_level_weights(self):
        datasets = np.asarray(["RxR", "RxR", "RxR", "R2R", "R2R"])
        scenes = np.asarray(["a", "a", "b", "c", "c"])
        episodes = np.asarray(["e1", "e1", "e2", "e3", "e4"])
        weights = domain_scene_episode_weights(datasets, scenes, episodes)
        self.assertAlmostEqual(float(weights.sum()), 5.0)
        self.assertAlmostEqual(float(weights[datasets == "RxR"].sum()), 2.5)
        self.assertAlmostEqual(float(weights[scenes == "a"].sum()), 1.25)
        self.assertAlmostEqual(float(weights[episodes == "e1"].sum()), 1.25)
        self.assertAlmostEqual(float(weights[0]), float(weights[1]))

    @staticmethod
    def fake_fit(*args, seeds, **kwargs):
        return [object() for _ in seeds], [f"init-{seed}" for seed in seeds], []

    @staticmethod
    def fake_predict(models, inputs, representation="semantic"):
        return np.ones(len(inputs["policy"]), dtype=np.float64)

    def test_nested_split_and_common_initialization(self):
        rows, inputs, target, scenes, datasets, episodes, folds = fixture()
        config = {
            "outer_folds": 5,
            "inner_folds": 4,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "seeds": [11, 12, 13],
            "learning_rate": 0.005,
            "dual_learning_rate": 0.05,
            "training_steps": 1,
            "inner_fold_salt": "rcsp-test",
        }
        with patch(
            "revealnav_mf3.rcsp_selection.fit_primal_dual_policy",
            side_effect=self.fake_fit,
        ), patch(
            "revealnav_mf3.rcsp_selection._predict",
            side_effect=self.fake_predict,
        ):
            result = nested_rcsp_fit(
                rows, inputs, target, scenes, datasets, episodes, folds, config
            )
        # Equal-budget baselines may correctly dominate the constant mock, but
        # every fold must still be constructed without scene leakage.
        self.assertIn(result["status"], {"NESTED_RCSP_PASS", "NESTED_RCSP_FAIL"})
        for record in result["outer_folds"]:
            self.assertFalse(set(record["fit_scenes"]) & set(record["evaluation_scenes"]))
            self.assertTrue(record["common_random_numbers_verified"])
            hashes = [
                trial["inner_cv"][0]["initialization_hashes"]
                for trial in record["trials"]
            ]
            self.assertTrue(all(value == hashes[0] for value in hashes[1:]))
        self.assertTrue(all(
            len(set(folds[scenes == scene])) == 1 for scene in set(scenes)
        ))

    def test_outer_target_isolation(self):
        rows, inputs, target, scenes, datasets, episodes, folds = fixture()
        config = {
            "outer_folds": 5, "inner_folds": 4,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "seeds": [11, 12, 13], "learning_rate": 0.005,
            "dual_learning_rate": 0.05, "training_steps": 1,
            "inner_fold_salt": "rcsp-target-isolation",
        }
        changed = target.copy()
        changed[folds == 0] = 0.9
        outputs = []
        with patch(
            "revealnav_mf3.rcsp_selection.fit_primal_dual_policy",
            side_effect=self.fake_fit,
        ), patch(
            "revealnav_mf3.rcsp_selection._predict",
            side_effect=self.fake_predict,
        ):
            for values in (target, changed):
                outputs.append(nested_rcsp_fit(
                    rows, inputs, values, scenes, datasets, episodes, folds, config
                ))
        self.assertEqual(
            outputs[0]["outer_folds"][0]["trials"],
            outputs[1]["outer_folds"][0]["trials"],
        )

    def test_fold_domain_budget_is_exact(self):
        rows, _, target, _, datasets, _, folds = fixture()
        mask = np.zeros(len(rows), dtype=bool)
        for fold in range(5):
            for dataset in ("RxR", "R2R"):
                mask[np.flatnonzero((folds == fold) & (datasets == dataset))[0]] = True
        result = rcsp_equal_budget_baselines(rows, target, mask, folds)
        self.assertEqual(result["budget"], 10)
        for stratum in result["fold_domain_matched"]["strata"]:
            self.assertEqual(stratum["budget"], 1)


if __name__ == "__main__":
    unittest.main()
