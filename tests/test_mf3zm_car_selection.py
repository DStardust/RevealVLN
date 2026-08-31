"""Nested split and criterion-alignment tests for MF3ZM-CAR v1."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.car import event_domain_weights
from revealnav_mf3.car_selection import nested_car_fit


def _synthetic_population():
    rows = []
    policy = []
    instruction = []
    history = []
    native = []
    runner = []
    engineered = []
    target = []
    scenes = []
    datasets = []
    episodes = []
    # Two domains occur in every scene, so every outer/inner stratum is
    # populated while the split remains a whole-scene split.
    for scene_index in range(10):
        scene = f"scene-{scene_index:02d}"
        for domain in ("RxR", "R2R"):
            value = 0.2 if (scene_index + (domain == "R2R")) % 3 else -0.2
            rows.append({
                "dataset": domain,
                "scene_id": scene,
                "episode_id": f"{domain}-episode-{scene_index}",
                "tier": "core",
                "target": value,
                "decision": {
                    "step": 1,
                    "native_margin": float(scene_index) / 10.0,
                    "policy_risk_adjusted_score": 0.6,
                },
            })
            target.append(value)
            scenes.append(scene)
            datasets.append(domain)
            episodes.append(f"{domain}-episode-{scene_index}")
            policy.append(np.linspace(0.1, 1.0, 10) + scene_index)
            base = np.ones(768, dtype=np.float64) * (1.0 + scene_index)
            instruction.append(base)
            history.append(base * 1.1)
            native.append(base)
            runner.append(base + 0.01)
            engineered.append(np.linspace(-1.0, 1.0, 28) + scene_index)
    scenes = np.asarray(scenes)
    # A deterministic assignment used by the real protocol would provide the
    # same fold to both domains because they share the scene string.
    from revealnav_mf3.nested_selection import deterministic_scene_folds
    _, mapping = deterministic_scene_folds(
        sorted(set(scenes)), 5, salt="car-test-folds"
    )
    folds = np.asarray([mapping[scene] for scene in scenes], dtype=np.int64)
    return (
        rows,
        {
            "policy": np.asarray(policy, dtype=np.float64),
            "instruction": np.asarray(instruction, dtype=np.float64),
            "history": np.asarray(history, dtype=np.float64),
            "native": np.asarray(native, dtype=np.float64),
            "runner": np.asarray(runner, dtype=np.float64),
        },
        np.asarray(target, dtype=np.float64),
        scenes,
        np.asarray(datasets),
        np.asarray(episodes),
        folds,
    )


class CARSelectionTest(unittest.TestCase):
    def test_nested_fit_keeps_shared_scene_together(self):
        rows, inputs, target, scenes, datasets, episodes, folds = (
            _synthetic_population()
        )
        config = {
            "outer_folds": 5,
            "inner_folds": 4,
            "weight_decay_grid": [0.0001, 0.001, 0.01],
            "seeds": [11, 12, 13],
            "learning_rate": 0.005,
            "dual_learning_rate": 0.05,
            "dual_cap": 100.0,
            "training_steps": 1,
            "inner_fold_salt": "car-test-inner",
            "use_cuda": False,
        }
        result = nested_car_fit(
            rows, inputs, target, scenes, datasets, episodes, folds, config,
            representation="semantic", risk_mode="hard", scene_constraint=True,
        )
        self.assertIn(result["status"], {"NESTED_CAR_PASS", "NESTED_CAR_FAIL"})
        self.assertTrue(result["outer_folds"])
        for record in result["outer_folds"]:
            self.assertEqual(record["scene_overlap"], [])
            self.assertFalse(
                set(record["fit_scenes"]) & set(record["evaluation_scenes"])
            )
            for inner in record["trials"][0]["inner_cv"]:
                self.assertEqual(inner["scene_overlap"], [])
                self.assertFalse(
                    set(inner["fit_scenes"]) & set(inner["evaluation_scenes"])
                )

    def test_domain_weights_used_by_car_are_not_scene_weights(self):
        values = np.asarray(["RxR"] * 3 + ["R2R"] * 2)
        weights = event_domain_weights(values)
        self.assertEqual(float(weights[0]), float(weights[1]))
        self.assertEqual(float(weights[3]), float(weights[4]))
        self.assertNotEqual(float(weights[0]), float(weights[3]))


if __name__ == "__main__":
    unittest.main()
