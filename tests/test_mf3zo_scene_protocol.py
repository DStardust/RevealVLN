import unittest

import numpy as np

from revealnav_mf3.mf3zo_pilot import PilotCandidate, select_balanced_candidates
from revealnav_mf3.mf3zo_probes import assign_scene_folds, ridge_scene_oof


class MF3ZOSceneProtocolTest(unittest.TestCase):
    def test_shared_scene_never_crosses_dataset_fold(self):
        scenes = np.asarray(["shared", "a", "b", "c", "d", "shared", "e"])
        folds, mapping = assign_scene_folds(scenes)
        self.assertEqual(folds[0], folds[5])
        self.assertEqual(folds[0], mapping["shared"])

    def test_outcome_free_balanced_selection(self):
        candidates = []
        for domain in ("R2R", "RxR"):
            for scene in range(10):
                for event in range(10):
                    identity = f"{domain}-{scene}-{event}"
                    candidates.append(PilotCandidate(
                        event_id=identity, dataset=domain, scene_id=f"scene-{scene}",
                        episode_id=identity, decision_step=event, source="fixture",
                        feature_path=f"fixture/{identity}.npz",
                    ))
        first = select_balanced_candidates(candidates)
        second = select_balanced_candidates(list(reversed(candidates)))
        self.assertEqual([value.event_id for value in first], [value.event_id for value in second])
        self.assertEqual(sum(value.dataset == "R2R" for value in first), 75)
        self.assertEqual(sum(value.dataset == "RxR" for value in first), 75)

    def test_held_target_cannot_change_held_prediction(self):
        scenes = np.asarray([f"scene-{index // 2}" for index in range(20)])
        folds, _ = assign_scene_folds(scenes)
        matrix = np.arange(40, dtype=float).reshape(20, 2)
        target = np.linspace(-1.0, 1.0, 20)
        prediction = ridge_scene_oof(matrix, target, folds)
        changed = target.copy()
        changed[folds == 0] += 10000.0
        changed_prediction = ridge_scene_oof(matrix, changed, folds)
        np.testing.assert_allclose(prediction[folds == 0], changed_prediction[folds == 0])


if __name__ == "__main__":
    unittest.main()

