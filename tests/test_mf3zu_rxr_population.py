import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from revealnav_mf3.mf3zu_protocol import (
    EXPECTED_CANDIDATE_ELIGIBLE_ROWS,
    EXPECTED_POPULATION_ROWS,
    EXPECTED_POPULATION_EPISODES,
    EXPECTED_POPULATION_SCENES,
    EXPECTED_SOURCE_EPISODES,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_SCENES,
    FOLDS,
    FORBIDDEN_POPULATION_FIELDS,
    build_population_rows,
    scene_fold_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


class Mf3zuRxrPopulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.targets, cls.summary = build_population_rows()

    def test_full_exact_support_population_has_frozen_counts(self):
        self.assertEqual(self.summary["source_rows"], EXPECTED_SOURCE_ROWS)
        self.assertEqual(
            self.summary["candidate_eligible_rows"],
            EXPECTED_CANDIDATE_ELIGIBLE_ROWS,
        )
        self.assertEqual(self.summary["population_rows"], EXPECTED_POPULATION_ROWS)
        self.assertEqual(self.summary["exact_target_rows"], EXPECTED_POPULATION_ROWS)
        self.assertGreater(self.summary["feature_row_physical_step_mismatch_rows"], 0)
        self.assertEqual(self.summary["source_episodes"], EXPECTED_SOURCE_EPISODES)
        self.assertEqual(self.summary["source_raw_scenes"], EXPECTED_SOURCE_SCENES)
        self.assertEqual(self.summary["episodes"], EXPECTED_POPULATION_EPISODES)
        self.assertEqual(self.summary["raw_scenes"], EXPECTED_POPULATION_SCENES)
        self.assertTrue(self.summary["exact_target_accessed_for_support_eligibility"])
        self.assertFalse(self.summary["target_value_in_sanitized_population"])
        self.assertFalse(self.summary["baseline_score_or_correctness_accessed"])
        self.assertFalse(self.summary["outcome_or_utility_accessed"])
        self.assertEqual(len(self.rows), EXPECTED_POPULATION_ROWS)
        self.assertEqual(len(self.targets), EXPECTED_POPULATION_ROWS)
        self.assertTrue(all(row["candidate_count"] >= 2 for row in self.rows))
        self.assertEqual(
            [row["event_id"] for row in self.rows],
            [row["event_id"] for row in self.targets],
        )

    def test_population_contains_no_target_or_outcome_field(self):
        for row in self.rows:
            with self.subTest(event_id=row["event_id"]):
                self.assertTrue(FORBIDDEN_POPULATION_FIELDS.isdisjoint(row))
                self.assertEqual(row["dataset"], "RxR")
                self.assertEqual(
                    row["population_selection_rule"],
                    "candidate_mask_count>=2_and_exact_target_feature_slot_active",
                )
                self.assertEqual(row["memory_required_label"], "NOT_YET_MATERIALIZED")

    def test_exact_targets_are_physically_separate(self):
        self.assertTrue(all("target_index" in row for row in self.targets))
        self.assertTrue(all("target_feature_slot" in row for row in self.targets))
        self.assertTrue(
            all(
                row["coordinate_system"] == "MF3B_candidate_feature_slot"
                for row in self.targets
            )
        )
        self.assertTrue(
            all(row["baseline_score_or_correctness_used"] is False for row in self.targets)
        )

    def test_feature_rows_are_explicitly_mapped_to_physical_steps(self):
        self.assertTrue(
            all(row["feature_row_equals_physical_step_assumed"] is False for row in self.rows)
        )
        self.assertTrue(
            all(row["decision_step"] == row["physical_decision_step"] for row in self.rows)
        )
        self.assertTrue(
            all(
                row["candidate_coordinate_binding_status"]
                == "UNBOUND_UNTIL_REPLAY_EMBEDDING_AND_SCORE_BYTE_MATCH"
                for row in self.rows
            )
        )

    def test_raw_scene_never_crosses_fold(self):
        seen = {}
        for row in self.rows:
            seen.setdefault(row["scene_id"], set()).add(row["scene_fold"])
        self.assertEqual(len(seen), EXPECTED_SOURCE_SCENES)
        self.assertTrue(all(len(folds) == 1 for folds in seen.values()))
        self.assertTrue(all(next(iter(folds)) in range(FOLDS) for folds in seen.values()))
        self.assertLessEqual(
            max(self.summary["fold_scene_counts"].values())
            - min(self.summary["fold_scene_counts"].values()),
            1,
        )

    def test_scene_hash_round_robin_is_order_invariant(self):
        scenes = ["scene-c", "scene-a", "scene-b", "scene-e", "scene-d", "scene-f"]
        first = scene_fold_mapping(scenes)
        second = scene_fold_mapping(reversed(scenes))
        self.assertEqual(first, second)
        counts = [sum(value == fold for value in first.values()) for fold in range(FOLDS)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_target_identity_is_isolated_from_sanitized_population(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            run = source_root / "runs/ep_1"
            run.mkdir(parents=True)
            feature = run / "online_feature.npz"
            mask = np.asarray(
                [[True, True, False], [True, False, False], [True, True, True]],
                dtype=bool,
            )

            def write_feature(target):
                np.savez(
                    feature,
                    candidate_mask=mask,
                    target_index=np.asarray(target, dtype=np.int64),
                    target_in_set=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
                )

            shadow = [
                {
                    "step": 5,
                    "current_local_action_ids": ["g0", "g1"],
                    "current_local_action_indices": [2, 3],
                    "native_action_id": "g0",
                    "teacher_action_id_label_only": "g1",
                    "teacher_action_index_label_only": 3,
                    "teacher_used_as_online_input": False,
                    "public_unseen_authorized": False,
                    "previous_hash": "0" * 64,
                    "record_hash": "a" * 64,
                },
                {
                    "step": 9,
                    "current_local_action_ids": ["g2", "g3", "g4"],
                    "current_local_action_indices": [4, 5, 6],
                    "native_action_id": "g2",
                    "teacher_action_id_label_only": "g4",
                    "teacher_action_index_label_only": 6,
                    "teacher_used_as_online_input": False,
                    "public_unseen_authorized": False,
                    "previous_hash": "a" * 64,
                    "record_hash": "b" * 64,
                },
            ]
            (run / "uad_shadow.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in shadow), encoding="utf-8"
            )
            (run / "base_trace.jsonl").write_text(
                json.dumps({"i": 5, "act": 4})
                + "\n"
                + json.dumps({"i": 9, "act": 4})
                + "\n",
                encoding="utf-8",
            )
            write_feature([0, -1, 2])
            manifest = {
                "public_unseen_authorized": False,
                "records": [
                    {
                        "scene_id": "scene-1",
                        "episode_id": "1",
                        "path": "runs/ep_1/online_feature.npz",
                        "sha256": hashlib.sha256(feature.read_bytes()).hexdigest(),
                    }
                ],
            }
            manifest_path = source_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            first, first_targets, first_summary = build_population_rows(
                manifest_path,
                source_root=source_root,
                enforce_frozen_counts=False,
                verify_feature_hashes=False,
            )
            write_feature([1, -1, 0])
            second, second_targets, second_summary = build_population_rows(
                manifest_path,
                source_root=source_root,
                enforce_frozen_counts=False,
                verify_feature_hashes=False,
            )
            self.assertEqual(first, second)
            self.assertEqual(first_summary, second_summary)
            self.assertNotEqual(first_targets, second_targets)
            self.assertEqual(len(first), 2)
            self.assertEqual(
                [(row["feature_row_index"], row["physical_decision_step"]) for row in first],
                [(0, 5), (2, 9)],
            )


if __name__ == "__main__":
    unittest.main()
