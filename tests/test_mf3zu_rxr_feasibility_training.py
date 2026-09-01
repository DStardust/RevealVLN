from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_mf3zu_rxr_feasibility_for_test",
    ROOT / "scripts/train_mf3zu_rxr_feasibility.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import MF3ZU feasibility trainer")
TRAIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)


class MF3ZURxRFeasibilityTrainingTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(11)
        scenes = np.asarray([f"scene-{index // 2}" for index in range(20)])
        folds = np.asarray([(index // 2) % 5 for index in range(20)], dtype=np.int64)
        candidate = rng.normal(size=(20, 2, 768)).astype(np.float32)
        base = rng.normal(scale=0.1, size=(20, 2)).astype(np.float32)
        memory = rng.normal(size=(20, 2, 78)).astype(np.float32)
        target = np.asarray([index % 2 for index in range(20)], dtype=np.int64)
        # Give the tiny fixture a learnable candidate-specific binding signal.
        memory[np.arange(20), target, -1] = 2.0
        memory[np.arange(20), 1 - target, -1] = -2.0
        return TRAIN.ProbeArrays(
            event_id=np.asarray([f"event-{index}" for index in range(20)]),
            scene_id=scenes,
            episode_id=np.asarray([f"episode-{index}" for index in range(20)]),
            decision_step=np.arange(20, dtype=np.int64),
            scene_fold=folds,
            candidate_action_ids=tuple(("candidate-0", "candidate-1") for _ in range(20)),
            candidate_features=candidate,
            base_scores=base,
            candidate_mask=np.ones((20, 2), dtype=bool),
            memory_features=memory,
            memory_count=np.asarray([1 + index % 2 for index in range(20)], dtype=np.int64),
            memory_required=np.asarray([index % 2 == 0 for index in range(20)], dtype=bool),
            target_index=target,
            source_feature_path=tuple("fixture.npz" for _ in range(20)),
            source_feature_row=np.arange(20, dtype=np.int64),
            population_rows_before_target=20,
            evidence_diagnostics={},
        )

    def test_fixed_three_arm_training_produces_complete_oof(self):
        data = self.fixture()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            scores, diagnostic = TRAIN.fixed_scene_oof_train(
                data,
                device=torch.device("cpu"),
                epochs=1,
                batch_size=8,
                bootstrap_replicates=20,
            )
        finally:
            torch.set_num_threads(previous_threads)
        self.assertTrue(diagnostic["complete_five_fold_oof"])
        self.assertFalse(diagnostic["checkpoint_written"])
        self.assertFalse(diagnostic["full_navigation_run"])
        self.assertEqual(len(diagnostic["folds"]), 5)
        np.testing.assert_array_equal(scores[TRAIN.ARM_CURRENT], data.base_scores)
        for arm in TRAIN.ARMS:
            self.assertTrue(np.isfinite(scores[arm][data.candidate_mask]).all())
        for fold in diagnostic["folds"]:
            self.assertTrue(fold["B_C_common_initialization"])
            self.assertTrue(fold["B_C_common_batch_order"])
            self.assertTrue(fold["shuffled_memory"]["held_donors_train_only"])
            self.assertTrue(fold["normalization_fit_train_fold_only"])

    def test_held_mutation_cannot_change_fold_normalizers(self):
        data = self.fixture()
        train = np.flatnonzero(data.scene_fold != 0)
        first_candidate, first_memory = TRAIN._fit_fold_normalizers(data, train)
        candidate = data.candidate_features.copy()
        memory = data.memory_features.copy()
        candidate[data.scene_fold == 0] = 1.0e6
        memory[data.scene_fold == 0] = -1.0e6
        mutated = replace(data, candidate_features=candidate, memory_features=memory)
        second_candidate, second_memory = TRAIN._fit_fold_normalizers(mutated, train)
        np.testing.assert_array_equal(first_candidate.mean, second_candidate.mean)
        np.testing.assert_array_equal(first_candidate.scale, second_candidate.scale)
        np.testing.assert_array_equal(first_memory.mean, second_memory.mean)
        np.testing.assert_array_equal(first_memory.scale, second_memory.scale)

    def test_replay_binding_reorders_source_slots_before_target_access(self):
        population_row = {
            "event_id": "RxR:scene:episode:3",
            "scene_id": "scene",
            "episode_id": "episode",
            "decision_step": 3,
            "candidate_action_ids": ["a", "b"],
            "active_candidate_feature_slots": [2, 5],
        }
        candidate = np.zeros((1, 2, 768), dtype=np.float32)
        candidate[0, 0, 0] = 2.0
        candidate[0, 1, 0] = 5.0
        population = {
            "event_id": np.asarray([population_row["event_id"]]),
            "candidate_mask": np.ones((1, 2), dtype=bool),
            "candidate_features": candidate,
            "base_scores": np.asarray([[2.0, 5.0]], dtype=np.float32),
            "candidate_action_ids": (("a", "b"),),
        }
        zero = [0.0] * 78
        evidence_row = {
            "event_id": population_row["event_id"],
            "scene_id": "scene",
            "episode_id": "episode",
            "decision_step": 3,
            "memory_required": False,
            "active_instruction_atom_ids": [],
            "retrieved_records": [],
            "candidate_action_ids": ["b", "a"],
            "active_candidate_feature_slots": [2, 5],
            "candidate_id_to_feature_slot": {"a": 2, "b": 5},
            "candidate_memory_features_by_slot": [
                {"candidate_action_id": "b", "feature_slot": 5, "feature": zero},
                {"candidate_action_id": "a", "feature_slot": 2, "feature": zero},
            ],
            "exact_target_artifact_opened": False,
            "ranking_label_read": False,
            "task_metric_read": False,
            "public_split_access": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.jsonl"
            evidence_path.write_text(json.dumps(evidence_row) + "\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "revision": TRAIN.REVISION,
                "status": "MF3ZU_RXR_EVIDENCE_MEMORY_FROZEN",
                "candidate_target_accessed": False,
                "outcome_or_utility_accessed": False,
                "exact_target_artifact_opened": False,
                "evidence_memory": {
                    "bytes": evidence_path.stat().st_size,
                    "sha256": TRAIN.sha256_file(evidence_path),
                },
            }), encoding="utf-8")
            TRAIN._join_frozen_evidence(
                [population_row], population, evidence_path, manifest_path
            )
        self.assertEqual(population["candidate_action_ids"], (("b", "a"),))
        self.assertEqual(population["ordered_candidate_feature_slots"], ((5, 2),))
        self.assertEqual(float(population["candidate_features"][0, 0, 0]), 5.0)
        self.assertEqual(float(population["base_scores"][0, 0]), 5.0)

    def test_result_atomically_embeds_complete_oof(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            result_path = root / "result.json"
            oof_rows = [{"event_id": "event"}]
            finalized = TRAIN._commit_embedded_result(
                result_path,
                {"status": "fixture"},
                oof_rows,
            )
            self.assertTrue(result_path.is_file())
            self.assertFalse(result_path.with_name(result_path.name + ".part").exists())
            observed = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(observed, finalized)
            inventory = finalized["OOF_predictions"]
            self.assertEqual(inventory["storage"], "embedded_in_result")
            self.assertEqual(inventory["rows"], oof_rows)
            self.assertEqual(inventory["row_count"], 1)
            canonical = TRAIN._canonical_oof_jsonl(oof_rows)
            self.assertEqual(inventory["canonical_jsonl_bytes"], len(canonical))
            self.assertEqual(
                inventory["canonical_jsonl_sha256"],
                TRAIN.hashlib.sha256(canonical).hexdigest(),
            )

    def test_result_rename_failure_exposes_no_final_or_oof(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            result_path = root / "result.json"
            original_replace = TRAIN.os.replace

            def fail_result_promotion(source, target):
                if Path(target) == result_path:
                    raise OSError("injected result promotion failure")
                return original_replace(source, target)

            with mock.patch.object(TRAIN.os, "replace", side_effect=fail_result_promotion):
                with self.assertRaisesRegex(OSError, "injected"):
                    TRAIN._commit_embedded_result(
                        result_path,
                        {"status": "fixture"},
                        [{"event_id": "performance"}],
                    )
            self.assertFalse(result_path.exists())
            self.assertFalse(result_path.with_name(result_path.name + ".part").exists())


if __name__ == "__main__":
    unittest.main()
