"""Small fixed-executor fixture for MF3ZN five-fold OOF training."""

from __future__ import annotations

import importlib.util
import contextlib
import io
from pathlib import Path
import unittest

import numpy as np
import torch

from revealnav_mf3.temporal_uad_schema import (
    CausalTemporalStep,
    TemporalSequence,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_mf3zn_tuad_for_test", ROOT / "scripts/train_mf3zn_tuad.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load MF3ZN training entrypoint")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class FixedTUADTrainingTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(9)
        rows, steps, feature_dim = 10, 3, 12
        actions, action_dim, action_feature_dim = 3, 5, 1
        event_id = np.asarray([f"event-{index}" for index in range(rows)])
        causal = {
            "event_id": event_id,
            "scene_id": np.asarray([f"scene-{index // 2}" for index in range(rows)]),
            "dataset": np.asarray(["R2R", "RxR"] * 5),
            "episode_id": np.asarray([f"episode-{index}" for index in range(rows)]),
            "lattice_id": np.asarray([f"lattice-{index}" for index in range(rows)]),
            "native_prefix_sha256": np.asarray([f"{index + 1:064x}" for index in range(rows)]),
            "sequence_features": rng.normal(
                size=(rows, steps, feature_dim)
            ).astype(np.float32),
            "sequence_mask": np.ones((rows, steps), dtype=bool),
        }
        factor = np.ones((rows, steps), dtype=bool)
        reveal_event = np.zeros((rows, steps), dtype=bool)
        reveal_event[:, -1] = True
        oracle = {
            "event_id": event_id,
            "delta_utility": np.zeros(rows, dtype=np.float64),
            "target_in_set": factor,
            "candidate_separated": factor,
            "evidence_closed": factor,
            "reveal_event": reveal_event,
            "expiry_event": np.zeros((rows, steps), dtype=bool),
            "factor_mask": np.ones((rows, steps), dtype=bool),
            "reveal_at_risk": np.ones((rows, steps), dtype=bool),
            "expiry_at_risk": np.ones((rows, steps), dtype=bool),
            "reveal_offset": np.linspace(-2.0, 2.0, rows).astype(np.float32),
            "expiry_offset": np.linspace(2.0, -2.0, rows).astype(np.float32),
        }
        native_embedding = rng.normal(size=(rows, action_dim)).astype(np.float32)
        action_embedding = rng.normal(
            size=(rows, actions, action_dim)
        ).astype(np.float32)
        action_embedding[:, 0] = native_embedding
        delta_utility = np.column_stack((
            np.zeros(rows), rng.uniform(-0.5, 0.5, size=(rows, 2))
        )).astype(np.float32)
        action = {
            "event_id": event_id,
            "lattice_id": causal["lattice_id"].copy(),
            "action_id": np.tile(
                np.asarray(["native", "alternative-1", "alternative-2"]),
                (rows, 1),
            ),
            "action_list_commitment_sha256": np.asarray("a" * 64),
            "native_embedding": native_embedding,
            "action_embedding": action_embedding,
            "action_features": rng.normal(
                size=(rows, actions, action_feature_dim)
            ).astype(np.float32),
            "action_mask": np.ones((rows, actions), dtype=bool),
            "is_native": np.tile(
                np.asarray([True, False, False]), (rows, 1)
            ),
            "delta_utility": delta_utility,
            "catastrophic": delta_utility <= TRAIN.CATASTROPHIC_THRESHOLD,
            "proposal_score": rng.normal(size=rows).astype(np.float32),
            "native_margin": np.abs(rng.normal(size=rows)).astype(np.float32),
        }
        return causal, oracle, action

    def test_all_fixed_controls_share_folds_seeds_and_complete_oof(self):
        causal, oracle, action = self.fixture()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            arrays, diagnostics = TRAIN.fixed_scene_oof_train(
                causal,
                oracle,
                action,
                device=torch.device("cpu"),
                stage_1_epochs=1,
                stage_2_epochs=1,
                bootstrap_replicates=100,
            )
        finally:
            torch.set_num_threads(previous_threads)
        self.assertTrue(diagnostics["complete_five_fold_oof"])
        self.assertFalse(diagnostics["model_selection_performed"])
        self.assertFalse(diagnostics["checkpoint_written"])
        self.assertFalse(diagnostics["public_authorization"])
        self.assertIn(
            diagnostics["development_result"]["status"],
            {"TUAD_DEVELOPMENT_PASS", "TUAD_DEVELOPMENT_FAIL"},
        )
        self.assertEqual(
            diagnostics["stop_b_applied"],
            diagnostics["development_result"]["status"]
            != "TUAD_DEVELOPMENT_PASS",
        )
        self.assertEqual(diagnostics["fixed_seeds"], [20260831, 20260832, 20260833])
        self.assertEqual(set(arrays["fold"].tolist()), set(range(5)))
        is_native = action["is_native"]
        for arm in (
            "TUAD_full", "current_only", "temporal_no_UAD_supervision",
            "oracle_UAD",
        ):
            q = arrays[f"q_{arm}"]
            self.assertTrue(np.isfinite(q).all())
            self.assertTrue(np.array_equal(q[is_native], np.zeros(len(q))))
            self.assertEqual(arrays[f"seed_q_{arm}"].shape[0], 3)
        runner = arrays["chosen_runner_only_support"]
        self.assertTrue(np.all(np.isin(runner, [0, 1])))
        self.assertTrue(np.array_equal(
            arrays["q_runner_only_support"][:, 2], np.zeros(len(runner))
        ))
        runner_diagnostics = [
            item for item in diagnostics["fits"]
            if item["arm"] == "runner-only-support"
        ]
        self.assertEqual(len(runner_diagnostics), 15)
        self.assertTrue(all(
            item["action_support"] == "native_plus_frozen_runner_only"
            for item in runner_diagnostics
        ))
        full_diagnostics = [
            item for item in diagnostics["fits"] if item["arm"] == "TUAD-full"
        ]
        self.assertTrue(all(
            item["temporal_encoder_utility_supervision"] is False
            for item in full_diagnostics
        ))

    def test_native_target_drift_fails_before_training(self):
        causal, oracle, action = self.fixture()
        action["delta_utility"][0, 0] = 0.1
        with self.assertRaisesRegex(Exception, "exact action-value"):
            TRAIN.fixed_scene_oof_train(
                causal, oracle, action, device=torch.device("cpu"),
                stage_1_epochs=1, stage_2_epochs=1, bootstrap_replicates=10,
            )

    def test_current_only_neutralizes_all_history_dynamics(self):
        causal, _, _ = self.fixture()
        features = torch.from_numpy(causal["sequence_features"])
        mask = torch.from_numpy(causal["sequence_mask"])
        expected, expected_mask = TRAIN._current_only_view(features, mask)
        mutated = features.clone()
        mutated[:, :-1] = 1_000.0
        mutated[:, -1, -7:] = -999.0
        observed, observed_mask = TRAIN._current_only_view(mutated, mask)
        self.assertTrue(torch.equal(expected, observed))
        self.assertTrue(torch.equal(expected_mask, observed_mask))

    def test_nonbinary_oracle_factor_is_rejected(self):
        causal, oracle, action = self.fixture()
        oracle["target_in_set"] = oracle["target_in_set"].astype(np.float32)
        with self.assertRaisesRegex(Exception, "oracle tensor"):
            TRAIN.fixed_scene_oof_train(
                causal, oracle, action, device=torch.device("cpu"),
                stage_1_epochs=1, stage_2_epochs=1, bootstrap_replicates=10,
            )

    def test_production_tensors_are_rebuilt_from_causal_and_exact_sources(self):
        step = CausalTemporalStep(
            step=2,
            native_action_id="native",
            candidate_action_ids=("native", "alt-1", "alt-2"),
            policy_features=np.asarray(
                [1.0, 0.5, 0.2, 0.1, 0.0], dtype=np.float64,
            ),
            instruction_embedding=np.asarray([0.1, 0.2], dtype=np.float64),
            checkpoint_embedding=np.asarray([0.3, 0.4], dtype=np.float64),
            action_embeddings=np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64,
            ),
        )
        record = TemporalSequence.create(
            dataset="RxR", scene_id="scene", episode_id="episode",
            decision_step=2, steps=(step,),
        )
        snapshot = {
            "dataset": "RxR", "scene_id": "scene", "episode_id": "episode",
            "decision_step": 2, "native_action_id": "native",
            "global_action_ids": ["native", "alt-1", "alt-2"],
            "executable_action_indices": [0, 1, 2],
            "policy_scores": [1.0, 0.5, 0.25],
            "native_prefix_sha256": "a" * 64,
        }
        plan = {"seal": {"events": [{
            "lattice_id": "lattice", "snapshot": snapshot,
            "alternative_action_ids": ["alt-1", "alt-2"],
        }]}}
        causal = TRAIN._causal_arrays_from_records((record,), plan)
        outcomes = []
        for action_id, arm_type, delta, catastrophic in (
            ("native", "native", 0.0, False),
            ("alt-1", "treatment", 0.2, False),
            ("alt-2", "treatment", -0.2, True),
        ):
            outcomes.append({
                "lattice_id": "lattice", "action_id": action_id,
                "arm_type": arm_type, "delta_utility": delta,
                "catastrophic": catastrophic,
            })
        action = TRAIN._action_arrays_from_sealed_sources(
            (record,), causal, plan, {
                "outcomes": outcomes,
                "action_list_commitment_sha256": "b" * 64,
            },
        )
        self.assertEqual(action["action_id"].tolist(), [["native", "alt-1", "alt-2"]])
        np.testing.assert_array_equal(action["delta_utility"], [[0.0, 0.2, -0.2]])
        np.testing.assert_array_equal(action["catastrophic"], [[False, False, True]])
        np.testing.assert_array_equal(action["action_features"][0, :, 0], [0.0, -0.5, -0.75])
        self.assertEqual(causal["native_prefix_sha256"].tolist(), ["a" * 64])

    def test_cli_exposes_no_caller_supplied_action_outcome_file(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                TRAIN.parse_args([
                    "--protocol", "p", "--identifiability", "i",
                    "--exact-lattice-audit", "e", "--actions", "forged.npz",
                    "--output-prefix", "out",
                ])


if __name__ == "__main__":
    unittest.main()
