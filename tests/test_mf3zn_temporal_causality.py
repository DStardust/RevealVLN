from __future__ import annotations

import unittest

import numpy as np

from revealnav_mf3.temporal_uad_features import (
    POLICY_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    TEMPORAL_SUMMARY_NAMES,
    causal_current_only_features,
    causal_sequence_feature_width,
    causal_sequence_features,
    causal_snapshot_summary,
    causal_temporal_summary,
    causal_temporal_summary_from_mapping,
    temporal_summary_bytes,
)
from revealnav_mf3.temporal_uad_labels import TemporalOracleLabel
from revealnav_mf3.temporal_uad_schema import (
    CausalTemporalStep,
    TemporalSequence,
)


def step(
    index: int,
    native: str,
    candidates: tuple[str, ...],
    score: float,
    margin: float,
    alignment: float,
    checkpoint: tuple[float, float],
) -> CausalTemporalStep:
    return CausalTemporalStep(
        step=index,
        native_action_id=native,
        candidate_action_ids=candidates,
        policy_features=np.asarray(
            [score + 1.0, margin, score, 0.1, alignment],
            dtype=np.float32,
        ),
        instruction_embedding=np.asarray([0.5, -0.5], dtype=np.float32),
        checkpoint_embedding=np.asarray(checkpoint, dtype=np.float32),
        action_embeddings=np.arange(
            len(candidates) * 2, dtype=np.float32,
        ).reshape(len(candidates), 2),
    )


def trace() -> tuple[CausalTemporalStep, ...]:
    return (
        step(0, "n", ("n", "a"), 0.0, 3.0, 0.1, (1.0, 0.0)),
        step(1, "n", ("n", "a", "b"), 2.0, 2.0, 0.2, (1.0, 0.0)),
        step(2, "a", ("a", "n", "b"), 4.0, 1.0, 0.4, (0.0, 1.0)),
    )


def sequence(decision_step: int = 2) -> TemporalSequence:
    return TemporalSequence.from_trace(
        dataset="RxR", scene_id="scene", episode_id="episode",
        decision_step=decision_step, trace_steps=trace(),
    )


class TemporalCausalityTest(unittest.TestCase):
    def test_fixed_temporal_summary_has_predeclared_semantics(self):
        summary = causal_temporal_summary(sequence())
        observed = dict(zip(TEMPORAL_SUMMARY_NAMES, summary, strict=True))
        self.assertAlmostEqual(observed["score_slope"], 2.0)
        self.assertAlmostEqual(observed["margin_slope"], -1.0)
        self.assertEqual(observed["candidate_birth_count"], 3.0)
        self.assertEqual(observed["candidate_expiry_count"], 0.0)
        self.assertAlmostEqual(observed["native_persistence"], 0.5)
        self.assertAlmostEqual(observed["runner_persistence"], 0.5)
        self.assertEqual(observed["rank_switch_count"], 1.0)
        self.assertAlmostEqual(observed["checkpoint_embedding_drift"], 0.5)
        self.assertAlmostEqual(
            observed["instruction_history_alignment_drift"], 0.15,
            places=6,
        )
        self.assertAlmostEqual(observed["candidate_set_jaccard"], 5.0 / 6.0)
        self.assertFalse(summary.flags.writeable)
        self.assertFalse(causal_snapshot_summary(sequence()).flags.writeable)

    def test_future_mutation_is_byte_identical_before_decision(self):
        baseline = sequence(decision_step=1)
        changed_future = (*trace()[:2], step(
            2, "b", ("b", "x", "n"), 999.0, 0.0, -0.9, (9.0, 9.0),
        ))
        changed = TemporalSequence.from_trace(
            dataset="RxR", scene_id="scene", episode_id="episode",
            decision_step=1, trace_steps=changed_future,
        )
        self.assertEqual(baseline.prefix_sha256, changed.prefix_sha256)
        self.assertEqual(
            temporal_summary_bytes(baseline), temporal_summary_bytes(changed),
        )

    def test_gru_rows_are_strictly_causal_under_future_mutation(self):
        baseline = causal_sequence_features(sequence())
        changed_trace = (*trace()[:2], step(
            2, "b", ("b", "x", "n"), 999.0, 0.0, -0.9, (9.0, 9.0),
        ))
        changed_sequence = TemporalSequence.from_trace(
            dataset="RxR", scene_id="scene", episode_id="episode",
            decision_step=2, trace_steps=changed_trace,
        )
        changed = causal_sequence_features(changed_sequence)
        self.assertTrue(np.array_equal(baseline[:2], changed[:2]))
        self.assertFalse(np.array_equal(baseline[2], changed[2]))
        self.assertFalse(baseline.flags.writeable)
        with self.assertRaises(ValueError):
            baseline.setflags(write=True)

    def test_candidate_rank_permutation_only_changes_rank_dynamics(self):
        first = step(
            0, "n", ("n", "a", "b"), 1.0, 1.0, 0.1, (1.0, 0.0),
        )
        actions = {
            "n": np.asarray([1.0, 0.0], dtype=np.float32),
            "a": np.asarray([0.0, 1.0], dtype=np.float32),
            "b": np.asarray([2.0, 2.0], dtype=np.float32),
        }

        def ranked_second(order: tuple[str, ...]) -> CausalTemporalStep:
            return CausalTemporalStep(
                step=1,
                native_action_id="n",
                candidate_action_ids=order,
                policy_features=first.policy_features,
                instruction_embedding=first.instruction_embedding,
                checkpoint_embedding=first.checkpoint_embedding,
                action_embeddings=np.stack([actions[value] for value in order]),
            )

        stable = TemporalSequence.create(
            dataset="RxR", scene_id="s", episode_id="stable",
            decision_step=1,
            steps=(first, ranked_second(("n", "a", "b"))),
        )
        reranked = TemporalSequence.create(
            dataset="RxR", scene_id="s", episode_id="reranked",
            decision_step=1,
            steps=(first, ranked_second(("n", "b", "a"))),
        )
        stable_matrix = causal_sequence_features(stable)
        reranked_matrix = causal_sequence_features(reranked)
        policy = len(POLICY_FEATURE_NAMES)
        checkpoint = first.checkpoint_embedding.size
        instruction = first.instruction_embedding.size
        action = first.action_embeddings.shape[1]
        structural_start = policy + checkpoint + instruction + 2 * action
        structural = {
            name: structural_start + index
            for index, name in enumerate(STRUCTURAL_FEATURE_NAMES)
        }
        # Policy, checkpoint, native embedding, and the order-invariant mean
        # of executable non-native embeddings remain identical.
        np.testing.assert_array_equal(
            stable_matrix[1, :structural_start],
            reranked_matrix[1, :structural_start],
        )
        unchanged = (
            "candidate_count", "candidate_birth_count",
            "candidate_expiry_count", "native_persistence_indicator",
            "candidate_set_jaccard", "step_delta",
        )
        for name in unchanged:
            self.assertEqual(
                stable_matrix[1, structural[name]],
                reranked_matrix[1, structural[name]],
            )
        self.assertEqual(
            stable_matrix[1, structural["runner_persistence_indicator"]], 1.0,
        )
        self.assertEqual(
            reranked_matrix[1, structural["runner_persistence_indicator"]], 0.0,
        )
        self.assertEqual(
            stable_matrix[1, structural["rank_change_indicator"]], 0.0,
        )
        self.assertEqual(
            reranked_matrix[1, structural["rank_change_indicator"]], 1.0,
        )
        self.assertEqual(
            stable_matrix.shape[1], causal_sequence_feature_width(stable),
        )
        self.assertEqual(
            stable_matrix.shape[1],
            policy + checkpoint + instruction + 2 * action
            + len(STRUCTURAL_FEATURE_NAMES),
        )

    def test_current_only_row_contains_no_previous_prefix_dynamics(self):
        final = trace()[-1]
        left = TemporalSequence.create(
            dataset="RxR", scene_id="scene", episode_id="left",
            decision_step=2, steps=(*trace()[:2], final),
        )
        different_history = (
            step(0, "x", ("x", "y"), 99.0, 9.0, -0.5, (9.0, 0.0)),
            step(1, "z", ("z", "a"), -9.0, 8.0, -0.4, (0.0, 9.0)),
        )
        right = TemporalSequence.create(
            dataset="RxR", scene_id="scene", episode_id="right",
            decision_step=2, steps=(*different_history, final),
        )
        np.testing.assert_array_equal(
            causal_current_only_features(left),
            causal_current_only_features(right),
        )

    def test_treatment_outcome_and_oracle_labels_cannot_change_tensor(self):
        causal = sequence()
        first_label = TemporalOracleLabel(
            target_in_set=(False, True, True),
            candidate_separated=(False, True, True),
            evidence_closed=(False, True, True),
            reveal_interval=(1, 2), expiry_step=2, resolvable=True,
        )
        second_label = TemporalOracleLabel(
            target_in_set=(True, True, True),
            candidate_separated=(True, True, True),
            evidence_closed=(True, True, True),
            reveal_interval=(0, 0), expiry_step=1, resolvable=False,
        )
        experiments = (
            {"causal": causal, "delta_utility": -0.5, "oracle": first_label},
            {"causal": causal, "delta_utility": 0.8, "oracle": second_label},
        )
        tensors = [
            causal_temporal_summary(value["causal"]) for value in experiments
        ]
        np.testing.assert_array_equal(tensors[0], tensors[1])
        self.assertEqual(
            experiments[0]["causal"].prefix_sha256,
            experiments[1]["causal"].prefix_sha256,
        )

    def test_mapping_tensor_builder_rejects_outcome_or_oracle_fields(self):
        value = sequence()
        mapping = {
            "dataset": value.dataset,
            "scene_id": value.scene_id,
            "episode_id": value.episode_id,
            "decision_step": value.decision_step,
            "steps": value.steps,
            "prefix_sha256": value.prefix_sha256,
        }
        np.testing.assert_array_equal(
            causal_temporal_summary_from_mapping(mapping),
            causal_temporal_summary(value),
        )
        for forbidden in ("delta_utility", "oracle_label", "future_trace"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    causal_temporal_summary_from_mapping({
                        **mapping, forbidden: None,
                    })


if __name__ == "__main__":
    unittest.main()
