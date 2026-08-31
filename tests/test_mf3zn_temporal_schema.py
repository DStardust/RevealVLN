from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from revealnav_mf3.temporal_uad_schema import (
    CausalTemporalStep,
    TemporalSequence,
    TEMPORAL_RECORD_LIST_SCHEMA,
    TEMPORAL_RECORD_LIST_STATUS,
    causal_prefix_sha256,
    temporal_record_list_from_mapping,
)


def make_step(
    step: int,
    *,
    candidates: tuple[str, ...] = ("native", "alt"),
    native: str = "native",
    offset: float = 0.0,
) -> CausalTemporalStep:
    return CausalTemporalStep(
        step=step,
        native_action_id=native,
        candidate_action_ids=candidates,
        policy_features=np.asarray(
            [1.0, 0.5, 0.25, 0.1, 0.2 + offset], dtype=np.float32,
        ),
        instruction_embedding=np.asarray(
            [0.25, 0.75 + offset], dtype=np.float32,
        ),
        checkpoint_embedding=np.asarray(
            [1.0 + offset, 2.0], dtype=np.float32,
        ),
        action_embeddings=np.arange(
            len(candidates) * 2, dtype=np.float32,
        ).reshape(len(candidates), 2) + offset,
    )


class CausalTemporalSchemaTest(unittest.TestCase):
    def test_arrays_are_defensively_copied_and_immutably_read_only(self):
        policy = np.asarray([1.0, 0.5], dtype=np.float32)
        instruction = np.asarray([0.25, 0.75], dtype=np.float32)
        checkpoint = np.asarray([2.0, 3.0], dtype=np.float32)
        actions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        value = CausalTemporalStep(
            0, "n", ("n", "a"), policy, instruction, checkpoint, actions,
        )
        policy[:] = -1.0
        instruction[:] = -4.0
        checkpoint[:] = -2.0
        actions[:] = -3.0
        np.testing.assert_array_equal(value.policy_features, [1.0, 0.5])
        np.testing.assert_array_equal(value.instruction_embedding, [0.25, 0.75])
        np.testing.assert_array_equal(value.checkpoint_embedding, [2.0, 3.0])
        np.testing.assert_array_equal(
            value.action_embeddings, [[1.0, 2.0], [3.0, 4.0]],
        )
        for array in (
            value.policy_features,
            value.instruction_embedding,
            value.checkpoint_embedding,
            value.action_embeddings,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_candidate_identity_and_embedding_alignment_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "native action"):
            CausalTemporalStep(
                0, "missing", ("n", "a"),
                np.ones(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            CausalTemporalStep(
                0, "n", ("n", "n"),
                np.ones(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "rows must align"):
            CausalTemporalStep(
                0, "n", ("n", "a"),
                np.ones(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.ones(2, dtype=np.float32),
                np.ones((3, 2), dtype=np.float32),
            )

    def test_prefix_hash_is_content_bound_and_verified(self):
        steps = (make_step(0), make_step(1, offset=0.1))
        sequence = TemporalSequence.create(
            dataset="RxR",
            scene_id="scene-a",
            episode_id="episode-1",
            decision_step=1,
            steps=steps,
        )
        self.assertEqual(
            sequence.prefix_sha256,
            causal_prefix_sha256(
                dataset="RxR",
                scene_id="scene-a",
                episode_id="episode-1",
                decision_step=1,
                steps=steps,
            ),
        )
        changed = TemporalSequence.create(
            dataset="RxR",
            scene_id="scene-a",
            episode_id="episode-1",
            decision_step=1,
            steps=(steps[0], make_step(1, offset=0.2)),
        )
        self.assertNotEqual(sequence.prefix_sha256, changed.prefix_sha256)
        instruction_changed_step = replace(
            steps[1],
            instruction_embedding=np.asarray([9.0, 9.0], dtype=np.float32),
        )
        instruction_changed = TemporalSequence.create(
            dataset="RxR",
            scene_id="scene-a",
            episode_id="episode-1",
            decision_step=1,
            steps=(steps[0], instruction_changed_step),
        )
        self.assertNotEqual(
            sequence.prefix_sha256, instruction_changed.prefix_sha256,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            TemporalSequence(
                dataset=sequence.dataset,
                scene_id=sequence.scene_id,
                episode_id=sequence.episode_id,
                decision_step=sequence.decision_step,
                steps=sequence.steps,
                prefix_sha256="0" * 64,
            )

    def test_instruction_width_is_fixed_within_sequence(self):
        first = make_step(0)
        second = replace(
            make_step(1),
            instruction_embedding=np.ones(3, dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "instruction embedding width"):
            TemporalSequence.create(
                dataset="RxR", scene_id="s", episode_id="e",
                decision_step=1, steps=(first, second),
            )

    def test_future_steps_are_rejected_or_strictly_truncated(self):
        trace = (make_step(0), make_step(1), make_step(2, offset=10.0))
        with self.assertRaisesRegex(ValueError, "future step"):
            TemporalSequence.create(
                dataset="R2R", scene_id="s", episode_id="e",
                decision_step=1, steps=trace,
            )
        prefix = TemporalSequence.from_trace(
            dataset="R2R", scene_id="s", episode_id="e",
            decision_step=1, trace_steps=trace,
        )
        self.assertEqual(tuple(step.step for step in prefix.steps), (0, 1))

    def test_mapping_schema_physically_rejects_leakage_fields(self):
        step = make_step(0)
        mapping = {
            "step": step.step,
            "native_action_id": step.native_action_id,
            "candidate_action_ids": step.candidate_action_ids,
            "policy_features": step.policy_features,
            "instruction_embedding": step.instruction_embedding,
            "checkpoint_embedding": step.checkpoint_embedding,
            "action_embeddings": step.action_embeddings,
        }
        self.assertIsInstance(
            CausalTemporalStep.from_mapping(mapping), CausalTemporalStep,
        )
        for forbidden in (
            "target", "target_in_set", "delta_utility", "future_frame",
            "navmesh", "pose", "oracle_state", "outcome",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    CausalTemporalStep.from_mapping({
                        **mapping, forbidden: 0,
                    })

    def test_serialized_record_list_rebuilds_and_rehashes_strict_schema(self):
        source_step = make_step(0)
        step = CausalTemporalStep(
            step=source_step.step,
            native_action_id=source_step.native_action_id,
            candidate_action_ids=source_step.candidate_action_ids,
            policy_features=source_step.policy_features.astype(np.float64),
            instruction_embedding=source_step.instruction_embedding.astype(np.float64),
            checkpoint_embedding=source_step.checkpoint_embedding.astype(np.float64),
            action_embeddings=source_step.action_embeddings.astype(np.float64),
        )
        sequence = TemporalSequence.create(
            dataset="RxR", scene_id="scene", episode_id="episode",
            decision_step=0, steps=(step,),
        )
        record = {
            "dataset": sequence.dataset,
            "scene_id": sequence.scene_id,
            "episode_id": sequence.episode_id,
            "decision_step": sequence.decision_step,
            "prefix_sha256": sequence.prefix_sha256,
            "steps": [{
                "step": step.step,
                "native_action_id": step.native_action_id,
                "candidate_action_ids": list(step.candidate_action_ids),
                "policy_features": step.policy_features.tolist(),
                "instruction_embedding": step.instruction_embedding.tolist(),
                "checkpoint_embedding": step.checkpoint_embedding.tolist(),
                "action_embeddings": step.action_embeddings.tolist(),
            }],
        }
        records, source = temporal_record_list_from_mapping({
            "schema_version": TEMPORAL_RECORD_LIST_SCHEMA,
            "status": TEMPORAL_RECORD_LIST_STATUS,
            "source_canonical_identity_sha256": "a" * 64,
            "records": [record],
            "public_split_access": False,
        })
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].prefix_sha256, sequence.prefix_sha256)
        np.testing.assert_array_equal(
            records[0].steps[0].action_embeddings,
            sequence.steps[0].action_embeddings,
        )
        self.assertEqual(source, "a" * 64)
        bad = dict(record)
        bad["future_frame"] = []
        with self.assertRaisesRegex(ValueError, "forbidden"):
            temporal_record_list_from_mapping({
                "schema_version": TEMPORAL_RECORD_LIST_SCHEMA,
                "status": TEMPORAL_RECORD_LIST_STATUS,
                "source_canonical_identity_sha256": "a" * 64,
                "records": [bad],
                "public_split_access": False,
            })


if __name__ == "__main__":
    unittest.main()
