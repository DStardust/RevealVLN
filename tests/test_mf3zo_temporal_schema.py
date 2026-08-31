import unittest

import numpy as np

from revealnav_mf3.mf3zo_temporal_schema import (
    CausalTemporalRecord,
    CausalTemporalStep,
    ORACLE_FIELDS,
    TemporalOracleLabel,
    causal_prefix_sha256,
    derive_uad,
    inference_tensors,
)


def make_step(step=0):
    return CausalTemporalStep(
        step=step,
        native_action_id="native",
        candidate_action_ids=("native", "runner"),
        policy_features=np.arange(10, dtype=np.float32),
        policy_feature_mask=np.ones(10, dtype=np.bool_),
        instruction_embedding=np.ones(768, dtype=np.float32),
        checkpoint_embedding=np.full(768, 2.0, dtype=np.float32),
        embedded_action_ids=("native", "runner"),
        action_embeddings=np.stack((
            np.full(768, 3.0, dtype=np.float32),
            np.full(768, 4.0, dtype=np.float32),
        )),
    )


def make_record():
    step = make_step()
    digest = causal_prefix_sha256("R2R", "scene", "episode", 0, (step,))
    return CausalTemporalRecord(
        dataset="R2R",
        scene_id="scene",
        episode_id="episode",
        decision_step=0,
        steps=(step,),
        prefix_sha256=digest,
    )


class MF3ZOTemporalSchemaTest(unittest.TestCase):
    def test_causal_record_and_tensor_contract(self):
        record = make_record()
        tensors = inference_tensors(record)
        self.assertEqual(tensors["policy_features"].shape, (1, 10))
        self.assertEqual(tensors["checkpoint_embedding"].shape, (1, 768))
        self.assertTrue(tensors["checkpoint_embedding_mask"].all())
        self.assertTrue(record.full_prefix_embedding_complete)

    def test_unavailable_oracle_is_explicit(self):
        label = TemporalOracleLabel(
            event_id="event",
            target_in_set=None,
            candidate_separated=None,
            evidence_closed=None,
            reveal_interval=None,
            expiry_step=None,
            resolvable=None,
            unavailable_fields=ORACLE_FIELDS,
            provenance="UNAVAILABLE: no verified review",
        )
        self.assertFalse(label.complete)

    def test_uad_is_deterministic_with_fixed_stability(self):
        states = derive_uad(
            (False, True, True, True, True),
            (False, False, True, True, True),
            (False, False, True, True, True),
        )
        self.assertEqual(tuple(value.value for value in states), ("U", "A", "A", "A", "D"))
        with self.assertRaises(ValueError):
            derive_uad((True,), (True,), (True,), stability_prefixes=2)


if __name__ == "__main__":
    unittest.main()

