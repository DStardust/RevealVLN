from __future__ import annotations

import unittest

import numpy as np

from revealnav_mf3.temporal_uad_labels import (
    TemporalOracleLabel,
    UAD_STABILITY_PREFIXES,
    UADState,
    derive_uad,
    reveal_interval_membership,
    validate_oracle_alignment,
)
from revealnav_mf3.temporal_uad_schema import (
    CausalTemporalStep,
    TemporalSequence,
)


def causal_sequence(length: int) -> TemporalSequence:
    steps = tuple(CausalTemporalStep(
        step=index,
        native_action_id="n",
        candidate_action_ids=("n", "a"),
        policy_features=np.asarray([1, 1, 1, 0, 0], dtype=np.float32),
        instruction_embedding=np.asarray([0, 1], dtype=np.float32),
        checkpoint_embedding=np.asarray([1, 0], dtype=np.float32),
        action_embeddings=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    ) for index in range(length))
    return TemporalSequence.create(
        dataset="RxR", scene_id="s", episode_id="e",
        decision_step=length - 1, steps=steps,
    )


class TemporalUADLabelTest(unittest.TestCase):
    def test_uad_is_deterministic_and_requires_fixed_k_stability(self):
        self.assertEqual(UAD_STABILITY_PREFIXES, 3)
        label = TemporalOracleLabel(
            target_in_set=(False, True, True, True, True),
            candidate_separated=(False, False, True, True, True),
            evidence_closed=(False, True, True, True, True),
            reveal_interval=(2, 3),
            expiry_step=4,
            resolvable=True,
        )
        self.assertEqual(
            derive_uad(label),
            (
                UADState.UNOBSERVED,
                UADState.AMBIGUOUS,
                UADState.AMBIGUOUS,
                UADState.AMBIGUOUS,
                UADState.DECISIVE,
            ),
        )
        self.assertEqual(
            derive_uad(
                label.target_in_set,
                label.candidate_separated,
                label.evidence_closed,
            ),
            derive_uad(label),
        )
        with self.assertRaises(TypeError):
            derive_uad(label, stability_k=2)  # type: ignore[call-arg]

    def test_occlusion_or_factor_regression_resets_decisive_streak(self):
        states = derive_uad(
            (True, True, True, False, True, True, True, True, True, True),
            (True, True, True, False, True, True, True, True, False, True),
            (True, True, True, False, True, True, True, True, True, True),
        )
        self.assertEqual(states[:4], (
            UADState.AMBIGUOUS,
            UADState.AMBIGUOUS,
            UADState.DECISIVE,
            UADState.UNOBSERVED,
        ))
        self.assertEqual(states[4:8], (
            UADState.AMBIGUOUS,
            UADState.AMBIGUOUS,
            UADState.DECISIVE,
            UADState.DECISIVE,
        ))
        self.assertEqual(states[8:], (
            UADState.AMBIGUOUS, UADState.AMBIGUOUS,
        ))

    def test_oracle_label_validates_types_and_alignment(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            TemporalOracleLabel(
                target_in_set=(True,), candidate_separated=(True, False),
                evidence_closed=(True,), reveal_interval=None,
                expiry_step=None, resolvable=False,
            )
        with self.assertRaisesRegex(TypeError, "boolean"):
            TemporalOracleLabel(
                target_in_set=(1,),  # type: ignore[arg-type]
                candidate_separated=(True,), evidence_closed=(True,),
                reveal_interval=None, expiry_step=None, resolvable=False,
            )
        sequence = causal_sequence(5)
        label = TemporalOracleLabel(
            target_in_set=(False, True, True, True, True),
            candidate_separated=(False, False, True, True, True),
            evidence_closed=(False, False, True, True, True),
            reveal_interval=(2, 3), expiry_step=4, resolvable=True,
        )
        validate_oracle_alignment(sequence, label)
        self.assertEqual(
            reveal_interval_membership(sequence, label),
            (False, False, True, True, False),
        )
        with self.assertRaisesRegex(ValueError, "length"):
            validate_oracle_alignment(
                causal_sequence(4), label,
            )


if __name__ == "__main__":
    unittest.main()
