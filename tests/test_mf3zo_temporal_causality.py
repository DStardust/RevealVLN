import unittest

import numpy as np

from revealnav_mf3.mf3zo_temporal_schema import (
    CausalTemporalRecord,
    CausalTemporalStep,
    causal_prefix_sha256,
    inference_tensors,
    reject_forbidden_inference_mapping,
)
from test_mf3zo_temporal_schema import make_record, make_step


class MF3ZOTemporalCausalityTest(unittest.TestCase):
    def test_future_step_is_rejected(self):
        step = make_step(step=1)
        digest = causal_prefix_sha256("R2R", "scene", "episode", 0, (step,))
        with self.assertRaises(ValueError):
            CausalTemporalRecord(
                dataset="R2R", scene_id="scene", episode_id="episode",
                decision_step=0, steps=(step,), prefix_sha256=digest,
            )

    def test_future_and_target_mutation_do_not_change_tensor(self):
        record = make_record()
        expected = {key: value.tobytes() for key, value in inference_tensors(record).items()}
        unrelated = {"future_frame": np.full((4, 4), 99.0), "delta_utility": -99.0}
        unrelated["future_frame"][:] = -12.0
        unrelated["delta_utility"] = 123.0
        observed = {key: value.tobytes() for key, value in inference_tensors(record).items()}
        self.assertEqual(expected, observed)

    def test_forbidden_keys_fail_closed(self):
        for key in ("target", "delta_utility", "future_candidates", "oracle_uad", "pose"):
            with self.assertRaises(ValueError):
                reject_forbidden_inference_mapping({key: 1})


if __name__ == "__main__":
    unittest.main()
