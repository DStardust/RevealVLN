import unittest

from revealnav_mf3.mf3zo_temporal_schema import (
    ORACLE_FIELDS,
    TemporalOracleLabel,
    inference_tensors,
)
from test_mf3zo_temporal_schema import make_record


class MF3ZOOracleIsolationTest(unittest.TestCase):
    def test_oracle_mutation_cannot_change_inference_tensor(self):
        record = make_record()
        before = {key: value.tobytes() for key, value in inference_tensors(record).items()}
        unavailable = TemporalOracleLabel(
            event_id="event", target_in_set=None, candidate_separated=None,
            evidence_closed=None, reveal_interval=None, expiry_step=None,
            resolvable=None, unavailable_fields=ORACLE_FIELDS,
            provenance="UNAVAILABLE: fixture",
        )
        available = TemporalOracleLabel(
            event_id="event", target_in_set=(True, True, True),
            candidate_separated=(True, True, True),
            evidence_closed=(True, True, True), reveal_interval=(0, 1),
            expiry_step=3, resolvable=True, unavailable_fields=(),
            provenance="verified fixture",
        )
        self.assertNotEqual(unavailable.complete, available.complete)
        after = {key: value.tobytes() for key, value in inference_tensors(record).items()}
        self.assertEqual(before, after)

    def test_missing_field_requires_unavailable_declaration(self):
        with self.assertRaises(ValueError):
            TemporalOracleLabel(
                event_id="event", target_in_set=None,
                candidate_separated=(True,), evidence_closed=(True,),
                reveal_interval=(0, 0), expiry_step=1, resolvable=True,
                unavailable_fields=(), provenance="invalid fixture",
            )


if __name__ == "__main__":
    unittest.main()
