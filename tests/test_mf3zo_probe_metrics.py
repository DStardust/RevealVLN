import unittest

import numpy as np

from revealnav_mf3.mf3zo_probes import (
    current_snapshot_features,
    matched_budget_baselines,
    oracle_feature_vector,
)
from revealnav_mf3.mf3zo_temporal_schema import TemporalOracleLabel
from test_mf3zo_temporal_schema import make_record


class MF3ZOProbeMetricTest(unittest.TestCase):
    def test_fixed_current_and_oracle_feature_dimensions(self):
        current = current_snapshot_features(make_record())
        label = TemporalOracleLabel(
            event_id="event", target_in_set=(True, True, True),
            candidate_separated=(True, True, True),
            evidence_closed=(True, True, True), reveal_interval=(0, 2),
            expiry_step=4, resolvable=True, unavailable_fields=(),
            provenance="verified fixture",
        )
        oracle = oracle_feature_vector(label, 2)
        self.assertEqual(current.shape, (16,))
        self.assertEqual(oracle.shape, (7,))
        np.testing.assert_array_equal(oracle[:3], (0.0, 0.0, 1.0))

    def test_matched_baselines_have_exact_fold_domain_budget(self):
        folds = np.repeat(np.arange(5), 8)
        datasets = np.tile(np.repeat(("R2R", "RxR"), 4), 5)
        selected = np.zeros(40, dtype=np.bool_)
        selected[::3] = True
        scores = np.linspace(0.0, 1.0, 40)
        margins = scores[::-1]
        identities = np.asarray([f"event-{index}" for index in range(40)])
        outputs = matched_budget_baselines(
            selected, scores, margins, identities, datasets, folds,
        )
        for mask in outputs.values():
            for fold in range(5):
                for domain in ("R2R", "RxR"):
                    stratum = (folds == fold) & (datasets == domain)
                    self.assertEqual(int(mask[stratum].sum()), int(selected[stratum].sum()))


if __name__ == "__main__":
    unittest.main()
