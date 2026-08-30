from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.action_aligned import FEATURE_NAMES
from revealnav_mf3.distributional_switch import (
    NATIVE_MARGIN_INDEX,
    DistributionalSwitchCritic,
    DistributionalSwitchGate,
    ensemble_checkpoint,
    quantile_switch_loss,
)


class DistributionalSwitchModelTest(unittest.TestCase):
    def test_feature_schema_excludes_domain_and_tier(self):
        lowered = {name.lower() for name in FEATURE_NAMES}
        self.assertNotIn("dataset", lowered)
        self.assertNotIn("benchmark", lowered)
        self.assertNotIn("tier", lowered)

    def test_quantiles_are_ordered_and_finite(self):
        torch.manual_seed(7)
        model = DistributionalSwitchCritic()
        prediction = model(torch.randn(8, len(FEATURE_NAMES)))
        self.assertTrue(torch.isfinite(prediction["lower_q20"]).all())
        self.assertTrue(torch.all(
            prediction["lower_q20"] <= prediction["median_q50"]
        ))
        self.assertTrue(torch.all(
            prediction["median_q50"] <= prediction["upper_q80"]
        ))

    def test_native_margin_anchor_is_nonpositive_and_monotone(self):
        model = DistributionalSwitchCritic()
        with torch.no_grad():
            for layer in (
                model.hidden, model.median_residual,
                model.lower_span, model.upper_span,
            ):
                layer.weight.zero_()
                layer.bias.zero_()
        features = torch.zeros(3, len(FEATURE_NAMES))
        features[:, NATIVE_MARGIN_INDEX] = torch.tensor([0.0, 0.5, 2.0])
        median = model(features)["median_q50"]
        self.assertAlmostEqual(float(median[0].detach()), 0.0, places=7)
        self.assertTrue(torch.all(median[1:] < 0.0))
        detached = median.detach()
        self.assertTrue(float(detached[2]) < float(detached[1]))

    def test_weighted_quantile_loss_is_finite_and_differentiable(self):
        model = DistributionalSwitchCritic()
        features = torch.zeros(4, len(FEATURE_NAMES))
        target = torch.tensor([-0.2, -0.1, 0.1, 0.3])
        weight = torch.tensor([1.0, 2.0, 1.0, 2.0])
        loss = quantile_switch_loss(model(features), target, weight)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_gate_round_trip_and_fixed_zero_rule(self):
        model = DistributionalSwitchCritic()
        with torch.no_grad():
            model.lower_span.bias.fill_(-20.0)
            model.upper_span.bias.fill_(-20.0)
            model.median_residual.bias.fill_(0.4)
        payload = ensemble_checkpoint([model], metadata={"seed": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.pt"
            torch.save(payload, path)
            gate = DistributionalSwitchGate(path)
            result = gate.evaluate(np.zeros(len(FEATURE_NAMES), dtype=np.float64))
        self.assertEqual(result["decision_rule"], "lower_q20_utility > 0")
        self.assertEqual(result["authorized"], result["lower_q20_utility"] > 0)
        self.assertGreater(result["lower_q20_utility"], 0.0)

    def test_malformed_features_fail_closed(self):
        model = DistributionalSwitchCritic()
        with self.assertRaises(ValueError):
            model(torch.zeros(1, len(FEATURE_NAMES) - 1))
        with self.assertRaises(ValueError):
            model(torch.full((1, len(FEATURE_NAMES)), float("nan")))

    def test_checkpoint_load_preserves_predictions(self):
        torch.manual_seed(29)
        model = DistributionalSwitchCritic()
        features = np.linspace(-0.2, 0.3, len(FEATURE_NAMES), dtype=np.float32)
        expected = model(torch.from_numpy(features[None]))
        payload = ensemble_checkpoint([model])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.pt"
            torch.save(payload, path)
            observed = DistributionalSwitchGate(path).evaluate(features)
        self.assertAlmostEqual(
            observed["lower_q20_utility"],
            float(expected["lower_q20"][0].detach()),
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
