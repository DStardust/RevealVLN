#!/usr/bin/env python3

import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf2r6 import (
    ReversibleAdvantageHead,
    ReversibleAdvantageHeadV631,
    ReversibleAdvantageLoss,
    ReversibleAdvantageLossV631,
    intervention_authorized,
    select_authorized_option,
)


class ReversibleAdvantageTest(unittest.TestCase):
    def test_shape_order_and_finite_loss(self):
        torch.manual_seed(7)
        model = ReversibleAdvantageHead(input_dim=8, projection_dim=4)
        values = [torch.randn(5, 8) for _ in range(6)]
        output = model(*values, torch.randn(5, 3))
        self.assertEqual(output.lower.shape, (5,))
        self.assertTrue(torch.all(output.lower <= output.median))
        self.assertTrue(torch.all(output.median <= output.upper))
        losses = ReversibleAdvantageLoss()(output, torch.randn(5))
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_dimension_drift_is_rejected(self):
        model = ReversibleAdvantageHead(input_dim=8, projection_dim=4)
        values = [torch.randn(2, 8) for _ in range(6)]
        with self.assertRaisesRegex(ValueError, "scalar dimension"):
            model(*values, torch.randn(2, 2))

    def test_class_balanced_sign_loss_is_finite(self):
        model = ReversibleAdvantageHead(input_dim=8, projection_dim=4)
        values = [torch.randn(4, 8) for _ in range(6)]
        output = model(*values, torch.randn(4, 3))
        losses = ReversibleAdvantageLoss(
            sign_weight=1.0, positive_weight=3.0
        )(output, torch.tensor([1.0, -1.0, -1.0, -1.0]))
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_v631_auxiliary_loss_is_fit_only_and_finite(self):
        model = ReversibleAdvantageHeadV631(
            input_dim=8, projection_dim=4, scalar_dim=20
        )
        values = [torch.randn(4, 8) for _ in range(6)]
        output = model(*values, torch.randn(4, 20))
        losses = ReversibleAdvantageLossV631(3.0, 3.0)(
            output, torch.tensor([1.0, -1.0, -1.0, -1.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_intervention_fails_closed(self):
        self.assertTrue(intervention_authorized(0.01, True, True))
        self.assertFalse(intervention_authorized(0.0, True, True))
        self.assertFalse(intervention_authorized(0.01, False, True))
        self.assertFalse(intervention_authorized(0.01, True, False))

    def test_multi_option_selection_is_conservative(self):
        options = [
            {"branch_id": "a", "calibrated_lower_quantile": 0.2,
             "return_executable": True, "option_live": True},
            {"branch_id": "b", "calibrated_lower_quantile": 0.4,
             "return_executable": False, "option_live": True},
            {"branch_id": "c", "calibrated_lower_quantile": 0.3,
             "return_executable": True, "option_live": True},
        ]
        self.assertEqual(select_authorized_option(options)["branch_id"], "c")
        for row in options:
            row["calibrated_lower_quantile"] = 0.0
        self.assertIsNone(select_authorized_option(options))


if __name__ == "__main__":
    unittest.main()
