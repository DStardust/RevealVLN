"""Model-level invariants for MF3ZN-TUAD v1."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.temporal_action_value import (
    NativeAnchoredActionValue,
    choose_native_inclusive_action,
    native_anchored_huber_loss,
)
from revealnav_mf3.temporal_uad_model import (
    RevealExpiryTargets,
    TemporalRevealExpiryEncoder,
    TemporalRevealExpiryLoss,
    freeze_temporal_encoder,
    last_causal_state,
)


class TemporalUADModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)

    def test_encoder_is_causal_and_hidden_size_is_frozen(self):
        model = TemporalRevealExpiryEncoder(input_dim=7)
        self.assertIn("fixed_input_projection", dict(model.named_buffers()))
        self.assertNotIn("fixed_input_projection", dict(model.named_parameters()))
        features = torch.randn(2, 5, 7)
        mask = torch.ones(2, 5, dtype=torch.bool)
        before = model(features, mask)
        changed = features.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100.0
        after = model(changed, mask)
        self.assertTrue(torch.equal(
            before.state_embedding[:, :3], after.state_embedding[:, :3]
        ))
        self.assertTrue(torch.equal(
            before.reveal_hazard_logit[:, :3], after.reveal_hazard_logit[:, :3]
        ))
        with self.assertRaises(ValueError):
            TemporalRevealExpiryEncoder(input_dim=7, hidden_dim=32)

    def test_mask_is_right_padded_and_last_state_is_decision_state(self):
        model = TemporalRevealExpiryEncoder(input_dim=3)
        features = torch.randn(2, 4, 3)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
        output = model(features, mask)
        state = last_causal_state(output, mask)
        self.assertTrue(torch.equal(state[0], output.state_embedding[0, 1]))
        self.assertTrue(torch.equal(state[1], output.state_embedding[1, 3]))
        bad = torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        with self.assertRaises(ValueError):
            model(features, bad)

    def test_stage_one_loss_has_only_factor_and_hazard_supervision(self):
        signature = inspect.signature(TemporalRevealExpiryEncoder.forward)
        self.assertNotIn("utility", signature.parameters)
        model = TemporalRevealExpiryEncoder(input_dim=4)
        features = torch.randn(3, 4, 4)
        mask = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool
        )
        output = model(features, mask)
        binary = torch.zeros(3, 4)
        targets = RevealExpiryTargets(
            target_in_set=binary,
            separation=binary,
            evidence=binary,
            reveal_event=binary,
            expiry_event=binary,
            factor_mask=mask,
            reveal_at_risk=mask,
            expiry_at_risk=mask,
        )
        loss = TemporalRevealExpiryLoss()(output, targets, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_freeze_prevents_stage_two_utility_gradient(self):
        encoder = TemporalRevealExpiryEncoder(input_dim=4)
        freeze_temporal_encoder(encoder)
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

    def test_native_value_bypasses_network_and_is_exact_zero(self):
        head = NativeAnchoredActionValue(action_embedding_dim=5, action_feature_dim=2)
        state = torch.randn(2, 64)
        native = torch.randn(2, 5)
        actions = torch.stack((native, torch.randn(2, 5)), dim=1)
        features = torch.randn(2, 2, 2)
        is_native = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)
        values = head(state, native, actions, features, is_native=is_native)
        self.assertTrue(torch.equal(values[:, 0], torch.zeros(2)))
        target = torch.tensor([[0.0, 0.4], [0.0, -0.2]])
        loss = native_anchored_huber_loss(
            values, target, torch.ones_like(is_native), is_native
        )
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
        ))

    def test_native_is_kept_on_zero_tie_and_alternative_requires_improvement(self):
        values = torch.tensor([[0.0, 0.0, -0.1], [0.0, 0.2, 0.1]])
        mask = torch.ones(2, 3, dtype=torch.bool)
        native = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.bool)
        chosen = choose_native_inclusive_action(values, mask, native)
        self.assertTrue(torch.equal(chosen, torch.tensor([0, 1])))

    def test_invalid_native_target_is_rejected(self):
        predicted = torch.zeros(1, 2)
        target = torch.tensor([[0.1, 0.2]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        native = torch.tensor([[1, 0]], dtype=torch.bool)
        with self.assertRaises(ValueError):
            native_anchored_huber_loss(predicted, target, mask, native)


if __name__ == "__main__":
    unittest.main()
