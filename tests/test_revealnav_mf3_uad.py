from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3 import (
    MF3B_SCOPE,
    NativeConditionedUAD,
    NativeConditionedUADOutput,
    PairwiseSwitchUtility,
    PolicyAnchoredTop2UAD,
    StructuredUADHeads,
    StructuredUADLoss,
    classify_shadow_outcome,
    current_local_action_indices,
    fuse_current_candidate_logits,
    native_conditioned_uad_loss,
    native_alternative_posterior_gain,
    median_native_conditioned_outputs,
    median_mad_lower_confidence,
    native_residual_logits,
    native_residual_uad_loss,
    pairwise_expected_utility,
    pairwise_switch_targets,
    pairwise_switch_utility_loss,
    policy_anchored_target_loss,
    policy_anchored_conditional_top2_loss,
    top2_conditional_advantage,
    top2_expected_switch_utility,
    top2_cost_sensitive_utility_loss,
    top2_posterior_advantage,
    top2_rescue_harm_logit,
    top2_rescue_harm_loss,
    top2_rescue_harm_ranked_loss,
    top2_switch_targets,
    top2_switch_utility_loss,
)


class StructuredUADTest(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(17)
        history = torch.randn(2, 3, 8)
        candidates = torch.randn(2, 3, 4, 8)
        mask = torch.tensor([
            [[True, True, False, False]] * 3,
            [[True, True, True, False]] * 3,
        ])
        instruction = torch.randn(2, 8)
        return history, candidates, mask, instruction

    def test_output_is_uad_only_and_normalized(self):
        model = StructuredUADHeads(feature_dim=8, hidden_dim=4)
        output = model(*self.inputs())
        self.assertEqual(output.target_logits.shape, (2, 3, 4))
        self.assertEqual(output.uad_probabilities.shape, (2, 3, 3))
        self.assertTrue(torch.allclose(
            output.uad_probabilities.sum(-1), torch.ones(2, 3), atol=1e-6
        ))
        self.assertTrue(torch.isneginf(output.target_logits[0, :, 2:]).all())
        self.assertFalse(hasattr(output, "option_cost"))
        self.assertFalse(hasattr(output, "checkpoint_value"))
        self.assertEqual(
            model.candidate_count_encoding, "count_over_count_plus_one"
        )
        self.assertFalse(MF3B_SCOPE["uses_branch_exploration"])

    def test_candidate_permutation_and_padding_are_invariant(self):
        model = StructuredUADHeads(feature_dim=8, hidden_dim=4).eval()
        history, candidates, mask, instruction = self.inputs()
        baseline = model(history[:1], candidates[:1, :, :2], mask[:1, :, :2], instruction[:1])
        permutation = torch.tensor([1, 0])
        permuted = model(
            history[:1], candidates[:1, :, :2][:, :, permutation],
            mask[:1, :, :2][:, :, permutation], instruction[:1],
        )
        self.assertTrue(torch.allclose(
            baseline.target_logits[:, :, permutation], permuted.target_logits,
            atol=1e-6, rtol=0.0,
        ))
        self.assertTrue(torch.allclose(
            baseline.uad_probabilities, permuted.uad_probabilities,
            atol=1e-6, rtol=0.0,
        ))

        padded_candidates = torch.randn(1, 3, 5, 8)
        padded_candidates[:, :, :2] = candidates[:1, :, :2]
        padded_mask = torch.zeros(1, 3, 5, dtype=torch.bool)
        padded_mask[:, :, :2] = True
        padded = model(history[:1], padded_candidates, padded_mask, instruction[:1])
        self.assertTrue(torch.allclose(
            baseline.target_logits, padded.target_logits[:, :, :2],
            atol=1e-6, rtol=0.0,
        ))
        self.assertTrue(torch.allclose(
            baseline.uad_probabilities, padded.uad_probabilities,
            atol=1e-6, rtol=0.0,
        ))

    def test_cpu_float64_batch_partition_gate(self):
        model = StructuredUADHeads(feature_dim=8, hidden_dim=4).double().eval()
        torch.manual_seed(71)
        history = torch.randn(1, 3, 8, dtype=torch.float64)
        candidates = torch.randn(1, 3, 2, 8, dtype=torch.float64)
        mask = torch.ones(1, 3, 2, dtype=torch.bool)
        instruction = torch.randn(1, 8, dtype=torch.float64)
        alone = model(history, candidates, mask, instruction)

        padded = torch.randn(2, 3, 5, 8, dtype=torch.float64)
        padded[0, :, :2] = candidates[0]
        padded_mask = torch.ones(2, 3, 5, dtype=torch.bool)
        padded_mask[0, :, 2:] = False
        together = model(
            torch.cat((history, torch.randn(1, 3, 8, dtype=torch.float64))),
            padded,
            padded_mask,
            torch.cat((instruction, torch.randn(1, 8, dtype=torch.float64))),
        )
        difference = torch.max(torch.abs(
            alone.uad_probabilities[0] - together.uad_probabilities[0]
        ))
        self.assertLessEqual(float(difference.detach()), 1e-10)
        self.assertTrue(torch.equal(
            alone.uad_probabilities[0].argmax(-1),
            together.uad_probabilities[0].argmax(-1),
        ))

    def test_uad_only_loss_is_finite_and_differentiable(self):
        model = StructuredUADHeads(feature_dim=8, hidden_dim=4)
        history, candidates, mask, instruction = self.inputs()
        output = model(history, candidates, mask, instruction)
        batch = {
            "candidate_mask": mask,
            "target_index": torch.tensor([[0, 1, -1], [2, 1, 0]]),
            "target_in_set": torch.tensor([[1., 1., -1.], [1., 1., 0.]]),
            "separation": torch.tensor([[1., 0., -1.], [1., 0., 0.]]),
            "evidence_complete": torch.tensor([[1., 1., -1.], [1., 0., 0.]]),
            "reveal_hazard": torch.tensor([[0., 1., -1.], [0., 1., 0.]]),
            "expiry_hazard": torch.tensor([[0., 0., -1.], [1., 0., 0.]]),
        }
        losses = StructuredUADLoss()(output, batch)
        self.assertEqual(
            set(losses), {"total", "target", "factors", "uad", "reveal", "expiry"}
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_out_of_range_target_is_rejected_cleanly(self):
        model = StructuredUADHeads(feature_dim=8, hidden_dim=4)
        history, candidates, mask, instruction = self.inputs()
        output = model(history, candidates, mask, instruction)
        labels = torch.zeros(2, 3)
        batch = {
            "candidate_mask": mask,
            "target_index": torch.tensor([[9, -1, -1], [-1, -1, -1]]),
            "target_in_set": labels,
            "separation": labels,
            "evidence_complete": labels,
            "reveal_hazard": labels,
            "expiry_hazard": labels,
        }
        with self.assertRaisesRegex(ValueError, "valid candidate"):
            StructuredUADLoss()(output, batch)


class ResidualFusionTest(unittest.TestCase):
    def test_posterior_gain_matches_factorization_and_rejects_bad_index(self):
        output = NativeConditionedUADOutput(
            native_error_logit=torch.tensor([[0.0]]),
            alternative_logits=torch.tensor([[[-float("inf"), 0.0, 0.0]]]),
        )
        index = torch.tensor([[1]], dtype=torch.long)
        gain = native_alternative_posterior_gain(output, index)
        self.assertTrue(torch.allclose(gain, torch.tensor([[-0.25]])))
        with self.assertRaises(ValueError):
            native_alternative_posterior_gain(
                output, torch.tensor([[3]], dtype=torch.long)
            )


class PairwiseSwitchUtilityTest(unittest.TestCase):
    def batch(self):
        return {
            "history_embeddings": torch.randn(1, 2, 8),
            "candidate_embeddings": torch.randn(1, 2, 3, 12),
            "candidate_mask": torch.tensor([
                [[True, True, True], [True, True, True]]
            ]),
            "instruction_embedding": torch.randn(1, 8),
            "native_scores": torch.tensor([
                [[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
            ]),
            "native_index": torch.tensor([[0, 0]], dtype=torch.long),
            "target_index": torch.tensor([[0, 2]], dtype=torch.long),
            "step_mask": torch.tensor([[True, True]]),
        }

    def test_targets_distinguish_harm_rescue_and_neither(self):
        labels, mask = pairwise_switch_targets(self.batch())
        self.assertEqual(labels[0, 0].tolist(), [0, 2, 2])
        self.assertEqual(labels[0, 1].tolist(), [0, 0, 1])
        self.assertEqual(mask.sum().item(), 4)

    def test_model_loss_and_expected_utility_are_finite(self):
        batch = self.batch()
        model = PairwiseSwitchUtility(8, 12, 4)
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        loss = pairwise_switch_utility_loss(output, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(pairwise_expected_utility(output).shape, (1, 2, 3))
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_top2_utility_uses_only_frozen_runner_up(self):
        batch = self.batch()
        model = PairwiseSwitchUtility(8, 12, 4)
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        score, runner, valid = top2_expected_switch_utility(output, batch)
        self.assertEqual(runner.tolist(), [[1, 1]])
        self.assertTrue(valid.all())
        self.assertEqual(score.shape, (1, 2))
        loss = top2_switch_utility_loss(output, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_cost_sensitive_top2_loss_is_finite(self):
        batch = self.batch()
        model = PairwiseSwitchUtility(8, 12, 4)
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        loss = top2_cost_sensitive_utility_loss(output, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_rescue_harm_loss_ignores_neither_and_is_finite(self):
        batch = self.batch()
        model = PairwiseSwitchUtility(8, 12, 4)
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        logit, runner, valid = top2_rescue_harm_logit(output, batch)
        self.assertEqual(logit.shape, (1, 2))
        self.assertEqual(runner.tolist(), [[1, 1]])
        self.assertTrue(valid.all())
        loss = top2_rescue_harm_loss(
            output, batch, rescue_positive_weight=2.0
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_ranked_rescue_harm_loss_is_finite(self):
        batch = self.batch()
        batch["target_index"] = torch.tensor([[0, 1]], dtype=torch.long)
        model = PairwiseSwitchUtility(8, 12, 4)
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        losses = top2_rescue_harm_ranked_loss(
            output, batch, rescue_positive_weight=2.0,
        )
        self.assertEqual(set(losses), {"total", "binary", "ranking"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertGreater(float(losses["ranking"].detach()), 0.0)
        losses["total"].backward()

    def test_ranked_rescue_harm_does_not_pair_across_episodes(self):
        batch = self.batch()
        batch["target_index"] = torch.tensor([[0, 1]], dtype=torch.long)
        batch["target_index"] = batch["target_index"].repeat(2, 1)
        for key in ("history_embeddings", "candidate_embeddings", "candidate_mask",
                    "instruction_embedding", "native_scores", "native_index",
                    "step_mask"):
            batch[key] = batch[key].repeat((2,) + (1,) * (batch[key].ndim - 1))
        model = PairwiseSwitchUtility(8, 12, 4)
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        losses = top2_rescue_harm_ranked_loss(
            output, batch, rescue_positive_weight=2.0,
        )
        self.assertTrue(torch.isfinite(losses["total"]))


class PolicyAnchoredTop2UADTest(unittest.TestCase):
    def _batch(self):
        return {
            "history_embeddings": torch.randn(1, 2, 768),
            "candidate_embeddings": torch.randn(1, 2, 3, 1536),
            "candidate_mask": torch.tensor([[[True, True, True],
                                               [True, True, False]]]),
            "instruction_embedding": torch.randn(1, 768),
            "native_scores": torch.tensor([[[2.0, 1.5, 0.0],
                                               [0.2, 0.5, -torch.inf]]]),
            "native_index": torch.tensor([[0, 1]]),
            "target_index": torch.tensor([[1, 1]]),
            "step_mask": torch.tensor([[True, True]]),
        }

    def test_zero_initialized_adapter_delegates_and_targets_top2(self):
        batch = self._batch()
        model = PolicyAnchoredTop2UAD(hidden_dim=16, correction_bound=1.0)
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        self.assertTrue(torch.equal(
            output.target_logits[batch["candidate_mask"]],
            batch["native_scores"][batch["candidate_mask"]],
        ))
        labels, runner, valid = top2_switch_targets(batch)
        self.assertEqual(runner.tolist(), [[1, 0]])
        self.assertEqual(labels.tolist(), [[1, 2]])
        self.assertTrue(valid.all())
        loss = policy_anchored_target_loss(output, batch)
        self.assertTrue(torch.isfinite(loss))
        advantage, proposed, eligible = top2_posterior_advantage(
            output, batch["native_scores"], batch["candidate_mask"],
            batch["native_index"],
        )
        self.assertTrue(torch.equal(proposed, runner))
        self.assertTrue(torch.equal(eligible, valid))
        self.assertTrue(torch.isfinite(advantage).all())

    def test_candidate_permutation_equivariance(self):
        batch = self._batch()
        model = PolicyAnchoredTop2UAD(hidden_dim=16)
        model.eval()
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.argsort(permutation)
        baseline = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        native = inverse[batch["native_index"]]
        permuted = model(
            batch["history_embeddings"],
            batch["candidate_embeddings"][:, :, permutation],
            batch["candidate_mask"][:, :, permutation],
            batch["instruction_embedding"],
            batch["native_scores"][:, :, permutation], native,
        )
        self.assertTrue(torch.allclose(
            baseline.target_logits[:, :, permutation], permuted.target_logits,
        ))

    def test_conditional_advantage_ignores_unrelated_candidate_denominator(self):
        batch = self._batch()
        model = PolicyAnchoredTop2UAD(hidden_dim=16)
        output = model(
            batch["history_embeddings"], batch["candidate_embeddings"],
            batch["candidate_mask"], batch["instruction_embedding"],
            batch["native_scores"], batch["native_index"],
        )
        conditional, runner, valid = top2_conditional_advantage(
            output, batch["native_scores"], batch["candidate_mask"],
            batch["native_index"],
        )
        expected = torch.tanh(torch.tensor([[-0.25, -0.15]]))
        self.assertTrue(torch.allclose(conditional, expected))
        self.assertEqual(runner.tolist(), [[1, 0]])
        self.assertTrue(valid.all())
        losses = policy_anchored_conditional_top2_loss(output, batch)
        losses["total"].backward()
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_median_mad_lower_confidence_penalizes_member_disagreement(self):
        members = torch.tensor([[0.2, 0.3], [0.4, 0.3], [0.3, 0.5]])
        score = median_mad_lower_confidence(members, mad_weight=1.0)
        self.assertTrue(torch.allclose(score, torch.tensor([0.2, 0.3])))
        with self.assertRaises(ValueError):
            median_mad_lower_confidence(members, mad_weight=-1.0)


class ResidualFusionPolicyTest(unittest.TestCase):
    def inputs(self):
        return {
            "native_logits": torch.tensor([[1., 2., 3., 4., 5.], [5., 4., 3., 2., 1.]]),
            "current_candidate_indices": torch.tensor([[2, 4, -1], [1, 3, -1]]),
            "target_scores": torch.tensor([[0.8, 0.2, 99.], [0.7, 0.1, 99.]]),
            "current_candidate_mask": torch.tensor([[True, True, False], [True, True, False]]),
            "decisive_probability": torch.tensor([0.9, 0.2]),
        }

    def test_authorized_residual_changes_only_current_candidates(self):
        values = self.inputs()
        native = values["native_logits"].clone()
        output = fuse_current_candidate_logits(
            **values, alpha=0.5, decisive_threshold=0.8, margin_threshold=0.5
        )
        self.assertTrue(output.authorized.tolist() == [True, False])
        self.assertTrue(torch.equal(output.logits[:, 0], native[:, 0]))
        self.assertTrue(torch.equal(output.logits[0, [0, 1, 3]], native[0, [0, 1, 3]]))
        self.assertTrue(torch.equal(output.logits[1], native[1]))
        self.assertAlmostEqual(
            float((output.logits[0] - native[0]).sum()), 0.0, places=6
        )

    def test_invalid_rows_delegate_exactly(self):
        for field, replacement in (
            ("current_candidate_indices", torch.tensor([[2, 2, -1], [1, 3, -1]])),
            ("target_scores", torch.tensor([[float("nan"), .2, 99.], [.7, .1, 99.]])),
        ):
            values = self.inputs()
            values[field] = replacement
            native = values["native_logits"].clone()
            output = fuse_current_candidate_logits(
                **values, alpha=0.5, decisive_threshold=0.8,
                margin_threshold=0.5,
            )
            self.assertFalse(bool(output.authorized[0]))
            self.assertTrue(torch.equal(output.logits[0], native[0]))

    def test_margin_gate_delegates_exactly(self):
        values = self.inputs()
        values["target_scores"][0, :2] = torch.tensor([0.51, 0.5])
        native = values["native_logits"].clone()
        output = fuse_current_candidate_logits(
            **values, alpha=0.5, decisive_threshold=0.8, margin_threshold=0.5
        )
        self.assertFalse(bool(output.authorized[0]))
        self.assertTrue(torch.equal(output.logits[0], native[0]))

    def test_native_action_outside_current_set_delegates_exactly(self):
        values = self.inputs()
        values["native_logits"][0, 1] = 9.0
        native = values["native_logits"].clone()
        output = fuse_current_candidate_logits(
            **values, alpha=0.5, decisive_threshold=0.8,
            margin_threshold=0.5,
        )
        self.assertFalse(bool(output.authorized[0]))
        self.assertTrue(torch.equal(output.logits[0], native[0]))


class NativeConditionedUADTest(unittest.TestCase):
    def test_separate_policy_token_dimension_is_supported(self):
        torch.manual_seed(5)
        model = NativeConditionedUAD(
            feature_dim=8, hidden_dim=4, candidate_feature_dim=12
        )
        output = model(
            torch.randn(1, 2, 8),
            torch.randn(1, 2, 3, 12),
            torch.tensor([[[True, True, False], [True, True, True]]]),
            torch.randn(1, 8),
            torch.tensor([[[1.0, 0.0, -float("inf")], [2.0, 1.0, 0.0]]]),
            torch.tensor([[0, 0]], dtype=torch.long),
        )
        self.assertEqual(output.alternative_logits.shape, (1, 2, 3))
        self.assertTrue(torch.isfinite(output.native_error_logit).all())

    def test_median_ensemble_is_elementwise_and_checks_shapes(self):
        from revealnav_mf3 import NativeConditionedUADOutput

        outputs = tuple(
            NativeConditionedUADOutput(
                native_error_logit=torch.full((1, 2), value),
                alternative_logits=torch.full((1, 2, 3), value),
            )
            for value in (3.0, 1.0, 2.0)
        )
        median = median_native_conditioned_outputs(outputs)
        self.assertTrue(torch.equal(
            median.native_error_logit, torch.full((1, 2), 2.0)
        ))
        self.assertTrue(torch.equal(
            median.alternative_logits, torch.full((1, 2, 3), 2.0)
        ))
        with self.assertRaisesRegex(ValueError, "shape drift"):
            median_native_conditioned_outputs((
                outputs[0], NativeConditionedUADOutput(
                    torch.zeros(1, 3), torch.zeros(1, 3, 3)
                ),
            ))

    def test_correction_masks_native_and_trains_on_errors(self):
        torch.manual_seed(29)
        model = NativeConditionedUAD(feature_dim=8, hidden_dim=4)
        history = torch.randn(2, 3, 8)
        candidates = torch.randn(2, 3, 4, 8)
        mask = torch.tensor([
            [[True, True, True, False]] * 3,
            [[True, True, False, False]] * 3,
        ])
        instruction = torch.randn(2, 8)
        scores = torch.tensor([
            [[3., 2., 1., -torch.inf]] * 3,
            [[1., 2., -torch.inf, -torch.inf]] * 3,
        ])
        native = torch.tensor([[0, 0, 0], [1, 1, 1]])
        output = model(history, candidates, mask, instruction, scores, native)
        self.assertEqual(output.native_error_logit.shape, (2, 3))
        self.assertEqual(output.alternative_logits.shape, (2, 3, 4))
        self.assertTrue(torch.isneginf(
            output.alternative_logits.gather(-1, native.unsqueeze(-1))
        ).all())
        batch = {
            "step_mask": torch.ones(2, 3, dtype=torch.bool),
            "candidate_mask": mask,
            "native_index": native,
            "target_index": torch.tensor([[0, 1, 0], [0, 1, 0]]),
        }
        losses = native_conditioned_uad_loss(
            output, batch, error_positive_weight=3.0
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_bounded_residual_keeps_native_slot_and_has_finite_loss(self):
        torch.manual_seed(37)
        model = NativeConditionedUAD(feature_dim=8, hidden_dim=4)
        history = torch.randn(1, 2, 8)
        candidates = torch.randn(1, 2, 3, 8)
        mask = torch.ones(1, 2, 3, dtype=torch.bool)
        instruction = torch.randn(1, 8)
        scores = torch.tensor([[[3., 2., 1.], [1., 3., 2.]]])
        native = torch.tensor([[0, 1]])
        output = model(history, candidates, mask, instruction, scores, native)
        fused, correction = native_residual_logits(
            output, scores, mask, correction_bound=2.0
        )
        self.assertTrue(torch.equal(
            correction.gather(-1, native.unsqueeze(-1)), torch.zeros(1, 2, 1)
        ))
        self.assertLessEqual(float(correction.detach().abs().max()), 2.0)
        batch = {
            "step_mask": torch.ones(1, 2, dtype=torch.bool),
            "candidate_mask": mask, "native_scores": scores,
            "native_index": native, "target_index": torch.tensor([[1, 1]]),
        }
        losses = native_residual_uad_loss(
            output, batch, correction_bound=2.0
        )
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()

    def test_candidate_permutation_preserves_error_and_alternatives(self):
        torch.manual_seed(31)
        model = NativeConditionedUAD(feature_dim=8, hidden_dim=4).eval()
        history = torch.randn(1, 2, 8)
        candidates = torch.randn(1, 2, 3, 8)
        mask = torch.ones(1, 2, 3, dtype=torch.bool)
        instruction = torch.randn(1, 8)
        scores = torch.randn(1, 2, 3)
        native = torch.tensor([[1, 1]])
        baseline = model(history, candidates, mask, instruction, scores, native)
        permutation = torch.tensor([2, 0, 1])
        permuted = model(
            history, candidates[:, :, permutation], mask[:, :, permutation],
            instruction, scores[:, :, permutation], torch.tensor([[2, 2]]),
        )
        self.assertTrue(torch.allclose(
            baseline.native_error_logit, permuted.native_error_logit,
            atol=1e-6, rtol=0.0,
        ))
        self.assertTrue(torch.allclose(
            baseline.alternative_logits[:, :, permutation],
            permuted.alternative_logits, atol=1e-6, rtol=0.0,
        ))

class ShadowBookkeepingTest(unittest.TestCase):
    def test_current_local_indices_exclude_stop_visited_and_nonlocal(self):
        indices = current_local_action_indices(
            [None, "local_a", "history", "local_b", "unstable"],
            [True, True, True, True, True],
            [False, False, True, False, False],
            {"local_a", "local_b"},
        )
        self.assertEqual(indices, (1, 3))

    def test_shadow_outcomes_are_mutually_exclusive(self):
        stable = (1, 3)
        cases = {
            (2, 1, 1): "RESCUE",
            (1, 3, 1): "HARM",
            (1, 1, 1): "AGREE_CORRECT",
            (3, 3, 1): "AGREE_INCORRECT",
            (0, 3, 1): "DISAGREE_NEITHER",
            (1, 1, 2): "INELIGIBLE",
        }
        for values, expected in cases.items():
            self.assertEqual(classify_shadow_outcome(*values, stable), expected)


if __name__ == "__main__":
    unittest.main()
