"""Small CPU tests for the matched architecture interface; no navigation run."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
from model import ExecutionAdaptation, ActionOutcomeMemory, START, STOP

torch.set_num_threads(2)
EVIDENCE = {}


def rollout(model, features, actions, actor_feature, native):
    memory = model.reset()
    states, logits = [], []
    for step in range(len(features)):
        previous = None if step == 0 else features[step - 1:step]
        action = torch.tensor([START]) if step == 0 else actions[step - 1:step]
        memory = model.update(features[step:step + 1], memory, previous, action)
        states.append(memory)
        logits.append(native + model.action_delta(actor_feature, memory))
    return torch.cat(states), torch.cat(logits)


class ModelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1209)
        self.models = {mode: ExecutionAdaptation(16, mode) for mode in ('DELTA', 'CURRENT')}
        self.models['CURRENT'].load_state_dict(self.models['DELTA'].state_dict())
        self.features = torch.randn(32, 16)
        self.actions = torch.ones(32, dtype=torch.long)
        self.actions[-1] = STOP
        self.actor_feature = torch.randn(1, 16)
        self.native = torch.randn(1, 4)

    def open_actor(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _, logits = rollout(model, self.features[:4], self.actions[:4], self.actor_feature, self.native)
        F.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def test_identical_keys_shapes_and_shared_initialization(self):
        left, right = (self.models[mode].state_dict() for mode in ('DELTA', 'CURRENT'))
        self.assertEqual(list(left), list(right))
        self.assertTrue(all(torch.equal(left[key], right[key]) for key in left))
        counts = {mode: sum(p.numel() for p in model.parameters()) for mode, model in self.models.items()}
        self.assertEqual(counts['DELTA'], counts['CURRENT'])
        EVIDENCE['same_state_dict_keys_shapes_and_values'] = True

    def test_zero_actor_preserves_native(self):
        for mode, model in self.models.items():
            states, logits = rollout(model, self.features, self.actions, self.actor_feature, self.native)
            self.assertEqual(tuple(states.shape), (32, 8, 64))
            self.assertTrue(torch.equal(logits, self.native.expand_as(logits)), mode)

    def test_delta_matches_original_update(self):
        original = ActionOutcomeMemory(16)
        original.load_state_dict(self.models['DELTA'].state_dict())
        expected = rollout(original, self.features, self.actions, self.actor_feature, self.native)[0]
        actual = rollout(self.models['DELTA'], self.features, self.actions, self.actor_feature, self.native)[0]
        self.assertTrue(torch.equal(expected, actual))

    def test_current_projection_uses_current_input(self):
        x, old, action = self.features[:1], self.features[1:2], torch.tensor([1])
        current = self.models['CURRENT']
        a = current.update(x, current.reset(), old, action)
        b = current.update(x, current.reset(), old + 5, action)
        self.assertTrue(torch.equal(a, b))
        expected = .01 * torch.tanh(current.writer(current.norm(x))
            + current.recurrent(current.reset().flatten(1))
            + current.change_writer(x) + current.executed_action(action))
        self.assertTrue(torch.equal(a.flatten(1), expected))
        with torch.no_grad():
            current.change_writer.weight.zero_()
        c = current.update(x, current.reset(), old, action)
        self.assertFalse(torch.equal(a, c))

    def test_start_reset_stop_and_episode_isolation(self):
        for mode, model in self.models.items():
            first = rollout(model, self.features, self.actions, self.actor_feature, self.native)[0]
            rollout(model, self.features + 7, self.actions, self.actor_feature, self.native)
            repeated = rollout(model, self.features, self.actions, self.actor_feature, self.native)[0]
            self.assertTrue(torch.equal(first, repeated), mode)
            with self.assertRaisesRegex(ValueError, 'START_REQUIRES_RESET'):
                model.update(self.features[:1], first[:1], None, torch.tensor([START]))
            with self.assertRaisesRegex(ValueError, 'STOP_HAS_NO_NEXT_OBSERVATION'):
                model.update(self.features[:1], model.reset(), self.features[:1], torch.tensor([STOP]))
            with self.assertRaisesRegex(ValueError, 'MOTION_REQUIRES_PREVIOUS'):
                model.update(self.features[:1], model.reset(), None, torch.tensor([1]))
            with self.assertRaisesRegex(ValueError, 'INVALID_PREVIOUS_ACTION'):
                model.update(self.features[:1], model.reset(), None, torch.tensor([5]))
            mixed = model.reset(2)
            mixed[1] = first[4]
            actual = model.update(self.features[:2], mixed, self.features[2:4], torch.tensor([START, 1]))
            reset = model.update(self.features[:1], model.reset(), None, torch.tensor([START]))
            torch.testing.assert_close(actual[:1], reset)

    def test_both_writers_have_effective_gradients_across_time(self):
        evidence = {}
        for mode, model in self.models.items():
            self.open_actor(model)
            features = torch.randn(96, 16, requires_grad=True)
            actions = torch.ones(96, dtype=torch.long)
            actions[-1] = STOP
            outputs = []
            def record(_module, _inputs, output):
                output.retain_grad()
                outputs.append(output)
            handle = model.change_writer.register_forward_hook(record)
            _, logits = rollout(model, features, actions, self.actor_feature, self.native)
            handle.remove()
            F.cross_entropy(logits[-1:], torch.tensor([2])).backward()
            gradients = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()
                if name in ('writer.weight', 'change_writer.weight', 'executed_action.weight', 'recurrent.weight')}
            self.assertTrue(all(value > 0 for value in gradients.values()), mode)
            self.assertGreater(float(features.grad[0].norm()), 0, mode)
            self.assertGreater(float(outputs[1].grad.norm()), 0, mode)
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            evidence[mode] = dict(sequence_length=96, loss_step=95, gradient_norms=gradients,
                first_feature_gradient_l2=float(features.grad[0].norm()),
                second_write_output_gradient_l2=float(outputs[1].grad.norm()))
        EVIDENCE['long_temporal_gradient'] = evidence

    def test_future_causality_with_nonzero_actor(self):
        evidence = {}
        for mode, model in self.models.items():
            self.open_actor(model)
            states, logits = rollout(model, self.features, self.actions, self.actor_feature, self.native)
            residual = float((logits - self.native).detach().abs().max())
            self.assertGreater(residual, 0, mode)
            changed_features = self.features.clone()
            changed_features[9:] += 4
            changed_actions = self.actions.clone()
            changed_actions[9:-1] = 2
            later_states, later_logits = rollout(model, changed_features, changed_actions, self.actor_feature, self.native)
            self.assertTrue(torch.equal(states[:9], later_states[:9]), mode)
            self.assertTrue(torch.equal(logits[:9], later_logits[:9]), mode)
            self.assertFalse(torch.equal(states[9:], later_states[9:]), mode)
            evidence[mode] = dict(actor_residual_max_abs=residual, changed_from_step=9,
                earlier_states_bitwise_equal=True, earlier_logits_bitwise_equal=True)
        EVIDENCE['future_causality'] = evidence


if __name__ == '__main__':
    began = time.time()
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(ModelContracts))
    paths = [Path(__file__).resolve(), Path(__file__).resolve().with_name('model.py')]
    print(json.dumps(dict(status='CPU_MODEL_CONTRACT_COMPLETE' if result.wasSuccessful() else 'CPU_MODEL_CONTRACT_FAILED',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        gpu_hours=0, torch_threads=torch.get_num_threads(), navigation_benefit='NOT_MEASURED',
        evidence=EVIDENCE, wall_seconds=time.time() - began,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}), indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)
