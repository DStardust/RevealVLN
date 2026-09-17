"""CPU-only contract tests. They do not certify an integrated GPU evaluator."""
import base64
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load('v4_cpu_audit', HERE / 'audit.py')
cycle = load('v4_frozen_cycle', HERE / 'cycle_policy.py')
common = load('v4_frozen_common', HERE.parent / 'r2r_ce_tiny_v1/common.py')
metrics = load('v4_frozen_metrics', HERE.parent / 'r2r_ce_tiny_v1/metrics.py')
RGB = bytes([63]) * (224 * 224 * 3)
PAYLOAD = dict(done=False, rgb=base64.b64encode(RGB).decode())


def decision(step=1, override=False):
    original = dict(instruction='Walk to the doorway.',
                    rgb_sha256=[hashlib.sha256(RGB).hexdigest()], executed=[])
    return dict(step=step, original_input=original, input_key=audit.raw_key(original),
                processed_sha256={k: hashlib.sha256(k.encode()).hexdigest()
                                  for k in audit.TENSOR_FIELDS},
                logits=[4., 2., 1., -1.], native_action='move_forward',
                executed_action='turn_left' if override else 'move_forward', override=override)


class ContractTests(unittest.TestCase):
    def test_episode_reset_does_not_share_memory(self):
        one, two = cycle.CycleRecovery(), cycle.CycleRecovery()
        args = ('go', [RGB], [], [4., 2., 1., -1.])
        self.assertEqual(one.choose(*args)[0], 'move_forward')
        self.assertEqual(one.choose(*args)[0], 'turn_left')
        self.assertEqual(two.choose(*args)[0], 'move_forward')
        one.reset()
        self.assertEqual(one.choose(*args)[0], 'move_forward')
        window = common.Window()
        window.receive(dict(PAYLOAD, instruction='one'))
        window.receive(PAYLOAD, executed='turn_left')
        window.receive(dict(PAYLOAD, instruction='two'))
        self.assertEqual((window.instruction, window.executed, window.images), ('two', [], [RGB]))

    def test_unrepeated_input_is_native_argmax(self):
        for values in ([0., 3., 2., 1.], [2., 2., 1., 0.], [0., 1., 2., 3.]):
            selected, row = cycle.CycleRecovery().choose('go', [RGB], [], values)
            self.assertEqual(selected, cycle.ACTIONS[max(range(4), key=values.__getitem__)])
            self.assertFalse(row['cycle_override'])

    def test_native_stop_never_overridden(self):
        policy = cycle.CycleRecovery()
        policy.choose('go', [RGB], [], [4., 2., 1., -1.])
        for _ in range(5):
            selected, row = policy.choose('go', [RGB], [], [1., 1., 1., 9.])
            self.assertEqual(selected, 'STOP')
            self.assertFalse(row['cycle_override'])

    def test_least_count_then_logit_then_original_index(self):
        policy = cycle.CycleRecovery()
        args = ('go', [RGB], [], [4., 2., 2., -1.])
        self.assertEqual([policy.choose(*args)[0] for _ in range(6)],
                         ['move_forward', 'turn_left', 'turn_right'] * 2)
        policy.reset()
        args = ('go', [RGB], [], [4., 1., 2., -1.])
        self.assertEqual([policy.choose(*args)[0] for _ in range(3)],
                         ['move_forward', 'turn_right', 'turn_left'])

    def test_actual_override_written_to_history(self):
        window = common.Window()
        window.receive(dict(PAYLOAD, instruction='go'))
        policy = cycle.CycleRecovery()
        args = (window.instruction, window.images, window.executed, [4., 2., 1., -1.])
        policy.choose(*args)
        selected, record = policy.choose(*args)
        self.assertTrue(record['cycle_override'])
        window.receive(PAYLOAD, executed=selected)
        self.assertEqual(window.executed, ['turn_left'])
        for _ in range(10):
            window.receive(PAYLOAD, executed='turn_right')
        self.assertEqual(window.executed, ['turn_right'] * 8)

    def test_500_decisions_include_stop_and_recovery(self):
        count, done = 0, False
        for _ in range(499):
            count, done = common.advance('turn_left', count, done)
        self.assertEqual((count, done), (499, False))
        self.assertEqual(common.advance('STOP', count, done), (500, True))
        self.assertEqual(common.advance('turn_right', count, done), (500, True))
        with self.assertRaises(ValueError):
            common.advance('move_forward', 500, True)

    def test_official_success_requires_stop_and_strict_distance(self):
        class Sim:
            distance = 2.
            position = [0., 0., 0.]

            def get_agent_state(self):
                return SimpleNamespace(position=metrics.np.asarray(self.position))

            def geodesic_distance(self, position, goals, episode):
                return self.distance

        sim = Sim()
        measures = metrics.OfficialMetrics(sim, [1., 0., 0.])
        self.assertEqual(measures.update(False)['success'], 0.)
        self.assertEqual(measures.update(True)['success'], 1.)
        sim.distance = 3.
        sim.position = [1., 0., 0.]
        self.assertEqual(measures.update(True)['success'], 0.)

    def test_timeout_or_missing_episode_cannot_admit_sr(self):
        for status, indices in [('RESOURCE_CENSORED', list(range(199))),
                                ('COMPLETE', list(range(199))),
                                ('COMPLETE', list(range(199)) + [198])]:
            with self.assertRaisesRegex(ValueError, 'NO_ADMITTED_METRICS'):
                audit.require_complete(status, indices, True, True)
        result = audit.blocked_result('No allocation')
        self.assertIsNone(result['delta_sr'])
        self.assertIsNone(result['native'])
        self.assertEqual(result['intervention_benefit'], 'unknown')

    def test_first_difference_fails_even_if_argmax_matches(self):
        a, b = decision(), decision()
        b['logits'][1] += .000001
        with self.assertRaises(audit.NumericInvalid) as raised:
            audit.check_pair([a], [b], {}, {})
        self.assertEqual(raised.exception.evidence['status'], 'NUMERIC_OR_TRANSPORT_INVALID')
        self.assertEqual(raised.exception.evidence['recovery']['step'], 1)

    def test_override_decision_itself_must_match(self):
        a, b = decision(), decision(override=True)
        self.assertEqual(audit.check_pair([a], [b], None, None)['first_override'], 1)
        b['processed_sha256']['pixel_values'] = '0' * 64
        with self.assertRaises(audit.NumericInvalid):
            audit.check_pair([a], [b], None, None)

    def test_no_override_requires_entire_trajectory_and_terminal(self):
        a = [decision(), decision(2)]
        terminal = dict(termination='MAX_STEPS', success=0.)
        self.assertIsNone(audit.check_pair(a, copy.deepcopy(a), terminal, terminal)['first_override'])
        for b, end in [(a[:1], terminal), (a, dict(terminal, success=1.)), (a, None)]:
            with self.assertRaises(audit.NumericInvalid):
                audit.check_pair(a, b, terminal, end)

    def test_all_seven_preprocessed_tensors_are_required(self):
        for field in audit.TENSOR_FIELDS:
            b = decision()
            del b['processed_sha256'][field]
            with self.assertRaises(audit.NumericInvalid):
                audit.check_prefix_decision(decision(), b)

    def test_raw_inputs_and_history_must_match(self):
        b = decision()
        b['original_input']['executed'] = ['turn_left']
        b['input_key'] = audit.raw_key(b['original_input'])
        with self.assertRaises(audit.NumericInvalid):
            audit.check_prefix_decision(decision(), b)

    def test_changed_parameters_cannot_admit_results(self):
        with self.assertRaises(audit.NumericInvalid):
            audit.require_complete('COMPLETE', list(range(200)), True, False)

    def test_cleanup_refuses_nonowner_or_reused_pid(self):
        initial = dict(pid=101, ppid=100, pgid=101, sid=101, start_ticks=50)
        audit.require_owned(initial, dict(initial), [dict(initial)], 100)
        for current, members, owner in [
            (dict(initial), [dict(initial)], 999),
            (dict(initial, start_ticks=51), [dict(initial)], 100),
            (None, [dict(initial)], 100),
            (dict(initial), [dict(initial, sid=999)], 100),
        ]:
            with self.assertRaises(ValueError):
                audit.require_owned(initial, current, members, owner)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if not args.attempt.isdigit():
        raise ValueError('Use a new numbered CPU attempt')
    out = HERE / 'cpu_attempts' / ('attempt_' + args.attempt)
    out.mkdir(exist_ok=False)
    for name in ('test_cpu.py', 'audit.py'):
        with (out / name).open('xb') as stream:
            stream.write((HERE / name).read_bytes())
    with (out / 'CPU_TEST_LOG.txt').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  scope='Frozen controller/window/official success plus new CPU audit contract',
                  integrated_gpu_evaluator_validated=False, gpu_started=False,
                  fit_replay_repeated=False, learnability_repeated=False)
    with (out / 'CPU_TEST_RESULT.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)
