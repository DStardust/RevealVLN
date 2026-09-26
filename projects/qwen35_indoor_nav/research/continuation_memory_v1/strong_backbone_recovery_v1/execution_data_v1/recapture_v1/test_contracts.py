"""Synthetic CPU contract fixtures, never evidence of a real model forward."""
import copy
import io
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from contracts import validate_chunk, admit_trajectory, summarize_coverage
from recapture_plan import requests_from_rows
from token_index import action_rows


def fixture():
    trace = dict(id=3, partition='FIT', kind='RECOVERY', cutoff=4,
        actions=[1, 2, 1, 3, 1, 2, 0], query_steps=[0, 4], rgb_sha256=list('abcdefg'))
    plans = requests_from_rows(list(action_rows(trace, 0)))
    return trace, plans


def capture(trace, plan, **changes):
    actions = [a['executed_action'] for a in plan['actions']]
    args = dict(header_ids=[90, 91], action_token_ids=[10, 11, 12, 13], eos_token_id=99,
        context_query_start=plan['query_start_step'], memory_replay_exclusive_end=plan['query_start_step'] + 1,
        actor_width=3)
    call = dict(generated_ids=[90, 91] + [10 + action for action in actions] + [99],
        actor_rows=[[0.1, 0.2, 0.3] for _ in actions], native_rows=[[0.1] * 4 for _ in actions])
    for key, value in changes.items():
        if key in call:
            call[key] = value
        else:
            args[key] = value
    return validate_chunk(trace, plan, **call, **args)


class ContractTests(unittest.TestCase):
    def test_header_action_rows_and_eos_alignment(self):
        trace, plans = fixture()
        value = capture(trace, plans[0])
        self.assertEqual(value['captured_actor_rows'], 4)
        self.assertEqual([r['generated_token_index'] for r in value['token_rows']], [2, 3, 4, 5])
        self.assertEqual(value['accepted_generated_ids'], [90, 91, 11, 12, 11, 13, 99])
        with self.assertRaises(ValueError):
            capture(trace, plans[0], generated_ids=[91, 90, 11, 12, 11, 13, 99])

    def test_no_unexecuted_action_or_eos_actor_rows(self):
        trace, plans = fixture()
        with self.assertRaises(ValueError):
            capture(trace, plans[1], generated_ids=[90, 91, 11, 12, 10, 99, 11])
        with self.assertRaises(ValueError):
            capture(trace, plans[1], actor_rows=[[0.1] * 3] * 4)
        value = capture(trace, plans[1], generated_ids=[90, 91, 11, 12, 10, 99, 99])
        self.assertEqual(value['ignored_eos_pad_suffix'], 1)

    def test_later_physical_memory_is_rejected(self):
        trace, plans = fixture()
        with self.assertRaises(ValueError):
            capture(trace, plans[0], memory_replay_exclusive_end=4)
        bad = copy.deepcopy(plans[0])
        bad['actions'][2]['allowed_memory_feature_index'] = 2
        with self.assertRaises(ValueError):
            capture(trace, bad)

    def test_unknown_mask_remains_unknown(self):
        trace, plans = fixture()
        value = capture(trace, plans[0])
        self.assertTrue(all(not row['original_action_in_registered_supervised_region'] for row in value['token_rows']))
        bad = copy.deepcopy(plans[0])
        bad['actions'][1]['original_existing_actor_supervision_known'] = True
        with self.assertRaises(ValueError):
            capture(trace, bad)

    def test_stop_has_no_observation_or_extra_budget(self):
        trace, plans = fixture()
        self.assertIsNone(capture(trace, plans[1])['token_rows'][-1]['physical_next_feature_index_audit_only'])
        bad = copy.deepcopy(plans[1])
        bad['actions'][-1]['physical_next_feature_index_audit_only'] = 7
        with self.assertRaises(ValueError):
            capture(trace, bad)
        bad_trace = dict(trace, actions=[1] * 500 + [0], rgb_sha256=['a'] * 501)
        with self.assertRaises(ValueError):
            capture(bad_trace, plans[0])

    def test_nonfinite_actor_or_native_is_rejected(self):
        trace, plans = fixture()
        with self.assertRaises(ValueError):
            capture(trace, plans[0], actor_rows=[[float('nan')] * 3] * 4)
        with self.assertRaises(ValueError):
            capture(trace, plans[0], native_rows=[[float('inf')] * 4] * 4)

    def test_complete_replay_group_and_partial_denominator(self):
        trace, plans = fixture()
        chunks = [capture(trace, plan) for plan in plans]
        with self.assertRaises(ValueError):
            admit_trajectory(trace, plans, chunks[:1], actual_actions=trace['actions'], actual_rgb_sha256=trace['rgb_sha256'])
        with self.assertRaises(ValueError):
            admit_trajectory(trace, plans, chunks, actual_actions=trace['actions'], actual_rgb_sha256=list('xxxxxxx'))
        group = admit_trajectory(trace, plans, chunks, actual_actions=trace['actions'], actual_rgb_sha256=trace['rgb_sha256'])
        self.assertEqual(group['new_actor_positions'], 5)
        self.assertEqual(group['new_terminal_stop_positions'], 1)
        self.assertEqual(group['new_positions_in_unknown_prefix'], 3)
        self.assertEqual(summarize_coverage(plans, [])['status'], 'INCOMPLETE')
        self.assertEqual(summarize_coverage(plans, [group])['status'], 'COMPLETE')
        with self.assertRaises(ValueError):
            summarize_coverage(plans, [group, group])

    def test_real_frozen_plan_denominator_without_model(self):
        plans = json.loads((HERE.parent / 'recapture_plan_001/REQUESTS.json').read_text())
        summary = summarize_coverage(plans, [])
        self.assertEqual(summary['totals']['planned_missing_actor_positions'], 21396)
        self.assertEqual(summary['totals']['planned_missing_stop_positions'], 133)
        self.assertEqual(summary['totals']['admitted_trajectories'], 0)
        self.assertEqual(summary['by_split_kind']['FIT:PRESERVATION']['planned_missing_stop_positions'], 102)
        self.assertEqual(summary['by_split_kind']['DEV:PRESERVATION']['planned_missing_stop_positions'], 31)


if __name__ == '__main__':
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    text = output.getvalue()
    print(text, end='')
    (HERE / 'CPU_TEST_RESULT.json').write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        scope='Synthetic contract fixtures plus real immutable plan denominators; no real model forward or simulator replay.',
        gpu_hours=0, optimizer_updates=0, actual_actor_features_captured=0, log=text), indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
