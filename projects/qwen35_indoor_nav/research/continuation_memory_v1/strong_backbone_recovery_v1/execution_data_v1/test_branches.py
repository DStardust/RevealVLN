import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_branches import prefix_record, select_prefix_sources


def example():
    entry = dict(id=0, house='house', partition='FIT', route_family='route',
        variant='natural', episode_id=1, trajectory_id=1, instruction='go')
    trace = [dict(event='reset', rgb_sha256='rgb0',
                  instruction_sha256=hashlib.sha256(b'go').hexdigest())]
    for step in range(5):
        if step in (0, 4):
            trace.append(dict(event='generation', environment_step=step,
                              input=dict(rgb_sha256='rgb' + str(step))))
        trace.append(dict(event='action', step=step + 1, executed_action=1 if step < 4 else 0,
            before_rgb_sha256='rgb' + str(step),
            after_rgb_sha256='rgb' + str(min(step + 1, 4))))
    return entry, trace


class PrefixTests(unittest.TestCase):
    def test_source_selection_ignores_scores_and_order(self):
        rows = [dict(id=i, house='h', partition='FIT', route_family=str(i // 2),
                     variant=('natural', 'turn_prefix')[i % 2]) for i in range(12)]
        before = [e['id'] for e in select_prefix_sources(rows)]
        for e in rows:
            e['candidate_success'] = e['id'] % 2
        self.assertEqual(before, [e['id'] for e in select_prefix_sources(rows[::-1])])
        self.assertEqual(len({e['route_family'] for e in select_prefix_sources(rows)}), 2)

    def test_house_leak_rejected(self):
        rows = [dict(id=i, house='h', partition=('FIT', 'DEV')[i],
                     route_family=str(i), variant='natural') for i in range(2)]
        with self.assertRaisesRegex(ValueError, 'HOUSE_SPLIT_LEAK'):
            select_prefix_sources(rows)

    def test_unseen_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ONLY_OFFICIAL_TRAIN'):
            select_prefix_sources([dict(id=0, partition='UNSEEN')])

    def test_terminal_query_is_pre_stop_and_counts_budget(self):
        entry, trace = example()
        p = prefix_record(entry, trace)
        self.assertEqual(p['prefix_actions'], [1] * 4)
        self.assertEqual(p['prefix_rgb_sha256'], ['rgb' + str(i) for i in range(5)])
        self.assertEqual(p['remaining_decisions'], 496)
        self.assertFalse(p['training_admission'])
        self.assertEqual(p['status'], 'PLANNED_NOT_EXECUTED')

    def test_broken_rgb_or_instruction_rejected(self):
        entry, trace = example()
        wrong = copy.deepcopy(trace)
        wrong[-1]['before_rgb_sha256'] = 'broken'
        with self.assertRaisesRegex(ValueError, 'RGB_CHAIN'):
            prefix_record(entry, wrong)
        entry['instruction'] = 'other'
        with self.assertRaisesRegex(ValueError, 'INSTRUCTION'):
            prefix_record(entry, trace)

    def test_action_after_stop_rejected(self):
        entry, trace = example()
        next(r for r in trace if r['event'] == 'action')['executed_action'] = 0
        with self.assertRaisesRegex(ValueError, 'ACTION_AFTER_STOP'):
            prefix_record(entry, trace)

    def test_query_does_not_create_new_observation(self):
        entry, trace = example()
        [r for r in trace if r['event'] == 'generation'][-1]['input']['rgb_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'QUERY_RGB'):
            prefix_record(entry, trace)


if __name__ == '__main__':
    unittest.main()
