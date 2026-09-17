import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fractions import Fraction
import unittest
from language import aliases, ast, parse, realize
from method import reorient_histories


ROLES = {
    'anchor_A': dict(mpcat40='chair', room='dining room', raw_match=dict(mode='exact', value='dining chair')),
    'anchor_B': dict(mpcat40='tv_monitor', room='bedroom', raw_match=dict(mode='exact', value='tv')),
    'terminal': dict(mpcat40='sink', room='kitchen', raw_match=dict(mode='exact', value='sink')),
}
# Category spelling is read from the registered finite vocabulary in setUp.
TASKS = {'task_A':dict(anchor='anchor_A', terminal='terminal', instruction='source'),
         'task_B':dict(anchor='anchor_B', terminal='terminal', instruction='source')}


class LanguageTests(unittest.TestCase):
    def setUp(self):
        from language import base
        self.roles = copy.deepcopy(ROLES)
        category = next(k for k, values in base.planner.RAW.items() if 'tv' in values)
        self.roles['anchor_B']['mpcat40'] = category

    def test_all_roundtrips_and_immutability(self):
        original = copy.deepcopy(TASKS)
        texts = set()
        for variant in range(5):
            rows = realize(self.roles, TASKS, variant)
            for name, task in rows.items():
                self.assertEqual(parse(task['instruction'], self.roles), ast(self.roles, TASKS[name]))
                texts.add(task['instruction'])
        self.assertEqual(len(texts), 10)
        self.assertEqual(TASKS, original)

    def test_semantic_counterexamples(self):
        text = realize(self.roles, TASKS, 0)['task_A']['instruction']
        for bad in [text.replace('two consecutive', 'two'), text.replace('two consecutive', 'one'),
                    text.replace('dining chair', 'chair'), text.replace('the sink', 'the nearest sink'),
                    text.replace('stop immediately', 'continue moving'), text.replace('First', 'Finally')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse(bad, self.roles)

    def test_role_swap_is_not_equivalent(self):
        text = realize(self.roles, TASKS, 0)['task_A']['instruction']
        swap = text.replace('the dining chair in the dining room', 'PLACEHOLDER').replace(
            'the sink in the kitchen', 'the dining chair in the dining room').replace('PLACEHOLDER', 'the sink in the kitchen')
        self.assertNotEqual(parse(swap, self.roles), ast(self.roles, TASKS['task_A']))

    def test_group_weights(self):
        rows = aliases(self.roles, TASKS, 'one-physical-family')
        self.assertEqual(sum(Fraction(x['relative_family_weight']) for x in rows), 1)
        self.assertEqual({x['physical_family_group'] for x in rows}, {'one-physical-family'})
        self.assertFalse(any(x['training_admission'] for x in rows))

    def test_registered_raw_vocabulary(self):
        from language import base
        for category, raws in base.planner.RAW.items():
            for raw in raws:
                roles = copy.deepcopy(self.roles)
                roles['anchor_A'] = dict(mpcat40=category, room='dining room', raw_match=dict(mode='exact', value=raw))
                for variant in range(5):
                    realize(roles, TASKS, variant)


class PathTests(unittest.TestCase):
    def test_yaw_transforms_keep_count_match(self):
        source = dict(yaw_bin=0, public_tail='LRLRLRLR', histories={
            'H_A':list('FLRF')+list('LRLRLRLR'),
            'H_B':list('RFFL')+list('LRLRLRLR'),
            'H_A_I':list('LFFR')+list('LRLRLRLR')})
        for yaw in (0, 6, 12, 18):
            rows = reorient_histories(source, yaw)
            self.assertEqual(len({tuple(v) for v in rows.values()}), 3)
            self.assertEqual(len({len(v) for v in rows.values()}), 1)


if __name__ == '__main__':
    unittest.main()

