import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('optional_test',HERE/'optional_filters.py')
filters=importlib.util.module_from_spec(spec);spec.loader.exec_module(filters)

class OptionalTests(unittest.TestCase):
    def example(self):
        return dict(closed_loop_A=dict(path='a'),closed_loop_B=dict(path='b'),closed_loop_irrelevant=dict(path='i'),
            roles={k:dict(room=r) for k,r in [('anchor_A','living'),('anchor_B','bedroom'),('terminal','living')]})
    def test_turn_only_not_motion_detour(self):
        flags=filters.flags(self.example(),{'a':['F','L'],'b':['F','R'],'i':list('LRLRLRLR')})
        self.assertFalse(flags['I_actual_closed_loop_has_at_least_2F'])
        self.assertFalse(flags['legacy_A_B_T_three_distinct_rooms'])
    def test_action_copy_not_independent_loop(self):
        flags=filters.flags(self.example(),{'a':['F','F'],'b':['R','F'],'i':['F','F']})
        self.assertTrue(flags['I_actual_closed_loop_has_at_least_2F'])
        self.assertFalse(flags['I_actual_closed_loop_actions_differ_from_A_and_B'])

if __name__=='__main__':unittest.main()
