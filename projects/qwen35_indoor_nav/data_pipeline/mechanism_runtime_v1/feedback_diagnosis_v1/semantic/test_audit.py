import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('failure_audit',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class AuditTests(unittest.TestCase):
    def test_empty_targets_not_missing_role(self):
        out=audit.summarize_target_calls([dict(role='a',targets=[]),dict(role='a',targets=[1])])
        self.assertEqual(out['a'],dict(calls=2,empty=1,nonempty=1,accepted_targets=1))
        self.assertNotIn('b',out)
    def test_zero_action_is_not_navigation(self):
        trace=dict(actions=[],observations=[dict(pose=dict(position=[0,0,0]))],complete=True,
            collisions=0,initial_position=[0,0,0],initial_yaw_bin=0)
        result=audit.action_stats(trace)
        self.assertEqual(result['length'],0)
        self.assertEqual(result['reset_position_error'],0)
    def test_missing_observation_fails_closed(self):
        with self.assertRaises(AssertionError):audit.action_stats(dict(actions=['F'],observations=[]))
    def test_geometry_does_not_infer_navmesh(self):
        result=audit.geometry([dict(u_position=[0,0,0])],dict(source_positions=[[0,0,0]],shared_metadata_levels=[1]),
            dict(objects={'1':dict(center=[10,1,0],region_id='1_2')},eligible={'anchor_A':[1]}))
        self.assertFalse(result['global_reachability_or_wrong_floor_proven'])
        self.assertTrue(result['starts'][0]['roles']['anchor_A']['all_unsnapped_ring_points_beyond_8m'])

if __name__=='__main__':unittest.main()
