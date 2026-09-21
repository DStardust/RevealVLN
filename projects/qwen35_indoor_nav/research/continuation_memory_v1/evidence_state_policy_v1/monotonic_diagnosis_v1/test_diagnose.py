import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import confusion,stop_failure,first_stationary_forward

class DiagnosisTests(unittest.TestCase):
    def test_empty_and_thresholds(self):
        self.assertIsNone(confusion([])['brier'])
        r=confusion([(.6,1),(.6,0),(.4,1),(.4,0)])
        self.assertEqual([r[k] for k in ('tp','fp','fn','tn')],[1]*4)
        with self.assertRaises(ValueError):confusion([(float('nan'),0)])
    def test_failure_types_keep_collision_separate(self):
        base=dict(safe_v16_label='FAIL',stopped=True,budget_exhausted=False,collisions=3)
        self.assertEqual(stop_failure(base,[0,0,1,0]),'STOP_WITHOUT_PRIOR_ANCHOR')
        self.assertEqual(stop_failure(base,[1,1,0,0]),'STOP_WITHOUT_CURRENT_TERMINAL')
        self.assertEqual(stop_failure(base,[1,1,1,1]),'READY_STOP_WITH_COLLISION')
        self.assertEqual(stop_failure(dict(base,stopped=False,budget_exhausted=True),[1,1,1,1]),'EXHAUSTED')
    def test_stationary_proxy_not_rotation_or_stop(self):
        trace=dict(actions=['L','F','S'],observations=[dict(pose=dict(position=[0,0,0])) for _ in range(3)])
        self.assertEqual(first_stationary_forward(trace,0),1)
        self.assertIsNone(first_stationary_forward(trace,2))
        trace['observations'][2]['pose']['position']=[.25,0,0]
        self.assertIsNone(first_stationary_forward(trace,0))

if __name__=='__main__':unittest.main()
