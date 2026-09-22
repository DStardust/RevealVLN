"""Tests for false-history attribution and descriptive failure categories."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import history_origin,stage_failure,confusion,write,HERE

class Tests(unittest.TestCase):
    def test_initial_error_distinguished_from_accumulation(self):
        truth=[[0,0,0,0]]*3;events=[[0,0]]*3
        a=history_origin([[.8,.8,0,0]]*3,truth,events,events,2)
        b=history_origin([[.1,.2,0,0],[.2,.4,0,0],[.4,.6,0,0]],truth,events,events,2)
        self.assertEqual(a['false_history_origin'],'initial_prior')
        self.assertEqual(b['false_history_origin'],'event_accumulation')
    def test_recognized_then_retained(self):
        z=[[.1,.1,0,0],[.1,.8,0,0],[.8,.9,0,0]];g=[[0,0,0,0],[0,1,0,0],[1,1,0,0]]
        e=[[0,0],[.8,0],[0,0]];y=[[0,0],[1,0],[0,0]]
        result=history_origin(z,g,e,y,2)
        self.assertTrue(result['first_witness_detected']);self.assertFalse(result['detected_then_lost']);self.assertFalse(result['history_missing_at_cutoff'])
    def test_future_witness_not_counted_at_cutoff(self):
        z=[[0,0,0,0],[0,0,0,0],[0,1,0,0]];g=z;e=[[0,0],[0,0],[1,0]]
        result=history_origin(z,g,e,e,1);self.assertIsNone(result['first_witness']);self.assertFalse(result['any_true_witness_detected'])
    def test_correct_state_can_still_wrongly_stop(self):
        task=dict(safe_v16_label='FAIL',stopped=True,budget_exhausted=False,collisions=0)
        self.assertEqual(stage_failure(task,[0,0,1,0],[.1,.1,.9,.09]),'STOP_DESPITE_CORRECT_MISSING_HISTORY')
        self.assertEqual(stage_failure(task,[0,0,1,0],[.9,.9,.9,.81]),'STOP_MISSING_HISTORY_WITH_FALSE_BELIEF')
    def test_empty_confusion_not_perfect_accuracy(self):
        self.assertEqual(confusion([])['n'],0);self.assertIsNone(confusion([])['brier'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,successful=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors)))
    raise SystemExit(not result.wasSuccessful())
