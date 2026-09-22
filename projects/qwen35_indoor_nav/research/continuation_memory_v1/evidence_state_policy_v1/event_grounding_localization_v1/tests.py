"""Check local score attribution and exact causal cache lookup boundaries."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import decomposition, input_signature, replay, action, write, HERE


class Checks(unittest.TestCase):
    def test_core_only_flip_is_not_attributed_to_state(self):
        left=dict(predicted_state=[1,1,1,1],method_logits=[0,0,2,5],executed_action='STOP')
        right=dict(predicted_state=[1,1,1,1],method_logits=[0,0,5,2],executed_action='turn_right')
        weights={a:torch.eye(4,dtype=torch.float64) for a in ('ORIGINAL','EVENT')}
        value=decomposition(left,right,weights)
        self.assertEqual(value['margin_delta']['state_branch'],0)
        self.assertTrue(value['core_only_restores_original'])
        self.assertFalse(value['state_only_restores_original'])

    def test_state_only_flip_has_zero_core_delta(self):
        left=dict(predicted_state=[1,1,1,1],method_logits=[0,0,2,3],executed_action='STOP')
        right=dict(predicted_state=[1,1,0,0],method_logits=[0,0,2,0],executed_action='turn_right')
        weight=torch.zeros(4,4,dtype=torch.float64);weight[3,3]=3
        value=decomposition(left,right,{a:weight for a in ('ORIGINAL','EVENT')})
        self.assertAlmostEqual(value['margin_delta']['core'],0)
        self.assertTrue(value['state_only_restores_original'])
        self.assertFalse(value['core_only_restores_original'])

    def test_lookup_matches_fields_not_hash_namespace(self):
        row=dict(instruction='task',rgb_sha256=['frame1','frame2'],executed_history=['turn_left'],input_key='runtime-hash')
        other=dict(row,input_key='cache-hash')
        self.assertEqual(input_signature(row),input_signature(other))
        self.assertNotEqual(input_signature(row),input_signature(dict(row,executed_history=['turn_right'])))
        self.assertNotEqual(input_signature(row),input_signature(dict(row,rgb_sha256=['frame2','frame1'])))

    def test_missing_causal_step_cannot_be_filled_by_future_or_partial_prefix(self):
        row=dict(raw=dict(instruction='task',rgb_sha256=['x'],executed_history=[]))
        value=replay([row],{}, {}, {}, {}, [], {})
        self.assertEqual(value['status'],'UNAVAILABLE_CAUSAL_FEATURES')
        self.assertEqual(value['missing_steps'],[0])

    def test_nonfinite_action_rejected(self):
        with self.assertRaises(ValueError):action(torch.tensor([0.,float('nan'),0.,0.]))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    write(HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        success=result.wasSuccessful(),cuda_initialized=torch.cuda.is_initialized(),optimizer_updates=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)
