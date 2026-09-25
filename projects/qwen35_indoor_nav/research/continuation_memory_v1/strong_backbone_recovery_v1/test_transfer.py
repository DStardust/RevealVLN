import copy
import unittest
from transfer_audit import audit_pair
from transfer_runtime import DenseRuntime
from memory_v2 import ExecutionMemory
import torch


def trace():
    return [dict(event='reset',rgb_sha256='initial'),
        dict(event='generation',environment_step=0,input={'ids':'x','rgb':'initial'},native_logits=[0.,2.,1.,0.],generated_ids=[[10]]),
        dict(event='action',step=1,executed_action=1,before_rgb_sha256='initial',after_rgb_sha256='next'),
        dict(event='generation',environment_step=1,input={'ids':'y','rgb':'next'},native_logits=[2.,0.,1.,0.],generated_ids=[[11]]),
        dict(event='action',step=2,executed_action=0,before_rgb_sha256='next',after_rgb_sha256='next')]


class TransferTests(unittest.TestCase):
    def test_complete_zero(self):
        self.assertTrue(audit_pair(trace(),trace(),{'success':1},{'success':1},zero=True)['full_trajectory_matched'])

    def test_finite_delta_is_separate_from_actions(self):
        changed=trace();changed[1]['native_logits'][1]+=0.001
        result=audit_pair(trace(),changed,{}, {})
        self.assertFalse(result['logits_bitwise_equal']);self.assertEqual(result['argmax_flip_count'],0)

    def test_first_intervention_input_is_checked(self):
        changed=trace();changed[2]['executed_action']=2
        self.assertEqual(audit_pair(trace(),changed,{}, {})['first_executed_override_step'],0)
        changed[1]['input']['ids']='corrupt'
        with self.assertRaisesRegex(ValueError,'PROCESSED_PREFIX'):audit_pair(trace(),changed,{}, {})

    def test_native_flip_rejected(self):
        changed=trace();changed[1]['native_logits']=[0.,1.,2.,0.]
        with self.assertRaisesRegex(ValueError,'ARGMAX_FLIP'):audit_pair(trace(),changed,{}, {})

    def test_transition_and_terminal_rejected(self):
        changed=trace();changed[2]['after_rgb_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'TRANSITION'):audit_pair(trace(),changed,{}, {})
        with self.assertRaisesRegex(ValueError,'TERMINAL'):audit_pair(trace(),trace(),{'success':0},{'success':1})

    def test_stop_does_not_update(self):
        runtime=object.__new__(DenseRuntime)
        with self.assertRaisesRegex(ValueError,'STOP_MUST_NOT'):runtime.observe(None,0)

    def test_runtime_matches_training_unroll_and_reset(self):
        torch.manual_seed(42);head=ExecutionMemory(16)
        sequence=torch.randn(2,5,16);actor=torch.randn(2,5,16)
        full=head(sequence,torch.tensor([5,5]),actor_features=actor)
        memory=head.reset(2)
        for i in range(5):
            memory=head.update(sequence[:,i],memory)
            self.assertTrue(torch.equal(memory,full['memory'][:,i]))
            self.assertTrue(torch.equal(head.action_delta(actor[:,i],memory),full['delta'][:,i]))
        self.assertEqual(head.reset(2).count_nonzero(),0)
        self.assertEqual(head.action_delta(actor[:,0],memory).count_nonzero(),0)


if __name__=='__main__':unittest.main()

