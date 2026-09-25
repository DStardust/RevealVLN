import copy
from pathlib import Path
import sys
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parent.parent)]
import torch
from confirmation_model import ConfirmationMemory, sequence_logits
from recovery_model import RecoveryMemory
from confirmation_review import summarize
from transfer_audit import audit_pair


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)

    def heads(self):
        full=ConfirmationMemory(16,'CONCAT');local=ConfirmationMemory(16,'LOCAL')
        with torch.no_grad():full.actor[-1].weight.normal_(0,.1)
        local.load_state_dict(full.state_dict());return full,local

    def test_concat_preserves_previous_equations_and_initialization(self):
        torch.manual_seed(42);old=RecoveryMemory(16,'CONCAT')
        torch.manual_seed(42);new=ConfirmationMemory(16,'CONCAT')
        self.assertTrue(all(torch.equal(t,new.state_dict()[k]) for k,t in old.state_dict().items()))
        x=torch.randn(1,16);m=torch.randn(1,8,64)
        self.assertTrue(torch.equal(old.update(x,m),new.update(x,m)))
        self.assertTrue(torch.equal(old.action_delta(x,m),new.action_delta(x,m)))

    def test_local_is_invariant_to_old_events_but_concat_can_use_them(self):
        full,local=self.heads();x=torch.randn(1,16);past=[torch.randn(1,16) for _ in range(8)]
        for head in (full,local):
            a=head.reset();b=head.reset()
            for v in past:a=head.update(v,a);b=head.update(-v,b)
            a=head.update(x,a);b=head.update(x,b)
            difference=(head.action_delta(x,a)-head.action_delta(x,b)).abs().max().item()
            self.assertEqual(difference==0,head.architecture=='LOCAL')
        self.assertTrue(torch.equal(full.reset(),torch.zeros(1,8,64)))

    def test_early_event_gradient_and_real_parameter_update(self):
        for model in self.heads():
            features=torch.randn(12,16,requires_grad=True)
            row=dict(memory_features=features,query_steps=torch.tensor([11]),actor_features=torch.randn(1,16),base_logits=torch.zeros(1,4))
            loss=sequence_logits(model,row)[0,0];loss.backward()
            self.assertEqual(features.grad[0].abs().sum().item()>0,model.architecture=='CONCAT')
            self.assertGreater(model.writer.weight.grad.abs().sum().item(),0)
            before=model.writer.weight.detach().clone();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);optimizer.step()
            self.assertFalse(torch.equal(before,model.writer.weight))

    def test_missing_or_partial_groups_never_produce_full_sr(self):
        p=dict(models={'CONCAT_s42':{},'LOCAL_s42':{}},seeds=[42]);entries=[dict(id=0,house='h'),dict(id=1,house='h')]
        outcomes={a:dict(success=1.,spl=.5,steps=10) for a in ['NATIVE',*p['models']]}
        row=dict(house='h',outcomes=outcomes)
        result=summarize({0:row},entries,p)
        self.assertIsNone(result['arms']['CONCAT_s42']['sr'])
        self.assertEqual(result['arms']['CONCAT_s42']['identification_bounds'],[.5,1.])
        del row['outcomes']['LOCAL_s42']
        with self.assertRaisesRegex(AssertionError,'MISSING_MODEL'):summarize({0:row},entries,p)

    def test_input_or_native_action_flip_rejected(self):
        trace=[dict(event='reset',rgb_sha256='a'),dict(event='generation',environment_step=0,input={'x':'same'},native_logits=[0.,1.,0.,0.],generated_ids=[[1]]),
               dict(event='action',executed_action=1,before_rgb_sha256='a',after_rgb_sha256='b')]
        result={'success':0}
        self.assertTrue(audit_pair(trace,trace,result,result)['full_trajectory_matched'])
        changed=copy.deepcopy(trace);changed[1]['native_logits']=[2.,1.,0.,0.]
        with self.assertRaisesRegex(ValueError,'NATIVE_ARGMAX_FLIP'):audit_pair(trace,changed,result,result)
        changed=copy.deepcopy(trace);changed[1]['input']['x']='different'
        with self.assertRaisesRegex(ValueError,'PROCESSED_PREFIX'):audit_pair(trace,changed,result,result)


if __name__=='__main__':unittest.main()
