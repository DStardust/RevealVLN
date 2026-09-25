"""Exercise actual optimization code on synthetic CPU fixtures; no model result."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import common as u
from memory import ExecutionMemory
from train_memory import objective,validate

class TrainingTests(unittest.TestCase):
    def test_all_arms_update_and_ours_reaches_old_features(self):
        torch.set_num_threads(2)
        for arm in ('BC','B2','OURS'):
            torch.manual_seed(42);m=ExecutionMemory(16);optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
            features=torch.randn(2,2,40,16,requires_grad=True)
            group=dict(features=features,lengths=torch.full((2,2),40),base_action_logits=torch.randn(2,2,40,4),
                action_known=torch.ones(2,2,40,dtype=torch.bool),action_targets=torch.ones(2,2,40,dtype=torch.long),
                preservation_mask=torch.zeros(2,2,40,dtype=torch.bool),state_targets=torch.zeros(2,2,40,4),
                state_known=torch.ones(2,2,40,4,dtype=torch.bool),action_pairs=torch.tensor([[1,2],[1,2]]),
                returns=torch.tensor([[[1.,0.],[0.,1.]],[[1.,1.],[1.,1.]]]),return_known=torch.ones(2,2,2,dtype=torch.bool))
            initial=m.actor[-1].weight.detach().clone()
            for step in range(2):
                loss,_=objective(m,group,arm);optimizer.zero_grad(set_to_none=True);features.grad=None;loss.backward()
                self.assertTrue(torch.isfinite(loss));optimizer.step()
            self.assertFalse(torch.equal(initial,m.actor[-1].weight))
            if arm=='OURS':
                self.assertGreater(m.writer.weight.grad.abs().sum().item(),0)
                self.assertGreater(features.grad[:,:,0].abs().sum().item(),0)

    def test_dev_cannot_train(self):
        with self.assertRaisesRegex(ValueError,'UNADMITTED_TRAINING_SPLIT'):
            validate(dict(split='INTERNAL_DEV',scope='STREAMVLN_EXECUTED_HISTORY_FEATURES'))

    def test_future_unknown_is_not_admitted(self):
        with self.assertRaisesRegex(ValueError,'FUTURE_INPUT_AUDIT_REQUIRED'):
            validate(dict(split='FIT',scope='STREAMVLN_EXECUTED_HISTORY_FEATURES',future_in_policy_features=None))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrainingTests))
    u.write(u.HERE/'TRAINING_CPU_TEST_RESULT.json',dict(tests=result.testsRun,successful=result.wasSuccessful(),
        scope='Synthetic CPU optimization of actual training objective; no StreamVLN method training or navigation claim'))
    raise SystemExit(not result.wasSuccessful())
