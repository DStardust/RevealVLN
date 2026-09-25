"""CPU tests of identity, causal learning and missing-label handling; no SR claims."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from memory import ExecutionMemory,interaction_loss,masked_state_loss


class MemoryTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);torch.set_num_threads(2)

    def test_zero_initialization_preserves_all_action_scores(self):
        m=ExecutionMemory(16);x=torch.randn(2,12,16);base=torch.randn(2,12,4)
        out=m(x,torch.tensor([12,12]));self.assertTrue(torch.equal(base,base+out['delta']))

    def test_reset_and_variable_lengths(self):
        m=ExecutionMemory(16);x=torch.randn(2,12,16)
        out=m(x,torch.tensor([4,12]));self.assertTrue(torch.equal(out['memory'][0,3],out['memory'][0,-1]))
        self.assertTrue(torch.equal(m.reset(),torch.zeros(1,8,64)))
        self.assertTrue(torch.equal(m(x,torch.tensor([4,12]))['memory'],out['memory']))

    def test_old_event_gradient_path(self):
        m=ExecutionMemory(16);x=torch.randn(1,40,16,requires_grad=True)
        y=m(x,torch.tensor([40]))['memory'][:,-1]
        m.state_reader(y.flatten(1)).sum().backward()
        self.assertGreater(x.grad[0,0].abs().sum().item(),0)
        self.assertGreater(m.writer.weight.grad.abs().sum().item(),0)

    def test_unknown_not_negative(self):
        logits=torch.randn(1,1,2,4,requires_grad=True);pairs=torch.tensor([[[1,2]]])
        y=torch.tensor([[[[1.,0.],[float('nan'),1.]]]]);known=torch.isfinite(y)
        loss=interaction_loss(logits,pairs,y,known);self.assertEqual(loss.item(),0);loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_interaction_changes_real_action_preferences(self):
        logits=torch.zeros(1,1,2,4,requires_grad=True);pairs=torch.tensor([[[1,2]]])
        y=torch.tensor([[[[1.,0.],[0.,1.]]]])
        loss=interaction_loss(logits,pairs,y,torch.ones_like(y,dtype=torch.bool));loss.backward()
        self.assertLess(logits.grad[0,0,0,1].item(),0)
        self.assertGreater(logits.grad[0,0,1,1].item(),0)

    def test_same_next_action_not_fabricated_conflict(self):
        with self.assertRaisesRegex(ValueError,'NO_IDENTIFIABLE_ACTION_FORK'):
            interaction_loss(torch.zeros(1,1,2,4),torch.tensor([[[1,1]]]),torch.zeros(1,1,2,2),torch.ones(1,1,2,2,dtype=torch.bool))

    def test_no_future_query_in_memory_or_actor_interface(self):
        import inspect
        self.assertEqual(list(inspect.signature(ExecutionMemory.update).parameters),['self','feature','memory'])
        self.assertEqual(list(inspect.signature(ExecutionMemory.action_delta).parameters),['self','feature','memory'])

    def test_b2_unknown_mask(self):
        x=torch.zeros(1,4,requires_grad=True);y=torch.tensor([[1.,0.,float('nan'),float('nan')]])
        loss=masked_state_loss(x,y,torch.isfinite(y));loss.backward()
        self.assertTrue(torch.equal(x.grad[0,2:],torch.zeros(2)))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MemoryTests))
    Path(__file__).with_name('CPU_TEST_RESULT.json').write_text(json.dumps(dict(tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful(),
        scope='CPU synthetic contracts only; no model/navigation/method benefit claim'),indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
