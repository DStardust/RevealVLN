"""CPU regression checks for nonempty preservation and real temporal gradients."""
import argparse
from pathlib import Path
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
from memory_v2 import ExecutionMemory
from preserve_objective_v3 import preservation_loss,sequence_logits
from preserve_review_v3 import summarize


class PreservationTests(unittest.TestCase):
    def test_old_native_teacher_disagreement_does_not_erase_preservation(self):
        base=torch.tensor([[0.,0.,0.,4.]])
        logits=(base+torch.tensor([[12.,0.,0.,0.]])).requires_grad_()
        loss,kl,margin=preservation_loss(logits,base,torch.tensor([3]))
        loss.backward()
        self.assertGreater(float(kl),0);self.assertGreater(float(logits.grad[0,0]),0)
        self.assertLess(float(logits.grad[0,3]),0)

    def test_empty_pool_is_error(self):
        with self.assertRaisesRegex(ValueError,'EMPTY_PRESERVATION'):
            preservation_loss(torch.zeros(0,4),torch.zeros(0,4),torch.zeros(0,dtype=torch.long))

    def test_gradient_reaches_old_event_and_update_reduces_loss(self):
        torch.manual_seed(42);model=ExecutionMemory(16)
        with torch.no_grad():model.actor[-1].weight.normal_(0,.03);model.actor[-1].bias[0]=5
        x=torch.randn(24,16,requires_grad=True)
        row=dict(memory_features=x,actor_features=torch.randn(1,16),base_logits=torch.tensor([[0.,0.,0.,3.]]),
            query_steps=torch.tensor([23]),targets=torch.tensor([3]))
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        before=sequence_logits(model,row);loss,*_=preservation_loss(before,row['base_logits'],row['targets']);loss.backward()
        self.assertGreater(float(x.grad[0].abs().sum()),0)
        self.assertGreater(float(model.writer.weight.grad.abs().sum()),0)
        old=model.actor[-1].bias.detach().clone();optimizer.step()
        after,*_=preservation_loss(sequence_logits(model,row),row['base_logits'],row['targets'])
        self.assertLess(float(after),float(loss));self.assertLess(float(model.actor[-1].bias[0]),float(old[0]))

    def test_native_zero_residual_has_zero_kl(self):
        torch.manual_seed(42);model=ExecutionMemory(16)
        row=dict(memory_features=torch.randn(3,16),actor_features=torch.randn(2,16),base_logits=torch.randn(2,4),
            query_steps=torch.tensor([0,2]))
        out=sequence_logits(model,row);self.assertTrue(torch.equal(out,row['base_logits']))
        loss,kl,margin=preservation_loss(out,row['base_logits'],out.argmax(-1))
        self.assertAlmostEqual(float(loss),0,places=6)

    def test_missing_episodes_never_become_full_sr(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);u.write(p/'DATA_MANIFEST.json',dict(episodes=[dict(id=1,house='A'),dict(id=2,house='B')]))
            u.write(p/'PROTOCOL.json',dict(expected_base_state_sha256='frozen',heads={}))
            result=summarize(p);self.assertFalse(result['evaluation_complete'])
            self.assertEqual(result['missing'],[1,2]);self.assertIsNone(result['arms']['B2']['sr_full'])
            self.assertEqual(result['arms']['B2']['identification_bounds'],[0,1])

    def test_duplicate_query_is_not_silent(self):
        model=ExecutionMemory(16)
        row=dict(memory_features=torch.randn(2,16),actor_features=torch.randn(2,16),base_logits=torch.zeros(2,4),query_steps=torch.tensor([1,1]))
        with self.assertRaisesRegex(ValueError,'MISALIGNED'):sequence_logits(model,row)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);start=time.time()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreservationTests))
    u.write(a.output,dict(status='PASSED' if result.wasSuccessful() else 'FAILED',tests=result.testsRun,
        errors=[str(e) for e in result.errors],failures=[str(e) for e in result.failures],seconds=time.time()-start,
        scope='CPU loss/backprop/missing-denominator regression; no GPU model or navigation result'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
