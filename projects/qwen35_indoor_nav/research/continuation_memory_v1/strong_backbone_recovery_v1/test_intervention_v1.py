"""CPU contracts for causal inputs and conservative one-time branch selection."""
import argparse
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import common as u
from intervention_gate_v1 import InterventionGate,choose_method,label,policy_features


class Tests(unittest.TestCase):
    def test_real_outcome_labels(self):
        self.assertEqual(label(False,True),1);self.assertEqual(label(True,False),2)
        self.assertEqual(label(True,True),0);self.assertEqual(label(False,False),0)
    def test_untrained_gate_retains_native(self):
        model=InterventionGate(24);self.assertFalse(bool(choose_method(model(torch.randn(16),torch.randn(4),torch.randn(4)))))
    def test_fixed_asymmetric_harm_cost(self):
        self.assertFalse(bool(choose_method(torch.tensor([0.,.5,0.]))))
        self.assertTrue(bool(choose_method(torch.tensor([0.,1.,0.]))))
    def test_features_have_no_future_or_identifiers(self):
        x=torch.arange(16).float();n=torch.tensor([2.,3.,4.,5.]);m=n+1
        f=policy_features(x,n,m);self.assertEqual(tuple(f.shape),(24,))
        self.assertTrue(torch.allclose(f,policy_features(x,n+100,m+100)))
    def test_true_backward_and_update(self):
        torch.manual_seed(42);model=InterventionGate(24);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        x=torch.randn(8,24);y=torch.tensor([0,1,2,1,2,1,0,2]);old=model.linear.weight.detach().clone()
        loss=torch.nn.functional.cross_entropy(model.linear(x),y);loss.backward()
        self.assertGreater(float(model.linear.weight.grad.norm()),0);optimizer.step()
        self.assertFalse(torch.equal(old,model.linear.weight))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    u.write(a.output,dict(status='PASSED' if r.wasSuccessful() else 'FAILED',tests=r.testsRun,failures=[str(x) for x in r.failures],errors=[str(x) for x in r.errors],scope='CPU contracts only, no efficacy claim'))
    raise SystemExit(0 if r.wasSuccessful() else 1)
