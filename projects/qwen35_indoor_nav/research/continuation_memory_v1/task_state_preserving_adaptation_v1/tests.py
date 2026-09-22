import sys,json,unittest
from pathlib import Path
from dataclasses import replace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from interface import *

class Tests(unittest.TestCase):
    def test_inactive_exact_logits(self):
        a=torch.tensor([[1.,2.,3.,4.],[-2.,-3.,-1.,-5.]])
        self.assertTrue(torch.equal(route_logits(a,-a,torch.zeros(2,dtype=torch.bool)),a))
    def test_active_stop_can_change(self):
        a=torch.tensor([[0.,0.,0.,2.]])
        p=torch.tensor([[3.,0.,0.,0.]])
        self.assertEqual(int(route_logits(a,p,torch.ones(1,dtype=torch.bool)).argmax()),0)
    def test_row_isolation(self):
        a=torch.zeros(2,4);p=torch.ones(2,4);r=route_logits(a,p,torch.tensor([True,False]))
        self.assertTrue(torch.equal(r[0],p[0]));self.assertTrue(torch.equal(r[1],a[1]))
    def test_nonfinite_is_error(self):
        with self.assertRaises(ValueError):route_logits(torch.zeros(1,4),torch.full((1,4),float('nan')),torch.zeros(1,dtype=torch.bool))
    def branch(self,label):return Branch('prefix','base','policy','env',0,450,0,True,True,label,'trace',40)
    def test_known_gain_harm_and_tie(self):
        a=self.branch('FAIL');b=replace(self.branch('PASS'),executed_first_action=1)
        self.assertEqual(branch_target(a,b).benefit,1)
        self.assertEqual(branch_target(replace(a,label='PASS'),replace(b,label='FAIL')).benefit,-1)
        self.assertEqual(branch_target(replace(a,label='PASS'),b).benefit,0)
    def test_unknown_mask(self):
        a=self.branch('FAIL');b=replace(self.branch('UNKNOWN'),executed_first_action=1)
        self.assertFalse(branch_target(a,b).known);self.assertIsNone(branch_target(a,b).benefit)
        self.assertFalse(branch_target(a,replace(b,label='FAIL',legal=False)).known)
    def test_pair_identity(self):
        a=self.branch('FAIL');b=replace(self.branch('PASS'),executed_first_action=1,prefix_sha256='other')
        with self.assertRaises(ValueError):branch_target(a,b)
    def test_router_backward_interface(self):
        torch.manual_seed(1209);net=InterventionRouter();f=torch.randn(2,2048);m=torch.randn(2,8,64,requires_grad=True)
        logits=net(f,m,torch.rand(2,4),torch.randn(2,4));logits.square().mean().backward()
        self.assertGreater(float(m.grad.abs().sum()),0);self.assertTrue(all(p.grad is not None for p in net.parameters()))

if __name__=='__main__':
    torch.set_num_threads(2);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    out=dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful(),cuda_initialized=torch.cuda.is_initialized(),optimizer_updates=0,trained_router=False,scope='Synthetic CPU interface/gradient tests; not a navigation or method experiment.')
    (Path(__file__).resolve().parent/'CPU_TEST_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
    raise SystemExit(not r.wasSuccessful())
