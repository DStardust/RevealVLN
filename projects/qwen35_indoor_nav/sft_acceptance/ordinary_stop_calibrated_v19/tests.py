"""CPU checks for trajectory loss, causal cutoff, masks and fixed motion outputs."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
import objective as o
import train as t
from calibration import fit_bias
class Tests(unittest.TestCase):
    def fixture(self):
        h=torch.eye(3,dtype=torch.float64);ref=torch.tensor([-2.,-3.,-1.],dtype=torch.float64);theta=torch.zeros(4,dtype=torch.float64,requires_grad=True)
        groups=t.groups_for([dict(house='a',negative=[0,1],positive=2)],['a']);return theta,h,ref,groups
    def test_actual_gradient_update_and_finite(self):
        theta,h,ref,g=self.fixture();loss,_=o.losses(theta,h,ref,**g);loss.backward();self.assertTrue(torch.isfinite(theta.grad).all());self.assertGreater(theta.grad[0],0);self.assertLess(theta.grad[2],0)
        before=theta.detach().clone();torch.optim.SGD([theta],lr=.1).step();self.assertFalse(torch.equal(theta,before))
    def test_bag_does_not_invent_length_prior(self):
        theta,h,ref,g=self.fixture();ref[1]=ref[0];_,parts=o.losses(theta,h,ref,**g);g['negative_mask'][0,1]=False;_,single=o.losses(theta,h,ref,**g);self.assertAlmostEqual(float(parts['negative_bag'].detach()),float(single['negative_bag'].detach()),places=12)
    def test_empty_negative_bag_no_nan(self):
        theta,h,ref,g=self.fixture();g['negative_mask'][:]=False;loss,_=o.losses(theta,h,ref,**g);loss.backward();self.assertTrue(torch.isfinite(theta.grad).all())
    def test_motion_mutation_rejected(self):
        a={'action_head.weight':torch.zeros(4,3),'action_head.bias':torch.zeros(4),'other':torch.ones(1)};b={k:v.clone() for k,v in a.items()};b['action_head.weight'][3,0]=1;u.validate_head(a,b);b['action_head.weight'][0,0]=1
        with self.assertRaises(AssertionError):u.validate_head(a,b)
    def test_masked_future_input_has_zero_gradient(self):
        theta,h,ref,g=self.fixture()
        with torch.no_grad():theta[:3].fill_(1)
        h=torch.cat([h,torch.ones(1,3,dtype=torch.float64)]).requires_grad_();ref=torch.cat([ref,torch.zeros(1,dtype=torch.float64)]);loss,_=o.losses(theta,h,ref,**g);loss.backward();self.assertGreater(float(h.grad[:3].norm()),0);self.assertTrue(torch.equal(h.grad[3],torch.zeros(3,dtype=torch.float64)))
    def test_full_budget_includes_stop(self):
        self.assertEqual(u.c.advance('STOP',499,False,500),(500,True))
    def test_calibration_budget_and_ties(self):
        groups=t.groups_for([dict(house='a',negative=[i],positive=None) for i in range(4)],['a'])
        current=torch.tensor([3.,2.,2.,-1.],dtype=torch.float64);ref=torch.tensor([1.,-1.,-1.,-1.],dtype=torch.float64);r=fit_bias(current,ref,groups)
        self.assertEqual(r['order_statistic_threshold'],2.)
        self.assertLessEqual(float(((current-r['bias_reduction'])>0).double().mean()),.25)
    def test_calibration_never_promotes_stop(self):
        groups=t.groups_for([dict(house='a',negative=[0],positive=None)],['a']);m=torch.tensor([-1.],dtype=torch.float64)
        self.assertEqual(fit_bias(m,m,groups)['bias_reduction'],0.)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));u.write(u.HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,passed=result.wasSuccessful(),scope='CPU contracts, not navigation benefit'));raise SystemExit(not result.wasSuccessful())
