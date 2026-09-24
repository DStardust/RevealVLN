"""CPU tests of the real objective, selection, masking and frozen motion contract."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch,common as u,objective as o,train as t
from evaluate import audit_prefix
torch.set_num_threads(2)
class Tests(unittest.TestCase):
    def fixture(self):
        rows=[dict(rank=0,house='fit',negative=[0,1],positive=2)]
        pp=[dict(rank=0,house='fit',negative=0,positive=2)]
        g=t.groups_for(rows,pp,['fit']);h=torch.eye(4,dtype=torch.float64);ref=torch.tensor([-.2,-1.,-1.,0.],dtype=torch.float64)
        th=torch.zeros(5,dtype=torch.float64,requires_grad=True);return h,ref,th,g
    def test_real_gradient_update(self):
        h,r,th,g=self.fixture();loss,_=o.losses(th,h,r,g);loss.backward()
        self.assertGreater(float(th.grad[0]),0);self.assertLess(float(th.grad[2]),0)
        before=th.detach().clone();torch.optim.SGD([th],lr=.1).step();self.assertFalse(torch.equal(th,before))
    def test_unused_input_zero_gradient(self):
        h,r,th,g=self.fixture();loss,_=o.losses(th,h,r,g);loss.backward();self.assertEqual(float(th.grad[3]),0)
    def test_individual_false_stop_penalty(self):
        h,r,th,g=self.fixture();_,a=o.losses(th,h,r,g)
        with torch.no_grad():th[0]=1
        _,b=o.losses(th,h,r,g);self.assertGreater(float(b['reference_continue_protection']),float(a['reference_continue_protection']))
    def test_empty_negative_safe(self):
        h,r,th,g=self.fixture();g['negative_mask'][:]=False
        loss,_=o.losses(th,h,r,g);loss.backward();self.assertTrue(torch.isfinite(th.grad).all())
    def test_duplicate_pair_does_not_reweight(self):
        rows=[dict(rank=0,house='fit',negative=[0],positive=1)];pp=[dict(rank=0,house='fit',negative=0,positive=1)]
        a=t.groups_for(rows,pp,['fit']);b=t.groups_for(rows,pp*3,['fit']);self.assertTrue(torch.equal(a['pair_weights'],b['pair_weights']))
    def test_diagnostic_house_excluded(self):
        rows=[dict(rank=0,house='fit',negative=[0],positive=1),dict(rank=1,house='check',negative=[2],positive=3)]
        pp=[dict(rank=r['rank'],house=r['house'],negative=r['negative'][0],positive=r['positive']) for r in rows]
        g=t.groups_for(rows,pp,['fit']);self.assertEqual(g['positive_indices'].tolist(),[1]);self.assertEqual(g['pair_negative'].tolist(),[0])
    def test_motion_mutation_rejected(self):
        a={'action_head.weight':torch.zeros(4,3),'action_head.bias':torch.zeros(4),'other':torch.ones(1)};b={k:v.clone() for k,v in a.items()};b['action_head.weight'][3,0]=1;u.validate_head(a,b);b['action_head.weight'][0,0]=1
        with self.assertRaises(AssertionError):u.validate_head(a,b)
    def test_stop_uses_last_budget_slot(self):self.assertEqual(u.c.advance('STOP',499,False,500),(500,True))
    def test_prefix_input_and_argmax_fail(self):
        a=dict(raw='same',processed='same',base_logits=[1.,0.,0.,0.],base_action='move_forward');self.assertEqual(audit_prefix(a,a),0)
        with self.assertRaises(ValueError):audit_prefix(a,dict(a,processed='wrong'))
        with self.assertRaises(ValueError):audit_prefix(a,dict(a,base_logits=[float('nan'),0.,0.,0.]))
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));u.write(u.HERE/'CPU_TEST_RESULT.json',dict(tests=r.testsRun,passed=r.wasSuccessful(),scope='CPU objective/transport tests, no navigation claim'));raise SystemExit(not r.wasSuccessful())
