"""CPU checks for genuine head gradients, masks and frozen base ownership."""
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
torch.set_num_threads(2)
obj=u.load('motion16_test_objective',u.HERE/'objective.py')
recovery=u.load('motion16_test_recovery',u.HERE/'prepare_recovery.py')
evaluation=u.load('motion16_test_evaluation',u.HERE/'evaluate.py')
class Tests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(1209);n,d=8,6;x=torch.randn(n,d,dtype=torch.float64);z=torch.randn(n,4,dtype=torch.float64);y=torch.tensor([0,0,0,0,1,1,1,1],dtype=torch.float64);sw=torch.ones(n,dtype=torch.float64)/n;my=torch.tensor([0,1,2,0,-1,-1,-1,-1]);mw=(my>=0).double()/4;theta=torch.zeros(4,d+1,dtype=torch.float64,requires_grad=True);return theta,x,z,y,sw,my,mw
    def test_real_gradient_and_update(self):
        a=self.fixture();opt=torch.optim.SGD([a[0]],lr=.1);loss,_=obj.loss(*a);loss.backward();self.assertGreater(float(a[0].grad[:3].norm()),0);self.assertGreater(float(a[0].grad[3].norm()),0);opt.step();self.assertGreater(float(a[0].norm()),0)
    def test_masked_labels_do_not_affect_loss_or_gradient(self):
        a=self.fixture();loss,_=obj.loss(*a);g=torch.autograd.grad(loss,a[0])[0];my=a[5].clone();my[4:]=2;b=(*a[:5],my,a[6]);other,_=obj.loss(*b);h=torch.autograd.grad(other,a[0])[0];self.assertTrue(torch.equal(g,h));self.assertEqual(float(loss),float(other))
    def test_base_mutation_rejected(self):
        old={'base':torch.ones(2),'action_head.weight':torch.zeros(4,2),'action_head.bias':torch.zeros(4)};new={k:v.clone() for k,v in old.items()};new['action_head.weight'][0,0]=1;u.validate_head(old,new);new['base'][0]=9
        with self.assertRaisesRegex(AssertionError,'NONHEAD'):u.validate_head(old,new)
    def test_logits_argmax_all_four(self):
        theta,x,z,*_=self.fixture();theta=theta.detach();theta[0,-1]=100;p=obj.logits(theta,x,z);self.assertTrue((p.argmax(1)==0).all())
    def test_recovery_prefix_excludes_unexecuted_stop(self):
        rows=[dict(executed_action=a) for a in ['turn_left','STOP']]
        private=[dict(action=a,collided=False,distance_to_goal=4) for a in ['turn_left','STOP']]
        result=recovery.trigger(rows,private);self.assertEqual(result['cutoff'],1);self.assertFalse(result['event_action_executed'])
        self.assertIsNone(recovery.trigger(rows[1:],private[1:]))
    def test_recovery_collisions_and_budget(self):
        rows=[dict(executed_action='move_forward') for _ in range(3)]
        private=[dict(action='move_forward',collided=True,distance_to_goal=4) for _ in rows]
        self.assertEqual(recovery.trigger(rows,private)['cutoff'],3)
        private[1]['collided']=False;self.assertIsNone(recovery.trigger(rows,private))
        neutral=dict(action='turn_left',collided=False,distance_to_goal=4)
        prefix=[dict(executed_action='turn_left')]*479
        private[1]['collided']=True
        self.assertIsNone(recovery.trigger(prefix+rows,[neutral]*479+private))
    def test_prefix_numerical_and_transport_audit(self):
        a=dict(raw={'key':'x'},processed={'sha':'y'},base_logits=[1.,0.,0.,0.],base_action='move_forward');b=dict(a,base_logits=[1.001,0.,0.,0.])
        self.assertGreater(evaluation.audit_prefix(a,b),0)
        with self.assertRaisesRegex(ValueError,'ARGMAX'):evaluation.audit_prefix(a,dict(b,base_action='STOP'))
        with self.assertRaisesRegex(ValueError,'INPUT'):evaluation.audit_prefix(a,dict(b,raw={'key':'z'}))
    def test_500_and_legal_stop(self):
        self.assertEqual(u.c.advance('turn_left',499,False,500),(500,True))
        self.assertEqual(u.c.advance('STOP',499,False,500),(500,True))
if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));u.write(u.HERE/'CPU_TEST_RESULT.json',dict(tests=r.testsRun,passed=r.wasSuccessful(),errors=len(r.errors),failures=len(r.failures),scope='CPU loss/mask/frozen-parameter contracts with real autograd; not a model-quality result'));raise SystemExit(not r.wasSuccessful())
