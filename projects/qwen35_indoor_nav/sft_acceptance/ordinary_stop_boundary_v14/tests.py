import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as u
import torch
from build_pairs import compile_pairs,pair_weights
from objective import ranking_loss,optimize
ev=u.load('stop14_test_evaluate',u.HERE/'evaluate.py')
rv=u.load('stop14_test_review',u.HERE/'review.py')
class Tests(unittest.TestCase):
    def test_pair_membership_and_nearest_real_step(self):
        labels=[];inputs=[]
        for i,(step,distance) in enumerate([(0,4.),(1,2.),(4,1.),(5,5.),(6,8.)]):
            labels.append(dict(record_id=str(i),house='fit',target=int(distance<3),occurrences=[dict(episode=1,step=step,house='fit',target=int(distance<3),distance=distance)]))
            inputs.append(dict(record_id=str(i),instruction='go',rgb_sha256=[str(i)],executed_actions=[]))
        p=compile_pairs(labels,inputs)
        self.assertEqual([(x['negative_index'],x['positive_index']) for x in p],[(0,1),(3,2)])
        self.assertEqual(pair_weights(p,['fit']).tolist(),[.5,.5])
    def test_ranking_does_not_shift_global_stop_bias(self):
        margin=torch.tensor([-.2,.5],dtype=torch.float64,requires_grad=True);bias=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
        loss=ranking_loss(margin+bias,torch.tensor([1]),torch.tensor([0]),torch.tensor([1.],dtype=torch.float64),1.);loss.backward()
        self.assertEqual(float(bias.grad),0.)
        self.assertGreater(float(margin.grad[0]),0);self.assertLess(float(margin.grad[1]),0)
    def test_pair_holdout_mask(self):
        p=[dict(episode=0,house='FIT'),dict(episode=1,house='CHECK')];w=pair_weights(p,['FIT'])
        self.assertEqual(w.tolist(),[1.,0.])
    def test_boundary_objective_true_update(self):
        h=torch.tensor([[-1.,0.],[1.,0.],[-2.,1.],[2.,1.]],dtype=torch.float64);z=torch.zeros(4,4,dtype=torch.float64)
        y=torch.tensor([0.,1.,0.,1.],dtype=torch.float64);w=torch.ones(4,dtype=torch.float64)/4;records=[]
        theta,scale,info=optimize(h,z,y,w,torch.tensor([1,3]),torch.tensor([0,2]),torch.tensor([.5,.5],dtype=torch.float64),dict(ranking_weight=1.,ranking_margin=1.,regularization=.001),records.append)
        self.assertGreater(float(theta.norm()),0);self.assertLess(records[-1]['loss'],records[0]['loss']);self.assertTrue(all(x['gradient_norm']>=0 for x in records))
    def test_parameter_scope(self):
        a={'action_head.weight':torch.zeros(4,8),'action_head.bias':torch.zeros(4),'frozen':torch.ones(2)};b={k:v.clone() for k,v in a.items()};b['action_head.weight'][3]=1;u.validate_head(a,b)
        b['action_head.weight'][0]=2
        with self.assertRaisesRegex(AssertionError,'MOTION'):u.validate_head(a,b)
    def test_prefix_numeric_separate(self):
        a=dict(raw={'x':1},processed={'y':2},base_logits=[1.,0.,0.,0.],base_action='move_forward');b=copy.deepcopy(a);b['base_logits'][0]+=.02
        self.assertAlmostEqual(ev.audit_prefix(a,b),.02)
        b['base_action']='STOP'
        with self.assertRaisesRegex(ValueError,'ARGMAX'):ev.audit_prefix(a,b)
    def test_raw_mismatch(self):
        a=dict(raw=1,processed=2,base_logits=[1.,0.,0.,0.],base_action='move_forward');b=dict(a,raw=3)
        with self.assertRaisesRegex(ValueError,'INPUT'):ev.audit_prefix(a,b)
    def test_budget_includes_stop(self):
        self.assertEqual(u.c.advance('STOP',499,False,500),(500,True))
        self.assertEqual(u.c.advance('turn_left',499,False,500),(500,True))
    def test_missing_denominator(self):
        import tempfile
        with tempfile.TemporaryDirectory() as p:
            r=rv.summarize(Path(p));self.assertIsNone(r['delta_sr']);self.assertIsNone(r['arms']['A']['full_sr']);self.assertEqual(r['arms']['A']['identification_bounds'],[0,1])
    def test_real_anchored_gradient_update(self):
        fit=u.load('stop14_test_optimizer',u.LINE/'sft_acceptance/ordinary_stop_row_v12/fit.py')
        h=torch.tensor([[1.,0.],[-1.,0.],[2.,1.],[-2.,1.]],dtype=torch.float64);z=torch.zeros(4,4,dtype=torch.float64);y=torch.tensor([1.,0.,1.,0.],dtype=torch.float64);w=torch.ones(4,dtype=torch.float64)/4
        theta,scale,info=fit.optimize(h,z,y,w)
        self.assertGreater(info['optimizer_iterations'],0);self.assertGreater(float(theta.norm()),0)
        before=fit.metrics(z[:,3],y,w);after=fit.metrics(h@theta[:-1]/scale+theta[-1],y,w);self.assertLess(after['bce'],before['bce'])
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
    u.write(u.HERE/'CPU_TEST_RESULT.json',dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful(),scope='CPU optimizer, parameter restriction, audit and missing denominators; not navigation'))
    raise SystemExit(0 if r.wasSuccessful() else 1)
