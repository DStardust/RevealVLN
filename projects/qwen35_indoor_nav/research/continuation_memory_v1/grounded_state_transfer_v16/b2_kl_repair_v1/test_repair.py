import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch.nn import functional as F
from shared import *
import objective as o

class Repair(unittest.TestCase):
    def test_only_supervised_conflict_removed_original_denominator(self):
        native=torch.tensor([[[3.,0,0,0],[0,3.,0,0],[3.,0,0,0],[0,0,0,3.]]])
        logits=torch.zeros_like(native,requires_grad=True);mask=torch.ones(1,4)
        labels=torch.tensor([[1,1,1,0]]);supervised=torch.tensor([[1.,1.,0.,1.]])
        old,conflicts=o.preservation_loss(logits,native,supervised,labels,mask,False)
        new,_=o.preservation_loss(logits,native,supervised,labels,mask,True)
        self.assertEqual(conflicts.tolist(),[[True,False,False,True]])
        g0=torch.autograd.grad(old,logits,retain_graph=True)[0];g1=torch.autograd.grad(new,logits)[0]
        self.assertTrue(torch.equal(g0[:,1:3],g1[:,1:3]));self.assertTrue(torch.equal(g1[:,[0,3]],torch.zeros_like(g1[:,[0,3]])))
        self.assertAlmostEqual(float(new),float(old)/2,places=6)
    def test_padding_and_zero_mask(self):
        x=torch.randn(1,3,4,requires_grad=True);n=torch.randn_like(x)
        loss,_=o.preservation_loss(x,n,torch.ones(1,3),torch.zeros(1,3,dtype=torch.long),torch.zeros(1,3),True)
        self.assertEqual(float(loss),0);loss.backward();self.assertEqual(float(x.grad.abs().sum()),0)
    def test_real_fit_losses_other_terms_unchanged_and_parameter_update(self):
        torch.set_num_threads(2);data=read(CONTROL/'DATA.json');f=next(f for f in data['families'] if f['split']=='FIT')
        cache=torch.load(CONTROL/'features/FEATURES.pt',weights_only=True);b=o.batch(f,'cpu');weights=read(CONTROL/'FIT_AUXILIARY_WEIGHTS.json')
        net=o.initialize(1209);before=c.model_identity(net)['sha256']
        l0,s0,d0=o.losses(net,cache,b,'B2',weights);l1,s1,d1=o.losses(net,cache,b,'B2Fix',weights)
        for key in ('action_ce','cutoff_ce','auxiliary','action_correct','state_supervisions'):self.assertEqual(s0[key],s1[key])
        self.assertTrue(torch.equal(d0['logits'],d1['logits']));self.assertGreater(s1['preservation_removed'],0)
        self.assertAlmostEqual(float(l0-l1),s0['preservation_kl']-s1['preservation_kl'],places=5)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001);l1.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None));optimizer.step()
        self.assertNotEqual(before,c.model_identity(net)['sha256'])
    def test_registry_uses_same_conditions_complete_group(self):
        from evaluate_continuations import registry_value
        reg=registry_value(read(CONTROL/'DATA.json')['raw_families'],read(HERE/'PROTOCOL.json'))
        self.assertEqual(reg['conditions'],read(CONTROL/'EVALUATION_REGISTRY.json')['conditions'])
        self.assertEqual(len(reg['slots']),1152);self.assertEqual(reg['models'],[f'{a}_{s}' for s in (1209,1210,1211) for a in ('B1','B2','B2Fix')])

if __name__=='__main__':unittest.main()
