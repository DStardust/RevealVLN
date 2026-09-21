"""Meaningful CPU tests of query firewall, early gradient and episode reset."""
import copy
import unittest
import torch
import objective as o

class Contract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(1209)
        self.net=o.initialize(1209)
        rows=[]
        for label in (0,1):
            rows.append(dict(features=list(range(12)),targets=[1]*11+[3 if label else 1],action_masks=[0]*11+[1],
                state_targets=[[label]*4]*12,state_masks=[1]*12,y=[label]*12,query_masks=[1]*12,
                query_contexts=[dict(zip(o.FIELDS,[1,1,0,1]))]*12,cutoff=11))
        self.batch=o.batch(dict(sequences=rows),'cpu');self.cache=dict(features=torch.randn(12,2048),logits=torch.randn(12,4))
        self.weights=dict(state=[[1,1]]*4,query=[1,1])
    def test_query_does_not_change_memory_or_action(self):
        _,_,a=o.losses(self.net,self.cache,self.batch,'Ours',self.weights)
        changed=dict(self.batch,query=1-self.batch['query'])
        _,_,b=o.losses(self.net,self.cache,changed,'Ours',self.weights)
        self.assertTrue(torch.equal(a['states'],b['states']));self.assertTrue(torch.equal(a['logits'],b['logits']))
    def test_early_write_receives_late_query_gradient(self):
        _,_,d=o.losses(self.net,self.cache,self.batch,'Ours',self.weights,retain_steps=(1,))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(d['query_logits'][:,-1],self.batch['y'][:,-1])
        gradient=torch.autograd.grad(loss,d['writes'][1])[0]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.norm()),0)
    def test_episode_reset_is_independent(self):
        x=self.cache['features'].unsqueeze(0);a=self.net.encode(x)[0];self.net.encode(x*2)
        self.assertTrue(torch.equal(a,self.net.encode(x)[0]))
    def test_padding_cannot_update_terminated_memory(self):
        x=self.cache['features'].unsqueeze(0);alive=torch.tensor([[True]*6+[False]*6]);a,_=o.encode_masked(self.net,x,alive)
        self.assertTrue(torch.equal(a[:,5],a[:,-1]))
    def test_b2_uses_analytic_query_composition(self):
        z=torch.tensor([[1.,1.,1.,1.],[0.,0.,1.,0.]])
        q=torch.tensor([[1.,1.,0.,1.]]*2)
        self.assertEqual(o.compose_state_probability(z,q).tolist(),[1.,0.])

if __name__=='__main__':unittest.main()
