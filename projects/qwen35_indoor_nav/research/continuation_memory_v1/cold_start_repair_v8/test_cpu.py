import json
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train
prior=train.c.load('v8_cpu_prior_train',HERE.parent/'multifamily_v7/train.py')


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(1209)

    def test_preservation_stops_before_first_supervised_action(self):
        net=train.models.MemoryPolicy(6,10,slots=2,width=4)
        states=torch.randn(2,170,2,4,requires_grad=True)
        native=torch.randn(2,170,4)
        loss=train.cold_start_preservation(net,states,native,156)
        gradient=torch.autograd.grad(loss,states)[0]
        self.assertGreater(float(loss.detach()),0)
        self.assertGreater(float(gradient[:,:156].norm()),0)
        self.assertEqual(int(torch.count_nonzero(gradient[:,156:])),0)
        altered=states.detach().clone();altered[:,156:]*=100
        self.assertTrue(torch.equal(loss.detach(),train.cold_start_preservation(net,altered,native,156)))

    def test_native_equivalence_and_actual_writer_gradient(self):
        net=train.models.MemoryPolicy(6,10,slots=2,width=4)
        features=torch.randn(2,170,6)
        states,_=net.encode(features)
        native=torch.randn(2,170,4)
        loss=train.cold_start_preservation(net,states,native,156)
        gradients=torch.autograd.grad(loss,[net.writer.weight,net.recurrent.weight,net.action.weight])
        self.assertTrue(all(bool(torch.isfinite(g).all()) and float(g.norm())>0 for g in gradients))
        with torch.no_grad():net.action.weight.zero_()
        self.assertLess(abs(float(train.cold_start_preservation(net,states.detach(),native,156).detach())),1e-6)

    def test_original_real_losses_plus_exactly_one_preservation_term(self):
        pilot=HERE.parent/'pilot'
        family=json.loads((pilot/'DATA.json').read_text())
        cache=torch.load(pilot/'run_002/FEATURES.pt',map_location='cpu',weights_only=True)
        net=train.models.MemoryPolicy(2048,max(t for q in family['queries'] for t in q)+1)
        batch=train.tensors(family,'cpu')
        states,_=net.encode(cache['features'][batch['prefix_indices']])
        preservation=train.cold_start_preservation(net,states,cache['logits'][batch['prefix_indices']],156)
        for mode in ('B1','B2','Ours'):
            old_loss,old_stats,_=prior.losses(net,cache,batch,mode)
            new_loss,new_stats,_=train.losses(net,cache,batch,mode)
            self.assertAlmostEqual(float(new_loss.detach()),float((old_loss+preservation).detach()),places=5)
            self.assertEqual(new_stats['action_owners'],old_stats['action_owners'])
            self.assertAlmostEqual(new_stats['action_ce'],old_stats['action_ce'],places=6)
            self.assertAlmostEqual(new_stats['crossed_result_bce'],old_stats['crossed_result_bce'],places=6)
            self.assertAlmostEqual(new_stats['exact_state_bce'],old_stats['exact_state_bce'],places=6)


if __name__=='__main__':unittest.main()
