import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import data
import train


class Tests(unittest.TestCase):
    def test_query_categories_do_not_share_local_integer_codes(self):
        item=dict(kind='observe',object_category='chair',room_category='office',min_pixels=256,consecutive_frames=2,order=0)
        query=dict(coordinate_frame='agent_relative_discrete',sequence=[item])
        first=data.query_tokens(query)
        query['sequence'][0]['object_category']='sink'
        second=data.query_tokens(query)
        self.assertNotEqual(first,second)
        self.assertIn('CATEGORY:chair',first)
        self.assertIn('CATEGORY:sink',second)
        self.assertEqual(data.split_houses(['a','b','c','d','e']),data.split_houses(['e','a','c','b','d']))

    def test_batched_loss_preserves_real_owner_masks_and_mean_reader(self):
        torch.set_num_threads(4)
        pilot=HERE.parent/'pilot'
        family=json.loads((pilot/'DATA.json').read_text())
        cache=torch.load(pilot/'run_002/FEATURES.pt',map_location='cpu',weights_only=True)
        old=data.load('v7_test_old_losses',pilot/'run.py')
        torch.manual_seed(1209)
        net=train.models.MemoryPolicy(2048,max(v for q in family['queries'] for v in q)+1)
        batch=train.tensors(family,'cpu')
        old_loss,old_stats,_=old.losses(net,cache,family,'Ours')
        new_loss,new_stats,_=train.losses(net,cache,batch,'Ours')
        self.assertAlmostEqual(float(old_loss),float(new_loss),places=5)
        self.assertAlmostEqual(old_stats['action_ce'],new_stats['action_ce'],places=5)
        self.assertAlmostEqual(old_stats['crossed_result_bce'],new_stats['crossed_result_bce'],places=5)
        self.assertEqual(new_stats['action_owners'],492)
        old_grad=torch.autograd.grad(old_loss,net.writer.weight)[0]
        new_grad=torch.autograd.grad(new_loss,net.writer.weight)[0]
        self.assertTrue(torch.allclose(old_grad,new_grad,atol=2e-6,rtol=2e-5))

    def test_padded_query_and_future_firewall(self):
        torch.manual_seed(1209)
        net=train.models.MemoryPolicy(6,10,slots=2,width=4)
        features=torch.randn(2,20,6)
        states,_=net.encode(features)
        memory=states[:,-1]
        before=memory.detach().clone()
        base=torch.randn(2,4)
        action=net.action_logits(memory,base).detach().clone()
        batch=dict(queries=torch.tensor([[1,2,0],[3,4,5]]),query_lengths=[2,3],cell_prefix=torch.tensor([0,1]),cell_query=torch.tensor([0,1]))
        packed=train.reader_logits(net,memory,batch)
        unpadded=torch.cat([net.reader(memory[i:i+1],batch['queries'][i:i+1,:length]) for i,length in enumerate([2,3])])
        self.assertTrue(torch.allclose(packed,unpadded,atol=1e-6))
        self.assertTrue(torch.equal(before,memory))
        self.assertTrue(torch.equal(action,net.action_logits(memory,base)))


if __name__=='__main__':unittest.main()
