"""Actual CPU gradients and boundary checks for the natural trajectory addition."""
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import data
import train


class NaturalTransferTests(unittest.TestCase):
    def test_selection_is_route_unique_and_house_isolated(self):
        sealed=train.c.read(HERE/'DATA.json')
        records=sealed['records']
        self.assertEqual(len({r['row']['physical_source_route_sha256'] for r in records}),len(records))
        self.assertFalse(set(sealed['fit_houses']) & set(sealed['check_houses']))
        self.assertEqual(sealed['decisions'],sum(len(r['targets']) for r in records))
        self.assertTrue(all(r['targets'][-1]==3 and 3 not in r['targets'][:-1] for r in records))

    def test_batched_padding_matches_individual_loss_and_gradient(self):
        torch.manual_seed(1209)
        net=train.models.MemoryPolicy(8,9,2,3,.99)
        cache=dict(features=torch.randn(8,8),logits=torch.randn(8,4))
        rows=[dict(features=[0,1,2,3,4],targets=[0,0,1,2,3]),dict(features=[5,6,7],targets=[1,0,3])]
        batch=train.ordinary_batch(rows,'cpu')
        loss,_=train.ordinary_loss(net,cache,batch)
        gradient=torch.autograd.grad(loss,net.writer.weight)[0]
        individual=torch.stack([train.ordinary_loss(net,cache,train.ordinary_batch([r],'cpu'))[0] for r in rows]).mean()
        other=torch.autograd.grad(individual,net.writer.weight)[0]
        torch.testing.assert_close(loss,individual)
        torch.testing.assert_close(gradient,other)
        self.assertGreater(float(gradient.norm()),0)

    def test_real_input_excludes_current_target_and_future_frames(self):
        sealed=train.c.read(HERE/'DATA.json')
        adapter=data.load('v9_cpu_adapter',data.LINE/'sft_acceptance/ordinary_baseline_v2/data.py')
        record=adapter.OrdinaryRecord(sealed['records'][0]['row'])
        for step in (0, min(12,len(record)-1),len(record)-1):
            value=record.decision_metadata(step)
            self.assertEqual(value['policy']['executed_actions'],list(record.actions[max(0,step-8):step]))
            self.assertEqual(value['control']['rgb_refs'],list(record._refs[max(0,step-1):step+1]))
            self.assertEqual(set(value['policy']),{'instruction','executed_actions'})
            self.assertEqual(value['supervision']['target_action'],record.actions[step])

    def test_actual_parameter_update_does_not_change_cached_encoder_features(self):
        torch.manual_seed(1209)
        net=train.models.MemoryPolicy(8,9,2,3,.99)
        cache=dict(features=torch.randn(12,8),logits=torch.randn(12,4))
        original=cache['features'].clone()
        rows=[dict(features=list(range(12)),targets=[0,1,2]*3+[0,1,3])]
        batch=train.ordinary_batch(rows,'cpu')
        before={k:v.clone() for k,v in net.state_dict().items()}
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        loss,_=train.ordinary_loss(net,cache,batch);loss.backward();opt.step()
        for key in ('writer.weight','recurrent.weight','action.weight'):
            self.assertFalse(torch.equal(before[key],net.state_dict()[key]))
        self.assertTrue(torch.equal(original,cache['features']))


if __name__=='__main__':unittest.main()
