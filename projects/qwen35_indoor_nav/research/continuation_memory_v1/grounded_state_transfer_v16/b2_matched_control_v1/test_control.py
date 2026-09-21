"""Real CPU losses/gradients plus scheduling and full denominator contracts."""
import copy
from collections import Counter
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import objective as o
from train import schedule_for_seed
from evaluate_continuations import registry_value,prefix_audit
from select_action import select


class Control(unittest.TestCase):
    def test_parent_and_variant_balance(self):
        families=[dict(parent_family_id=str(i),family_id=str(i)+'_'+v) for i in range(32) for v in ('present','absent') if v=='present' or i<27]
        s=schedule_for_seed(families,[[0],[1]],[0,1],1209)
        self.assertEqual(s,schedule_for_seed(families,[[0],[1]],[0,1],1209))
        self.assertEqual(len(s),1200)
        self.assertEqual(set(Counter(r['parent_family'] for r in s).values()),{37,38})
        for i in range(27):
            n=Counter(r['family'] for r in s if r['parent_family']==str(i))
            self.assertLessEqual(abs(n[str(i)+'_present']-n[str(i)+'_absent']),1)

    def test_no_nonfit_ordinary(self):
        with self.assertRaises(ValueError):schedule_for_seed([dict(parent_family_id='p',family_id='p')],[[7]],[0],1209)

    def test_registry_and_shards(self):
        families=[dict(family_id=str(i),parent_family_id=str(i%8),house='DEV',split='DEV',stratum='terminal_present' if i<8 else 'terminal_absent') for i in range(16)]
        reg=registry_value(families,dict(seeds=[1209,1210,1211],arms=['B1','B2','Terminal']))
        self.assertEqual(len(reg['slots']),1152)
        self.assertEqual(len(reg['conditions']),128)
        self.assertEqual(reg['expected_prefix_comparisons'],768)
        for gpu in range(8):
            subset=set(range(128)[gpu::8]);self.assertEqual(len(subset),16)
            self.assertEqual(sum(r['condition'] in subset for r in reg['slots']),144)

    def test_terminal_loss_ignores_other_bits_and_reaches_memory(self):
        torch.set_num_threads(2);net=o.initialize(1209)
        row=dict(features=list(range(12)),targets=[0]*11+[3],action_masks=[1]*12,
                 state_targets=[[0,0,int(i%2),0] for i in range(12)],state_masks=[1]*12,
                 y=[0]*12,query_masks=[1]*12,query_contexts=[dict(zip(o.FIELDS,[0,1,0,1]))]*12,cutoff=10)
        family=dict(sequences=[row]);b=o.batch(family,'cpu');cache=dict(features=torch.randn(12,2048),logits=torch.randn(12,4))
        weights=dict(state=[[1.,1.]]*4,query=[1.,1.])
        _,stats,d=o.losses(net,cache,b,'Terminal',weights,retain_steps=(1,))
        changed=copy.deepcopy(b);changed['state'][...,[0,1,3]]=1-changed['state'][...,[0,1,3]]
        _,_,other=o.losses(net,cache,changed,'Terminal',weights)
        self.assertTrue(torch.equal(d['auxiliary_loss'],other['auxiliary_loss']))
        grad=torch.autograd.grad(d['auxiliary_loss'],d['writes'][1],retain_graph=True)[0]
        self.assertGreater(float(grad.norm()),0)
        self.assertEqual(stats['state_supervisions'],12)
        self.assertEqual(stats['query_supervisions'],0)
        old=net.writer.weight.detach().clone();optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
        d['auxiliary_loss'].backward();optimizer.step()
        self.assertFalse(torch.equal(old,net.writer.weight))

    def test_final_selector_overrides_native_stop(self):
        self.assertEqual(select([0,0,0,9],[9,0,0,0])['executed_action'],'move_forward')
        with self.assertRaises(ValueError):select([0,0,0,1],[0,0,0,float('nan')])


if __name__=='__main__':unittest.main()
