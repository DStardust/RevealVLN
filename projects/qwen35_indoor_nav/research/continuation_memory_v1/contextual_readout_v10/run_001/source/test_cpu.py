"""Actual CPU readout gradients on immutable real causal features; no GPU run."""
import json
from pathlib import Path
import sys
import unittest
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from model import MemoryPolicy
import train


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        source=HERE.parent/'natural_transfer_v9'
        cls.data=json.loads((source/'DATA.json').read_text())
        cls.cache={k:v.float() for k,v in torch.load(source/'run_001/FEATURES.pt',map_location='cpu',weights_only=True).items()}
        cls.row=next(r for r in cls.data['records'] if r['partition']=='fit')

    def inputs(self):
        ids=self.row['features']
        return self.cache['features'][ids],self.cache['logits'][ids],torch.tensor(self.row['targets'])

    def test_initial_residual_is_exactly_zero_for_any_memory(self):
        torch.manual_seed(1209)
        net=MemoryPolicy(2048,32)
        features,base,_=self.inputs()
        memory=torch.randn(len(features),8,64)
        self.assertTrue(torch.equal(net.action_logits(memory,base,features),base))

    def test_real_updates_reach_writer_and_current_feature_after_zero_init(self):
        torch.manual_seed(1209)
        net=MemoryPolicy(2048,32)
        features,base,target=self.inputs()
        before={k:v.clone() for k,v in net.state_dict().items()}
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=0)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            states,_=net.encode(features[None])
            loss=F.cross_entropy(net.action_logits(states[0],base,features),target)
            loss.backward();opt.step()
        for key in ('writer.weight','recurrent.weight','action.0.weight','action.2.weight'):
            self.assertFalse(torch.equal(before[key],net.state_dict()[key]),key)
        # Current observations can change the correction while memory is fixed.
        held=states[0,0:1].expand(len(features),-1,-1).detach()
        residual=net.action_logits(held,base,features)-base
        self.assertGreater(float(residual.detach().std(0).max()),1e-6)

    def test_short_window_control_is_independent_of_memory(self):
        torch.manual_seed(1209)
        net=MemoryPolicy(2048,32,no_memory=True)
        torch.nn.init.normal_(net.action[-1].weight,std=.01)
        features,base,_=self.inputs()
        first=torch.randn(len(features),8,64,requires_grad=True)
        second=torch.randn_like(first)
        a=net.action_logits(first,base,features)
        b=net.action_logits(second,base,features)
        self.assertTrue(torch.equal(a,b))
        self.assertIsNone(torch.autograd.grad(a.sum(),first,allow_unused=True)[0])

    def test_training_reader_cannot_change_causal_memory_or_action(self):
        torch.manual_seed(1209)
        net=MemoryPolicy(2048,32)
        features,base,_=self.inputs()
        states,_=net.encode(features[None]);memory=states[:,-1]
        before=memory.clone()
        a=net.action_logits(memory,base[-1:],features[-1:])
        net.reader(memory,torch.tensor([[1,2,3,4]]))
        net.reader(memory,torch.tensor([[5,6,7,8]]))
        b=net.action_logits(memory,base[-1:],features[-1:])
        self.assertTrue(torch.equal(memory,before))
        self.assertTrue(torch.equal(a,b))

    def test_actual_training_losses_gradients_and_no_memory_control(self):
        source=HERE.parent/'multifamily_v7'
        data=json.loads((source/'DATA.json').read_text())
        cache={k:v.float() for k,v in torch.load(source/'run_001/FEATURES.pt',map_location='cpu',weights_only=True).items()}
        family=next(f for f in data['families'] if f['split']=='fit')
        batch=train.special.tensors(family,'cpu')
        natural=train.ordinary_batch([self.row],'cpu')
        torch.manual_seed(1209)
        net=MemoryPolicy(2048,len(data['query_vocabulary']))
        initial={k:v.clone() for k,v in net.state_dict().items()}
        for arm in ('N0','B1','B2','Ours'):
            net.load_state_dict(initial);net.no_memory=arm=='N0'
            optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                special_loss,_,_=train.special.losses(net,cache,batch,arm)
                ordinary_loss,_=train.ordinary_loss(net,self.cache,natural)
                loss=special_loss+ordinary_loss
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward();optimizer.step()
            audit=train.gradient_audit(net,cache,batch,arm)
            self.assertEqual(audit['no_memory_control'],arm=='N0')
            self.assertEqual(torch.equal(initial['writer.weight'],net.writer.weight),arm=='N0')
            self.assertEqual(torch.equal(initial['recurrent.weight'],net.recurrent.weight),arm=='N0')
            self.assertFalse(torch.equal(initial['action.2.weight'],net.action[2].weight))


if __name__=='__main__':unittest.main()
