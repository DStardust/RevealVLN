"""Frozen 113-parameter STOP residual, clamped nonnegative at inference. Original state and motion logits stay frozen."""
import torch
from torch import nn

class StopReadout(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+20000)
            self.net=nn.Sequential(nn.Linear(5,16),nn.Tanh(),nn.Linear(16,1))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,logits,state):
        if logits.shape!=state.shape or logits.shape[-1]!=4:raise ValueError('READOUT_INPUT_SHAPE')
        margin=logits[...,3:4]-logits[...,:3].max(-1,keepdim=True).values
        delta=self.net(torch.cat((state,margin),-1)).clamp_min(0)
        return torch.cat((logits[...,:3],logits[...,3:4]+delta),-1)

class StopPolicy(nn.Module):
    def __init__(self,frozen,seed):
        super().__init__();self.frozen=frozen;self.readout=StopReadout(seed)
        for parameter in self.frozen.parameters():parameter.requires_grad_(False)
    def reset(self,batch_size,device):return self.frozen.reset(batch_size,device)
    def step(self,feature,native_logits,state):
        with torch.no_grad():logits,updated,detail=self.frozen.step(feature,native_logits,state)
        repaired=self.readout(logits,detail['state'])
        return repaired,updated,detail
