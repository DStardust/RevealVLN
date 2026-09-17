"""Frozen causal feature prototype: eight recurrent slots read by the action head.

Future queries are accepted only by reader(); no KV/history text or task oracle.
"""
import torch
from torch import nn


class MemoryPolicy(nn.Module):
    def __init__(self,feature_width,vocabulary_size,slots=8,width=64,retention=.99):
        super().__init__()
        self.slots,self.width,self.retention=slots,width,retention
        size=slots*width
        self.feature_norm=nn.LayerNorm(feature_width,elementwise_affine=False)
        self.writer=nn.Linear(feature_width,size)
        self.recurrent=nn.Linear(size,size,bias=False)
        self.action=nn.Linear(size,4,bias=False)
        self.state_head=nn.Linear(size,4)
        self.query_embedding=nn.Embedding(vocabulary_size,width)
        self.query_gru=nn.GRU(width,width,batch_first=True)
        self.result_head=nn.Sequential(nn.Linear(size+width,128),nn.Tanh(),nn.Linear(128,1))

    def reset(self,batch_size,device):
        return torch.zeros(batch_size,self.slots,self.width,device=device)

    def update(self,feature,memory,retain_write=False):
        old=memory.flatten(1)
        written=self.writer(self.feature_norm(feature))
        if retain_write: written.retain_grad()
        value=self.retention*old+(1-self.retention)*torch.tanh(written+self.recurrent(old))
        return value.reshape_as(memory),written

    def action_logits(self,memory,base_logits):
        return base_logits+self.action(memory.flatten(1))

    def encode(self,causal_features,retain_steps=()):
        memory=self.reset(causal_features.shape[0],causal_features.device)
        states=[];writes={}
        for t in range(causal_features.shape[1]):
            memory,write=self.update(causal_features[:,t],memory,t in retain_steps)
            states.append(memory)
            if t in retain_steps: writes[t]=write
        return torch.stack(states,1),writes

    def reader(self,memory,query_tokens):
        # Called after the causal recurrent forward; query cannot mutate memory.
        _,hidden=self.query_gru(self.query_embedding(query_tokens))
        return self.result_head(torch.cat([memory.flatten(1),hidden[-1]],-1)).squeeze(-1)
