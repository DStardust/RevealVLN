"""Matched fresh heads: concatenation versus attention and centered memory read."""
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from torch import nn
from memory_v2 import ExecutionMemory


class RecoveryMemory(ExecutionMemory):
    def __init__(self,width,arm):
        super().__init__(width)
        assert arm in ('CONCAT','EVIDENCE')
        self.arm=arm
        self.query=nn.Linear(width,64,bias=False)
        self.key=nn.Linear(64,64,bias=False)

    def action_delta(self,feature,memory):
        current=self.norm(feature)
        if self.arm=='CONCAT':return self.actor(torch.cat((current,memory.flatten(1)),-1))
        weights=torch.softmax((self.key(memory)*self.query(current)[:,None,:]).sum(-1)/math.sqrt(64),-1)
        context=(8*weights[:,:,None]*memory).flatten(1)
        return self.actor(torch.cat((current,context),-1))-self.actor(torch.cat((current,torch.zeros_like(context)),-1))


def sequence_logits(model,row):
    lookup={int(t):i for i,t in enumerate(row['query_steps'])}
    memory=model.reset();deltas=[]
    for t,x in enumerate(row['memory_features']):
        memory=model.update(x[None],memory)
        if t in lookup:
            q=lookup[t];deltas.append(model.action_delta(row['actor_features'][q:q+1],memory)[0])
    assert len(deltas)==len(lookup) and deltas
    # Transformers 4.45.1 casts generation scores to FP32 before processors,
    # even though the frozen backbone itself runs in BF16.
    return row['base_logits'].float()+torch.stack(deltas)
