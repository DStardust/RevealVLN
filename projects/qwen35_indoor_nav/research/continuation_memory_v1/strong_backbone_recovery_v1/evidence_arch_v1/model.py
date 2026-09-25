"""CPU-ready architecture candidate; not connected to active rollout jobs.

Adds query-dependent memory reading and a centered correction to the existing
8x64 writer. All weights start fresh. No base model or old adapter is loaded.
"""
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from memory_v2 import ExecutionMemory


class EvidenceMemory(ExecutionMemory):
    def __init__(self,feature_width):
        super().__init__(feature_width)
        self.query=nn.Linear(feature_width,64,bias=False)
        self.key=nn.Linear(64,64,bias=False)
        self.actor[-1]=nn.Linear(128,4,bias=False)
        nn.init.zeros_(self.actor[-1].weight)
        self.intervention=nn.Sequential(nn.Linear(512+8,64),nn.SiLU(),nn.Linear(64,3))
        nn.init.zeros_(self.intervention[-1].weight)
        nn.init.zeros_(self.intervention[-1].bias)

    def read_memory(self,feature,memory):
        query=self.query(self.norm(feature))
        weights=torch.softmax((self.key(memory)*query[:,None,:]).sum(-1)/math.sqrt(64),dim=-1)
        context=(weights[:,:,None]*memory*8).flatten(1)
        return context,weights

    def correction(self,feature,memory):
        context,weights=self.read_memory(feature,memory);current=self.norm(feature)
        with_memory=self.actor(torch.cat((current,context),dim=-1))
        without_memory=self.actor(torch.cat((current,torch.zeros_like(context)),dim=-1))
        return with_memory-without_memory,context,weights

    def action_delta(self,feature,memory):
        # Compatible with the existing causal unroll/action-supervision interface.
        return self.correction(feature,memory)[0]

    def policy_logits(self,feature,memory,native_logits):
        delta,context,weights=self.correction(feature,memory)
        centered=(native_logits-native_logits.mean(-1,keepdim=True))/10
        intervention_logits=self.intervention(torch.cat((context,centered,delta),dim=-1))
        probability=intervention_logits.softmax(-1)
        accept=probability[:,1]>2*probability[:,2]
        final=native_logits+torch.where(accept[:,None],delta,torch.zeros_like(delta))
        return dict(logits=final,proposed_logits=native_logits+delta,delta=delta,memory_attention=weights,
            intervention_logits=intervention_logits,accept=accept)
