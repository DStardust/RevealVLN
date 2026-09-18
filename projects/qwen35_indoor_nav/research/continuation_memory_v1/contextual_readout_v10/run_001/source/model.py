"""One candidate repair: condition the residual readout on the current feature.

Base encoder, recurrent update, memory size, and training query reader are reused.
The short-window control computes the same memory but removes it from the action
readout; its unused recurrent parameters must not be claimed as trained by BC.
"""
import importlib.util
from pathlib import Path
import torch
from torch import nn

spec=importlib.util.spec_from_file_location('v10_prior_memory',Path(__file__).resolve().parent.parent/'query_reader_repair_v2/model.py')
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


class MemoryPolicy(prior.MemoryPolicy):
    def __init__(self,feature_width,vocabulary_size,slots=8,width=64,retention=.99,*,no_memory=False):
        super().__init__(feature_width,vocabulary_size,slots,width,retention)
        self.no_memory=no_memory
        self.action=nn.Sequential(nn.Linear(feature_width+slots*width,128),nn.Tanh(),nn.Linear(128,4,bias=False))
        nn.init.zeros_(self.action[-1].weight)

    def action_logits(self,memory,base_logits,current_feature):
        history=torch.zeros_like(memory) if self.no_memory else memory
        inputs=torch.cat([self.feature_norm(current_feature),history.flatten(1)],dim=-1)
        return base_logits+self.action(inputs)
