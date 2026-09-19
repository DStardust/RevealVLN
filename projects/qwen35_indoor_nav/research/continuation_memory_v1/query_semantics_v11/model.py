"""V10 runtime memory/action path; new training-only structured-query reader."""
import importlib.util
from pathlib import Path
import torch
from torch import nn

spec=importlib.util.spec_from_file_location('v11_prior_model',Path(__file__).resolve().parent.parent/'contextual_readout_v10/model.py')
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


class MemoryPolicy(prior.MemoryPolicy):
    def __init__(self,feature_width,vocabulary_size,slots=8,width=64,retention=.99,*,no_memory=False):
        super().__init__(feature_width,vocabulary_size,slots,width,retention,no_memory=no_memory)
        del self.query_embedding
        del self.query_gru
        self.result_head=nn.Sequential(nn.Linear(slots*width+4,128),nn.Tanh(),nn.Linear(128,1))

    def reader(self,memory,query_features):
        assert query_features.ndim==2 and query_features.shape[-1]==4
        return self.result_head(torch.cat([memory.flatten(1),query_features],-1)).squeeze(-1)
