"""Run the exact committed 100000-step causal memory policy on ordinary RGB inputs."""
import sys
from pathlib import Path
import common as u
import torch

def architecture():
    path=u.LINE/'research/continuation_memory_v1/evidence_state_policy_v1'
    saved_common=sys.modules['common'];saved_path=list(sys.path)
    try:
        legacy=u.load('ordinary100k_evidence_common',path/'common.py');sys.modules['common']=legacy
        mod=u.load('ordinary100k_evidence_model',path/'model.py')
    finally:
        sys.modules['common']=saved_common;sys.path[:]=saved_path
    return mod.EvidencePolicy

class Candidate:
    def __init__(self,cfg,device):
        self.device=device;path=Path(cfg['memory_checkpoint'])
        assert u.sha(path)==cfg['memory_checkpoint_sha256'],'MEMORY_CHECKPOINT_CHANGED'
        self.net=architecture()(1209,'MONOTONIC').to(device).eval()
        self.net.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True)
        for p in self.net.parameters():p.requires_grad_(False)
        assert u.c.model_identity(self.net)['sha256']==cfg['memory_state_sha256'],'LOADED_MEMORY_STATE_CHANGED'
        self.reset()
    def reset(self):self.state=self.net.reset(1,self.device);self.updates=0
    def identity(self):return u.c.model_identity(self.net)
    def state_identity(self):return {k:u.tensor_hash(v.clone(memory_format=torch.contiguous_format)) for k,v in self.state._asdict().items()}
    def first(self,h,z):
        out,_,_=self.net.step(h,z,self.net.reset(h.shape[0],h.device));return out[0]
    def step(self,h,z):
        before=self.state_identity();out,self.state,detail=self.net.step(h,z,self.state);self.updates+=1
        return out[0],dict(memory_before=before,memory_after=self.state_identity(),memory_updates=self.updates,
            predicted_state=detail['state'][0].cpu().tolist(),event_probabilities=detail['event_logits'][0].sigmoid().cpu().tolist())
