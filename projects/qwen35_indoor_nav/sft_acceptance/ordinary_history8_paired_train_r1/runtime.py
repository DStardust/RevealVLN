"""Shared frozen-policy initialization and four/eight-frame data for both arms."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[1]
ROOT=LINE.parents[1]
BRIDGE=HERE.parent/'ordinary_onpolicy_fp32_master_v8/initial_fp32_master_from_best4000.pt'
BRIDGE_SHA='aa4e3d3bb9be97e4b906bc56834c3cc008c17eeef52ed51c24ac0bbb052466e5'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def configure():
    speed=HERE.parent/'ordinary_speedup_10x_v1'
    for path in (speed/'official_einops_0_8_1/deps',speed/'official_fla_0_5_2/deps'):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    import fla.ops.gated_delta_rule
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    bound=modeling.torch_chunk_gated_delta_rule
    cells=dict(zip(bound.__code__.co_freevars,(v.cell_contents for v in bound.__closure__)))
    assert getattr(cells.get('implementation'),'__module__','').startswith('fla.')
def initialize():
    import torch
    assert sha(BRIDGE)==BRIDGE_SHA
    model=load('paired_model',HERE/'model.py')
    policy=model.build_policy(1209)
    state=torch.load(BRIDGE,map_location='cpu',weights_only=True)
    model.load_trainable(policy,state['trainable'])
    assert len(state['trainable'])==28 and all(p.dtype==torch.float32 for p in policy.parameters() if p.requires_grad)
    for name,p in policy.named_parameters():
        if p.requires_grad:assert torch.equal(p.detach().cpu(),state['trainable'][name])
    policy.eval()
    return model,policy,state
def ordinary():
    return load('paired_ordinary_data',HERE.parent/'ordinary_prefix_history8_v1/data.py')
def stores(rows):
    data=ordinary()
    # Metadata cache bounded by this finite FIT snapshot; no pixel tensor cache.
    # Avoid repeatedly revalidating every route after an arbitrary 128-record eviction.
    class PrefixStore(data.SampleStore):
        def record(self,record_idx):
            if type(record_idx) is not int or not 0<=record_idx<len(self.rows):raise ValueError('RECORD_INDEX')
            if record_idx not in self.records:
                row=self.rows[record_idx]
                if row['split']!='FIT':raise ValueError('NON_FIT_ROW')
                self.records[record_idx]=data.decoder['OrdinaryRecord'](row)
            return self.records[record_idx]
    return {'control_recent2':data.base['SampleStore'](rows),'treatment_prefix8':PrefixStore(rows)}
def fingerprint(policy):
    import torch
    h=hashlib.sha256()
    for name,p in policy.named_parameters():
        if p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def save(path,value,exclusive=False):
    path=Path(path);dest=path if exclusive else path.with_suffix(path.suffix+'.tmp')
    with dest.open('x' if exclusive else 'w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    if not exclusive:dest.replace(path)


