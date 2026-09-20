"""Frozen V5 loader copied with V16 identity path; one cache/live forward implementation."""
import gc
import os
import sys
import time
from pathlib import Path
import torch
from v16_common import *

def load_policy(session, p):
    speed = c.LINE/'sft_acceptance/ordinary_speedup_10x_v1'
    sys.path.insert(0, str(speed/'official_einops_0_8_1/deps'))
    sys.path.insert(0, str(speed/'official_fla_0_5_2/deps'))
    import fla.ops.gated_delta_rule
    import torch
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(p['model_memory_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    bound = modeling.torch_chunk_gated_delta_rule
    cells = dict(zip(bound.__code__.co_freevars, (x.cell_contents for x in bound.__closure__)))
    dispatch = getattr(cells.get('implementation'), '__module__', '')
    assert dispatch.startswith('fla.'), 'FLA_DISPATCH_CHANGED'
    model = c.load('v5_frozen_model', c.TRAIN/'model.py')
    assert c.sha(Path(p['checkpoint'])) == p['checkpoint_sha256'], 'CHECKPOINT_HASH'
    state = torch.load(p['checkpoint'], map_location='cpu', weights_only=True)
    assert state['binding']['protocol_sha256'] == p['training_protocol_sha256']
    assert state['binding']['sample_index_sha256'] == p['sample_index_sha256']
    assert state['cursor']['updates'] == 4000
    assert all(bool(torch.isfinite(t).all()) for t in state['trainable'].values())
    policy = model.build_policy(p['seed'])
    model.load_trainable(policy, state['trainable'])
    policy.eval()
    del state
    gc.collect()
    initial = c.model_identity(policy)
    c.write(session/'STATE_INITIAL.json', initial, True)
    import importlib.metadata
    versions = {}
    for package in ['torch','transformers','peft','triton','flash-linear-attention','numpy']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = 'unavailable'
    base_files = {str(x.relative_to(model.MODEL)): c.sha(x) for x in model.MODEL.iterdir()
                  if x.is_file() and x.suffix in ('.json','.safetensors','.jinja')}
    tokenizer = policy.processor.tokenizer.backend_tokenizer.to_str().encode()
    import hashlib
    identity = dict(session=session.name, pid=os.getpid(), gpu_uuid=p['gpu_uuid'], physical_gpu=p['gpu'],
        software=versions, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
        checkpoint_sha256=p['checkpoint_sha256'], base_files=base_files,
        model_source_sha256=c.sha(c.TRAIN/'model.py'), loaded_state_sha256=initial['sha256'],
        tokenizer_sha256=hashlib.sha256(tokenizer).hexdigest(), processor=policy.processor.to_dict(),
        image_processor=policy.processor.image_processor.to_dict(), attention=policy.base.config._attn_implementation,
        fla_dispatch=dispatch, triton_selected_config='unavailable before compilation; cache metadata saved at session end',
        dtype='unchanged BF16 base and FP32 action head', source_hashes=p['source_hashes'],
        protocol_sha256=p.get('v16_protocol_sha256', c.sha(HERE/'PROTOCOL.json')), model_loads=1, optimizer_updates=0)
    c.write(session/'RUNTIME_IDENTITY.json', identity, True)
    return model, policy, initial

class Forward:
    """No target, query, pose or program state in model inputs."""
    def __init__(self,model,policy,store,records):
        self.policy=policy;self.captured=[]
        self.dataset=model.DecisionDataset(records,store,policy.processor)
        self.collate=model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
        self.hook=policy.action_head.register_forward_pre_hook(lambda module,args:self.captured.append(args[0]))
    def prepare(self,index):
        began=time.perf_counter();batch=self.collate([self.dataset[index]])
        batch.pop('targets');batch.pop('weights')
        processed={k:c.tensor_identity(v) for k,v in batch.items()}
        return batch,processed,time.perf_counter()-began
    def __call__(self,index):
        batch,processed,preprocess=self.prepare(index);started=time.perf_counter()
        logits=self.policy.forward_batch(**{k:v.cuda() for k,v in batch.items()});torch.cuda.synchronize()
        if len(self.captured)!=1:raise ValueError('ACTION_FEATURE_CAPTURE')
        feature=self.captured.pop().float();logits=logits.float()
        if not bool(torch.isfinite(feature).all()) or not bool(torch.isfinite(logits).all()):raise ValueError('NONFINITE_BASE_FORWARD')
        return feature,logits,processed,dict(preprocess_seconds=preprocess,inference_seconds=time.perf_counter()-started)
    def close(self):self.hook.remove()

class RawStore:
    def __init__(self,data):
        from functools import lru_cache
        self.data=data
        loader=load('v16_raw_array_reader',LINE/'data_pipeline/mechanism_runtime_v1/loader.py')
        @lru_cache(maxsize=128)
        def pixels(ref):
            info=data['contents'][ref];path=loader.safe_path(LINE,info['line_relative_path']);blob=path.read_bytes()
            if loader.sha(blob)!=info['file_sha256']:raise ValueError('RAW_ARRAY_FILE_HASH')
            return loader.npy_pixels(blob,ref[7:],'rgb')
        self.pixels=pixels
    def get(self,index,t):
        from PIL import Image
        row=self.data['features'][index]
        return dict(instruction=row['instruction'],images=[Image.frombytes('RGB',(224,224),self.pixels(ref)) for ref in row['rgb_refs']],executed=row['executed'])

def warmup(forward,data,config):
    indices=[i for i,r in enumerate(data['features']) if r['split'] in ('FIT','DEV')]
    selected=[indices[int(i*len(indices)/min(len(indices),config['warmup_indices']))] for i in range(min(len(indices),config['warmup_indices']))]
    hashes=[]
    with torch.inference_mode():
        for _ in range(config['warmup_repeats']):
            for i in selected:
                f,l,p,t=forward(i);hashes.append(dict(index=i,processed=p,feature=c.tensor_identity(f),logits=l[0].cpu().tolist()))
    return hashes
