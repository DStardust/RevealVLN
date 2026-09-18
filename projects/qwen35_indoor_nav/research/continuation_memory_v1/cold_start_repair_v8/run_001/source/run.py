"""Reuse the real frozen V7 features; train only the fixed cold-start repair."""
from pathlib import Path
import sys
import time
import traceback
import torch

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
PRIOR=HERE.parent/'multifamily_v7'
sys.path.insert(0,str(HERE))
import train
c=train.c


@torch.no_grad()
def preservation_readback(run,cache,data,config):
    features=cache['features'].to('cuda:0').float()
    native=cache['logits'].to('cuda:0').float()
    rows=[]
    for seed in config['seeds']:
        for arm in config['arms']:
            key=f'{arm}_{seed}'
            measured={}
            for version,folder in (('original',PRIOR/'run_001'),('repair',run)):
                c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='PRESERVATION_READBACK',key=key,version=version))
                net=train.models.MemoryPolicy(2048,len(data['query_vocabulary']),8,64,.99).cuda().eval()
                net.load_state_dict(torch.load(folder/f'{key}_MEMORY.pt',map_location='cpu',weights_only=True))
                expected=c.read(folder/'RESULT.json')['runs'][key]['final_state_sha256']
                assert c.model_identity(net)['sha256']==expected,'READBACK_CHECKPOINT_CHANGED'
                values=[]
                for family in data['families']:
                    indices=torch.tensor([p['features'][:156] for p in family['prefixes']],device='cuda:0')
                    states,_=net.encode(features[indices])
                    value=train.cold_start_preservation(net,states,native[indices],156)
                    assert bool(torch.isfinite(value))
                    values.append(dict(family_id=family['family_id'],split=family['split'],kl=float(value)))
                measured[version]=values
                del net
            rows.append(dict(key=key,values=measured))
    c.write(run/'PRESERVATION_READBACK.json',dict(same_causal_inputs=True,optimizer_updates=0,
        qwen_forwards=0,prefix_decisions=156,runs=rows,
        interpretation='Matched cached-input KL measures the training repair; it is not navigation benefit.'),True)


def main(run):
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    config=c.read(HERE/'PROTOCOL.json')
    for path,digest in config['source_hashes'].items():assert c.sha(LINE/path)==digest,'SOURCE_CHANGED:'+path
    assert config['cold_start_decisions']==156 and config['preservation_weight']==1
    c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='LOAD_FROZEN_FEATURE_CACHE'))
    identity=c.read(PRIOR/'run_001/FEATURE_RESULT.json')
    assert c.sha(PRIOR/'run_001/FEATURES.pt')==identity['file_sha256']
    assert c.sha(PRIOR/'DATA.json')==identity['data_sha256']
    data=c.read(PRIOR/'DATA.json')
    first_action=min(x['step'] for f in data['families'] if f['split']=='fit' for cell in f['cells'] for x in cell['tail'] if x['mask'])
    assert first_action==config['cold_start_decisions'],'TRAINING_SUPPORT_CHANGED'
    cache=torch.load(PRIOR/'run_001/FEATURES.pt',map_location='cpu',weights_only=True)
    assert all(bool(torch.isfinite(t).all()) for t in cache.values())
    c.write(run/'FEATURE_RESULT.json',dict(identity,real_qwen_forwards=0,
        reused_from=str((PRIOR/'run_001/FEATURE_RESULT.json').relative_to(LINE)),
        original_real_forwards=identity['real_qwen_forwards'],cache_recomputed=False),True)
    c.write(run/'RUNTIME_IDENTITY.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(),gpu_uuid=c.read(run/'CONFIG.json')['gpu_uuid'],
        dtype='float32 memory, unchanged frozen cached feature/logit values',
        qwen_loaded_in_this_run=False,base_state_sha256=identity['base_state_sha256'],
        feature_file_sha256=identity['file_sha256'],tf32=torch.backends.cuda.matmul.allow_tf32,
        qwen_attention_and_fla_dispatch='not executed; immutable V7 cache provenance applies',
        protocol_sha256=c.sha(HERE/'PROTOCOL.json')),True)
    train.train(run,config,cache,data)
    preservation_readback(run,cache,data,config)


if __name__=='__main__':
    run=Path(sys.argv[1])
    try:main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
