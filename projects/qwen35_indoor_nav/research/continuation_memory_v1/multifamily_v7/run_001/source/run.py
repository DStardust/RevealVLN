"""Load best4k once, freeze causal features, then run the fixed matched comparison."""
from functools import lru_cache
import gc
import json
from pathlib import Path
import sys
import time
import traceback
import torch

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
sys.path.insert(0,str(HERE))
import data as source
import train
c=train.c


def features(run,config,data):
    from evaluate import load_policy
    from PIL import Image
    loader=source.load('v7_pixels',LINE/'data_pipeline/mechanism_runtime_v1/loader.py')
    runtime=c.read(run/'CONFIG.json')
    c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='LOAD_FROZEN_BEST4K'))
    model,policy,initial=load_policy(run,runtime)
    original=c.read(next((LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5/sessions').glob('session_*/STATE_INITIAL.json')))
    assert initial['sha256']==original['sha256'],'BASE_IDENTITY_CHANGED'
    for parameter in policy.parameters():parameter.requires_grad_(False)
    collate=model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
    @lru_cache(maxsize=256)
    def pixels(reference):
        item=data['contents'][reference]
        path=loader.safe_path(LINE,item['line_relative_path'])
        blob=path.read_bytes()
        assert loader.sha(blob)==item['file_sha256'],'CONTENT_CHANGED'
        return loader.npy_pixels(blob,reference[7:],'rgb')
    class Store:
        def get(self,index,t):
            record=data['features'][index]
            return dict(instruction=record['instruction'],
                images=[Image.frombytes('RGB',(224,224),pixels(ref)) for ref in record['rgb_refs']],
                executed=record['executed'])
    records=[dict(record_idx=i,t=0,target=0,weight=1.) for i in range(len(data['features']))]
    dataset=model.DecisionDataset(records,Store(),policy.processor)
    capture=[]
    handle=policy.action_head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach().cpu().clone()))
    feature_rows,logit_rows=[],[]
    parts=[]
    try:
        with torch.inference_mode():
            for i in range(len(records)):
                batch=collate([dataset[i]])
                batch.pop('targets');batch.pop('weights')
                processed={k:c.tensor_identity(v) for k,v in batch.items()}
                logits=policy.forward_batch(**{k:v.to('cuda:0') for k,v in batch.items()})
                torch.cuda.synchronize()
                assert len(capture)==1
                value=capture.pop()
                assert bool(torch.isfinite(value).all()) and bool(torch.isfinite(logits).all())
                feature_rows.append(value);logit_rows.append(logits.detach().float().cpu())
                c.append(run/'FEATURE_INPUTS.jsonl',dict(index=i,input_key=data['features'][i]['key'],processed=processed))
                c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='FEATURES',completed=i+1,total=len(records)))
                if len(feature_rows)==2048 or i+1==len(records):
                    part=dict(features=torch.cat(feature_rows),logits=torch.cat(logit_rows))
                    path=run/f'FEATURE_PART_{len(parts):03d}.pt'
                    torch.save(part,path)
                    final=c.model_identity(policy)
                    assert final['sha256']==initial['sha256'],'FROZEN_STATE_CHANGED'
                    c.write(path.with_suffix('.json'),dict(end_index=i+1,rows=len(part['features']),
                        sha256=c.sha(path),state_sha256=final['sha256'],unchanged=True),True)
                    parts.append(path);feature_rows.clear();logit_rows.clear()
        loaded=[torch.load(path,map_location='cpu',weights_only=True) for path in parts]
        cache={key:torch.cat([part[key] for part in loaded]) for key in ('features','logits')}
        torch.save(cache,run/'FEATURES.pt')
        c.write(run/'FEATURE_RESULT.json',dict(real_qwen_forwards=len(records),feature_shape=list(cache['features'].shape),
            file_sha256=c.sha(run/'FEATURES.pt'),data_sha256=c.sha(HERE/'DATA.json'),base_state_sha256=initial['sha256'],
            encoder_parameters_unchanged=True,encoder_updates=0,future_query_in_encoder=False,
            input_contract='original instruction, last <=2 RGB, last <=8 actually executed motion actions'),True)
    finally:handle.remove()
    pixels.cache_clear()
    del policy,model,dataset,initial,final,capture,loaded,part,logits,batch
    gc.collect();torch.cuda.empty_cache()
    return cache


def main(run):
    config=c.read(HERE/'PROTOCOL.json')
    for path,digest in config['source_hashes'].items():
        assert c.sha(LINE/path)==digest,'SOURCE_CHANGED:'+path
    data=c.read(HERE/'DATA.json')
    cache=features(run,config,data)
    train.train(run,config,cache,data)


if __name__=='__main__':
    run=Path(sys.argv[1])
    try:main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
