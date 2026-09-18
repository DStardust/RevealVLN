"""One frozen encoder load for real natural trajectories, then matched memory training."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
sys.path.insert(0, str(HERE))
import train
c = train.c


def features(run, data):
    loader = c.load('v9_frozen_loader', LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5/evaluate.py')
    adapter = c.load('v9_feature_adapter', LINE/'sft_acceptance/ordinary_sync_recovery_v1/data.py')
    c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='LOAD_FROZEN_BEST4K'))
    model, policy, initial = loader.load_policy(run,c.read(run/'CONFIG.json'))
    prior = c.read(HERE.parent/'multifamily_v7/run_001/FEATURE_RESULT.json')
    assert initial['sha256'] == prior['base_state_sha256'],'BASE_IDENTITY_CHANGED'
    for parameter in policy.parameters(): parameter.requires_grad_(False)
    rows = [r['row'] for r in data['records']]
    store = adapter.SampleStore(rows)
    samples = [dict(record_idx=i,t=t,target=target,weight=1.) for i,r in enumerate(data['records']) for t,target in enumerate(r['targets'])]
    assert len(samples) == data['decisions']
    dataset = model.DecisionDataset(samples,store,policy.processor)
    collate = model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
    capture = []
    handle = policy.action_head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach().cpu().clone()))
    feature_rows, logit_rows, parts = [], [], []
    try:
        with torch.inference_mode():
            for i, sample in enumerate(samples):
                batch = collate([dataset[i]])
                batch.pop('targets');batch.pop('weights')
                processed = {k:c.tensor_identity(v) for k,v in batch.items()}
                logits = policy.forward_batch(**{k:v.cuda() for k,v in batch.items()})
                torch.cuda.synchronize()
                assert len(capture)==1
                feature = capture.pop()
                assert bool(torch.isfinite(feature).all()) and bool(torch.isfinite(logits).all())
                feature_rows.append(feature);logit_rows.append(logits.detach().float().cpu())
                record = store._cache[sample['record_idx']]
                metadata = record.decision_metadata(sample['t'])
                raw = dict(instruction=metadata['policy']['instruction'], rgb_sha256=[Path(x).stem for x in metadata['control']['rgb_refs']],
                    executed_actions=metadata['policy']['executed_actions'])
                key = hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
                c.append(run/'FEATURE_INPUTS.jsonl',dict(index=i,record_index=sample['record_idx'],step=sample['t'],raw=raw,input_key=key,processed=processed))
                c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='FEATURES',completed=i+1,total=len(samples)))
                if len(feature_rows)==2048 or i+1==len(samples):
                    path=run/f'FEATURE_PART_{len(parts):03d}.pt'
                    torch.save(dict(features=torch.cat(feature_rows),logits=torch.cat(logit_rows)),path)
                    c.write(path.with_suffix('.json'),dict(end_index=i+1,rows=len(feature_rows),sha256=c.sha(path)),True)
                    parts.append(path);feature_rows.clear();logit_rows.clear()
        final = c.model_identity(policy)
        assert final['sha256'] == initial['sha256'],'FROZEN_STATE_CHANGED'
        loaded = [torch.load(path,map_location='cpu',weights_only=True) for path in parts]
        cache = {key:torch.cat([part[key] for part in loaded]) for key in ('features','logits')}
        torch.save(cache,run/'FEATURES.pt')
        c.write(run/'FEATURE_RESULT.json',dict(real_qwen_forwards=len(samples),feature_shape=list(cache['features'].shape),
            file_sha256=c.sha(run/'FEATURES.pt'),data_sha256=c.sha(HERE/'DATA.json'),base_state_sha256=initial['sha256'],
            final_state_sha256=final['sha256'],encoder_parameters_unchanged=True,encoder_updates=0,
            future_query_in_encoder=False,current_target_in_encoder=False,
            input_contract='Original natural instruction; last <=2 RGB; last <=8 actually executed motion actions'),True)
    finally:
        handle.remove()
    del policy,model,dataset,store,capture,initial,final,loaded,batch,logits,feature
    gc.collect();torch.cuda.empty_cache()
    return cache


def main(run):
    config = c.read(HERE/'PROTOCOL.json')
    for path,digest in config['source_hashes'].items():assert c.sha(LINE/path)==digest,'SOURCE_CHANGED:'+path
    natural_data = c.read(HERE/'DATA.json')
    natural_cache = features(run,natural_data)
    source = HERE.parent/'multifamily_v7'
    special_data = c.read(source/'DATA.json')
    special_cache = torch.load(source/'run_001/FEATURES.pt',map_location='cpu',weights_only=True)
    train.train(run,config,special_cache,special_data,natural_cache,natural_data)


if __name__ == '__main__':
    run = Path(sys.argv[1])
    try:main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
