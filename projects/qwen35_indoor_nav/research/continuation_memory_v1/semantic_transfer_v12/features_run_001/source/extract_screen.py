"""One frozen best4k load for a finite, preselected current-witness screen."""
from functools import lru_cache
from pathlib import Path
import sys
import time
import traceback

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[2]
V5 = LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'
sys.path.insert(0, str(V5))
import common as c
from evaluate import load_policy


def main(run):
    config = c.read(HERE/'FEATURE_PROTOCOL.json')
    for relative, digest in config['source_hashes'].items():
        assert c.sha(LINE/relative)==digest, relative
    data = c.read(HERE/'SCREEN.json')
    prior = HERE.parent/'multifamily_v7'
    old_identity = c.read(prior/'run_001/FEATURE_RESULT.json')
    assert c.sha(prior/'run_001/FEATURES.pt')==old_identity['file_sha256']
    old = torch.load(prior/'run_001/FEATURES.pt', map_location='cpu', weights_only=True)
    output = dict(features=torch.empty(len(data['features']),2048,dtype=torch.float32),
                  logits=torch.empty(len(data['features']),4,dtype=torch.float32))
    reuse = {int(k):v for k,v in data['reused_v7_feature_indices'].items()}
    for index, source in reuse.items():
        for key in output:
            output[key][index] = old[key][source].float()
    del old
    pending = [i for i in range(len(data['features'])) if i not in reuse]
    assert pending, 'NO_NEW_INPUTS_TO_ENCODE'
    c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='LOAD_FROZEN_BEST4K'))
    model, policy, initial = load_policy(run,c.read(run/'CONFIG.json'))
    assert initial['sha256']==old_identity['base_state_sha256'], 'ENCODER_IDENTITY_CHANGED'
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    loaders = c.load('v12_screen_pixels',LINE/'data_pipeline/mechanism_runtime_v1/loader.py')

    @lru_cache(maxsize=128)
    def pixels(reference):
        item = data['contents'][reference]
        blob = loaders.safe_path(LINE,item['line_relative_path']).read_bytes()
        assert loaders.sha(blob)==item['file_sha256'], 'CONTENT_CHANGED'
        return loaders.npy_pixels(blob,reference[7:],'rgb')

    class Store:
        def get(self,index,t):
            record = data['features'][index]
            return dict(instruction=record['instruction'],
                images=[Image.frombytes('RGB',(224,224),pixels(ref)) for ref in record['rgb_refs']],
                executed=record['executed'])

    records = [dict(record_idx=i,t=0,target=0,weight=1.) for i in pending]
    dataset = model.DecisionDataset(records,Store(),policy.processor)
    collate = model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,
                                policy.query_sid,policy.base.config.image_token_id)
    captured = []
    handle = policy.action_head.register_forward_pre_hook(
        lambda module,args: captured.append(args[0].detach().cpu().clone()))
    began = time.monotonic()
    try:
        with torch.inference_mode():
            for position,index in enumerate(pending):
                batch = collate([dataset[position]])
                batch.pop('targets');batch.pop('weights')
                processed = {k:c.tensor_identity(v) for k,v in batch.items()}
                logits = policy.forward_batch(**{k:v.to('cuda:0') for k,v in batch.items()})
                torch.cuda.synchronize()
                assert len(captured)==1
                feature = captured.pop()
                assert tuple(feature.shape)==(1,2048)
                assert bool(torch.isfinite(feature).all()) and bool(torch.isfinite(logits).all())
                output['features'][index] = feature[0].float()
                output['logits'][index] = logits[0].detach().float().cpu()
                c.append(run/'FEATURE_INPUTS.jsonl',dict(index=index,input_key=data['features'][index]['key'],processed=processed))
                c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='FEATURES',completed=position+1,
                    expected=len(pending),reused_rows=len(reuse)))
        final = c.model_identity(policy)
        assert final['sha256']==initial['sha256'], 'FROZEN_ENCODER_MUTATED'
        assert all(bool(torch.isfinite(t).all()) for t in output.values())
        torch.save(output,run/'FEATURES.pt')
        c.write(run/'FEATURE_RESULT.json',dict(status='FINITE_CAUSAL_SCREEN_FEATURES_COMPLETE',
            real_qwen_forwards=len(pending),reused_rows=len(reuse),
            source_cache_sha256=old_identity['file_sha256'],source_reuse_map=data['reused_v7_feature_indices'],
            file_sha256=c.sha(run/'FEATURES.pt'),screen_sha256=c.sha(HERE/'SCREEN.json'),
            screen_protocol_sha256=c.sha(HERE/'SCREEN_PROTOCOL.json'),
            feature_protocol_sha256=c.sha(HERE/'FEATURE_PROTOCOL.json'),
            base_state_sha256=initial['sha256'],final_state_sha256=final['sha256'],
            parameters_unchanged=True,model_loads=1,optimizer_updates=0,
            forward_seconds=time.monotonic()-began,
            policy_inputs='Original instruction, <=2 RGB, <=8 executed actions; no event labels or future input.',
            original_training_admission=False),True)
    finally:
        handle.remove()


if __name__=='__main__':
    run=Path(sys.argv[1])
    try:
        main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
