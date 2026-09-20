"""One frozen best4k load; only causal observations enter the Qwen encoder."""
from functools import lru_cache
from pathlib import Path
import sys
import time
import traceback
import torch
from PIL import Image
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
sys.path.insert(0,str(LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c
from evaluate import load_policy


def main(run):
    config=c.read(HERE/'FEATURE_PROTOCOL.json')
    for path,digest in config['source_hashes'].items():assert c.sha(LINE/path)==digest,path
    data=c.read(HERE/'DATA.json')
    c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='LOAD_FROZEN_BEST4K'))
    model,policy,initial=load_policy(run,c.read(run/'CONFIG.json'))
    assert initial['sha256']=='a87292acedfbf3fc8159259c23017fa1eec47a2ebfd39782c46e088bcbacef6b'
    for p in policy.parameters():p.requires_grad_(False)
    loaders=c.load('v15_pixels',LINE/'data_pipeline/mechanism_runtime_v1/loader.py')
    @lru_cache(maxsize=128)
    def pixels(reference):
        item=data['contents'][reference];blob=loaders.safe_path(LINE,item['line_relative_path']).read_bytes()
        assert loaders.sha(blob)==item['file_sha256']
        return loaders.npy_pixels(blob,reference[7:],'rgb')
    class Store:
        def get(self,index,t):
            r=data['features'][index]
            return dict(instruction=r['instruction'],images=[Image.frombytes('RGB',(224,224),pixels(ref)) for ref in r['rgb_refs']],executed=r['executed'])
    records=[dict(record_idx=i,t=0,target=0,weight=1.) for i in range(len(data['features']))]
    dataset=model.DecisionDataset(records,Store(),policy.processor)
    collate=model.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
    output=dict(features=torch.empty(len(records),2048),logits=torch.empty(len(records),4));captured=[]
    hook=policy.action_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().clone()))
    began=time.monotonic()
    try:
        with torch.inference_mode():
            for i in range(len(records)):
                batch=collate([dataset[i]]);batch.pop('targets');batch.pop('weights')
                processed={k:c.tensor_identity(v) for k,v in batch.items()}
                logits=policy.forward_batch(**{k:v.to('cuda:0') for k,v in batch.items()});torch.cuda.synchronize()
                assert len(captured)==1;feature=captured.pop()
                assert bool(torch.isfinite(feature).all()) and bool(torch.isfinite(logits).all())
                output['features'][i]=feature[0].float();output['logits'][i]=logits[0].detach().float().cpu()
                c.append(run/'FEATURE_INPUTS.jsonl',dict(index=i,input_key=data['features'][i]['key'],processed=processed))
                c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='FEATURES',completed=i+1,total=len(records)))
                if (i+1)%2048==0 or i+1==len(records):
                    path=run/f'PART_{i+1:06d}.pt'
                    torch.save({k:v[max(0,i+1-((i+1)%2048 or 2048)):i+1].clone() for k,v in output.items()},path)
        final=c.model_identity(policy);assert final['sha256']==initial['sha256']
        torch.save(output,run/'FEATURES.pt')
        c.write(run/'FEATURE_RESULT.json',dict(status='CAUSAL_FROZEN_FEATURES_COMPLETE',real_qwen_forwards=len(records),
            model_loads=1,optimizer_updates=0,base_state_sha256=initial['sha256'],final_state_sha256=final['sha256'],
            parameters_unchanged=True,data_sha256=c.sha(HERE/'DATA.json'),feature_protocol_sha256=c.sha(HERE/'FEATURE_PROTOCOL.json'),
            file_sha256=c.sha(run/'FEATURES.pt'),forward_seconds=time.monotonic()-began,
            policy_inputs=['instruction','last2_raw_RGB','last8_executed_motion_actions'],future_query_in_encoder=False),True)
    finally:hook.remove()


if __name__=='__main__':
    run=Path(sys.argv[1])
    try:main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
