"""Frozen feature extraction only: hook existing action head, no forward changes."""
import gc,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def main():
    c.verify();assert c.read(HERE/'CPU_TEST_RESULT.json')['status']=='PASS'
    speed=c.LINE/'sft_acceptance/ordinary_speedup_10x_v1'
    sys.path.insert(0,str(speed/'official_einops_0_8_1/deps'));sys.path.insert(0,str(speed/'official_fla_0_5_2/deps'))
    import fla.ops.gated_delta_rule
    import torch
    from PIL import Image
    torch.cuda.set_device(0);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(25*1024**3/torch.cuda.get_device_properties(0).total_memory)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    bound=modeling.torch_chunk_gated_delta_rule
    cells=dict(zip(bound.__code__.co_freevars,(x.cell_contents for x in bound.__closure__)))
    assert getattr(cells.get('implementation'),'__module__','').startswith('fla.')
    m=c.load('frozen_model',c.MODEL);state=torch.load(c.BEST,map_location='cpu',weights_only=True)
    assert c.sha(c.BEST)=='c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775'
    policy=m.build_policy(1209);m.load_trainable(policy,state['trainable']);policy.eval()
    before=m.trainable_state(policy);del state;gc.collect()
    inputs=c.rows(HERE/'POLICY_INPUTS.jsonl');assert len(inputs)==5487
    class Store:
        def get(self,i,t):
            assert t==0
            row=inputs[i];assert set(row)=={'record_id','instruction','rgb_sha256','executed_actions'}
            images=[]
            for digest in row['rgb_sha256']:
                with Image.open(c.DATA/'content'/(digest+'.png')) as im:images.append(im.convert('RGB'))
            return dict(instruction=row['instruction'],executed=row['executed_actions'],images=images)
    dataset=m.DecisionDataset([dict(record_idx=i,t=0,target=0,weight=1.) for i in range(len(inputs))],Store(),policy.processor)
    collate=m.make_collate(policy.processor.tokenizer.pad_token_id,policy.exec_sid,policy.query_sid,policy.base.config.image_token_id)
    captured=[]
    def hook(module,args):
        assert len(args)==1 and args[0].shape==(1,2048);captured.append(args[0].detach().cpu().clone())
    handle=policy.action_head.register_forward_pre_hook(hook)
    hh=[];zz=[];began=time.monotonic()
    with torch.inference_mode():
        for i in range(len(inputs)):
            assert time.monotonic()-began<1200 and i<6000
            batch=collate([dataset[i]]);batch.pop('targets');batch.pop('weights')
            z=policy.forward_batch(**{k:v.to('cuda:0') for k,v in batch.items()}).cpu()
            assert len(captured)==1;h=captured.pop();hh.append(h);zz.append(z)
            if (i+1)%100==0 or i+1==len(inputs):
                c.write(HERE/'EXTRACTION_PROGRESS.json',dict(status='EXTRACTING',unix=time.time(),completed=i+1,total=len(inputs),wall_seconds=time.monotonic()-began),False)
                print(json.dumps(dict(completed=i+1,total=len(inputs))),flush=True)
    handle.remove();after=m.trainable_state(policy)
    assert set(before)==set(after) and all(torch.equal(before[k],after[k]) for k in before)
    features=torch.cat(hh);logits=torch.cat(zz)
    assert torch.isfinite(features).all() and torch.isfinite(logits).all()
    assert not (HERE/'FEATURES.pt').exists()
    torch.save(dict(record_ids=[x['record_id'] for x in inputs],features=features,logits=logits,source_checkpoint_sha256=c.sha(c.BEST)),HERE/'FEATURES.pt')
    c.write(HERE/'EXTRACTION_RESULT.json',dict(status='COMPLETE_PENDING_PARITY',unix=time.time(),forwards=len(inputs),feature_shape=list(features.shape),parameter_updates=0,parameters_exact_unchanged=True,wall_seconds=time.monotonic()-began,feature_sha256=c.sha(HERE/'FEATURES.pt'),peak_torch_allocated_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':main()
