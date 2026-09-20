"""FIT/DEV-only golden forwards; no optimizer or method efficacy measurement."""
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
import encoder
import objective as o
from select_action import select

def main(run):
    config=read(run/'PROTOCOL.json');data=read(run/'DATA.json')
    if any(f['split'] not in ('FIT','DEV') for f in data['families']):raise ValueError('GOLDEN_PILOT_MUST_EXCLUDE_TEST')
    folder=run/'numerical_pilot';folder.mkdir(exist_ok=False)
    config=dict(config,source_hashes=read(run/'SOURCE_LOCK.json')['files'],v16_protocol_sha256=sha(run/'PROTOCOL.json'))
    began=time.monotonic();model,policy,initial=encoder.load_policy(folder,config)
    for p in policy.parameters():p.requires_grad_(False)
    forward=encoder.Forward(model,policy,encoder.RawStore(data),[dict(record_idx=i,t=0,target=0,weight=1.) for i in range(len(data['features']))])
    rows=[];heads={seed:o.initialize(seed).cuda().eval() for seed in config['seeds']};caches=[]
    try:
        with torch.inference_mode():
            warm=encoder.warmup(forward,data,config);write(folder/'WARMUP.json',warm,True)
            count=min(16,len(data['features']));indices=[int(i*len(data['features'])/count) for i in range(count)]
            immutable(folder/'GOLDEN_REGISTRY.json',dict(indices=indices,raw_keys=[data['features'][i]['key'] for i in indices],selection='evenly spaced causal FIT/DEV feature indices, no scores'))
            for i in indices:
                f,l,p,t=forward(i);caches.append((f.clone(),l.clone(),p))
            for i,(f,l,p) in zip(indices,caches):
                actual,logits,processed,timing=forward(i)
                row=dict(index=i,raw=data['features'][i],processed=p,processed_inputs_equal=p==processed,
                    feature_sha256=c.tensor_identity(f),live_feature_sha256=c.tensor_identity(actual),
                    feature_max_abs_delta=float((actual-f).abs().max()),feature_relative_l2=float((actual-f).norm()/f.norm().clamp_min(1e-12)),
                    native_logits=l[0].cpu().tolist(),live_native_logits=logits[0].cpu().tolist(),
                    max_logit_delta=float((logits-l).abs().max()),argmax_flip=int(logits.argmax()!=l.argmax()),
                    method_comparisons=[],**timing)
                for seed,net in heads.items():
                    memory,_=net.update(f,net.reset(1,'cuda'));other,_=net.update(actual,net.reset(1,'cuda'))
                    a=net.action_logits(memory,l,f);b=net.action_logits(other,logits,actual)
                    row['method_comparisons'].append(dict(seed=seed,head_state=c.model_identity(net)['sha256'],max_delta=float((a-b).abs().max()),argmax_flip=int(a.argmax()!=b.argmax()),
                        cache=select(l[0].cpu().tolist(),a[0].cpu().tolist()),live=select(logits[0].cpu().tolist(),b[0].cpu().tolist())))
                append(folder/'FORWARDS.jsonl',row);rows.append(row)
                if not row['processed_inputs_equal'] or row['argmax_flip']:raise ValueError('FIT_DEV_NUMERICAL_OR_INPUT_CORRECTNESS')
        final=c.model_identity(policy)
        if final['sha256']!=initial['sha256']:raise ValueError('BASE_PARAMETERS_CHANGED')
        atomic_torch(folder/'GOLDEN_FEATURES.pt',dict(features=torch.cat([f.cpu() for f,_,_ in caches]),logits=torch.cat([l.cpu() for _,l,_ in caches])))
        write(folder/'RESULT.json',dict(status='FIT_DEV_GOLDEN_FORWARD_COMPLETE',golden_inputs=len(rows),real_qwen_forwards=len(warm)+2*len(rows),
            base_loaded=True,base_parameters_unchanged=True,base_state_sha256=initial['sha256'],optimizer_updates=0,
            all_processed_inputs_equal=all(r['processed_inputs_equal'] for r in rows),native_argmax_flips=sum(r['argmax_flip'] for r in rows),
            max_feature_delta=max(r['feature_max_abs_delta'] for r in rows),max_logit_delta=max(r['max_logit_delta'] for r in rows),
            seconds=time.monotonic()-began,scope='Generator/numerical pilot, no trained method comparison, no TEST inputs'),True)
    finally:
        forward.close();final=c.model_identity(policy)
        write(folder/'STATE_SEAL.json',dict(base_unchanged=final['sha256']==initial['sha256'],base_state_sha256=final['sha256']),True)

if __name__=='__main__':main(Path(sys.argv[1]))
