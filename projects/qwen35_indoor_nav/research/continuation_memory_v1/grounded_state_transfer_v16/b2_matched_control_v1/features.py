"""Real causal Qwen features, fixed cache/live path and segment-level state seal."""
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
import encoder

def main(run):
    import torch
    cfg=runtime_config(run);verify_lock(read(run/'SOURCE_LOCK.json'));cfg.update(source_hashes=read(run/'SOURCE_LOCK.json')['files'],v16_protocol_sha256=sha(run/'PROTOCOL.json'))
    out=run/'features';out.mkdir(exist_ok=True)
    if (out/'FEATURE_RESULT.json').exists():return
    session=out/f'session_{len(list(out.glob("session_*")))+1:03d}';session.mkdir()
    model,policy,initial=encoder.load_policy(session,cfg)
    if initial['sha256']!='a87292acedfbf3fc8159259c23017fa1eec47a2ebfd39782c46e088bcbacef6b':raise ValueError('BASE_STATE_IDENTITY')
    for p in policy.parameters():p.requires_grad_(False)
    data=read(run/'DATA.json');samples=[dict(record_idx=i,t=0,target=0,weight=1.) for i in range(len(data['features']))]
    forward=encoder.Forward(model,policy,encoder.RawStore(data),samples);began=time.monotonic()
    try:
        with torch.inference_mode():
            write(session/'WARMUP.json',encoder.warmup(forward,data,cfg),True)
            gold=[]
            for i in range(cfg['golden_inputs']):
                a,l,p,_=forward(i);b,k,q,_=forward(i)
                row=dict(index=i,processed_equal=p==q,feature_delta=float((a-b).abs().max()),native_delta=float((l-k).abs().max()),native_flip=int(l.argmax()!=k.argmax()))
                gold.append(row)
                if p!=q or row['native_flip']:raise ValueError('NUMERIC_GOLDEN_INPUT_FAILURE')
            write(session/'GOLDEN_INPUTS.json',gold,True)
            old_result=read(PRIOR_RUN/'features/FEATURE_RESULT.json')
            old_path=PRIOR_RUN/'features/FEATURES.pt'
            if sha(old_path)!=old_result['file_sha256'] or old_result['base_state_sha256']!=initial['sha256']:raise ValueError('REUSED_CACHE_IDENTITY')
            old_cache=torch.load(old_path,map_location='cpu',weights_only=True)
            old_data=read(PRIOR_RUN/'DATA.json');old_lookup={r['key']:i for i,r in enumerate(old_data['features'])}
            old_processed={}
            for path in (PRIOR_RUN/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
                if (path.parent/'STATE_SEAL.json').exists():
                    for row in c.records(path):old_processed[row['index']]=row['processed']
            features=[];native=[];reused=0;new_forwards=0;cache_parity=[]
            for i in range(len(samples)):
                old_index=old_lookup.get(data['features'][i]['key'])
                if old_index is None:
                    f,l,p,t=forward(i);new_forwards+=1
                else:
                    f=old_cache['features'][old_index:old_index+1];l=old_cache['logits'][old_index:old_index+1]
                    p=old_processed[old_index];t=dict(reused=True,source_index=old_index);reused+=1
                    if len(cache_parity)<cfg['golden_inputs']:
                        ff,ll,pp,_=forward(i)
                        row=dict(index=i,processed_equal=p==pp,feature_delta=float((f-ff.cpu()).abs().max()),native_delta=float((l-ll.cpu()).abs().max()),native_flip=int(l.argmax()!=ll.argmax()))
                        cache_parity.append(row)
                        if not row['processed_equal']:raise ValueError('REUSED_CACHE_INPUT_MISMATCH')
                features.append(f.cpu());native.append(l.cpu())
                append(session/'FEATURES_INPUTS_000000.jsonl',dict(index=i,raw_key=data['features'][i]['key'],processed=p,**t))
                if i%20==0:write(run/'FEATURE_PROGRESS.json',dict(completed=i+1,total=len(samples),seconds=time.monotonic()-began))
            payload=dict(features=torch.cat(features),logits=torch.cat(native))
            atomic_torch(session/'FEATURES.pt',payload)
        final=c.model_identity(policy)['sha256']
        if final!=initial['sha256']:raise ValueError('BASE_CHANGED')
        write(session/'STATE_SEAL.json',dict(base_unchanged=True,base_state_sha256=final),True)
        if (out/'FEATURES.pt').exists():raise ValueError('UNSEALED_FINAL_CACHE_EXISTS')
        import os
        os.link(session/'FEATURES.pt',out/'FEATURES.pt')
        ordinary=read(OLD/'features/FEATURE_RESULT.json')
        if ordinary['base_state_sha256']!=final:raise ValueError('ORDINARY_BASE_MISMATCH')
        path=OLD/'features/ORDINARY_FEATURES.pt'
        if sha(path)!=ordinary['ordinary_sha256']:raise ValueError('ORDINARY_CACHE_CHANGED')
        write(out/'FEATURE_RESULT.json',dict(windows=len(samples),parameters_unchanged=True,base_state_sha256=final,
            file_sha256=sha(out/'FEATURES.pt'),ordinary_path=str(path),ordinary_sha256=ordinary['ordinary_sha256'],
            seconds=time.monotonic()-began,base_updates=0,golden=gold,reused_windows=reused,new_window_forwards=new_forwards,cache_session_parity=cache_parity),True)
    finally:
        forward.close()
        if not (session/'STATE_SEAL.json').exists():
            final=c.model_identity(policy)['sha256'];write(session/'STATE_SEAL.json',dict(base_unchanged=initial['sha256']==final,base_state_sha256=final),True)

if __name__=='__main__':main(Path(sys.argv[1]))
