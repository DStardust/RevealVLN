"""Shard real frozen Qwen forward calls; only fully sealed chunks can resume."""
import os,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
encoder=local_module("encoder")
extractor=load('scale_original_feature_parts',V16/'extract_features.py')
def main(run):
 cfg=runtime_config(run);cfg['source_hashes']=read(run/'SOURCE_LOCK.json')['files'];cfg['v16_protocol_sha256']=sha(run/'PROTOCOL.json')
 data=read(run/'TRAIN_DATA.json');out=run/'features';out.mkdir(exist_ok=True)
 done=extractor.sealed_parts(out,'FEATURES');chunks=[i for i in range(0,len(data['features']),1024) if (i//1024)%len(cfg['devices'])==cfg['gpu'] and i not in done]
 if not chunks:return
 session=out/f"session_gpu{cfg['gpu']}_{time.time_ns()}";session.mkdir()
 model,policy,initial=encoder.load_policy(session,cfg)
 if initial['sha256']!='a87292acedfbf3fc8159259c23017fa1eec47a2ebfd39782c46e088bcbacef6b':raise ValueError('BASE_IDENTITY')
 for p in policy.parameters():p.requires_grad_(False)
 samples=[dict(record_idx=i,t=0,target=0,weight=1.) for i in range(len(data['features']))]
 forward=encoder.Forward(model,policy,encoder.RawStore(data),samples);began=time.monotonic()
 try:
  with torch.inference_mode():
   # Exactly the same old FIT/DEV warmup indices used by live evaluation.
   golden=read(run/'WARMUP_DATA.json');warm=encoder.warmup(forward,golden,cfg);write(session/'WARMUP.json',warm,True)
   checks=[]
   for i in range(4):
    f,l,p,t=forward(i);g,m,q,u=forward(i)
    row=dict(index=i,processed=p,processed_equal=p==q,feature_delta=float((f-g).abs().max()),native_delta=float((l-m).abs().max()),native_flip=int(l.argmax()!=m.argmax()),native_logits=l[0].cpu().tolist())
    checks.append(row)
    if not row['processed_equal'] or row['native_flip']:raise ValueError('GOLDEN_INPUT_NUMERIC_FAILURE')
   write(session/'GOLDEN_INPUTS.json',checks,True)
   for start in chunks:
    values=[];native=[];log=session/f'FEATURES_INPUTS_{start:06d}.jsonl'
    for i in range(start,min(start+1024,len(samples))):
     f,l,p,t=forward(i);values.append(f.cpu());native.append(l.cpu())
     append(log,dict(index=i,raw_key=data['features'][i]['key'],processed=p,**t))
     if i%20==0:write(out/f"PROGRESS_{cfg['gpu']}.json",dict(gpu=cfg['gpu'],last_index=i,session_forward_count=len(values),chunk=start,total_chunks=len(chunks),completed_chunks=len(list(session.glob('FEATURES_PART_*.json'))),unix=time.time()))
    path=session/f'FEATURES_PART_{start:06d}.pt';atomic_torch(path,dict(features=torch.cat(values),logits=torch.cat(native)))
    write(path.with_suffix('.json'),dict(start=start,end=start+len(values),sha256=sha(path)),True)
    if time.monotonic()-began>cfg['max_session_hours']*3600-600:break
 finally:
  forward.close();final=c.model_identity(policy)
  write(session/'STATE_SEAL.json',dict(base_unchanged=final['sha256']==initial['sha256'],base_state_sha256=final['sha256']),True)
  from kernel_metadata import record
  record(session/'TRITON_CACHE_METADATA.json')
  if final['sha256']!=initial['sha256']:raise ValueError('BASE_CHANGED')
if __name__=='__main__':main(Path(sys.argv[1]))
