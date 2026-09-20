"""New causal cache and FIT/DEV numerical pilot through the live encoder path."""
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v16_common import *
import encoder

class SegmentBoundary(Exception):
    pass

def main(run):
    config=read(run/'PROTOCOL.json');config['source_hashes']=read(run/'SOURCE_LOCK.json')['files'];config['v16_protocol_sha256']=sha(run/'PROTOCOL.json')
    data=read(run/'DATA.json');out=run/'features';out.mkdir(exist_ok=True)
    if (out/'FEATURE_RESULT.json').exists():
        if sha(out/'FEATURES.pt')!=read(out/'FEATURE_RESULT.json')['file_sha256']:raise ValueError('FEATURE_RESULT_CORRUPT')
        return
    session=out/f'session_{len(list(out.glob("session_*")))+1:03d}';session.mkdir()
    model,policy,initial=encoder.load_policy(session,config)
    expected='a87292acedfbf3fc8159259c23017fa1eec47a2ebfd39782c46e088bcbacef6b'
    if initial['sha256']!=expected:raise ValueError('BASE_STATE_MISMATCH')
    for p in policy.parameters():p.requires_grad_(False)
    samples=[dict(record_idx=i,t=0,target=0,weight=1.) for i in range(len(data['features']))]
    forward=encoder.Forward(model,policy,encoder.RawStore(data),samples)
    began=time.monotonic();completed=0
    try:
        warm=encoder.warmup(forward,data,config);write(session/'WARMUP.json',warm,True)
        golden=[i for i,r in enumerate(data['features']) if r['split'] in ('FIT','DEV')][:config['golden_inputs']]
        with torch.inference_mode():
            observations=[]
            import objective as o
            from select_action import index
            head=o.initialize(1209).cuda().eval()
            for i in golden:
                f1,l1,p1,_=forward(i);f2,l2,p2,_=forward(i)
                m1,_=head.update(f1,head.reset(1,'cuda'));m2,_=head.update(f2,head.reset(1,'cuda'))
                a=head.action_logits(m1,l1,f1);b=head.action_logits(m2,l2,f2)
                row=dict(index=i,raw_key=data['features'][i]['key'],processed_equal=p1==p2,processed=p1,
                    feature_max_abs_delta=float((f1-f2).abs().max()),native_max_abs_delta=float((l1-l2).abs().max()),
                    method_max_abs_delta=float((a-b).abs().max()),native_argmax_flip=int(l1.argmax()!=l2.argmax()),
                    method_argmax_flip=int(a.argmax()!=b.argmax()),native_logits=[l1[0].cpu().tolist(),l2[0].cpu().tolist()],
                    feature_relative_l2=float((f1-f2).norm()/f1.norm().clamp_min(1e-12)),head_state_sha256=c.model_identity(head)['sha256'])
                observations.append(row)
                if not row['processed_equal'] or row['native_argmax_flip']:raise ValueError('FIT_DEV_NUMERICAL_PILOT_PREFIX_INVALID')
            write(session/'GOLDEN_INPUTS.json',observations,True);del head
            deadline=began+config['max_session_hours']*3600-600
            completed=extract(out,session,'FEATURES',forward,len(samples),data,deadline)
            # Refresh the exact shared ordinary pool through this same frozen numerical path.
            natural=read(HERE.parent/'natural_transfer_v9/DATA.json')
            adapter=load('v16_ordinary_input_adapter',LINE/'sft_acceptance/ordinary_sync_recovery_v1/data.py')
            store=adapter.SampleStore([r['row'] for r in natural['records']])
            ordinary_samples=[dict(record_idx=i,t=t,target=y,weight=1.) for i,r in enumerate(natural['records']) for t,y in enumerate(r['targets'])]
            forward.close();forward=encoder.Forward(model,policy,store,ordinary_samples)
            ordinary_completed=extract(out,session,'ORDINARY_FEATURES',forward,len(ordinary_samples),None,deadline)
        final=c.model_identity(policy)
        if final['sha256']!=initial['sha256']:raise ValueError('BASE_CHANGED')
        write(session/'STATE_SEAL.json',dict(base_unchanged=True,base_state_sha256=final['sha256']),True)
        assemble(out,'FEATURES',len(samples));assemble(out,'ORDINARY_FEATURES',len(ordinary_samples))
        forward_count=sum(len(c.records(p)) for p in out.glob('session_*/*_INPUTS_*.jsonl'))
        forward_count+=sum(len(read(p)) for p in out.glob('session_*/WARMUP.json'))+2*sum(len(read(p)) for p in out.glob('session_*/GOLDEN_INPUTS.json'))
        immutable(out/'FEATURE_RESULT.json',dict(real_qwen_forwards=forward_count,
            windows=len(samples),ordinary_windows=len(ordinary_samples),parameters_unchanged=True,base_state_sha256=initial['sha256'],
            file_sha256=sha(out/'FEATURES.pt'),ordinary_sha256=sha(out/'ORDINARY_FEATURES.pt'),
            data_sha256=sha(run/'DATA.json'),seconds=time.monotonic()-began,base_optimizer_updates=0,
            golden_input_rows=observations,sessions=[str(p.relative_to(run)) for p in out.glob('session_*')]))
    except SegmentBoundary:
        write(session/'SEGMENT_BOUNDARY.json',dict(reason='planned complete feature-chunk boundary',seconds=time.monotonic()-began),True)
    finally:
        forward.close()
        from kernel_metadata import record
        record(session/'TRITON_CACHE_METADATA.json')
        if not (session/'STATE_SEAL.json').exists():
            final=c.model_identity(policy)
            write(session/'STATE_SEAL.json',dict(base_unchanged=final['sha256']==initial['sha256'],base_state_sha256=final['sha256']),True)

def sealed_parts(out,name):
    chosen={}
    for session in sorted(out.glob('session_*')):
        seal=session/'STATE_SEAL.json'
        if not seal.exists() or not read(seal)['base_unchanged']:continue
        for p in session.glob(name+'_PART_*.json'):
            row=read(p);path=p.with_suffix('.pt')
            if sha(path)!=row['sha256']:raise ValueError('FEATURE_PART_CORRUPT')
            if row['start'] in chosen:raise ValueError('DUPLICATE_FEATURE_PART')
            chosen[row['start']]=(row,path)
    return chosen

def extract(out,session,name,forward,count,data,deadline):
    existing=sealed_parts(out,name);calls=0
    for start in range(0,count,1024):
        if start in existing:continue
        values=[];native=[];log=session/(name+f'_INPUTS_{start:06d}.jsonl')
        for i in range(start,min(start+1024,count)):
            f,l,p,t=forward(i);calls+=1;values.append(f.cpu());native.append(l.cpu())
            append(log,dict(index=i,raw_key=data['features'][i]['key'] if data else None,processed=p,**t))
            if i%20==0:write(out/'PROGRESS.json',dict(kind=name,completed=i+1,total=count,unix=time.time()))
        path=session/(name+f'_PART_{start:06d}.pt')
        atomic_torch(path,dict(features=torch.cat(values),logits=torch.cat(native)))
        write(path.with_suffix('.json'),dict(start=start,end=start+len(values),sha256=sha(path)),True)
        if time.monotonic()>deadline:raise SegmentBoundary()
    return calls

def assemble(out,name,count):
    parts=sealed_parts(out,name);end=0;values=[]
    for start,(row,path) in sorted(parts.items()):
        if start!=end:raise ValueError('FEATURE_PART_GAP')
        values.append(torch.load(path,map_location='cpu',weights_only=True));end=row['end']
    if end!=count:raise ValueError('FEATURE_COUNT_MISMATCH')
    payload={k:torch.cat([x[k] for x in values]) for k in ('features','logits')};path=out/(name+'.pt')
    if path.exists():
        previous=torch.load(path,map_location='cpu',weights_only=True)
        if any(not torch.equal(previous[k],v) for k,v in payload.items()):raise ValueError('ASSEMBLED_CACHE_CHANGED')
    else:atomic_torch(path,payload)

if __name__=='__main__':main(Path(sys.argv[1]))
