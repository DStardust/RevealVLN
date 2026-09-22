"""Only expanded FIT updates; no evaluation imports or score-based decisions."""
import os,sys,time,random,platform
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent))
from common import c,read,write,sha,digest,immutable,atomic_torch
from model import EvidencePolicy
import objective
append=c.append

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'].cpu())
    if r['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])

def load_resume(path,net,opt,expected):
    meta=read(path.with_suffix('.json'))
    if meta['sha256']!=sha(path) or meta['binding']!=expected:raise ValueError('CHECKPOINT_BINDING_CHANGED')
    value=torch.load(path,map_location=next(net.parameters()).device,weights_only=False)
    if value['binding']!=expected or value['step']!=value['cursor'] or value['step']!=meta['step']:raise ValueError('RESUME_CURSOR')
    net.load_state_dict(value['model']);opt.load_state_dict(value['optimizer']);restore_rng(value['rng'])
    return value['step']

def save(path,net,opt,step,binding):
    if path.exists() or path.with_suffix('.json').exists():raise FileExistsError(path)
    atomic_torch(path,dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,cursor=step,binding=binding,rng=rng_state()))
    immutable(path.with_suffix('.json'),dict(step=step,binding=binding,sha256=sha(path)))

def candidates(run):
    paths=list((run/'train/recovery').glob('STEP_*.pt'))+list((run/'checkpoints').glob('STEP_*/RESUME.pt'))
    return [p for p in paths if p.with_suffix('.json').exists()]

def commit_milestone(run,net,opt,step,binding,initial):
    out=run/'checkpoints'/f'STEP_{step:06d}';out.mkdir(parents=True,exist_ok=True)
    # A partial milestone is preserved in its failed attempt, never mistaken for committed output.
    if (out/'COMMIT.json').exists():return
    if any(out.iterdir()):
        out.rename(out.with_name(out.name+'_incomplete_'+str(time.time_ns())));out.mkdir()
    save(out/'RESUME.pt',net,opt,step,binding)
    atomic_torch(out/'HEAD.pt',net.state_dict())
    result=dict(mode='MONOTONIC',arm='EXPANDED',seed=1209,updates=step,initial=initial,
        final=c.model_identity(net)['sha256'],checkpoint_sha256=sha(out/'HEAD.pt'),base_updates=0)
    immutable(out/'HEAD_RESULT.json',result)
    immutable(out/'COMMIT.json',dict(step=step,head_sha256=sha(out/'HEAD.pt'),resume_sha256=sha(out/'RESUME.pt'),
        binding=binding,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

def main(run):
    cfg=read(run/'PROTOCOL.json');prepared=read(run/'PREPARED.json');source=Path(cfg['source_run'])
    for p,h in prepared['files'].items():
        if sha(Path(p))!=h:raise ValueError('INPUT_CHANGED:'+p)
    for rel,h in read(run/'SOURCE_LOCK.json')['files'].items():
        if sha(D.parent.parents[2]/rel)!=h:raise ValueError('SOURCE_CHANGED:'+rel)
    out=run/'train';out.mkdir(exist_ok=True);(out/'recovery').mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(1209);np.random.seed(1209);torch.manual_seed(1209)
    net=EvidencePolicy(1209,'MONOTONIC').cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    initial=c.model_identity(net)['sha256'];binding=dict(prepared_sha256=sha(run/'PREPARED.json'),seed=1209,arm='EXPANDED',mode='MONOTONIC',target_updates=100000,initial=initial)
    paths=candidates(run)
    if paths:
        path=max(paths,key=lambda p:read(p.with_suffix('.json'))['step']);cursor=load_resume(path,net,opt,binding)
    else:
        path=Path(cfg['source_resume']);cursor=load_resume(path,net,opt,prepared['source_checkpoint_binding'])
        if cursor!=1200 or initial!=prepared['source_checkpoint_binding']['initial']:raise ValueError('MIGRATION_START_IDENTITY')
    for group in opt.param_groups:
        if group['lr']!=cfg['learning_rate'] or group['weight_decay']!=cfg['weight_decay']:raise ValueError('OPTIMIZER_CHANGED')
    attempt=out/f'attempt_{len(list(out.glob("attempt_*")))+1:03d}';attempt.mkdir()
    immutable(attempt/'START.json',dict(resumed_from=str(path),resume_sha256=sha(path),cursor=cursor,binding=binding,
        initial=initial,resumed_model=c.model_identity(net)['sha256'],base_loaded=False,base_updates=0,
        python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,device=os.environ.get('CUDA_VISIBLE_DEVICES'),
        dtype=str(next(net.parameters()).dtype),optimizer_steps=sorted({int(x['step']) for x in opt.state.values()})))
    if cursor%5000==0:commit_milestone(run,net,opt,cursor,binding,initial)
    data=read(source/'TRAIN_DATA.json');allowed=set(data['arms']['EXPANDED'])
    @lru_cache(maxsize=32)
    def family(fid):
        if fid not in allowed:raise ValueError('NON_EXPANDED_FAMILY')
        info=data['family_files'][fid];p=source/info['path']
        if sha(p)!=info['sha256']:raise ValueError('FAMILY_CHANGED')
        f=read(p)
        if f['split']!='FIT':raise ValueError('NONFIT_FAMILY')
        return f
    cache={k:v.float().cuda() for k,v in torch.load(source/'features/FEATURES.pt',map_location='cpu',weights_only=True).items()}
    ordinary={k:v.float().cuda() for k,v in torch.load(prepared['feature_result']['ordinary_path'],map_location='cpu',weights_only=True).items()}
    natural=read(D.parent.parent/'natural_transfer_v9/DATA.json');schedule=read(run/'SCHEDULE.json');weights=read(source/'SHARED_FIT_WEIGHTS.json')
    began=time.monotonic()
    write(out/'PROGRESS.json',dict(step=cursor,target=100000,attempt=attempt.name,unix=time.time(),status='TRAINING'))
    for i in range(cursor,100000):
        item=schedule[i]
        if any(natural['records'][j]['partition']!='fit' for j in item['ordinary']):raise ValueError('NONFIT_OPTIMIZATION')
        b=objective.batch(family(item['family']),'cuda');opt.zero_grad(set_to_none=True)
        loss,stats,detail=objective.losses(net,cache,b,weights)
        bc=objective.ordinary_loss(net,ordinary,[natural['records'][j] for j in item['ordinary']]);total=loss+bc
        if not torch.isfinite(total):raise ValueError('NONFINITE_LOSS')
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):raise ValueError('NONFINITE_GRADIENT')
        norms={n:float(p.grad.norm()) for n,p in net.named_parameters() if p.grad is not None};opt.step()
        append(attempt/'STEPS.jsonl',dict(step=i+1,family=item['family'],split='FIT',loss=float(total.detach()),ordinary_ce=float(bc.detach()),gradients=norms,**stats))
        write(out/'PROGRESS.json',dict(step=i+1,target=100000,attempt=attempt.name,unix=time.time(),loss=float(total.detach()),new_updates=i+1-1200,
            seconds=time.monotonic()-began,seconds_per_update=(time.monotonic()-began)/(i+1-cursor),status='TRAINING',base_updates=0))
        if (i+1)%cfg['checkpoint_every']==0:commit_milestone(run,net,opt,i+1,binding,initial)
        elif (i+1)%cfg['recovery_every']==0:
            save(out/'recovery'/f'STEP_{i+1:06d}.pt',net,opt,i+1,binding)
            recent=sorted(p for p in (out/'recovery').glob('STEP_*.pt') if p.with_suffix('.json').exists())
            for p in recent[:-2]:p.unlink();p.with_suffix('.json').unlink()
    immutable(out/'RESULT.json',dict(status='TRAINING_COMPLETE',updates=100000,new_updates=98800,base_updates=0,initial=initial,final=c.model_identity(net)['sha256'],checkpoints=20))
    print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
