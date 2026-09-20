"""Nine final1200 models; full optimizer/RNG recovery, no TEST loss access."""
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import objective as o
from v16_common import *

FITDEV_SCOPE='FIT16_TRAIN_DEV2_DIAGNOSTIC'

def training_families(data,config):
    fit=[f for f in data['families'] if f['split']=='FIT']
    if len(fit)!=16:raise ValueError('FIT_FAMILY_COUNT')
    if config.get('scope')==FITDEV_SCOPE:
        if data.get('training_admission')!='FIT16_DEV2_TRAINING_AUTHORIZED':raise ValueError('FITDEV_TRAINING_NOT_ADMITTED')
        if any(f['split'] not in ('FIT','DEV') for f in data['families']):raise ValueError('TEST_OR_UNKNOWN_SPLIT_IN_FITDEV_DATA')
        if sorted(f['family_id'] for f in fit)!=sorted(config['authorized_training_families']):raise ValueError('FITDEV_TRAINING_FAMILY_IDENTITY')
        dev=sorted(f['family_id'] for f in data['families'] if f['split']=='DEV')
        if dev!=sorted(config['authorized_diagnostic_families']):raise ValueError('FITDEV_DIAGNOSTIC_FAMILY_IDENTITY')
    elif data.get('training_admission')!='V16_REGISTERED_SCOPE_ONLY':raise ValueError('FORMAL_TRAINING_NOT_ADMITTED')
    return fit

def schedule_for_seed(family_ids,old_schedule,fit_ordinary,seed):
    ids=sorted(family_ids);schedule=ids*75
    if len(schedule)!=1200:raise ValueError('TRAINING_SCHEDULE_LENGTH')
    random.Random(seed).shuffle(schedule)
    natural=[old_schedule[i%len(old_schedule)] for i in range(1200)]
    allowed=set(fit_ordinary)
    if any(i not in allowed for selected in natural for i in selected):raise ValueError('NONFIT_ORDINARY_SCHEDULE')
    return [dict(family=f,ordinary=o) for f,o in zip(schedule,natural)]

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def save_checkpoint(path,net,optimizer,step,schedule,binding):
    atomic_torch(path,dict(model=net.state_dict(),optimizer=optimizer.state_dict(),step=step,schedule_cursor=step,
        rng=rng_state(),schedule_sha256=digest(schedule),binding=binding))
    immutable(path.with_suffix('.json'),dict(step=step,file_sha256=sha(path),binding=binding))

def restore_checkpoint(path,net,optimizer,schedule,binding):
    seal=read(path.with_suffix('.json'))
    if seal['file_sha256']!=sha(path) or seal['binding']!=binding:raise ValueError('CHECKPOINT_IDENTITY')
    # Trusted locally-created full RNG/optimizer checkpoint; never accept arbitrary external pickle.
    record=torch.load(path,map_location=next(net.parameters()).device,weights_only=False)
    if record['binding']!=binding or record['schedule_sha256']!=digest(schedule):raise ValueError('RESUME_BINDING_CHANGED')
    if record['step']!=record['schedule_cursor']:raise ValueError('RESUME_CURSOR')
    net.load_state_dict(record['model'],strict=True);optimizer.load_state_dict(record['optimizer']);restore_rng(record['rng'])
    return record['step']

def main(run):
    session_began=time.monotonic()
    config=read(run/'PROTOCOL.json');verify_lock(read(run/'SOURCE_LOCK.json'))
    data=read(run/'DATA.json');fit=training_families(data,config)
    # No TEST or DEV batch, loss, diagnostic or model selection is instantiated here.
    weights=o.class_weights(fit);immutable(run/'FIT_AUXILIARY_WEIGHTS.json',weights)
    feature_result=read(run/'features/FEATURE_RESULT.json');feature_path=run/'features/FEATURES.pt'
    if not feature_result['parameters_unchanged'] or sha(feature_path)!=feature_result['file_sha256']:raise ValueError('FEATURE_CACHE_IDENTITY')
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';cache={k:v.float().to(device) for k,v in torch.load(feature_path,weights_only=True).items()}
    ordinary_folder=HERE.parent/'natural_transfer_v9';ordinary=read(ordinary_folder/'DATA.json')
    oldpath=run/'features/ORDINARY_FEATURES.pt'
    if sha(oldpath)!=feature_result['ordinary_sha256']:raise ValueError('ORDINARY_CACHE_IDENTITY')
    ordinary_cache={k:v.float().to(device) for k,v in torch.load(oldpath,weights_only=True).items()}
    fit_ordinary=[i for i,r in enumerate(ordinary['records']) if r['partition']=='fit']
    if any(ordinary['records'][i]['row']['scene_group'] in {f['house'] for f in data['families']} for i in fit_ordinary):raise ValueError('ORDINARY_HOUSE_LEAK')
    batches={f['family_id']:o.batch(f,device) for f in fit};folder=run/'train';folder.mkdir(exist_ok=True)
    binding=dict(protocol=sha(run/'PROTOCOL.json'),sources=sha(run/'SOURCE_LOCK.json'),data=sha(run/'DATA.json'),
        features=sha(feature_path),ordinary_data=sha(ordinary_folder/'DATA.json'),ordinary_features=sha(oldpath))
    results=[]
    for seed in config['seeds']:
        old_schedule=read(ordinary_folder/'run_001'/f'SCHEDULE_{seed}.json')['ordinary_indices']
        schedule=schedule_for_seed(batches,old_schedule,fit_ordinary,seed)
        immutable(folder/f'SCHEDULE_{seed}.json',schedule)
        for arm in config['arms']:
            tag=f'{arm}_{seed}';target=folder/tag;target.mkdir(exist_ok=True)
            if (target/'RESULT.json').exists():
                result=read(target/'RESULT.json')
                if result['updates']!=1200 or sha(target/'FINAL.pt')!=result['checkpoint_sha256']:raise ValueError('FINAL_MODEL_CHANGED')
                results.append(result);continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            net=o.initialize(seed).to(device);initial=c.model_identity(net)['sha256']
            optimizer=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            start=0;checkpoints=sorted((p for p in target.glob('attempt_*/CHECKPOINT_*.pt') if p.with_suffix('.json').exists()),key=lambda p:(read(p.with_suffix('.json'))['step'],str(p)))
            model_binding=dict(binding,initial_state=initial,arm=arm,seed=seed)
            if checkpoints:start=restore_checkpoint(checkpoints[-1],net,optimizer,schedule,model_binding)
            attempt=len(list(target.glob('attempt_*')))+1;attempt_dir=target/f'attempt_{attempt:03d}';attempt_dir.mkdir()
            log=attempt_dir/'STEPS.jsonl';began=time.monotonic()
            for step in range(start,1200):
                row=schedule[step];optimizer.zero_grad(set_to_none=True)
                special,stats,_=o.losses(net,cache,batches[row['family']],arm,weights)
                bc=o.ordinary_loss(net,ordinary_cache,[ordinary['records'][i] for i in row['ordinary']]);loss=special+bc
                if not bool(torch.isfinite(loss)):raise ValueError('NONFINITE_LOSS')
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters()):raise ValueError('NONFINITE_GRADIENT')
                grad=float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in net.parameters() if p.grad is not None])))
                optimizer.step()
                append(log,dict(step=step+1,family=row['family'],loss=float(loss.detach()),ordinary_ce=float(bc.detach()),gradient_norm=grad,**stats))
                write(run/'TRAIN_PROGRESS.json',dict(tag=tag,step=step+1,completed_models=len(results),seconds=time.monotonic()-began))
                if (step+1)%200==0:
                    save_checkpoint(attempt_dir/f'CHECKPOINT_{step+1:04d}.pt',net,optimizer,step+1,schedule,model_binding)
                    if step+1<1200 and time.monotonic()-session_began>config['max_session_hours']*3600-600:return
            final=c.model_identity(net)['sha256']
            if initial==final:raise ValueError('NO_PARAMETER_UPDATE')
            if (target/'FINAL.pt').exists():
                previous=torch.load(target/'FINAL.pt',map_location='cpu',weights_only=True)
                if any(not torch.equal(previous[k],v.detach().cpu()) for k,v in net.state_dict().items()):raise ValueError('UNSEALED_FINAL_DIFFERS_FROM_FINAL_CHECKPOINT')
            else:atomic_torch(target/'FINAL.pt',{k:v.detach().cpu() for k,v in net.state_dict().items()})
            result=dict(tag=tag,arm=arm,seed=seed,updates=1200,base_updates=0,initial_state_sha256=initial,
                final_state_sha256=final,checkpoint_sha256=sha(target/'FINAL.pt'),binding=model_binding,
                test_losses_or_scores_read=False,training_seconds=time.monotonic()-began,
                actual_attempt_updates=sum(len(c.records(p)) for p in target.glob('attempt_*/STEPS.jsonl')))
            write(target/'RESULT.json',result,True);results.append(result)
            del net,optimizer;torch.cuda.empty_cache()
            if len(results)<9 and time.monotonic()-session_began>config['max_session_hours']*3600-600:return
    immutable(folder/'RESULT.json',dict(models=results,total_updates=10800,base_updates=0,final_step=1200))

if __name__=='__main__':main(Path(sys.argv[1]))
