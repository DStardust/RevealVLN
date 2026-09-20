"""Fixed two-size, three-seed B1 / strong B2 / crossed-supervision comparison."""
from pathlib import Path
import random
import sys
import time
import traceback
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import objective as o
c=o.c


@torch.no_grad()
def ordinary_evaluate(net,cache,records):
    rows=[]
    for start in range(0,len(records),8):
        chosen=records[start:start+8];n=max(len(r['features']) for r in chosen);device=cache['features'].device
        indices=torch.tensor([r['features']+[0]*(n-len(r['features'])) for r in chosen],device=device)
        features=cache['features'][indices];states,_=net.encode(features)
        base=cache['logits'][indices];logits=net.action_logits(states.flatten(0,1),base.flatten(0,1),features.flatten(0,1)).view(*indices.shape,4)
        for i,r in enumerate(chosen):
            k=len(r['features']);target=torch.tensor(r['targets'],device=device);current=logits[i,:k];native=base[i,:k]
            rows.append(dict(record_id=r['row']['record_id'],decisions=k,correct=int((current.argmax(-1)==target).sum()),
                base_correct=int((native.argmax(-1)==target).sum()),ce=float(F.cross_entropy(current,target)),
                base_ce=float(F.cross_entropy(native,target)),premature_stops=int((current[:-1].argmax(-1)==3).sum()),
                terminal_stop_correct=bool(current[-1].argmax()==3)))
    return dict(rows=rows,routes=len(rows),correct=sum(r['correct'] for r in rows),decisions=sum(r['decisions'] for r in rows),
        mean_route_ce=sum(r['ce'] for r in rows)/len(rows),scope='Previously exposed natural teacher routes; not navigation SR')


def main(run):
    config=c.read(HERE/'TRAIN_PROTOCOL.json')
    for path,digest in config['source_hashes'].items():assert c.sha(c.LINE/path)==digest,path
    assert c.read(HERE/'CPU_TEST_RESULT.json')['passed']
    feature_run=HERE/'features_run_001';identity=c.read(feature_run/'FEATURE_RESULT.json')
    assert identity['parameters_unchanged'] and identity['real_qwen_forwards']==9312
    assert c.sha(feature_run/'FEATURES.pt')==identity['file_sha256']
    torch.cuda.set_device(0);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0'
    data=c.read(HERE/'DATA.json')
    cache={k:v.float().to(device) for k,v in torch.load(feature_run/'FEATURES.pt',map_location='cpu',weights_only=True).items()}
    natural_folder=HERE.parent/'natural_transfer_v9'
    natural=c.read(natural_folder/'DATA.json')
    assert c.sha(natural_folder/'run_001/FEATURES.pt')==c.read(natural_folder/'run_001/FEATURE_RESULT.json')['file_sha256']
    ordinary={k:v.float().to(device) for k,v in torch.load(natural_folder/'run_001/FEATURES.pt',map_location='cpu',weights_only=True).items()}
    batches={f['family_id']:o.batch(f,device) for f in data['families']}
    c.write(run/'RUNTIME_IDENTITY.json',dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        dtype='float32 memory/heads and cached causal features',gpu_uuid=config['gpu_uuid'],
        feature_cache_sha256=identity['file_sha256'],base_state_sha256=identity['base_state_sha256'],
        base_model_loads=0,base_optimizer_updates=0,tf32=False,deterministic_algorithms=True),True)
    results=[]
    for size,ids in config['data_sizes'].items():
        for seed in config['seeds']:
            rng=random.Random(seed);schedule=[]
            while len(schedule)<config['steps_per_arm']:
                cycle=list(ids);rng.shuffle(cycle);schedule.extend(cycle)
            schedule=schedule[:config['steps_per_arm']]
            ordinary_schedule=c.read(natural_folder/'run_001'/f'SCHEDULE_{seed}.json')['ordinary_indices']
            assert len(ordinary_schedule)==config['steps_per_arm']
            assert all(natural['records'][i]['partition']=='fit' for ids_in_step in ordinary_schedule for i in ids_in_step)
            c.write(run/f'SCHEDULE_{size}_{seed}.json',dict(families=schedule,ordinary_indices=ordinary_schedule,
                exposure_counts={f:schedule.count(f) for f in ids}),True)
            for arm in config['arms']:
                tag=f'{size}_{arm}_{seed}';net=o.initialize(seed).to(device)
                initial=c.model_identity(net);optimizer=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
                began=time.monotonic()
                for step,(family_id,ordinary_ids) in enumerate(zip(schedule,ordinary_schedule),1):
                    optimizer.zero_grad(set_to_none=True)
                    special,stats,_=o.losses(net,cache,batches[family_id],arm)
                    bc=o.ordinary_loss(net,ordinary,[natural['records'][i] for i in ordinary_ids])
                    loss=special+bc;assert bool(torch.isfinite(loss))
                    loss.backward();assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters())
                    optimizer.step()
                    c.append(run/(tag+'_STEPS.jsonl'),dict(step=step,family_id=family_id,loss=float(loss.detach()),ordinary_ce=float(bc.detach()),**stats))
                    c.write(run/'PROGRESS.json',dict(unix=time.time(),phase='TRAIN',tag=tag,step=step,total=config['steps_per_arm'],completed_models=len(results)))
                net.eval()
                selected=list(batches.values())
                measured=o.evaluate(net,cache,selected,arm)
                check_ordinary=ordinary_evaluate(net,ordinary,[r for r in natural['records'] if r['partition']=='check'])
                final=c.model_identity(net);assert final['sha256']!=initial['sha256']
                path=run/(tag+'_MEMORY.pt');torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},path)
                reloaded=o.initialize(seed);reloaded.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
                assert c.model_identity(reloaded)['sha256']==final['sha256']
                row=dict(tag=tag,data_size=size,arm=arm,seed=seed,updates=config['steps_per_arm'],
                    initial_state_sha256=initial['sha256'],final_state_sha256=final['sha256'],checkpoint_sha256=c.sha(path),
                    families=measured,ordinary_teacher_check=check_ordinary,seconds=time.monotonic()-began,
                    checkpoint_readback_verified=True,base_updates=0,autonomous_navigation_evaluated=False)
                c.write(run/(tag+'_RESULT.json'),row,True);results.append(dict(tag=tag,result=tag+'_RESULT.json'))
                del net,optimizer,reloaded,loss,special,bc
                torch.cuda.empty_cache()
    c.write(run/'RESULT.json',dict(status='MATCHED_TWO_SIZE_MEMORY_PILOT_COMPLETE',runs=results,
        actual_optimizer_updates=len(results)*config['steps_per_arm'],base_optimizer_updates=0,
        new_qwen_forwards=0,independent_test_houses=1,publication_ready=False),True)


if __name__=='__main__':
    run=Path(sys.argv[1])
    try:main(run)
    except BaseException as exc:
        c.write(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()),True)
        raise
