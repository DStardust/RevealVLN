"""Freeze the six-model registry before training or reading new scores."""
from pathlib import Path
import random
import sys
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import common as u
import torch
from confirmation_model import ConfirmationMemory


def main():
    old=HERE.parent/'recovery_action_v1/runs/action_001';run=HERE/'runs/confirmation_001'
    run.mkdir(parents=True,exist_ok=False)
    previous=u.read(old/'data/ADMISSION.json');pool_path=old/'data/POOLS.pt'
    assert u.sha(pool_path)==previous['pools_sha256']
    pool=torch.load(pool_path,map_location='cpu',weights_only=True)
    p=u.read(old/'PROTOCOL.json');p.update(id='RECOVERY_CONFIRMATION_V2',previous_run=str(old),seeds=[42,43,44],
        models={f'{a}_s{s}':dict(architecture=a,seed=s) for s in (42,43,44) for a in ('CONCAT','LOCAL')},
        heads={},arms=['CONCAT','LOCAL'],planned_unseen_episodes=1400,planned_dev_episodes=448,
        phases=['TRAIN','DEV','UNSEEN','REVIEW'],primary_comparison='CONCAT vs LOCAL within each seed; CONCAT vs NATIVE replication',
        architecture_package='Same concatenation head; LOCAL uses a unit-gain tanh(writer(current)) embedding, no added recurrence',
        interpretation='LOCAL retains StreamVLN history; parameter storage matched, effective recurrence differs by design',
        selection='Final step 3000, all seeds reported; no DEV/unseen checkpoint or hyperparameter selection',
        limitations=['Exposed unseen200, not blind or full1839','Does not isolate data vs preservation loss contributions',
                     'LOCAL retains the same stored parameter set; recurrent matrix unused; unit current-write gain avoids a 100x attenuation',
                     'Seed42 repeats the discovery seed; seeds43/44 test training variability on the same exposed episodes'])
    sources={str(pool_path):previous['pools_sha256'],str(old/'data/ADMISSION.json'):u.sha(old/'data/ADMISSION.json')}
    for s in p['seeds']:
        data=run/'data'/str(s);data.mkdir(parents=True)
        rec=[i for i,r in enumerate(pool['rows']) if r['partition']=='FIT' and r['kind']=='RECOVERY']
        ordinary=[i for i,r in enumerate(pool['rows']) if r['partition']=='FIT' and r['kind']=='PRESERVATION']
        rng=random.Random(s);schedule=[]
        for step in range(p['steps']):
            if step%len(rec)==0:rng.shuffle(rec)
            if step%len(ordinary)==0:rng.shuffle(ordinary)
            schedule.append([rec[step%len(rec)],ordinary[step%len(ordinary)]])
        u.write(data/'SCHEDULE.json',schedule)
        torch.manual_seed(s);initial=ConfirmationMemory(3584,'CONCAT');torch.save(initial.state_dict(),data/'INITIAL.pt')
        if s==42:
            assert schedule==u.read(old/'data/SCHEDULE.json')
            old_initial=torch.load(old/'data/INITIAL.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(v,old_initial[k]) for k,v in initial.state_dict().items())
        admission=dict(previous,pools_path=str(pool_path),seed=s,initial_sha256=u.sha(data/'INITIAL.pt'),schedule_sha256=u.sha(data/'SCHEDULE.json'))
        u.write(data/'ADMISSION.json',admission)
        for f in data.iterdir():sources[str(f)]=u.sha(f)
    for phase in ('dev','unseen'):
        for name in ('DATA_MANIFEST.json','PROTOCOL.json'):
            f=old/phase/name;sources[str(f)]=u.sha(f)
    for f in (old/'dev/prefixes').glob('*.json'):sources[str(f)]=u.sha(f)
    p['shared_source_files']=sources
    u.write(run/'PROTOCOL.json',p)
    u.write(run/'DATA_MANIFEST.json',u.read(old/'DATA_MANIFEST.json'))
    u.write(run/'STATUS.json',dict(status='READY',phase='TRAIN',gpu_hours=0))
    u.write(run/'PREFLIGHT.json',dict(status='PREPARED_CPU_TEST_PENDING',probe_gpu_hours=0,
        source_pool_sha256=previous['pools_sha256'],fresh_initialization=True,seed42_initial_and_schedule_equal_previous=True))
    print(run)


if __name__=='__main__':main()
