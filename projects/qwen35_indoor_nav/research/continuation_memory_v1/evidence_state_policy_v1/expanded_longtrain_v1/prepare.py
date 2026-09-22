"""Bind immutable inputs and extend the existing FIT schedule without replaying1200updates."""
import random
from collections import Counter,defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def extend(data,old,target,seed):
    groups=defaultdict(list)
    for fid in data['arms']['EXPANDED']:groups[data['family_files'][fid]['parent']].append(fid)
    for ids in groups.values():ids.sort()
    rng=random.Random(seed);visits=Counter();parents=sorted(groups);rows=[]
    while len(rows)<target:
        order=parents.copy();rng.shuffle(order)
        for parent in order:
            if len(rows)==target:break
            fid=groups[parent][visits[parent]%len(groups[parent])];visits[parent]+=1
            rows.append(dict(family=fid,ordinary=old[len(rows)%len(old)]['ordinary']))
    if rows[:len(old)]!=old:raise ValueError('SCHEDULE_PREFIX_CHANGED')
    return rows

def main(run):
    cfg=config(run);source=Path(cfg['source_run'])
    if (run/'PREPARED.json').exists():
        for p,h in read(run/'PREPARED.json')['files'].items():
            if sha(Path(p))!=h:raise ValueError('PREPARED_INPUT_CHANGED:'+p)
        return
    data=read(source/'TRAIN_DATA.json');old=read(source/'SCHEDULES.json')['EXPANDED']['1209']
    schedule=extend(data,old,cfg['steps'],cfg['seed']);immutable(run/'SCHEDULE.json',schedule)
    natural=read(PARENT.parent/'natural_transfer_v9/DATA.json')
    if any(natural['records'][j]['partition']!='fit' for item in schedule for j in item['ordinary']):raise ValueError('NONFIT_ORDINARY_SAMPLE')
    counts=Counter(x['family'] for x in schedule);parents={data['family_files'][fid]['parent'] for fid in counts}
    if set(counts)!=set(data['arms']['EXPANDED']):raise ValueError('UNEXPOSED_EXPANDED_VARIANT')
    bound=read(source/'BINDING.json');feature=bound['feature_result'];resume=Path(cfg['source_resume'])
    meta=read(resume.with_suffix('.json'))
    if sha(resume)!=cfg['source_resume_sha256'] or meta['sha256']!=cfg['source_resume_sha256'] or meta['step']!=1200 or meta['binding']['inputs']!=digest(bound):raise ValueError('SOURCE_RESUME_IDENTITY')
    paths=[source/n for n in ('TRAIN_DATA.json','SHARED_FIT_WEIGHTS.json','SCHEDULES.json','BINDING.json','DATA.json','WARMUP_DATA.json','DATA_AUDIT.json','EVALUATION_REGISTRY.json')]
    paths += [source/'features/FEATURES.pt',source/'features/FEATURE_RESULT.json',Path(feature['ordinary_path']),PARENT.parent/'natural_transfer_v9/DATA.json',resume,resume.with_suffix('.json'),run/'SCHEDULE.json',run/'PROTOCOL.json',run/'SOURCE_LOCK.json']
    files={str(p):sha(p) for p in paths}
    if files[str(source/'features/FEATURES.pt')]!=feature['file_sha256'] or files[feature['ordinary_path']]!=feature['ordinary_sha256']:raise ValueError('CACHE_CHANGED')
    if not feature['parameters_unchanged'] or feature['base_updates']!=0:raise ValueError('CACHE_BASE_NOT_FROZEN')
    immutable(run/'PREPARED.json',dict(files=files,source_binding_digest=digest(bound),source_checkpoint_binding=meta['binding'],
        first1200_identical=True,variants=len(counts),parents=len(parents),total_updates=100000,new_updates=98800,
        feature_result=feature,exposure=dict(counts),ordinary_schedule_period=1200))
    print('PREPARED',len(schedule),len(counts),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
