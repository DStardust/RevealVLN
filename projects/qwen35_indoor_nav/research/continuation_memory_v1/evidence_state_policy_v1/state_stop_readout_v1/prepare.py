"""Bind old FIT/DEV, ordinary pool and three MONOTONIC finals; no new test data."""
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def main(run):
    cfg=config(run);source=Path(cfg['source_models']);verify_lock(read(source/'SOURCE_LOCK.json'))
    old_binding=read(CPU/'BINDING.json')
    for path,expected in old_binding['files'].items():
        if sha(Path(path))!=expected:raise ValueError('TRAINING_ASSET_CHANGED:'+path)
    data=read(CPU/'DATA.json');ordinary_path=PARENT.parent/'natural_transfer_v9/DATA.json';ordinary=read(ordinary_path)
    ordinary_cache=Path(old_binding['feature_result']['ordinary_path'])
    if sha(ordinary_cache)!=old_binding['feature_result']['ordinary_sha256']:raise ValueError('ORDINARY_CACHE_CHANGED')
    dev_houses={f['house'] for f in data['raw_families'] if f['split']=='DEV'}
    if dev_houses&{r['row']['scene_group'] for r in ordinary['records'] if r['partition']=='fit'}:raise ValueError('HOUSE_LEAK')
    files={CPU/'DATA.json':run/'WARMUP_DATA.json',CPU/'SCHEDULES.json':run/'SCHEDULES.json',
        CONTROL/'features/FEATURES.pt':run/'features/FEATURES.pt',CONTROL/'features/FEATURE_RESULT.json':run/'features/FEATURE_RESULT.json',
        ordinary_path:run/'ORDINARY_DATA.json',ordinary_cache:run/'features/ORDINARY_FEATURES.pt'}
    # DATA.json is also written separately below.
    for path in (CONTROL/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
        files[path]=run/path.relative_to(CONTROL);files[path.parent/'STATE_SEAL.json']=run/path.parent.relative_to(CONTROL)/'STATE_SEAL.json'
    receipts=[]
    for seed in cfg['seeds']:
        tag=f'MONOTONIC_{seed}';record=read(source/'train'/tag/'RESULT.json');path=source/'train'/tag/'FINAL.pt'
        if record['updates']!=1200 or sha(path)!=record['checkpoint_sha256']:raise ValueError('SOURCE_MODEL_CHANGED')
        for name in ('FINAL.pt','RESULT.json'):files[source/'train'/tag/name]=run/'train'/tag/name
        receipts.append(dict(model=tag,sha256=record['checkpoint_sha256'],state_sha256=record['final']))
    for src,dst in files.items():
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha(dst)!=sha(src):raise ValueError('BOUND_COPY_CHANGED')
        else:shutil.copyfile(src,dst)
    immutable(run/'DATA.json',data)
    from evaluate_continuations import registry_value
    reg=registry_value(data['raw_families'],cfg);immutable(run/'EVALUATION_REGISTRY.json',reg)
    old_conditions=read(PARENT/'gpu_runtime_r1/runs/gpu_001/EVALUATION_REGISTRY.json')['conditions']
    if [{k:v for k,v in x.items() if k!='available'} for x in reg['conditions']]!=old_conditions:raise ValueError('DEV_CONDITIONS_CHANGED')
    immutable(run/'MODEL_REUSE.json',dict(models=receipts,base_updates=0,old_model_updates=0,repair_trainable_parameters=113))
    immutable(run/'DATA_AUDIT.json',dict(data_sha256=sha(run/'DATA.json'),registry_sha256=sha(run/'EVALUATION_REGISTRY.json'),
        planned_slots=len(reg['slots']),fit_variants=sum(f['split']=='FIT' for f in data['families']),dev_variants=16,
        new_house_test_data_loaded=False,original_action_and_state_labels_unchanged=True))
    immutable(run/'BINDING.json',dict(files={str(p):sha(p) for p in list(files.values())+[run/'DATA.json',run/'EVALUATION_REGISTRY.json']},
        original_cpu_binding_sha256=sha(CPU/'BINDING.json')))

if __name__=='__main__':main(Path(sys.argv[1]))
