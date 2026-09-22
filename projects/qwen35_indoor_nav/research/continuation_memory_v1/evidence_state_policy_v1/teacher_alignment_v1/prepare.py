"""Freeze FIT-only alternative-teacher plans; preserve original trajectory pool."""
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from evaluator_v16 import legacy,evaluate,state_sequence

def first_ready(row):
    return next(t for t in range(row['cutoff'],len(row['targets']))
                if row['state_masks'][t] and row['state_targets'][t][3])

def supervision_audit(data):
    counts=Counter();contexts=defaultdict(list)
    for f in data['families']:
        if f['split']!='FIT':continue
        for index,row in enumerate(f['sequences']):
            for t,mask in enumerate(row['action_masks']):
                if not mask:continue
                counts['decisions']+=1
                counts['stop']+=row['targets'][t]==3
                counts['ready_continue']+=bool(row['state_targets'][t][3] and row['targets'][t]!=3)
                if row['task']=='task_A' and row['history'].startswith('missing'):
                    counts['missing_main_decisions']+=1
                    counts['missing_main_ready_continue']+=bool(row['state_targets'][t][3] and row['targets'][t]!=3)
                contexts[tuple(row['features'][:t+1])].append(dict(family=f['family_id'],sequence=index,
                    task=row['task'],history=row['history'],step=t,action=row['targets'][t]))
    conflicts=[dict(context_sha256=digest(key),labels=value) for key,value in contexts.items()
               if len({x['action'] for x in value})>1]
    return dict(counts=dict(counts),conflicting_complete_causal_contexts=conflicts,
                note='Complete recurrent feature histories, not only recent-window collisions.')

def copy_bound(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if sha(src)!=sha(dst):raise ValueError('COPY_CHANGED:'+str(dst))
    else:shutil.copyfile(src,dst)

def main(run):
    cfg=config(run);bound=read(CPU/'BINDING.json')
    for p,h in bound['files'].items():
        if sha(Path(p))!=h:raise ValueError('ORIGINAL_ASSET_CHANGED:'+p)
    data=read(CPU/'DATA.json');raw={f['family_id']:f for f in data['raw_families']}
    ordinary=PARENT.parent/'natural_transfer_v9/DATA.json'
    dev={f['house'] for f in data['families'] if f['split']=='DEV'}
    if dev&{r['row']['scene_group'] for r in read(ordinary)['records'] if r['partition']=='fit'}:raise ValueError('HOUSE_LEAK')
    files={CPU/'SCHEDULES.json':run/'SCHEDULES.json',
        CPU/'DATA.json':run/'WARMUP_DATA.json',ordinary:run/'ORDINARY_DATA.json',
        Path(bound['feature_result']['ordinary_path']):run/'features/ORDINARY_FEATURES.pt',
        CONTROL/'features/FEATURES.pt':run/'features/FEATURES.pt',CONTROL/'features/FEATURE_RESULT.json':run/'features/FEATURE_RESULT.json'}
    if sha(Path(bound['feature_result']['ordinary_path']))!=bound['feature_result']['ordinary_sha256']:raise ValueError('ORDINARY_FEATURE_CHANGED')
    for p in (CONTROL/'features').glob('session_*/FEATURES_INPUTS_*.jsonl'):
        files[p]=run/p.relative_to(CONTROL);files[p.parent/'STATE_SEAL.json']=run/p.parent.relative_to(CONTROL)/'STATE_SEAL.json'
    for src,dst in files.items():copy_bound(src,dst)
    immutable(run/'DATA.json',data)
    plans={};overlays=[];selected=0
    for f in data['families']:
        if f['split']!='FIT':continue
        family=raw[f['family_id']];compiler=legacy.Compiler(**family['compiler'])
        for index,row in enumerate(f['sequences']):
            if not any(row['action_masks']):continue
            selected+=1;ref=row['source_trace'];path=LINE/ref['path']
            if sha(path)!=ref['sha256']:raise ValueError('TEACHER_TRACE_CHANGED')
            trace=read(path)
            if evaluate(compiler,trace,row['task'],row['cutoff'])['safe_v16_label']!='PASS':raise ValueError('SOURCE_TEACHER_NOT_PASS')
            if state_sequence(compiler,trace['observations'],row['task'])!=row['state_targets']:raise ValueError('SOURCE_STATE_CHANGED')
            t=first_ready(row)
            if t==len(row['targets'])-1:continue
            actions=trace['actions'][:t]+['S']
            if len(actions)>500 or 'S' in actions[:-1]:raise ValueError('PLAN_STOP_BUDGET')
            key=digest([f['family_id'],ref,t])[:24]
            plans.setdefault(key,dict(id=key,family_id=f['family_id'],house=f['house'],split='FIT',source_trace=ref,
                stop_step=t,actions=actions,tasks=[]))
            task=dict(task=row['task'],cutoff=row['cutoff'])
            if task not in plans[key]['tasks']:plans[key]['tasks'].append(task)
            overlays.append(dict(family_id=f['family_id'],sequence=index,stop_step=t,plan=key))
    plan=dict(plans=sorted(plans.values(),key=lambda p:(p['house'],p['family_id'],p['id'])),overlays=overlays,
              selected_teacher_rows=selected,changed_teacher_rows=len(overlays),new_physical_executions=len(plans),
              selection='FIT-only first ready at/after takeover; no method scores or DEV labels used')
    immutable(run/'TEACHER_PLAN.json',plan);immutable(run/'ORIGINAL_SUPERVISION_AUDIT.json',supervision_audit(data))
    from evaluate_continuations import registry_value
    reg=registry_value(data['raw_families'],cfg)
    prior=read(PARENT/'state_stop_readout_positive_v3/runs/positive_001/EVALUATION_REGISTRY.json')
    if reg['conditions']!=prior['conditions'] or len(reg['slots'])!=768:raise ValueError('DEV_CONDITIONS_CHANGED')
    immutable(run/'EVALUATION_REGISTRY.json',reg)
    immutable(run/'DATA_AUDIT.json',dict(data_sha256=sha(run/'DATA.json'),registry_sha256=sha(run/'EVALUATION_REGISTRY.json'),
        fit_variants=59,fit_parents=32,fit_houses=4,dev_variants=16,dev_parents=8,dev_houses=1,
        unchanged_original_pool=True,cache_regenerated=False,independent_test_accessed=False))
    immutable(run/'PREPARE_RESULT.json',dict(planned_replays=len(plans),changed_teacher_rows=len(overlays),
        base_loaded=False,new_optimizer_updates=0,source_binding_sha256=sha(CPU/'BINDING.json')))
    print(read(run/'PREPARE_RESULT.json'),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
