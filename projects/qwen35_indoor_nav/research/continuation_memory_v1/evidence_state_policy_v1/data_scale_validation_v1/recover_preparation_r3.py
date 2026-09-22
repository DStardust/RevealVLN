"""Seal the completed CPU compilation after fixing registry import; no regeneration."""
import argparse,shutil,sys,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *

def copy_verified(source,target,expected):
    if sha(source)!=expected:raise ValueError('RECOVERY_SOURCE_CHANGED:'+str(source))
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(target)!=expected:raise ValueError('RECOVERY_DESTINATION_CHANGED:'+str(target))
    else:
        shutil.copyfile(source,target)
        if sha(target)!=expected:raise ValueError('RECOVERY_COPY_CHANGED')

def main(source,run):
    cfg=read(HERE/'PROTOCOL.json')
    if read(source/'PROTOCOL.json')!=cfg:raise ValueError('RECOVERY_PROTOCOL_CHANGED')
    if list((source/'features').glob('session_*')) or list((source/'train').glob('*/FINAL.pt')):raise ValueError('NOT_CPU_ONLY_RECOVERY')
    progress=read(source/'PREPARE_PROGRESS.json')
    if progress['complete']!=1491 or progress['total']!=1491:raise ValueError('COMPILATION_INCOMPLETE')
    # Verify old source identities against preserved source bytes where R3 changes wiring.
    source_lock=read(source/'SOURCE_LOCK.json')
    for rel,h in source_lock['files'].items():
        p=LINE/rel
        if sha(p)!=h:
            if p.parent!=HERE or sha(HERE/'source_revisions/r2'/p.name)!=h:raise ValueError('UNEXPLAINED_OLD_SOURCE_CHANGE:'+rel)
    if sha(Path(cfg['expanded_data']))!=cfg['expanded_sha256']:raise ValueError('RAW_DATA_IDENTITY')
    run.mkdir(parents=True,exist_ok=True)
    immutable(run/'PROTOCOL.json',cfg);immutable(run/'SOURCE_LOCK.json',read(HERE/'SOURCE_LOCK.json'))
    data=read(source/'TRAIN_DATA.json');old=read(Path(cfg['old_data']));extra=read(Path(cfg['expanded_data']))['raw_families']
    raw=[f for f in old['raw_families'] if f['split']=='FIT']+extra
    extra_ids={f['family_id'] for f in extra}
    byid={f['family_id']:f for f in raw};counts=Counter();raw_references={};audit_arrays={}
    if set(byid)!=set(data['family_files']) or len(data['arms']['OLD'])!=59 or len(data['arms']['EXPANDED'])!=1550:raise ValueError('COMPILED_FAMILY_DENOMINATOR')
    for i,(fid,info) in enumerate(data['family_files'].items()):
        p=source/info['path'];copy_verified(p,run/info['path'],info['sha256']);f=read(p)
        if f['family_id']!=fid or f['house']!=byid[fid]['house'] or f['split']!='FIT':raise ValueError('COMPILED_FAMILY_IDENTITY')
        if fid in extra_ids:
            for row in f['sequences']:
                counts['causal_steps_with_reuse']+=len(row['features']);counts['action_labels']+=sum(row['action_masks'])
        for row in f['sequences']:
            if any(j<0 or j>=len(data['features']) or data['features'][j]['split']!='FIT' for j in row['features']):raise ValueError('COMPILED_INPUT_SPLIT')
            ref=row['source_trace'];raw_references[ref['path']]=ref['sha256']
        if i%100==0:write(run/'RECOVERY_PROGRESS.json',dict(families_verified=i+1,total=len(byid),stage='verify_shards',unix=time.time()))
    for f in extra:
        p=LINE/f['audit']['path']
        if sha(p)!=f['audit']['sha256']:raise ValueError('RAW_ADMISSION_CHANGED')
        for path,h in read(p)['arrays'].items():
            if path in audit_arrays and audit_arrays[path]!=h:raise ValueError('ARRAY_SOURCE_CONFLICT')
            audit_arrays[path]=h
    for i,(p,h) in enumerate({**raw_references,**audit_arrays}.items()):
        if sha(LINE/p)!=h:raise ValueError('RAW_EVIDENCE_CHANGED:'+p)
        if i%1000==0:write(run/'RECOVERY_PROGRESS.json',dict(stage='verify_raw_bytes',files_verified=i+1,total=len(raw_references)+len(audit_arrays),unix=time.time()))
    copies=['TRAIN_DATA.json','SCHEDULES.json','SHARED_FIT_WEIGHTS.json','WARMUP_DATA.json','DATA.json']
    bindings={n:sha(source/n) for n in copies}
    for n,h in bindings.items():copy_verified(source/n,run/n,h)
    evaluator=local_module('evaluate_continuations');reg=evaluator.registry_value(read(run/'DATA.json')['raw_families'],cfg)
    immutable(run/'EVALUATION_REGISTRY.json',reg)
    natural=read(PARENT.parent/'natural_transfer_v9/DATA.json');ordinary_fit={r['row']['scene_group'] for r in natural['records'] if r['partition']=='fit'}
    olddev={f['house'] for f in old['raw_families'] if f['split']!='FIT'};ordinary_check={r['row']['scene_group'] for r in natural['records'] if r['partition']!='fit'}
    if {f['house'] for f in raw}&(set(cfg['houses'])|olddev|ordinary_check):raise ValueError('FIT_HOUSE_LEAK')
    if ordinary_fit&(set(cfg['houses'])|olddev):raise ValueError('ORDINARY_HOUSE_LEAK')
    immutable(run/'DATA_AUDIT.json',dict(old_parents=32,old_variants=59,new_parents=816,new_variants=1491,expanded_parents=848,expanded_variants=1550,new_houses=sorted({f['house'] for f in extra}),heldout=cfg['houses'],heldout_exposed=True,arrays_verified=progress['arrays_verified'],array_files_rehashed=len(audit_arrays),source_traces_rehashed=len(raw_references),features=len(data['features']),counts=counts,ordinary_FIT_overlap_expansion=sorted(ordinary_fit&{f['house'] for f in extra}),new_physical_executions=0,data_sha256=sha(run/'DATA.json'),registry_sha256=sha(run/'EVALUATION_REGISTRY.json')))
    schedules=read(run/'SCHEDULES.json');exposure={}
    for arm,ss in schedules.items():
        exposure[arm]={s:dict(unique_variants=len({r['family'] for r in rows}),unique_parents=len({data['family_files'][r['family']]['parent'] for r in rows}),variant_occurrences=dict(Counter(r['family'] for r in rows))) for s,rows in ss.items()}
    immutable(run/'EXPOSURE_SCHEDULE.json',exposure)
    immutable(run/'RECOVERY_SEAL.json',dict(source=str(source),source_files=bindings,source_lock_sha256=sha(source/'SOURCE_LOCK.json'),failure_log_sha256=sha(source/'attempts/prepare_gpu0_001/stdout.log'),source_families=len(byid),reused_windows=len(data['features']),labels_or_sampling_changed=False,old_run_read_only=True,recovery='Only registry import repaired; compiled outputs copied byte-identically, all original source traces and array bytes rehashed.'))
    names=copies+['DATA_AUDIT.json','EVALUATION_REGISTRY.json','EXPOSURE_SCHEDULE.json','RECOVERY_SEAL.json']
    immutable(run/'PREPARED.json',dict(files={n:sha(run/n) for n in names},family_files=len(byid),feature_windows=len(data['features'])))
    write(run/'PREPARE_PROGRESS.json',dict(complete=1491,total=1491,features=len(data['features']),done=True,reused_from=str(source)))
    print(dict(status='RECOVERED_COMPLETE_PREPARATION',families=len(byid),features=len(data['features']),planned_slots=len(reg['slots'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--run',required=True,type=Path);a=p.parse_args();main(a.source.resolve(),a.run.resolve())
