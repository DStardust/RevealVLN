"""Deterministic CPU-only disjoint scaleout of real, still-unused bank programs."""
import collections
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
WF=HERE.parent
BE=WF/'batch_execution_v1'
ROOT=next(p for p in HERE.parents if p.name=='vla')
ORIGINAL=WF/'auto_generation_v1/queue.py'
ORIGINAL_SHA='529e8e77c7e37a3f1f2b408d89a192752197cf93ec163e1b1781689a2053c12c'
DEVICES=(3,4,5,7)
QUEUES=(WF/'auto_generation_v1/queue_v1/QUEUE.json',WF/'auto_generation_gpu1_v1/queue_v1/QUEUE.json')
MAX_BATCHES=128

def sha(path):
    path=Path(path);assert path.is_absolute() and path.resolve()==path and path.is_relative_to(ROOT),'CANONICAL_PROJECT_SCOPE'
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    after=path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'SOURCE_CHANGED'
    return h.hexdigest()
def read(path):return json.loads(path.read_text())
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def load(name,path,expected):
    assert sha(path)==expected,'FROZEN_CPU_METHOD_SOURCE_CHANGED'
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
q=load('scaleout_original_semantic_dedup',ORIGINAL,ORIGINAL_SHA)
def valid_row(row,fit):
    assert row['split']=='FIT' and row['house_id'] in fit,'FIT_ONLY'
    pos=row['configuration']['u_position']
    assert len(pos)==3 and all(type(x) in (int,float) and math.isfinite(x) for x in pos),'FINITE_FULL_3D_POSITION'
    assert row['component_provenance'] and row['runtime_allowed'] is False and row['executable'] is False
    return row
def hub_key(row):return (row['house_id'],tuple(row['configuration']['u_position']))
def prior_matches(row,prior):
    return [p for p in prior if row['candidate_id']==p['row']['candidate_id'] or q.same_program(row,p['row'])]
def plan(entries,prior):
    """Source capacity only; no runtime success/outcome is a selection feature.

    The longest remaining physical-hub group is consumed first, so a bank's
    last two hubs can mix with other banks rather than force fake new hubs.
    Ties are house/full-position, then the original frozen source ordering.
    """
    remaining=collections.OrderedDict();ledger=[];seen=[]
    for entry in entries:
        row=entry['row'];item={k:v for k,v in entry.items() if k!='row'}
        item['candidate_id']=row['candidate_id'];ledger.append(item)
        matches=prior_matches(row,prior)
        if matches:
            item.update(status='EXCLUDED_ALL_PRIOR_FROZEN_CONFIGS_NO_RETRY',
                prior=[{k:v for k,v in x.items() if k!='row'} for x in matches]);continue
        if any(q.same_program(row,x) for x in seen):item['status']='SEMANTIC_DUPLICATE';continue
        seen.append(row);item['status']='PENDING_NO_COMPLETE_THREE_HUB_BATCH'
        remaining.setdefault(hub_key(row),[]).append((entry,item))
    batches=[]
    while len(batches)<MAX_BATCHES:
        keys=sorted((k for k,v in remaining.items() if v),key=lambda k:(-len(remaining[k]),k))
        chosen=[]
        for key in keys:
            entry,item=remaining[key][0]
            if all(q.distinct_hub(entry['row'],x[1][0]['row']) for x in chosen):chosen.append((key,(entry,item)))
            if len(chosen)==3:break
        if len(chosen)!=3:break
        number=300+len(batches);gpu=DEVICES[len(batches)%len(DEVICES)]
        batch={'id':'batch_'+str(number),'gpu':gpu,'lane_index':len(batches)//4,
            'candidate_ids':[x[1][0]['row']['candidate_id'] for x in chosen],
            'source_items':[{k:v for k,v in x[1][0].items() if k!='row'} for x in chosen]}
        batches.append(batch)
        for key,(entry,item) in chosen:
            remaining[key].pop(0);item.update(status='FROZEN_NEW_SCALEOUT_BATCH',batch_id=batch['id'],gpu=gpu)
    # This bound is admission, not a reason to silently omit remaining entries.
    return batches,ledger
def assert_no_rolling_inputs(lock):
    for key in lock:
        path=Path(key)
        if path.is_relative_to(BE):
            relative=path.relative_to(BE).parts
            if len(relative)>=3 and relative[0].startswith('batch_') and relative[1]=='run_v1' and int(relative[0][6:].split('r')[0])>=100:
                assert relative[2:] == ('EXECUTION_CONFIG.json',),'ROLLING_RUN_FILES_NOT_ALLOWED_IN_SCALEOUT_SOURCE'

def freeze():
    out=HERE/'queue_v1';out.mkdir(exist_ok=False)
    split=WF.parents[2]/'data_pipeline/ordinary_scale_v1/SPLIT_FREEZE.json';fit=set(read(split)['FIT'])
    common={str(split):sha(split)};prior=[];exposure=[]
    # Only immutable prospective configs are read. In particular no rolling
    # batch100/202 HEAD, journal, PROGRESS, result or temporary prefix is used.
    cfg_paths=sorted(BE.glob('batch_*/run_v1/EXECUTION_CONFIG.json'))
    cfg_paths.append(WF/'short_revisit_v3/run_v1/EXECUTION_CONFIG.json')
    for path in cfg_paths:
        h=sha(path);cfg=read(path);assert sha(path)==h
        common[str(path)]=h
        source={'configuration':str(path),'configuration_sha256':h,
            'candidate_ids':[r['candidate_id'] for r in cfg['candidates']],
            'status_scope':'EXCLUDE_REGARDLESS_OF_SUCCESS_FAILURE_PARTIAL_OR_RESERVATION',
            'runtime_observed_by_this_freeze':False}
        exposure.append(source)
        for row in cfg['candidates']:
            prior.append({'row':row,'candidate_id':row['candidate_id'],'configuration':str(path)})
    for path in QUEUES:
        queue=read(path);source_lock=path.parent/'SOURCE_LOCK.json';bindings=read(source_lock)
        assert bindings[str(path)]==sha(path)
        assert queue['candidate_count']==36 and queue['batch_count']==12
        for batch in queue['batches']:
            expected=str(BE/batch['id']/'run_v1/EXECUTION_CONFIG.json')
            actual={r['candidate_id'] for r in prior if r['configuration']==expected}
            assert set(batch['candidate_ids'])==actual,'PRIOR_QUEUE_RESERVATION_MISSING'
        for p in (path,source_lock):common[str(p)]=sha(p)
    source_locks={};verified={};entries=[]
    for pool_order,pool in enumerate(q.POOLS):
        draft=read(pool/'CONFIG_DRAFT.json');assert draft['runtime_allowed'] is False and draft['training_allowed'] is False
        bindings=read(pool/'SOURCE_LOCK.json');source_locks[str(pool)]=bindings
        for path,h in bindings.items():
            if path in verified:assert verified[path]==h,'CONFLICTING_SOURCE_SHA'
            else:assert sha(Path(path))==h,path;verified[path]=h
        for path in (pool/'CONFIG_DRAFT.json',pool/'SOURCE_LOCK.json'):common[str(path)]=sha(path)
        for index,row in enumerate(draft['candidates']):
            valid_row(row,fit)
            entries.append({'row':row,'source_snapshot':str(pool),'source_index':index,'pool_order':pool_order})
    batches,ledger=plan(entries,prior)
    assert batches,'NO_NEW_COMPLETE_THREE_HUB_BATCH'
    v=load('scaleout_frozen_language',q.VERBALIZER,q.VERBALIZER_SHA)
    lookup={(e['source_snapshot'],e['source_index']):e['row'] for e in entries};changes=[]
    for batch in batches:
        assert not (BE/batch['id']).exists(),'EXISTING_BATCH_NOT_REUSED'
        snapshot=out/batch['id'];snapshot.mkdir()
        rows=[]
        for item in batch['source_items']:
            row,revision=q.normalize(lookup[item['source_snapshot'],item['source_index']],v)
            rows.append(row);changes.append({'batch_id':batch['id'],'candidate_id':row['candidate_id'],'changes':revision['changes']})
        template=copy.deepcopy(read(Path(batch['source_items'][0]['source_snapshot'])/'CONFIG_DRAFT.json'))
        template.update(node='Q35N_SPECIAL_SCALEOUT_V1_'+batch['id'].upper(),candidates=rows,gpu_device=batch['gpu'],
            runtime_allowed=False,executable=False,training_allowed=False,requires_main_agent_admission=True)
        save(snapshot/'CONFIG_DRAFT.json',template)
        batch.update(prepared_snapshot=str(snapshot),source_snapshot=str(snapshot),source_indices=[0,1,2])
    queue={'schema_version':'q35n.special_scaleout_queue.v1','batches':batches,'batch_count':len(batches),
        'candidate_count':3*len(batches),'gpu_devices':list(DEVICES),'each_lane_wall_seconds':43200,
        'per_batch_supervisor_seconds':3900,'per_batch_factory_seconds':3600,'per_batch_actions':60000,
        'per_batch_disk_bytes':7*1024**3,'max_programmed_batches':MAX_BATCHES,
        'transport_limits_and_holder_authorization_required':True,'source_selection_uses_runtime_outcomes':False,
        'auto_retry':False,'cross_lane_candidate_reassignment':False,'stop_lane_on_transport_failure':True,
        'candidate_is_accepted_family':False,'new_hub_collection_performed':False,
        'same_hub_and_house_programs_correlated':True,'runtime_allowed':False,'scientific_pass':False}
    save(out/'QUEUE.json',queue);save(out/'EXPOSURE.json',exposure)
    lane_paths=[]
    for gpu in DEVICES:
        lane=out/('GPU_'+str(gpu)+'_QUEUE.json');lane_paths.append(lane)
        selected_batches=[b for b in batches if b['gpu']==gpu]
        save(lane,dict(queue,batches=selected_batches,batch_count=len(selected_batches),
            candidate_count=3*len(selected_batches),gpu_devices=[gpu],parent_queue=str(out/'QUEUE.json')))
    save(out/'SELECTION_LEDGER.json',ledger);save(out/'LANGUAGE_CHANGES.json',changes)
    for path in [*HERE.glob('*.py'),HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',ORIGINAL,q.VERBALIZER,
        ORIGINAL.parent/'SHA256SUMS',out/'QUEUE.json',out/'EXPOSURE.json',out/'SELECTION_LEDGER.json',out/'LANGUAGE_CHANGES.json',*lane_paths]:
        common[str(path)]=sha(path)
    assert_no_rolling_inputs(common);save(out/'SOURCE_LOCK.json',common)
    lock_counts=[];lock_bytes=[]
    for batch in batches:
        merged={}
        for item in batch['source_items']:
            for path,h in source_locks[item['source_snapshot']].items():
                assert path not in merged or merged[path]==h
                merged[path]=h
        merged.update(common);snapshot=Path(batch['prepared_snapshot'])
        for path in (snapshot/'CONFIG_DRAFT.json',out/'SOURCE_LOCK.json'):merged[str(path)]=sha(path)
        assert_no_rolling_inputs(merged)
        size=sum(Path(p).stat().st_size for p in merged)
        assert len(merged)<=1900,'LEAVE_RUNTIME_CLOSURE_HEADROOM_BELOW_2048'
        assert size<=30*1024**3,'LEAVE_AUDIT_HEADROOM_BELOW_32_GIB'
        save(snapshot/'SOURCE_LOCK.json',merged);lock_counts.append(len(merged));lock_bytes.append(size)
    selected=[lookup[item['source_snapshot'],item['source_index']] for b in batches for item in b['source_items']]
    result={'status':'SCALEOUT_SOURCES_FROZEN_NOT_RUNTIME_PREPARED','raw_bank_candidates':len(entries),
        'unreserved_candidates':sum(x['status'] in ('FROZEN_NEW_SCALEOUT_BATCH','PENDING_NO_COMPLETE_THREE_HUB_BATCH') for x in ledger),
        'candidate_count':len(selected),'batch_count':len(batches),
        'batches_per_gpu':dict(collections.Counter(b['gpu'] for b in batches)),
        'selection_statuses':dict(collections.Counter(x['status'] for x in ledger)),
        'fit_houses':len({r['house_id'] for r in selected}),'physical_hubs':len({hub_key(r) for r in selected}),
        'max_snapshot_source_files':max(lock_counts),'max_snapshot_source_bytes':max(lock_bytes),
        'old_runtime_sources_untouched':True,'new_hubs':0,'physical_families':0,'gpu_operations':0,'scientific_pass':False}
    save(out/'result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':freeze()
