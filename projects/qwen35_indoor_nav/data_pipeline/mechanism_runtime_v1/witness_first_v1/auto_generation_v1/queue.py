"""CPU-only prospective queue. No retries, no simulator or GPU operations."""
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
POOLS=[WF/'multi_program_bank_v1/snapshot_v3',WF/'multi_program_bank_v2/language_ready_v1',
       WF/'multi_program_bank_v3/language_ready_v1',WF/'new_hub_bank_v1/language_ready_v1']
CORE=WF/'multi_program_bank_v1/core.py'
CORE_SHA='21cff82e02315dcc3616fa523687f09ea51ebcc6b934176bed1e316e30962dd8'
VERBALIZER=WF/'language_realization_v1/verbalizer.py'
VERBALIZER_SHA='ae5308f1c6257a846c20ddf42d05629b3b373b144517b283447273d9bbae6c1b'

def sha(path):
    p=Path(path);assert p.is_absolute() and p.resolve()==p and p.is_relative_to(ROOT)
    before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'SOURCE_CHANGED'
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def load(name,path,expected=None):
    if expected:assert sha(path)==expected,'FROZEN_SOURCE_CHANGED'
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def semantic(row):
    def role(n):return (row['roles'][n]['mpcat40'],row['roles'][n]['room'])
    return (row['house_id'],tuple(row['configuration']['u_position']),tuple(sorted((role('anchor_A'),role('anchor_B')))),role('terminal'))
def same_program(a,b):
    x,y=semantic(a),semantic(b)
    return x[0]==y[0] and math.dist(x[1],y[1])<1 and x[2:]==y[2:]
def distinct_hub(a,b):
    return a['house_id']!=b['house_id'] or math.dist(a['configuration']['u_position'],b['configuration']['u_position'])>=1
def attempted_ids(records):
    physical=set();entered=set()
    for r in records:
        p=r['payload']
        if r['kind']=='freeze':entered.update(k for k,v in p.items() if isinstance(v,dict) and v.get('attempt',0)>0)
        if r['kind']=='action_completed':physical.add(p['bundle'])
    return entered|physical,physical
def normalize(row,v):
    out=copy.deepcopy(row);revision=v.propose_revision(out)
    out['tasks']=revision['proposed_tasks']
    out['component_provenance']['auto_generation_language_revision']=revision
    a=copy.deepcopy(row);b=copy.deepcopy(out)
    for x in (a,b):
        for task in x['tasks'].values():task.pop('instruction')
        x['component_provenance'].pop('auto_generation_language_revision',None)
    assert a==b,'LANGUAGE_MUST_NOT_CHANGE_STRUCTURE'
    return out,revision
def choose(pool_entries,prior,max_batches=12):
    assert type(max_batches) is int and 1<=max_batches<=12
    ledger=[];pool_queues=[];seen=[]
    for pool,entries in pool_entries:
        groups=collections.OrderedDict()
        for index,row in entries:
            item={'source_snapshot':str(pool),'source_index':index,'candidate_id':row['candidate_id']}
            ledger.append(item)
            matches=[p for p in prior if row['candidate_id']==p['row']['candidate_id'] or same_program(row,p['row'])]
            if matches:
                item['status']='EXCLUDED_PRIOR_ATTEMPT_NO_RETRY' if any(p['attempted'] for p in matches) else 'RESERVED_PRIOR_FROZEN_NOT_PHYSICAL'
                item['prior']=[{k:v for k,v in p.items() if k!='row'} for p in matches];continue
            if any(same_program(row,p) for p in seen):item['status']='DUPLICATE_SEMANTIC_PROGRAM';continue
            seen.append(row);item['status']='PENDING_QUEUE_BUDGET'
            key=(row['house_id'],tuple(row['configuration']['u_position']))
            groups.setdefault(key,[]).append((index,row,item))
        ordered=[]
        keys=sorted(groups)
        for depth in range(max(map(len,groups.values()),default=0)):
            for key in keys:
                if depth<len(groups[key]):ordered.append(groups[key][depth])
        pool_queues.append((pool,ordered))
    batches=[]
    # Pool round-robin makes the first 12 batches cover the available source
    # houses/hubs, not merely exhaust one recently successful source.
    while len(batches)<max_batches:
        progressed=False
        for pool,queue in pool_queues:
            selected=[]
            for entry in queue:
                if all(distinct_hub(entry[1],old[1]) for old in selected):selected.append(entry)
                if len(selected)==3:break
            if len(selected)!=3:continue
            batch={'id':'batch_'+str(100+len(batches)),'gpu':2,'source_snapshot':str(pool),
                   'source_indices':[x[0] for x in selected],'candidate_ids':[x[1]['candidate_id'] for x in selected]}
            batches.append(batch);progressed=True
            ids=set(batch['candidate_ids']);queue[:]=[x for x in queue if x[1]['candidate_id'] not in ids]
            for _,_,item in selected:item.update(status='FROZEN_FOR_NEW_BATCH',batch_id=batch['id'])
            if len(batches)==max_batches:break
        if not progressed:break
    return batches,ledger

def freeze():
    out=HERE/'queue_v1';out.mkdir(exist_ok=False)
    c=load('auto_queue_frozen_core',CORE,CORE_SHA);v=load('auto_queue_verbalizer',VERBALIZER,VERBALIZER_SHA)
    lock={};prior=[];history=[]
    paths=sorted(BE.glob('batch_*/run_v1/EXECUTION_CONFIG.json'))
    paths.append(WF/'short_revisit_v3/run_v1/EXECUTION_CONFIG.json')
    for path in paths:
        raw,h=c.stable(path);lock[str(path)]=h;cfg=json.loads(raw)
        run=path.parent;label=run.parent.name
        entries=physical=set();evidence='NO_COMMITTED_JOURNAL_RESERVATION_ONLY'
        if (run/'journal/HEAD.json').exists():
            cfg,records=c.read_committed_prefix(run,out/('exposure_'+label),lock)
            entries,physical=attempted_ids(records);evidence='VERIFIED_COMMITTED_PREFIX'
        for row in cfg['candidates']:
            prior.append({'row':row,'candidate_id':row['candidate_id'],'run_root':str(run),
                'attempted':row['candidate_id'] in entries,'actual_action_observed':row['candidate_id'] in physical,'evidence':evidence})
        history.append({'run_root':str(run),'frozen_ids':[r['candidate_id'] for r in cfg['candidates']],
            'entered_attempt_ids':sorted(entries),'physical_action_ids':sorted(physical),'evidence':evidence})
    pool_entries=[]
    for pool in POOLS:
        cfg=read(pool/'CONFIG_DRAFT.json');assert cfg['runtime_allowed'] is False and cfg['training_allowed'] is False
        source_lock=read(pool/'SOURCE_LOCK.json')
        for p,h in source_lock.items():assert sha(Path(p))==h,p
        for p in (pool/'CONFIG_DRAFT.json',pool/'SOURCE_LOCK.json'):lock[str(p)]=sha(p)
        pool_entries.append((pool,list(enumerate(cfg['candidates']))))
    batches,ledger=choose(pool_entries,prior)
    assert len(batches)==12,'LESS_THAN_TWELVE_COMPLETE_THREE_HUB_BATCHES'
    changes=[]
    for batch in batches:
        assert not (BE/batch['id']).exists(),'EXISTING_BATCH_MUST_NOT_BE_REUSED'
        source=Path(batch['source_snapshot']);draft=read(source/'CONFIG_DRAFT.json');rows=[]
        for index in batch['source_indices']:
            row,revision=normalize(draft['candidates'][index],v);rows.append(row)
            changes.append({'batch_id':batch['id'],'candidate_id':row['candidate_id'],'changes':revision['changes']})
        snapshot=out/batch['id'];snapshot.mkdir()
        save(snapshot/'CONFIG_DRAFT.json',dict(draft,candidates=rows,node='Q35N_AUTO_GENERATION_V1_'+batch['id'].upper()))
        batch['prepared_snapshot']=str(snapshot)
    save(out/'EXPOSURE.json',{'history':history,'all_failures_preserved':True,'no_retry':True,'scientific_pass':False})
    save(out/'SELECTION_LEDGER.json',ledger);save(out/'LANGUAGE_CHANGES.json',changes)
    queue={'schema_version':'q35n.auto_generation_queue.v1','gpu_devices':[2],'holder_operations_allowed':False,
        'batches':batches,'batch_count':len(batches),'candidate_count':sum(len(b['candidate_ids']) for b in batches),
        'queue_wall_seconds':43200,'per_job_max_seconds':5400,'transport_upper_seconds':4200,'audit_seconds':1200,'per_batch_disk_bytes':7*1024**3,
        'supervisor_seconds':3900,'factory_seconds':3600,'per_batch_actions':60000,
        'stop_lane_on_transport_failure':True,'auto_retry':False,'same_hub_programs_independent':False,
        'candidate_pool_count':len(ledger),'candidate_is_accepted_family':False,'runtime_allowed':False,'scientific_pass':False}
    save(out/'QUEUE.json',queue)
    for p in [*HERE.glob('*.py'),HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',CORE,VERBALIZER,
              out/'QUEUE.json',out/'EXPOSURE.json',out/'SELECTION_LEDGER.json',out/'LANGUAGE_CHANGES.json']:
        lock[str(p)]=sha(p)
    save(out/'SOURCE_LOCK.json',lock)
    for batch in batches:
        source=Path(batch['source_snapshot']);snapshot=Path(batch['prepared_snapshot'])
        merged=read(source/'SOURCE_LOCK.json');merged.update(lock)
        for p in (out/'SOURCE_LOCK.json',snapshot/'CONFIG_DRAFT.json'):merged[str(p)]=sha(p)
        save(snapshot/'SOURCE_LOCK.json',merged)
    print(json.dumps({'batches':len(batches),'candidates':36,'candidate_pool':len(ledger),
        'selection_statuses':dict(collections.Counter(x['status'] for x in ledger)),
        'max_snapshot_lock':max(len(read(Path(b['prepared_snapshot'])/'SOURCE_LOCK.json')) for b in batches),
        'gpu_operations':0},indent=2))
if __name__=='__main__':freeze()
