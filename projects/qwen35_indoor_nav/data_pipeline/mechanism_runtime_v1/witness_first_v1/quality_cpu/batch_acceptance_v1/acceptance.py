"""Read-only runtime evidence extraction and versioned preliminary batch gate.

Writes only new review artifacts beneath this module. No GPU/process operations.
Post-run file seals establish integrity, not a fictitious pre-action timestamp;
pre-action configuration provenance is separately established from INPUT_LOCK,
PROCESS, locked source and the verified journal's genesis/phase ordering.
"""
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
QUALITY_ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('sealed_family_quality',QUALITY_ROOT/'quality.py')
quality=importlib.util.module_from_spec(spec)
expected={x.split(None,1)[1]:x.split(None,1)[0] for x in (QUALITY_ROOT/'SHA256SUMS').read_text().splitlines()}
if hashlib.sha256((QUALITY_ROOT/'quality.py').read_bytes()).hexdigest()!=expected['quality.py']:raise ImportError('SEALED_QUALITY_CHANGED')
spec.loader.exec_module(quality)
LINE=quality.LINE
PROJECT=LINE.parents[1]

def require(ok,reason):
    if not ok:raise ValueError(reason)

def path_safe(path):
    p=Path(path).absolute();require(p==p.resolve() and p.is_relative_to(PROJECT),'PATH_SCOPE_OR_SYMLINK');return p

def sha(path):
    path=path_safe(path);h=hashlib.sha256()
    require(path.stat().st_size<=8*1024**3,'FILE_CAP')
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(path_safe(path).read_text())
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def save(path,value):
    path=path_safe(path);require(path.is_relative_to(HERE),'OUTPUT_SCOPE')
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)

def parse_journal(raw,head,config):
    records=[];previous='0'*64
    for index,line in enumerate(raw.splitlines()):
        row=json.loads(line);require(canonical(row)==line,'JOURNAL_CANONICAL')
        body={k:v for k,v in row.items() if k!='hash'}
        require(row['seq']==index and row['prev']==previous,'JOURNAL_ORDER')
        require(hashlib.sha256(canonical(body)).hexdigest()==row['hash'],'JOURNAL_HASH')
        previous=row['hash'];records.append(row)
    require(bool(records) and raw.endswith(b'\n'),'EMPTY_OR_PARTIAL_JOURNAL')
    require(head['count']==len(records) and head['last_hash']==previous and head['byte_length']==len(raw),'HEAD_MISMATCH')
    require(records[0]['kind']=='__config__' and records[0]['payload']==config,'GENESIS_CONFIG')
    require(head['config_hash']==hashlib.sha256(canonical(config)).hexdigest(),'GENESIS_CONFIG_HASH')
    return records

def phase_evidence(records,bundle_id):
    active=None;frozen=None;frozen_seq=None;cert_started=None;trace_records=[];first_action=None;actions=0
    for row in records:
        kind,payload=row['kind'],row['payload']
        if kind=='budget':
            active=payload['active']
            if active==[bundle_id,'certification'] and cert_started is None:cert_started=row['seq']
        elif kind=='freeze' and bundle_id in payload:
            state=payload[bundle_id]
            if state.get('candidate') is not None and frozen is None:
                frozen=state['candidate'];frozen_seq=row['seq']
        elif kind=='action_completed' and payload.get('bundle')==bundle_id:
            first_action=row['seq'] if first_action is None else first_action;actions+=1
        elif kind=='trace_saved' and payload.get('bundle')==bundle_id and active==[bundle_id,'certification']:
            trace_records.append(payload)
    return {'frozen_candidate':frozen,'frozen_seq':frozen_seq,'certification_started_seq':cert_started,
            'certification_trace_records':trace_records,'first_confirmed_action_seq':first_action,
            'confirmed_actions':actions}

def gpu_snapshot(snapshot,worker,uuid):
    require(snapshot['uuid']==uuid,'GPU_UUID')
    procs={int(p):v for p,v in snapshot['processes'].items()}
    require(all(isinstance(v,dict) and isinstance(v.get('mib'),int) and v['mib']>=0 and v.get('type') in ('C','G','C+G') for v in procs.values()),'GRAPHICS_INVENTORY_SCHEMA')
    external=[v['mib'] for pid,v in procs.items() if pid!=worker]
    require(all(v<=768 for v in external) and sum(external)<=2048,'EXTERNAL_MEMORY_LIMIT')
    upper=snapshot['memory_mib']-sum(external)
    require(0<=upper<4096 and sum(v['mib'] for v in procs.values())<=snapshot['memory_mib'],'OWN_MEMORY_OR_ACCOUNTING')
    return {'own_memory_upper_mib':upper,'own_renderer_absent':worker not in procs}

def inspect_run(run_root):
    run=path_safe(run_root);require(run.is_relative_to(LINE),'RUN_OUTSIDE_LINE')
    cfg=read(run/'EXECUTION_CONFIG.json');lock=read(run/'INPUT_LOCK.json');proc=read(run/'PROCESS.json')
    final=read(run/'SUPERVISOR_RESULT.json');result=read(run/'result.json');budget=read(run/'BUDGET_FINAL.json');store=read(run/'STORE_CLOSE_AUDIT.json')
    require(lock.get(str(run/'EXECUTION_CONFIG.json'))==sha(run/'EXECUTION_CONFIG.json'),'CFG_NOT_IN_INPUT_LOCK')
    require(len(lock)<=1024 and sum(path_safe(p).stat().st_size for p in lock)<=32*1024**3,'SOURCE_READ_CAP')
    for p,value in lock.items():require(sha(p)==value,'LOCKED_SOURCE_CHANGED:'+p)
    require((run/'INPUT_LOCK.json').stat().st_mtime<=proc['started_unix'] and (run/'EXECUTION_CONFIG.json').stat().st_mtime<=proc['started_unix'],'LOCAL_PRESTART_FILE_ORDER')
    raw=(run/'journal/events.jsonl').read_bytes();records=parse_journal(raw,read(run/'journal/HEAD.json'),cfg)
    samples=[json.loads(x) for x in (run/'RESOURCE_SAMPLES.jsonl').read_text().splitlines() if x]
    require(bool(samples),'RESOURCE_SAMPLES_MISSING')
    for sample in samples:
        gpu_snapshot(sample,proc['pid'],cfg['gpu_uuid'])
        require(sample['elapsed']<cfg['supervision_wall_seconds'],'WALL_RESOURCE_LIMIT')
        if 'disk_bytes' in sample:require(sample['disk_bytes']<7*1024**3,'DISK_RESOURCE_LIMIT')
    final_gpu=gpu_snapshot(final['gpu_after'],proc['pid'],cfg['gpu_uuid'])
    require(final.get('cleanup_complete') is True and final_gpu['own_renderer_absent'],'GPU_CLEANUP_NOT_PROVEN')
    require(final.get('error') is None and final.get('external_processes_stopped')==0 and final.get('holders_touched') is False,'SUPERVISOR_RESOURCE_ERROR')
    require(store.get('audit_pass') is True and store.get('poisoned') is False,'STORE_FAILURE')
    require(budget.get('active') is None,'BUDGET_NOT_CLOSED')
    quality.bridge.BudgetLedger(budget['limits'],state=budget,clock=lambda:budget['last_clock'])
    split_paths=[Path(p) for p,value in lock.items() if value==cfg['split_sha256']]
    require(len(split_paths)==1,'SPLIT_SOURCE_IDENTITY')
    split=read(split_paths[0])
    require(all(row['house_id'] in split['FIT'] for row in cfg['candidates']),'NON_FIT_CANDIDATE')
    require(len({r['candidate_id'] for r in cfg['candidates']})==len(cfg['candidates']),'DUPLICATE_FROZEN_ID')
    evidence={'run_root':str(run),'config_sha256':sha(run/'EXECUTION_CONFIG.json'),'input_lock_sha256':sha(run/'INPUT_LOCK.json'),
              'journal_sha256':sha(run/'journal/events.jsonl'),'preaction_configuration_proof':{
                  'genesis_seq':0,'genesis_equals_locked_config':True,'source_hashes_verified':True,
                  'process_started_unix':proc['started_unix'],'input_lock_mtime_unix':(run/'INPUT_LOCK.json').stat().st_mtime,
                  'scope':'trusted_local_clock_and_locked_worker_journal_order_not_external_timestamp_authority'},
              'gpu_final':dict(final_gpu,gpu_after=final['gpu_after']),
              'source_worker_result':result,'source_supervisor_result':final,'budget':budget,'store':store}
    evidence['source_files']=dict(lock)
    for filename in ('EXECUTION_CONFIG.json','INPUT_LOCK.json','PROCESS.json','SUPERVISOR_RESULT.json','result.json',
                     'BUDGET_FINAL.json','STORE_CLOSE_AUDIT.json','RESOURCE_SAMPLES.jsonl','GPU_BEFORE.json',
                     'journal/events.jsonl','journal/HEAD.json'):
        evidence['source_files'][str(run/filename)]=sha(run/filename)
    return run,cfg,records,evidence

def build_family_evidence(run_root,bundle_id,output_dir,postprocess_result=None):
    require(isinstance(bundle_id,str) and Path(bundle_id).name==bundle_id and bundle_id not in ('.','..'),'BUNDLE_ID')
    out=path_safe(output_dir);require(out.is_relative_to(HERE) and out!=HERE,'OUTPUT_SCOPE');out.mkdir(parents=True,exist_ok=False)
    response={'quality_pass':False,'evidence_verified':False,'grade':'MISSING_OR_REJECTED_EVIDENCE','errors':[],'scientific_pass':False}
    try:
        run,cfg,records,source=inspect_run(run_root)
        row=next(r for r in cfg['candidates'] if r['candidate_id']==bundle_id)
        response.update(house_id=row['house_id'],hub_position=row['configuration']['u_position'],control_type=cfg.get('control_type'))
        phases=phase_evidence(records,bundle_id)
        save(out/'SOURCE_AUDIT.json',dict(source,phase_evidence=phases))
        folder=run/'bundles'/bundle_id
        for name in ('FROZEN_CANDIDATE.json','CERTIFICATE.json','READBACK.json','result.json','export_v4/MANIFEST.json'):
            require((folder/name).is_file(),'MISSING_FAMILY_EVIDENCE:'+name)
        candidate=read(folder/'FROZEN_CANDIDATE.json')
        require(candidate==phases['frozen_candidate'],'FROZEN_CANDIDATE_NOT_JOURNAL_BOUND')
        require(phases['frozen_seq']<phases['certification_started_seq'],'CANDIDATE_NOT_FROZEN_BEFORE_CERTIFICATION')
        traces=[]
        require(len(phases['certification_trace_records'])==27,'CERTIFICATION_NOT_27_COMPLETE_TRACES')
        for item in phases['certification_trace_records']:
            path=folder/'traces'/('%06d.json'%item['index'])
            require(item['complete'] is True and sha(path)==item['sha256'],'CERTIFICATION_TRACE_BINDING')
            traces.append(str(path))
        resource=dict(source['gpu_final'],external_processes_signalled=0,original_supervisor_path=str(run/'SUPERVISOR_RESULT.json'))
        control={'control_type':cfg.get('control_type'),'house_split':'FIT','source_scope':'actual_physical_fit',
                 'source_config_sha256':source['config_sha256'],'source_journal_sha256':source['journal_sha256'],
                 'language_generation':'controlled_template_only','source_phase_proof':phases}
        # Other control types need explicit frozen ranges; never invent them.
        if 'control_evidence' in row:control.update(row['control_evidence'])
        save(out/'RESOURCE_FINAL.json',resource);save(out/'CONTROL_EVIDENCE.json',control)
        evidence={'candidate':str(folder/'FROZEN_CANDIDATE.json'),'certificate':str(folder/'CERTIFICATE.json'),
                  'readback':str(folder/'READBACK.json'),'result':str(path_safe(postprocess_result)) if postprocess_result else str(folder/'result.json'),
                  'store_close':str(run/'STORE_CLOSE_AUDIT.json'),'budget_final':str(run/'BUDGET_FINAL.json'),
                  'resource_final':str(out/'RESOURCE_FINAL.json'),'control_evidence':str(out/'CONTROL_EVIDENCE.json'),
                  'replay_paths':traces,'house_split':'FIT','source_scope':'actual_physical_fit'}
        export=folder/'export_v4';files={Path(v) for k,v in evidence.items() if k not in ('replay_paths','house_split','source_scope')}
        files.update(map(Path,traces));files.add(export/'SHA256SUMS')
        for line in (export/'SHA256SUMS').read_text().splitlines():files.add(export/line.split('  ',1)[1])
        contents=read(export/'CONTENT_INDEX.json')
        files.update(LINE/item['line_relative_path'] for item in contents.values())
        evidence['sealed_files']={str(path_safe(p)):sha(p) for p in sorted(files)}
        evidence['sealed_files'].update(source['source_files'])
        evidence['sealed_files'][str(out/'SOURCE_AUDIT.json')]=sha(out/'SOURCE_AUDIT.json')
        evidence['postrun_seal_created_unix']=time.time()
        evidence['preaction_proof_path']=str(out/'SOURCE_AUDIT.json')
        save(out/'EVIDENCE.json',evidence)
        response=quality.audit_family(export,evidence)
        response.update(source_and_phase_binding_verified=True,preaction_configuration_verified=True,
                        attempt_id=bundle_id,run_root=str(run))
    except (ValueError,OSError,KeyError,TypeError,StopIteration) as error:
        response['errors'].append(f'{type(error).__name__}: {error}')
    save(out/'FAMILY_REPORT.json',response)
    return response

def independent_hubs(rows):
    """Connected components for same-house distances <1 m; exactly 1 m distinct."""
    groups=[]
    for row in rows:
        p=row['hub_position'];require(len(p)==3 and all(type(x) in (int,float) and math.isfinite(x) for x in p),'INVALID_HUB')
        hits=[g for g in groups if any(x['house_id']==row['house_id'] and math.dist(x['hub_position'],p)<1. for x in g)]
        merged=[row]
        for g in hits:merged.extend(g);groups.remove(g)
        groups.append(merged)
    return groups

def gate_from_verified_batches(batches):
    errors=[];qualified=[];types=Counter()
    if len(batches)!=2:errors.append('NEED_TWO_FROZEN_BATCHES')
    for batch in batches:
        if batch.get('bindings_verified') is not True:errors.append('UNVERIFIED_BATCH_BINDINGS')
        attempted=batch['attempts'];passed=[r for r in attempted if r.get('quality_pass') is True and r.get('source_and_phase_binding_verified') is True]
        if len(independent_hubs(attempted))<3:errors.append('BATCH_NEEDS_THREE_ATTEMPTED_DISTINCT_HUBS')
        if len(independent_hubs(passed))<2:errors.append('BATCH_NEEDS_TWO_QUALIFIED_DISTINCT_HUBS')
        qualified.extend(passed);types.update(r['control_type'] for r in passed)
    hubs=independent_hubs(qualified);houses={r['house_id'] for r in qualified}
    if len(hubs)<6:errors.append('NEED_SIX_QUALIFIED_PHYSICAL_HUBS')
    if len(houses)<3:errors.append('NEED_THREE_FIT_HOUSES')
    return {'preliminary_cross_house_production_pass':not errors,'scope':'initial_cross_house_repeatable_production_only',
            'distinct_qualified_hubs':len(hubs),'fit_houses':len(houses),'control_type_counts':dict(types),'errors':errors,
            'statistical_stability_pass':False,'model_generalization_pass':False,'scientific_pass':False}

def audit_batches(run_roots,output_dir):
    """Production-run audit wrapper; derives reports itself, never trusts supplied PASS."""
    out=path_safe(output_dir);require(out.is_relative_to(HERE) and out!=HERE,'OUTPUT_SCOPE');out.mkdir(parents=True,exist_ok=False)
    require(len(run_roots)==2 and len(set(map(str,run_roots)))==2,'TWO_DISTINCT_RUNS_REQUIRED')
    batches=[]
    for index,run_root in enumerate(run_roots):
        run,cfg,records,source=inspect_run(run_root);attempts=[]
        frozen={r['candidate_id'] for r in cfg['candidates']}
        actual=set(source['budget']['bundles'])
        require(frozen==actual,'FROZEN_ATTEMPT_SET_NOT_COMPLETE')
        require({r['candidate_id'] for r in source['source_worker_result']['bundles']}==frozen,'RUN_RESULTS_OMIT_ATTEMPTS')
        for row in cfg['candidates']:
            report=build_family_evidence(run,row['candidate_id'],out/(str(index)+'_'+row['candidate_id']))
            report.setdefault('house_id',row['house_id']);report.setdefault('hub_position',row['configuration']['u_position'])
            report.setdefault('control_type',cfg.get('control_type'));attempts.append(report)
        batches.append({'bindings_verified':True,'attempts':attempts,'source':source})
    result=gate_from_verified_batches(batches);save(out/'BATCH_REPORT.json',dict(result,batches=batches));return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',required=True);parser.add_argument('--bundle',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(build_family_evidence(args.run_root,args.bundle,args.output),indent=2))
