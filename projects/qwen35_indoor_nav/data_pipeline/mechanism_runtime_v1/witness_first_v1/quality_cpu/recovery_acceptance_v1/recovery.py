"""Read-only, separately graded recovery of fully certified physical families.

Never writes the source run, resumes its Journal, instantiates its ContentStore,
or fabricates original runtime terminal records. Training admission is main-only.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import time

HERE=Path(__file__).resolve().parent
LINE=HERE.parents[4]
PROJECT=LINE.parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

sem=load('recovery_semantics_v1',HERE/'semantics.py')
acceptance_path=HERE.parent/'batch_acceptance_v1/acceptance.py'
if hashlib.sha256(acceptance_path.read_bytes()).hexdigest()!='a2cae535b132b652c289d753bcf11dc2f55413bb942fd731528198a8ab4fa7d8':
    raise ValueError('SEALED_ACCEPTANCE_SOURCE_CHANGED')
old=load('recovery_original_acceptance',acceptance_path)
require=old.require

def safe(path):
    path=Path(path).absolute()
    require(path==path.resolve() and path.is_relative_to(PROJECT),'SOURCE_PATH_SCOPE_OR_SYMLINK')
    return path

def stamp(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)

class Reader:
    def __init__(self):self.seals={};self.bytes_read=0
    def read(self,path,retain=True):
        path=safe(path);before=path.lstat()
        require(stat.S_ISREG(before.st_mode),'SOURCE_NOT_REGULAR')
        require(before.st_size<= (256*1024**2 if retain else 8*1024**3),'FILE_READ_CAP')
        digest=hashlib.sha256();chunks=[]
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        with os.fdopen(fd,'rb') as stream:
            require(stamp(os.fstat(stream.fileno()))==stamp(before),'OPENED_SOURCE_CHANGED')
            for block in iter(lambda:stream.read(1024**2),b''):
                digest.update(block);self.bytes_read+=len(block)
                require(self.bytes_read<=64*1024**3,'TOTAL_READ_CAP')
                if retain:chunks.append(block)
            require(stamp(os.fstat(stream.fileno()))==stamp(before),'SOURCE_CHANGED_DURING_READ')
        require(stamp(path.lstat())==stamp(before),'SOURCE_CHANGED_AFTER_READ')
        value=digest.hexdigest();previous=self.seals.setdefault(str(path),value)
        require(previous==value,'SOURCE_CHANGED_BETWEEN_READS')
        return b''.join(chunks) if retain else value
    def json(self,path):return json.loads(self.read(path))
    def verify_again(self):
        for path in list(self.seals):self.read(path,False)

def committed_prefix(raw,head,cfg):
    n=head['byte_length'];require(type(n) is int and 0<n<=len(raw),'HEAD_BYTE_RANGE')
    records=old.parse_journal(raw[:n],head,cfg)
    return records,{'committed_bytes':n,'committed_records':len(records),
        'uncommitted_tail_bytes':len(raw)-n,'uncommitted_tail_sha256':hashlib.sha256(raw[n:]).hexdigest(),
        'tail_used_for_labels_or_budget':False}

def family_completion(records,ident,candidate):
    phase=old.phase_evidence(records,ident)
    require(phase['frozen_candidate']==candidate,'FROZEN_CANDIDATE_NOT_COMMITTED')
    require(phase['frozen_seq']<phase['certification_started_seq'],'FREEZE_AFTER_CERTIFICATION')
    require(len(phase['certification_trace_records'])==27,'COMPLETE_CERTIFICATION_REQUIRED')
    certified=None;closed=None
    for row in records:
        if row['kind']=='freeze' and row['payload'].get(ident,{}).get('status')=='certified':
            certified=row['seq'] if certified is None else certified
        if row['kind']=='budget' and certified is not None and row['seq']>certified:
            budget=row['payload'];entry=budget['bundles'].get(ident,{})
            if budget['active'] is None and entry.get('certification',{}).get('finished') is not None:
                closed=row;break
    require(closed is not None,'NO_COMMITTED_FAMILY_COMPLETION_BUDGET')
    budget=closed['payload'];sem.bridge.BudgetLedger(budget['limits'],state=budget,clock=lambda:budget['last_clock'])
    counts={'discovery':0,'certification':0};active=None;previous_total=0
    for row in records[:closed['seq']+1]:
        if row['kind']=='budget':
            state=row['payload'];active=state['active'];total=state['total_reserved_actions']
            require(total>=previous_total and total-previous_total<=1,'BUDGET_REFUND_OR_NONUNIT_RESERVATION')
            previous_total=total
            sem.bridge.BudgetLedger(state['limits'],state=state,clock=lambda:state['last_clock'])
        elif row['kind']=='action_completed' and row['payload'].get('bundle')==ident:
            require(active is not None and active[0]==ident,'ACTION_OUTSIDE_ACTIVE_FAMILY')
            name=active[1];counts[name]+=1
            require(state['bundles'][ident][name]['actions']==counts[name],'ACTION_WITHOUT_PRIOR_DURABLE_RESERVATION')
    for name in counts:
        entry=budget['bundles'][ident][name]
        require(entry['finished'] is not None and entry['actions']==counts[name],'INCOMPLETE_OR_UNCONFIRMED_FAMILY_ACTION')
        require(entry['actions']<=budget['limits'][name+'_actions'],'FAMILY_ACTION_BUDGET')
        require(entry['finished']-entry['started']<=budget['limits'][name+'_seconds'],'FAMILY_WALL_BUDGET')
    require(budget['last_clock']-budget['created']<=budget['limits']['total_seconds'],'TOTAL_WALL_AT_COMPLETION')
    return phase,{'certified_seq':certified,'closed_budget_seq':closed['seq'],
        'family_budget':budget['bundles'][ident],'confirmed_actions_by_phase':counts,
        'budget_snapshot_at_family_completion':budget,'all_confirmed_actions_durably_reserved_first':True}

def scan_store(root,reader,limit=6*1024**3):
    root=safe(root);before=root.stat();entries=sorted(root.iterdir());total=0;files=[];problems=[]
    require(len(entries)<=100000,'STORE_FILE_COUNT_CAP')
    for path in entries:
        info=path.lstat();total+=info.st_size
        if not stat.S_ISREG(info.st_mode):problems.append({'name':path.name,'error':'NONREGULAR_ENTRY'});continue
        raw=reader.read(path)
        item={'name':path.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};files.append(item)
        match=re.fullmatch(r'([0-9a-f]{64})\.(rgb|semantic)\.npy',path.name)
        if not match:problems.append({'name':path.name,'error':'FAILURE_PARTIAL_OR_FOREIGN_ENTRY'});continue
        try:sem.loader_module.npy_pixels(raw,match[1],match[2])
        except (ValueError,KeyError,TypeError) as e:problems.append({'name':path.name,'error':str(e)})
    require(stamp(root.stat())==stamp(before) and [p.name for p in sorted(root.iterdir())]==[p.name for p in entries],'STORE_CHANGED_DURING_SCAN')
    return {'type':'READ_ONLY_POST_TERMINATION_CONTENT_RESEAL_V1','content_inventory_pass':not problems and total<=limit,
        'file_count':len(entries),'actual_bytes_all_entries':total,'limit_bytes':limit,'files':files,'problems':problems,
        'original_store_close_exists':(root.parent/'STORE_CLOSE_AUDIT.json').exists(),'original_poisoned_state_known':False,
        'not_original_store_close':True,'not_original_accounting_or_fsync_ack_proof':True}

def sampled_resources(run,cfg,proc,final,folder,reader):
    require(final['cleanup_complete'] is True and str(proc['pid']) not in final['gpu_after']['processes'],'OWN_RENDERER_NOT_TERMINATED')
    require(final['gpu_after']['uuid']==cfg['gpu_uuid'],'FINAL_GPU_IDENTITY')
    require(final['external_processes_stopped']==0 and final['holders_touched'] is False,'UNAUTHORIZED_EXTERNAL_MUTATION')
    samples=[json.loads(line) for line in reader.read(run/'RESOURCE_SAMPLES.jsonl').splitlines()]
    require(samples and all(samples[i]['elapsed']<samples[i+1]['elapsed'] for i in range(len(samples)-1)),'RESOURCE_SAMPLES_ORDER')
    for sample in samples:
        old.gpu_snapshot(sample,proc['pid'],cfg['gpu_uuid'])
        require(sample['elapsed']<cfg['supervision_wall_seconds'],'SAMPLED_WALL_LIMIT')
        if 'disk_bytes' in sample:require(sample['disk_bytes']<7*1024**3,'SAMPLED_DISK_LIMIT')
    done=(folder/'result.json').stat().st_mtime;start=proc['started_unix'];end=(run/'SUPERVISOR_RESULT.json').stat().st_mtime
    require(start<done<end,'FAMILY_COMPLETION_NOT_BEFORE_TERMINATION')
    # This is deliberately *not* an exact monotonic-to-wall clock conversion.
    return {'all_recorded_resource_samples_pass':True,'samples':len(samples),'family_result_local_mtime':done,
        'process_started_unix':start,'supervisor_result_local_mtime':end,
        'maximum_observed_sample_gap_seconds':max((b['elapsed']-a['elapsed'] for a,b in zip(samples,samples[1:])),default=0),
        'original_resource_error':final['error'],'failed_trigger_snapshot_available':False,
        'continuous_resource_compliance_proven':False,'precise_family_sample_time_alignment_proven':False,
        'scope':'all_persisted_samples_and_trusted_local_file_order_only_not_original_supervisor_pass',
        'own_renderer_absent_at_original_terminal':True}

def audit_recovery(run_root,ident,output):
    run=safe(run_root);out=safe(output)
    require(out.is_relative_to(HERE) and out!=HERE,'OUTPUT_SCOPE');out.mkdir(parents=True,exist_ok=False)
    reader=Reader();report={'recovery_content_pass':False,'quality_pass':False,'original_batch_pass':False,
        'training_admission':False,'requires_main_training_admission':True,'scientific_pass':False,'model_gain_pass':False,
        'grade':'RECOVERY_EVIDENCE_INSUFFICIENT','errors':[],'run_root':str(run),'candidate_id':ident}
    report['auditor_sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'recovery.py',HERE/'semantics.py',acceptance_path,sem.SOURCE)}
    try:
        require(Path(ident).name==ident and ident not in ('.','..'),'BUNDLE_ID')
        cfg=reader.json(run/'EXECUTION_CONFIG.json');lock=reader.json(run/'INPUT_LOCK.json');proc=reader.json(run/'PROCESS.json');final=reader.json(run/'SUPERVISOR_RESULT.json')
        require(final.get('error') is not None and final['returncode']!=0,'RECOVERY_REQUIRES_CENSORED_SOURCE')
        require(lock.get(str(run/'EXECUTION_CONFIG.json'))==reader.seals[str(run/'EXECUTION_CONFIG.json')],'CONFIG_NOT_LOCKED')
        require(len(lock)<=1024 and sum(safe(p).stat().st_size for p in lock)<=32*1024**3,'SOURCE_LOCK_CAP')
        for path,expected in lock.items():require(reader.read(path,False)==expected,'LOCKED_SOURCE_CHANGED:'+path)
        require(max((run/n).stat().st_mtime for n in ('EXECUTION_CONFIG.json','INPUT_LOCK.json'))<=proc['started_unix'],'CONFIG_NOT_PREACTION')
        rows=[r for r in cfg['candidates'] if r['candidate_id']==ident];require(len(rows)==1,'FROZEN_CANDIDATE_ID');row=rows[0]
        splits=[p for p,h in lock.items() if h==cfg['split_sha256']];require(len(splits)==1,'SPLIT_IDENTITY')
        require(row['house_id'] in reader.json(splits[0])['FIT'],'NON_FIT_HOUSE')
        folder=run/'bundles'/ident;candidate=reader.json(folder/'FROZEN_CANDIDATE.json');cert=reader.json(folder/'CERTIFICATE.json');readback=reader.json(folder/'READBACK.json');result=reader.json(folder/'result.json')
        require(result.get('physical_certified') is True and result.get('error') is None,'BUNDLE_NOT_COMPLETED')
        require(result['candidate_id']==ident and result['house_id']==row['house_id'],'BUNDLE_RESULT_IDENTITY')
        records,prefix=committed_prefix(reader.read(run/'journal/events.jsonl'),reader.json(run/'journal/HEAD.json'),cfg)
        require(all(r['payload']['limits']==cfg['budget'] for r in records if r['kind']=='budget'),'BUDGET_LIMITS_DIFFER_FROM_FROZEN_CONFIG')
        phase,completion=family_completion(records,ident,candidate)
        require(candidate['context']['house_id']==row['house_id'] and candidate['context']['asset_config']==row['assets'],'FAMILY_SOURCE_CONTEXT')
        manifest=reader.json(folder/'export_v4/MANIFEST.json');cc=manifest['compiler_config']
        require(manifest['family_id']==ident and cc['tasks']==row['tasks'] and cc['eligible']==row['expected_eligible'],'EXPORT_FROZEN_TASK_IDENTITY')
        require(cc['roles']=={k:[v['mpcat40'],v['room']] for k,v in row['roles'].items()},'EXPORT_FROZEN_ROLE_IDENTITY')
        traces=[]
        for entry in phase['certification_trace_records']:
            p=folder/'traces'/('%06d.json'%entry['index']);require(reader.read(p,False)==entry['sha256'] and entry['complete'] is True,'COMMITTED_TRACE_FILE_CHANGED');traces.append(str(p))
        semantics=sem.audit_semantics(folder/'export_v4',candidate,cert,readback,traces,cfg['control_type'],reader.read)
        semantics.pop('read_bytes_counted',None)
        report.update(semantics=semantics,committed_prefix=prefix,family_completion=completion)
        inventory=scan_store(run/'content',reader)
        save(out/'RECOVERY_CONTENT_RESEAL.json',inventory)
        save(out/'FAMILY_COMPLETION_BUDGET.json',completion)
        resources=sampled_resources(run,cfg,proc,final,folder,reader)
        save(out/'RESOURCE_EVIDENCE.json',resources)
        report['resource_evidence']=resources
        require(inventory['content_inventory_pass'],'READ_ONLY_STORE_RESEAL_FAILED')
        reader.verify_again()
        report.update(recovery_content_pass=True,grade='RECOVERED_FAMILY_CONTENT_VERIFIED_RUNTIME_CENSORED',
            semantics=semantics,committed_prefix=prefix,family_completion=completion,resource_evidence=resources,
            source_and_phase_binding_verified=True,preaction_configuration_verified=True,
            required_main_decision='allow content-verified recovered FIT family into training with recovery provenance; no original batch or continuous resource PASS',
            known_limits=['original process did not emit STORE_CLOSE/BUDGET_FINAL/result',
                'saved GPU sampling is not continuous compliance; failed trigger snapshot absent',
                'language remains controlled templates; no model improvement established'])
    except (KeyError,ValueError,OSError,TypeError,StopIteration,AssertionError) as e:report['errors'].append(type(e).__name__+': '+str(e))
    report['read_bytes_counted']=reader.bytes_read
    save(out/'RECOVERY_SOURCE_SEALS.json',reader.seals);save(out/'REPORT.json',report)
    return report

def save(path,value):
    path=safe(path);require(path.is_relative_to(HERE),'OUTPUT_SCOPE')
    with path.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(audit_recovery(a.run,a.bundle,a.output),indent=2))
