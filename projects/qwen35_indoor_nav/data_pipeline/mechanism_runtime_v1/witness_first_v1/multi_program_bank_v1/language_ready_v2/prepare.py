"""New preaction cohort only; every candidate byte-equivalent at JSON level."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'language_ready_v1'
WF=HERE.parents[1]
BE=WF/'batch_execution_v1'
ROOT=next(p for p in HERE.parents if p.name=='vla')
ASSIGNMENTS=[('batch_03r1',1,[0,1,2]),('batch_04r1',5,[3,4,5]),('batch_05r1',1,[6,7,8]),('batch_06r1',5,[9,10,11])]

def sha(path):
    path=Path(path);assert path.resolve()==path and path.is_relative_to(ROOT)
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):
    assert p.parent==HERE
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def derive(source,old_cohort):
    cfg=copy.deepcopy(source)
    cfg['node']='Q35N_MULTI_PROGRAM_LANGUAGE_READY_V2'
    cfg['evaluation_cohort_manifest']=str(HERE/'COHORT_V2.json')
    assert cfg['candidates']==source['candidates']
    cohort=copy.deepcopy(old_cohort)
    cohort['cohort_id']='Q35N_MULTI_PROGRAM_LANGUAGE_READY_FOUR_BATCHES_R1_V2'
    cohort['transport_version']='GPU1_readiness_v1__GPU5_exact_identity_transport_v4_input_file_cap2048'
    for member,(name,gpu,indices) in zip(cohort['members'],ASSIGNMENTS):
        assert member['gpu_device']==gpu and member['source_indices']==indices
        member['batch_id']=name;member['run_root']=str(BE/name/'run_v1')
    cohort['declared_run_roots']=[m['run_root'] for m in cohort['members']]
    cohort['source_configuration']=str(HERE/'CONFIG_DRAFT.json')
    cohort['source_configuration_sha256']=None
    cohort['predeclared_unix']=None
    cohort['superseded_unstarted_cohort']={'path':str(OLD/'COHORT_V2.json'),'reason':'INPUT_LOCK_CLOSURE_COUNT_ONLY','not_current_cohort':True}
    return cfg,cohort
def preaction_check(run):
    allowed={'INPUT_LOCK.json','EXECUTION_CONFIG.json'}
    present=sorted(p.name for p in run.iterdir()) if run.exists() else []
    assert set(present)<=allowed,'OLD_MEMBER_HAS_RUNTIME_OR_UNKNOWN_EVIDENCE:'+str(run)
    return {'run_root':str(run),'observed_run_files':present,'physical_attempts':0,
        'not_launched':True,'basis':'only preparation files or absent run root; main confirms no launches'}
def main():
    assert not (HERE/'SOURCE_LOCK.json').exists(),'NO_OVERWRITE'
    lock=read(OLD/'SOURCE_LOCK.json')
    for path,h in lock.items():assert sha(Path(path))==h,path
    for folder in (OLD,BE/'gpu5_transport_v3'):
        for line in (folder/'SHA256SUMS').read_text().splitlines():
            h,name=line.split(None,1);assert sha(folder/name)==h
    source=read(OLD/'CONFIG_DRAFT.json');prior=read(OLD/'COHORT_V2.json')
    old_members=[preaction_check(Path(p)) for p in prior['declared_run_roots']]
    cfg,cohort=derive(source,prior)
    assert cfg['executable'] is False and cfg['runtime_allowed'] is False and cfg['training_allowed'] is False
    save(HERE/'CONFIG_DRAFT.json',cfg)
    cohort['source_configuration_sha256']=sha(HERE/'CONFIG_DRAFT.json');cohort['predeclared_unix']=time.time()
    cohort['superseded_unstarted_cohort']['sha256']=sha(OLD/'COHORT_V2.json')
    save(HERE/'COHORT_V2.json',cohort)
    save(HERE/'CHANGE_LEDGER.json',{'reason':'INPUT_LOCK_CLOSURE_COUNT_ONLY','source':str(OLD),
        'old_prepared_lock_counts':{'batch_03':1015,'batch_04':1026},
        'all_12_candidates_identical':cfg['candidates']==source['candidates'],
        'instruction_role_action_eligible_balance_changes':0,'old_members':old_members,
        'old_cohort_provenance_only_not_current':True,'quality_and_action_and_time_threshold_changes':0,
        'scientific_pass':False})
    deps=[OLD/'SOURCE_LOCK.json',OLD/'result.json',OLD/'SHA256SUMS',
        BE/'gpu5_transport_v3/transport.py',BE/'gpu5_transport_v3/prepare.py',BE/'gpu5_transport_v3/SHA256SUMS',
        BE/'gpu5_transport_v2/transport.py',BE/'gpu5_transport_v2/prepare.py',
        HERE/'prepare.py',HERE/'test_prepare.py',HERE/'CONFIG_DRAFT.json',HERE/'COHORT_V2.json',HERE/'CHANGE_LEDGER.json']
    for member in old_members:
        deps.extend(Path(member['run_root'])/name for name in member['observed_run_files'])
    for p in deps:lock[str(p)]=sha(p)
    assert len(lock)<=2048
    save(HERE/'SOURCE_LOCK.json',lock)
    result={'status':'PREACTION_COHORT_REVISED_NOT_EXECUTABLE','candidates':12,'candidate_changes':0,
        'source_lock_entries':len(lock),'current_cohort':str(HERE/'COHORT_V2.json'),
        'cohort_sha256':sha(HERE/'COHORT_V2.json'),'old_members_physical_attempts':0,
        'gpu_operations':0,'scientific_pass':False}
    save(HERE/'result.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
