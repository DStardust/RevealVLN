"""Finite wording-only revision and prospective four-batch cohort, CPU only."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
WF=HERE.parents[1]
ROOT=next(p for p in HERE.parents if p.name=='vla')
SOURCE=HERE.parent/'handoff_v1'
LANGUAGE=WF/'language_realization_v1'
GATE=WF/'quality_cpu/batch_acceptance_v1/gate_v2'
BE=WF/'batch_execution_v1'
ASSIGNMENTS=[('batch_03',1,[0,1,2]),('batch_04',5,[3,4,5]),('batch_05',1,[6,7,8]),('batch_06',5,[9,10,11])]

def sha(path):
    path=Path(path);assert path.resolve()==path and path.is_relative_to(ROOT)
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    return h.hexdigest()
def read(path):return json.loads(path.read_text())
def save(path,value):
    assert path.is_relative_to(HERE)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def load_language():
    entries={r.split(None,1)[1]:r.split(None,1)[0] for r in (LANGUAGE/'SHA256SUMS').read_text().splitlines()}
    assert sha(LANGUAGE/'verbalizer.py')==entries['verbalizer.py']
    s=importlib.util.spec_from_file_location('frozen_finite_verbalizer',LANGUAGE/'verbalizer.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def invariant(row):
    value=copy.deepcopy(row)
    for task in value['tasks'].values():task.pop('instruction')
    value['component_provenance'].pop('language_revision',None)
    return value
def revise(source,verbalizer):
    result=copy.deepcopy(source);changes=[]
    for old,row in zip(source['candidates'],result['candidates']):
        revision=verbalizer.propose_revision(old)
        row['tasks']=revision['proposed_tasks']
        row['component_provenance']['language_revision']=revision
        assert invariant(old)==invariant(row),'STRUCTURE_OR_ACTION_CHANGED'
        changes.append(revision)
    return result,changes
def cohort(cfg,config_hash):
    members=[]
    assert len(cfg['candidates'])==12
    assert len({r['canonical_program_id'] for r in cfg['candidates']})==12
    for name,gpu,indices in ASSIGNMENTS:
        rows=[cfg['candidates'][i] for i in indices]
        for i,a in enumerate(rows):
            for b in rows[i+1:]:assert a['house_id']!=b['house_id'] or math.dist(a['configuration']['u_position'],b['configuration']['u_position'])>=1
        members.append({'batch_id':name,'gpu_device':gpu,'source_indices':indices,
            'run_root':str(BE/name/'run_v1'),'candidate_ids':[r['candidate_id'] for r in rows],
            'canonical_program_ids':[r['canonical_program_id'] for r in rows],
            'hubs':[{'house_id':r['house_id'],'hub_position':r['configuration']['u_position']} for r in rows]})
    return {'schema_version':'q35n.production_cohort.v2','criteria_version':'AT_LEAST_TWO_BATCHES_V2',
        'cohort_id':'Q35N_MULTI_PROGRAM_LANGUAGE_READY_FOUR_BATCHES_V1',
        'mechanism_version':'winding_v1__multi_program_bank_v1__finite_language_realization_v1',
        'transport_version':'GPU1_readiness_v1__GPU5_versioned_exact_identity_transport_with_per_batch_source_lock',
        'declared_run_roots':[m['run_root'] for m in members],'members':members,
        'source_configuration':str(HERE/'CONFIG_DRAFT.json'),'source_configuration_sha256':config_hash,
        'predeclared_unix':time.time(),'all_members_remain_in_verdict_including_failures':True,
        'prior_failed_batches_retained_in_exposure_not_retroactively_in_this_cohort':True,
        'planned_distinct_hubs':6,'planned_distinct_FIT_houses':6,'planned_programs':12,
        'same_hub_programs_statistically_independent':False,'original_quality_thresholds_unchanged':True,
        'runtime_allowed':False,'training_allowed':False,'scientific_pass':False}

def main():
    assert not (HERE/'SOURCE_LOCK.json').exists(),'NO_OVERWRITE'
    lock=read(SOURCE/'SOURCE_LOCK.json')
    for path,h in lock.items():assert sha(Path(path))==h,path
    cfg,changes=revise(read(SOURCE/'CONFIG_DRAFT.json'),load_language())
    cfg.update(node='Q35N_MULTI_PROGRAM_LANGUAGE_READY_V1',evaluation_cohort_manifest=str(HERE/'COHORT_V2.json'))
    assert cfg['runtime_allowed'] is False and cfg['executable'] is False and cfg['training_allowed'] is False
    save(HERE/'CONFIG_DRAFT.json',cfg)
    save(HERE/'LANGUAGE_CHANGES.json',{'source':str(SOURCE/'CONFIG_DRAFT.json'),'revisions':changes,
        'changed_instructions':sum(len(x['changes']) for x in changes),'structural_invariants_verified':12,
        'candidate_ids_preserved':True,'scientific_pass':False})
    save(HERE/'COHORT_V2.json',cohort(cfg,sha(HERE/'CONFIG_DRAFT.json')))
    dependencies=[SOURCE/'CONFIG_DRAFT.json',SOURCE/'SOURCE_LOCK.json',LANGUAGE/'verbalizer.py',LANGUAGE/'SHA256SUMS',
        LANGUAGE/'SCHEMA.json',WF.parent.parent/'mechanism_scale_v1/planner.py',GATE/'gate.py',GATE/'SPEC_ZH.md',GATE/'SHA256SUMS',
        BE/'gpu5_transport_v2/transport.py',BE/'gpu5_transport_v2/prepare.py',HERE/'prepare.py',HERE/'test_prepare.py',
        HERE/'CONFIG_DRAFT.json',HERE/'LANGUAGE_CHANGES.json',HERE/'COHORT_V2.json']
    for path in dependencies:lock[str(path)]=sha(path)
    assert len(lock)<=1024,'SOURCE_LOCK_CAP'
    save(HERE/'SOURCE_LOCK.json',lock)
    save(HERE/'result.json',{'status':'LANGUAGE_AND_COHORT_FROZEN_NOT_EXECUTABLE','candidates':12,
        'changed_instructions':sum(len(x['changes']) for x in changes),'cohort_batches':4,'cohort_independent_hubs':6,
        'cohort_FIT_houses':6,'source_lock_entries':len(lock),'source_lock_within_1024':True,
        'remaining_source_lock_slots_before_batch_specific_closure':1024-len(lock),
        'cohort_sha256':sha(HERE/'COHORT_V2.json'),'gpu_operations':0,'scientific_pass':False})
    print(json.dumps(read(HERE/'result.json'),indent=2))

if __name__=='__main__':main()
