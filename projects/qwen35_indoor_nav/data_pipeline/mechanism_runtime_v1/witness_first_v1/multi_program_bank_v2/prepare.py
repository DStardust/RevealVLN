"""Frozen V1 algorithm transported to the next fully-closed three-house scout."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
WF=HERE.parent
ROOT=next(p for p in HERE.parents if p.name=='vla')
OLD=WF/'multi_program_bank_v1'
SCOUT=WF/'bulk_source_v1/shard_00/run_v1'
HOUSES=['5q7pvUzZiYa','759xd9YjKW5','7y3sRwLe3Va']
ORIGINAL_SHA='0c5a26e23a453e07e8893242bc13e42e7ffc033efa8238ba3929d7075ecdc4b5'
CORE_SHA='21cff82e02315dcc3616fa523687f09ea51ebcc6b934176bed1e316e30962dd8'
LANG_SHA='ae5308f1c6257a846c20ddf42d05629b3b373b144517b283447273d9bbae6c1b'

def sha(path):
    path=Path(path);assert path.resolve()==path and path.is_relative_to(ROOT)
    before=path.stat();assert before.st_size<=8*1024**3;h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    return h.hexdigest()
def read(path):return json.loads(path.read_text())
def save(path,value):
    assert path.resolve()==path and path.is_relative_to(HERE)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def patches():
    return [
      ("HERE/'core.py'",repr(str(OLD/'core.py'))),
      ("SCOUTS=[c.WF/'scout_v1/run_v1',c.WF/'scout_next_v1/shard_0/run_v1']","SCOUTS=[Path("+repr(str(SCOUT))+")]"),
      ("SPIN_RUNS=[c.WF/'batch_execution_v1'/name/'run_v1' for name in ('batch_00','batch_01r1','batch_02r2')]","SPIN_RUNS=[]  # New houses have no previously certified full-turn witnesses."),
      ("out=HERE/'snapshot_v3'","out=HERE/'snapshot_v1'"),
      ("HERE/'prepare.py',HERE/'test_core.py'","Path("+repr(str(OLD/'prepare.py'))+"),Path("+repr(str(OLD/'test_core.py'))+"),HERE/'prepare.py',HERE/'test_prepare.py'"),
      ("'node':'Q35N_MULTI_PROGRAM_BANK_V1'","'node':'Q35N_MULTI_PROGRAM_BANK_V2_CLOSED_NEW_HOUSES'"),
    ]
def adapt(source):
    assert hashlib.sha256(source.encode()).hexdigest()==ORIGINAL_SHA,'FROZEN_PREPARE_CHANGED'
    out=source
    for old,new in patches():
        expected=2 if old=="HERE/'core.py'" else 1
        assert out.count(old)==expected,(old,out.count(old));out=out.replace(old,new)
    reverse=out
    for old,new in reversed(patches()):reverse=reverse.replace(new,old)
    assert reverse==source,'NONREVERSIBLE_TRANSPORT'
    return out
def closure(source=SCOUT):
    result=read(source/'result.json');assert result['status']=='SCOUT_CLOSED' and result['error'] is None
    audit=read(source/'STORE_CLOSE_AUDIT.json');assert audit['audit_pass'] is True and audit['poisoned'] is False
    cfg=read(source/'EXECUTION_CONFIG.json');assert [c['house_id'] for c in cfg['candidates']]==HOUSES
    # shard_00 runtime binds its parent's SOURCE_LOCK, not a run INPUT_LOCK.
    manifest=source.parents[1]/'SOURCE_LOCK.json'
    lock=read(manifest);assert len(lock)<=2048
    assert sum(Path(p).stat().st_size for p in lock)<=32*1024**3
    for p,h in lock.items():assert sha(Path(p))==h,'SCOUT_INPUT_CHANGED:'+p
    assert cfg['source_lock_sha256']==sha(manifest),'RUNTIME_SOURCE_MANIFEST_BINDING'
    approval=source.parents[1]/'MAIN_AGENT_APPROVAL_SHARD_00.json'
    assert cfg['main_agent_approval_sha256']==sha(approval)
    assert read(approval)=={'approved':True,'shard':0,'gpu':1,'source_lock_sha256':sha(manifest)}
    prepared=read(source.parent/'PREPARED_CONFIG.json')
    prepared.update(runtime_allowed=True,executable=True,runtime_adapter_ready=True,
        main_agent_approval_sha256=sha(approval),source_lock_sha256=sha(manifest))
    assert cfg==prepared,'EXACT_PREPARED_RUNTIME_CONFIG'
    proc=read(source/'PROCESS.json')
    for p in (manifest,approval,source/'EXECUTION_CONFIG.json'):
        assert p.stat().st_mtime<=proc['started_unix'],'SOURCE_NOT_PREACTION'
    supervisor=read(source/'SUPERVISOR_RESULT.json')
    assert supervisor['error'] is None and supervisor['cleanup_complete'] is True
    split=[Path(p) for p,h in lock.items() if h==cfg['split_sha256']]
    assert len(split)==1
    assert set(HOUSES)<=set(read(split[0])['FIT'])
    for house in HOUSES:assert read(source/'houses'/house/'result.json')['status']=='SCOUT_COMPONENT_BANK_COMPLETE'
    for p in (manifest,approval):lock[str(p)]=sha(p)
    for name in ('EXECUTION_CONFIG.json','PROCESS.json','SUPERVISOR_RESULT.json','result.json','STORE_CLOSE_AUDIT.json'):
        lock[str(source/name)]=sha(source/name)
    return lock
def load_language():
    path=WF/'language_realization_v1/verbalizer.py';assert sha(path)==LANG_SHA
    s=importlib.util.spec_from_file_location('bank_v2_finite_language',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def normalize(cfg,language):
    new=copy.deepcopy(cfg);changes=[]
    for old,row in zip(cfg['candidates'],new['candidates']):
        revision=language.propose_revision(old);row['tasks']=revision['proposed_tasks'];row['component_provenance']['language_revision']=revision
        check=copy.deepcopy(row);check['tasks']=copy.deepcopy(old['tasks']);check['component_provenance'].pop('language_revision')
        if 'language_revision' in old['component_provenance']:
            check['component_provenance']['language_revision']=copy.deepcopy(old['component_provenance']['language_revision'])
        assert check==old,'LANGUAGE_CHANGED_STRUCTURE';changes.append(revision)
    return new,changes
def main():
    assert not (HERE/'snapshot_v1').exists() and not (HERE/'language_ready_v1').exists(),'NO_OVERWRITE'
    assert sha(OLD/'core.py')==CORE_SHA
    source=OLD/'prepare.py';adapted=adapt(source.read_text());closure_lock=closure()
    save(HERE/'CLOSED_SOURCE_ACCEPTANCE.json',{'source':str(SCOUT),'houses':HOUSES,'input_sources_verified':len(closure_lock),
        'source_result_sha256':sha(SCOUT/'result.json'),'source_input_manifest_sha256':sha(SCOUT.parents[1]/'SOURCE_LOCK.json'),
        'all_houses_FIT':True,'active_sources_used':False,'algorithm_unchanged':True,'scientific_pass':False})
    module=types.ModuleType('multi_bank_v2_private_transport');module.__file__=str(HERE/'prepare.py')
    exec(compile(adapted,str(source)+'::closed_new_house_transport','exec'),module.__dict__)
    module.main()
    bank=HERE/'snapshot_v1';cfg=read(bank/'CONFIG_DRAFT.json');lock=read(bank/'SOURCE_LOCK.json')
    for p,h in closure_lock.items():assert sha(Path(p))==h;lock[p]=h
    normalized,revisions=normalize(cfg,load_language())
    normalized['node']='Q35N_MULTI_PROGRAM_BANK_V2_FINITE_LANGUAGE'
    out=HERE/'language_ready_v1';out.mkdir()
    save(out/'CONFIG_DRAFT.json',normalized)
    groups={}
    for row in normalized['candidates']:groups.setdefault((row['house_id'],row['hub_index']),[]).append(row)
    ordered=[]
    for i in range(max(map(len,groups.values()),default=0)):
        for key,rows in sorted(groups.items()):
            if i<len(rows):ordered.append(rows[i]['candidate_id'])
    save(out/'LANGUAGE_CHANGES.json',{'revisions':revisions,'changed_instructions':sum(len(r['changes']) for r in revisions),
        'structural_changes':0,'scientific_pass':False})
    save(out/'NEXT12.json',{'candidate_ids':ordered[:12],'selection_rule':'deterministic round-robin sorted house/hub; rank within same hub unchanged',
        'new_physical_families':0,'runtime_allowed':False,'requires_main_freeze_and_full27_replays':True,'same_hub_variants_independent':False})
    deps=[source,OLD/'core.py',OLD/'SHA256SUMS',HERE/'prepare.py',HERE/'test_prepare.py',HERE/'CLOSED_SOURCE_ACCEPTANCE.json',
        WF/'language_realization_v1/verbalizer.py',WF/'language_realization_v1/SHA256SUMS',WF/'language_realization_v1/SCHEMA.json',
        WF.parent.parent/'mechanism_scale_v1/planner.py',out/'CONFIG_DRAFT.json',out/'LANGUAGE_CHANGES.json',out/'NEXT12.json']
    deps.extend(bank/name for name in ('CONFIG_DRAFT.json','SOURCE_LOCK.json','PROPOSAL_LEDGER.json','COUNT_LEDGER.json','SEMANTIC_DEDUP_LEDGER.json','SPIN_SOURCE_LEDGER.json','SPIN_DECISION_EVIDENCE.json','result.json'))
    for p in deps:lock[str(p)]=sha(p)
    save(out/'SOURCE_LOCK.json',lock)
    summary={'status':'NEW_CLOSED_HOUSE_BANK_AND_LANGUAGE_READY_NOT_PHYSICAL','houses':HOUSES,
        'candidates':len(cfg['candidates']),'hubs_with_candidates':len(groups),'source_lock_entries':len(lock),
        'next12':len(ordered[:12]),'actual_full_spin_witnesses':0,'missing_spin_neutrality':'UNTESTED_REQUIRES_RUNTIME',
        'gpu_operations':0,'new_physical_families':0,'scientific_pass':False}
    save(out/'result.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
