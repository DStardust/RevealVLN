"""CPU postprocessing only, after the four-new-hub scout genuinely closes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent;WF=HERE.parent
PREVIOUS=WF/'multi_program_bank_v2/prepare.py'
PREVIOUS_SHA='61e2e9dc840c11f5fb1f1dfdec46579f9e2f78324f2010a02871f4a5257e5b02'
SOURCE=WF/'new_hub_scout_v1/run_v1'
HOUSE='8WUmhLawc2A'

def patches():
    return [
        ("SCOUT=WF/'bulk_source_v1/shard_00/run_v1'","SCOUT=WF/'new_hub_scout_v1/run_v1'"),
        ("HOUSES=['5q7pvUzZiYa','759xd9YjKW5','7y3sRwLe3Va']","HOUSES=['8WUmhLawc2A']"),
        ("# shard_00 runtime binds its parent's SOURCE_LOCK, not a run INPUT_LOCK.","# New-hub scout stores an exact copy of its parent SOURCE_LOCK in INPUT_LOCK."),
        ("manifest=source.parents[1]/'SOURCE_LOCK.json'","manifest=source.parent/'SOURCE_LOCK.json'"),
        ("lock=read(manifest);assert len(lock)<=2048","lock=read(manifest);assert len(lock)<=2048\n    assert read(source/'INPUT_LOCK.json')==lock,'EXACT_RUNTIME_INPUT_MANIFEST'"),
        ("approval=source.parents[1]/'MAIN_AGENT_APPROVAL_SHARD_00.json'","approval=source.parent/'MAIN_AGENT_APPROVAL.json'"),
        ("{'approved':True,'shard':0,'gpu':1,'source_lock_sha256':sha(manifest)}",
         "{'approved':True,'gpu':2,'source_lock_sha256':sha(manifest),'node':'Q35N_NEW_HUB_SCOUT_FOUR_V1','new_hubs':4,'training_allowed':False}"),
        ("for p in (manifest,approval,source/'EXECUTION_CONFIG.json'):","for p in (manifest,approval,source/'EXECUTION_CONFIG.json',source/'INPUT_LOCK.json'):"),
        ("for p in (manifest,approval):lock[str(p)]=sha(p)","for p in (manifest,approval,source/'INPUT_LOCK.json'):lock[str(p)]=sha(p)"),
        ("sha(SCOUT.parents[1]/'SOURCE_LOCK.json')","sha(SCOUT.parent/'SOURCE_LOCK.json')"),
        ("Q35N_MULTI_PROGRAM_BANK_V2_CLOSED_NEW_HOUSES","Q35N_NEW_HUB_BANK_V1"),
        ("Q35N_MULTI_PROGRAM_BANK_V2_FINITE_LANGUAGE","Q35N_NEW_HUB_BANK_V1_FINITE_LANGUAGE"),
        ("# New houses have no previously certified full-turn witnesses.","# These four new physical hubs have no certified full-turn witnesses."),
        ("deps=[source,OLD/'core.py'","deps=[Path("+repr(str(PREVIOUS))+"),Path("+repr(str(PREVIOUS.parent/'SHA256SUMS'))+"),source,OLD/'core.py'"),
    ]
def adapt(source):
    if hashlib.sha256(source.encode()).hexdigest()!=PREVIOUS_SHA:raise ValueError('FROZEN_V2_PREPARE_CHANGED')
    output=source
    for old,new in patches():
        assert output.count(old)==1,(old,output.count(old));output=output.replace(old,new)
    reverse=output
    for old,new in reversed(patches()):reverse=reverse.replace(new,old)
    assert reverse==source,'NONREVERSIBLE_SOURCE_ADAPTER'
    return output
def require_four_actual_hubs(cfg,house_result,frozen):
    plan=cfg['new_hub_plan'];assert plan['house_id']==HOUSE and plan['limit']==4
    expected=plan['selected_positions'];excluded=plan['excluded_positions']
    assert len(expected)==4 and len({tuple(p) for p in expected})==4
    assert house_result['house_id']==HOUSE and house_result['status']=='SCOUT_COMPONENT_BANK_COMPLETE'
    assert house_result['selected_hubs']==4 and len(house_result['hubs'])==4
    hubs=frozen['hubs'];assert len(hubs)==4 and {tuple(h['position']) for h in hubs}==set(map(tuple,expected))
    assert frozen['saved_before_any_house_action'] is True
    assert frozen['geometry']['all_four_runtime_geometry_retained'] is True
    assert frozen['geometry']['frozen_selected_positions']==expected
    for i,hub in enumerate(hubs):
        pos=hub['position'];assert len(pos)==3 and all(type(x) in (int,float) and math.isfinite(x) for x in pos)
        assert hub['yaw_bin']==0
        assert all(math.dist(pos,q)>=1 for q in excluded+[h['position'] for h in hubs[:i]]),'NOT_A_NEW_PHYSICAL_HUB'
    assert len(cfg['candidates'])==1 and cfg['candidates'][0]['house_id']==HOUSE
    assert cfg['candidates'][0]['split']=='FIT' and cfg['candidates'][0]['source_positions']==expected
    assert plan['scene_group_id']=='mp3d:'+HOUSE and plan['house_split']=='FIT'

def load():
    module=types.ModuleType('new_hub_bank_private_v2');module.__file__=str(HERE/'prepare.py')
    exec(compile(adapt(PREVIOUS.read_text()),str(PREVIOUS)+'::new_hub_bank_v1','exec'),module.__dict__)
    original_closure=module.closure
    def closure(source=SOURCE):
        assert source==SOURCE,'ONLY_FIXED_NEW_HUB_SOURCE'
        # Missing/active sources fail before any bank/config/output is created.
        lock=original_closure(source)
        cfg=module.read(source/'EXECUTION_CONFIG.json')
        result=module.read(source/'houses'/HOUSE/'result.json')
        frozen_path=source/'houses'/HOUSE/'FROZEN_HUB_CONFIGS.json'
        require_four_actual_hubs(cfg,result,module.read(frozen_path))
        supervisor=module.read(source/'SUPERVISOR_RESULT.json')
        assert supervisor['returncode']==0 and supervisor['external_processes_stopped']==0 and supervisor['holders_touched'] is False
        launch=module.read(source/'LAUNCH_RESULT.json')
        assert launch['supervisor_returned'] is True and launch['error'] is None
        for path in (frozen_path,source/'houses'/HOUSE/'result.json',source/'LAUNCH_RESULT.json',
                     HERE/'prepare.py',HERE/'test_prepare.py',HERE/'SOURCE_LOCK.json',HERE/'SHA256SUMS'):
            lock[str(path)]=module.sha(path)
        return lock
    module.closure=closure
    return module
def main(argv=None):
    parser=argparse.ArgumentParser(description='CPU bank build only after four-hub natural closure; no GPU operations')
    parser.add_argument('--check-only',action='store_true',help='read-only source readiness and hashes; creates no candidate files')
    args=parser.parse_args(argv)
    module=load()
    if args.check_only:
        lock=module.closure()
        print(json.dumps({'status':'SOURCE_CLOSED_AND_BOUND_READY_FOR_CPU_BANK','verified_sources':len(lock),
            'new_physical_families':0,'gpu_operations':0,'scientific_pass':False},indent=2))
    else:module.main()
if __name__=='__main__':main()
