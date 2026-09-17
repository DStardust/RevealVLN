"""Natural closure only -> original strict bank, with explicit 0..4 coverage."""
import copy
import importlib.util
import math
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('scale_scout_bank_common',HERE/'common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
ORIGINAL=c.WF/'multi_program_bank_v1/prepare.py'
ORIGINAL_SHA='0c5a26e23a453e07e8893242bc13e42e7ffc033efa8238ba3929d7075ecdc4b5'
CORE=c.WF/'multi_program_bank_v1/core.py'
CORE_SHA='21cff82e02315dcc3616fa523687f09ea51ebcc6b934176bed1e316e30962dd8'

def coverage(cfg,house_result,frozen):
    plan=cfg['new_hub_plan'];positions=plan['selected_positions'];house=plan['house_id']
    c.require(plan['limit']==4 and len(positions)==4 and len({tuple(p) for p in positions})==4,'FOUR_FROZEN_POSITIONS')
    c.require(house_result['status']=='SCOUT_COMPONENT_BANK_COMPLETE' and house_result['house_id']==house,'NATURAL_HOUSE_CLOSURE_REQUIRED')
    c.require(frozen.get('saved_before_any_house_action') is True,'PREACTION_HUB_FREEZE')
    hubs=frozen['hubs'];geometry=frozen['geometry'];checks=geometry['geometry_checks']
    c.require(geometry['frozen_selected_positions']==positions,'EXACT_SELECTED_POSITIONS')
    c.require(len(checks)==4 and {tuple(x['position']) for x in checks}==set(map(tuple,positions)),'ALL_POSITIONS_GEOMETRY_TERMINAL')
    c.require(0<=len(hubs)<=4 and house_result['selected_hubs']==len(hubs),'ACTUAL_HUB_COUNT')
    c.require(len(house_result['hubs'])==len(hubs) and {r['hub'] for r in house_result['hubs']}==set(range(len(hubs))),'ALL_SELECTED_HUBS_COMPLETE')
    actual={tuple(h['position']):h for h in hubs};c.require(len(actual)==len(hubs),'DUPLICATE_ACTUAL_HUB')
    for i,h in enumerate(hubs):
        p=h['position'];c.require(p in positions and h['yaw_bin']==0,'NO_HUB_SUBSTITUTION')
        c.require(all(math.dist(p,q)>=1 for q in plan['excluded_positions']+[v['position'] for v in hubs[:i]]),'NO_HUB_OVERLAP')
    rows=[]
    for p in positions:
        g=next(v for v in checks if v['position']==p)
        if tuple(p) in actual:
            c.require(g['status']=='GEOMETRY_CHECKED_NOT_VISIBILITY_CERTIFIED' and g.get('reachable_groups',0)>0 and g.get('snapped') is not None and math.dist(p,g['snapped'])<=1e-5,'ACCEPTED_GEOMETRY_PROOF')
            status='ACTUAL_COMPONENT_SCOUT_COMPLETE_NOT_FAMILY'
        elif g['status']=='SOURCE_COORDINATE_NOT_EXACT_NAVMESH':status=g['status']
        elif g['status']=='GEOMETRY_CHECKED_NOT_VISIBILITY_CERTIFIED' and g.get('reachable_groups')==0:status='NO_REACHABLE_ROLE_GROUP'
        else:raise ValueError('UNEXPLAINED_UNSELECTED_POSITION')
        rows.append(dict(position=p,status=status,geometry=g))
    return dict(requested_hubs=4,actual_hubs=len(hubs),house_id=house,positions=rows,
        source_grade='NATURAL_CLOSED_SUBSET_WITH_FULL_POSITION_LEDGER',new_physical_families=0,scientific_pass=False)

def closure(job):
    job=c.job_root(job);runtime=c.load('scale_bank_runtime_checks',HERE/'runtime.py');cfg,_,_=runtime.check(job)
    run=job/'run_v1';c.require(c.read(run/'EXECUTION_CONFIG.json')==cfg,'EXACT_EXECUTION_CONFIG')
    lock=c.verify_lock(job/'SOURCE_LOCK.json');c.require(c.read(run/'INPUT_LOCK.json')==lock,'EXACT_RUNTIME_INPUT_LOCK')
    result=c.read(run/'result.json');c.require(result['status']=='SCOUT_CLOSED' and result['error'] is None,'NATURAL_SOURCE_CLOSED')
    audit=c.read(run/'STORE_CLOSE_AUDIT.json');c.require(audit['audit_pass'] is True and audit['poisoned'] is False,'CONTENT_CLOSED_UNPOISONED')
    sup=c.read(run/'SUPERVISOR_RESULT.json');c.require(sup['returncode']==0 and sup['error'] is None and sup['cleanup_complete'] is True,'SUPERVISOR_CLOSED')
    lease=c.read(run/'LEASE_RESULT.json');c.require(lease['execute_returned'] is True and lease['error'] is None and lease['holder_restored'] is True,'LEASE_CLOSED')
    launch=c.read(run/'LAUNCH_RESULT.json');c.require(launch['supervisor_returned'] is True and launch['error'] is None,'LAUNCH_CLOSED')
    restored=c.read(run/'RESTORATION.json');c.require(restored['restored'] is True and restored.get('remain_on_exit_restored') is True,'HOLDER_RESTORED')
    house=cfg['new_hub_plan']['house_id'];folder=run/'houses'/house
    report=coverage(cfg,c.read(folder/'result.json'),c.read(folder/'FROZEN_HUB_CONFIGS.json'))
    c.require(cfg['candidates'][0]['split']=='FIT' and house in c.read(c.LINE/'data_pipeline/ordinary_scale_v1/SPLIT_FREEZE.json')['FIT'],'FIT_ONLY')
    proc=c.read(run/'PROCESS.json')
    for path in [job/'SOURCE_LOCK.json',job/'MAIN_AGENT_APPROVAL.json',run/'EXECUTION_CONFIG.json',run/'INPUT_LOCK.json']:
        c.require(path.stat().st_mtime<=proc['started_unix'],'SOURCE_NOT_PREACTION');lock[str(path)]=c.sha(path)
    for path in [run/name for name in ['result.json','STORE_CLOSE_AUDIT.json','SUPERVISOR_RESULT.json','LEASE_RESULT.json','RESTORATION.json','LAUNCH_RESULT.json','PROCESS.json']]+[folder/'result.json',folder/'FROZEN_HUB_CONFIGS.json']:
        lock[str(path)]=c.sha(path)
    return cfg,lock,report

def adapted(job):
    c.require(c.sha(ORIGINAL)==ORIGINAL_SHA and c.sha(CORE)==CORE_SHA,'FROZEN_BANK_SOURCE_CHANGED')
    run=c.job_root(job)/'run_v1';output=job/'bank';source=ORIGINAL.read_text()
    changes=[('HERE=Path(__file__).resolve().parent','HERE=Path('+repr(str(output))+')'),
        ("spec=importlib.util.spec_from_file_location('multi_bank_core',HERE/'core.py')","spec=importlib.util.spec_from_file_location('multi_bank_core',Path("+repr(str(CORE))+"))"),
        ("SCOUTS=[c.WF/'scout_v1/run_v1',c.WF/'scout_next_v1/shard_0/run_v1']",'SCOUTS=[Path('+repr(str(run))+')]'),
        ("SPIN_RUNS=[c.WF/'batch_execution_v1'/name/'run_v1' for name in ('batch_00','batch_01r1','batch_02r2')]",'SPIN_RUNS=[]'),
        ("out=HERE/'snapshot_v3'","out=HERE/'snapshot_v1'"),
        ("for path in [HERE/'core.py',HERE/'prepare.py',HERE/'test_core.py',",'for path in [Path('+repr(str(CORE))+'),Path('+repr(str(ORIGINAL))+'),Path('+repr(str(HERE/'bank.py'))+'),')]
    value=source
    for old,new in changes:value=c.exact(value,old,new)
    reverse=value
    for old,new in reversed(changes):reverse=reverse.replace(new,old)
    c.require(reverse==source,'BANK_ALGORITHM_EXACT_REVERSE');return value

def main(job,check_only=False):
    job=c.job_root(job);cfg,lock,report=closure(job)
    if check_only:return report
    c.require(report['actual_hubs']>=1,'NO_ACTUAL_HUBS_NO_BANK')
    out=job/'bank';out.mkdir(exist_ok=False);c.save(out/'COVERAGE.json',report)
    source=adapted(job);module=types.ModuleType('scale_new_hub_original_bank');module.__file__=str(ORIGINAL)
    exec(compile(source,str(ORIGINAL)+'::new_hub_scale','exec'),module.__dict__)
    module.main()  # original journal, every trace/digest/instance/count/query checks
    bank=out/'snapshot_v1';proposal=c.read(bank/'CONFIG_DRAFT.json');lock.update(c.read(bank/'SOURCE_LOCK.json'))
    c.require(c.sha(c.WF/'multi_program_bank_v2/prepare.py')=='61e2e9dc840c11f5fb1f1dfdec46579f9e2f78324f2010a02871f4a5257e5b02','FROZEN_NORMALIZER')
    normalize=c.load('scale_bank_frozen_normalizer',c.WF/'multi_program_bank_v2/prepare.py')
    result,changes=normalize.normalize(proposal,normalize.load_language());ready=out/'language_ready_v1';ready.mkdir()
    c.save(ready/'CONFIG_DRAFT.json',result);c.save(ready/'LANGUAGE_CHANGES.json',dict(revisions=changes,structural_changes=0))
    groups={}
    for row in result['candidates']:groups.setdefault((row['house_id'],row['hub_index']),[]).append(row)
    ordered=[rows[i]['candidate_id'] for i in range(max(map(len,groups.values()),default=0)) for key,rows in sorted(groups.items()) if i<len(rows)]
    c.save(ready/'NEXT12.json',dict(candidate_ids=ordered[:12],selection_rule='sorted house/hub round robin; original within-hub rank',new_physical_families=0,runtime_allowed=False,same_hub_variants_independent=False))
    c.save(ready/'result.json',dict(status='CLOSED_SOURCE_PROGRAM_BANK_READY_NOT_FAMILIES',candidates=len(ordered),actual_hubs=report['actual_hubs'],requested_hubs=4,
        hubs_with_candidates=len(groups),house_id=report['house_id'],new_physical_families=0,scientific_pass=False))
    for path in [out/'COVERAGE.json',job/'SOURCE_LOCK.json',HERE/'bank.py',ORIGINAL,CORE,c.WF/'multi_program_bank_v2/prepare.py',c.WF/'language_realization_v1/verbalizer.py']+list(bank.glob('*.json'))+list(ready.glob('*.json')):lock[str(path)]=c.sha(path)
    for path,h in lock.items():c.require(c.sha(path)==h,'SOURCE_CHANGED_DURING_BANK')
    c.save(ready/'SOURCE_LOCK.json',lock)
    return c.read(ready/'result.json')

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    print(json.dumps(main(a.job,a.check_only),indent=2))
