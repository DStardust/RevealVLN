"""CPU-only next three distinct physical hubs from the frozen new-hub NEXT12."""
import importlib.util
import json
import math
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('batch08_private_clock_transport',HERE/'transport.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
SNAPSHOT=t.WF/'new_hub_bank_v1/language_ready_v1'
def selection(snapshot):
    draft=json.loads((snapshot/'CONFIG_DRAFT.json').read_text());nxt=json.loads((snapshot/'NEXT12.json').read_text())
    assert draft['runtime_allowed'] is False and draft['training_allowed'] is False and nxt['runtime_allowed'] is False
    assert len(nxt['candidate_ids'])==len(set(nxt['candidate_ids']))==12
    mapping={r['candidate_id']:(i,r) for i,r in enumerate(draft['candidates'])};assert len(mapping)==len(draft['candidates'])
    chosen=[];indices=[];positions=[];ledger=[]
    for pos,key in enumerate(nxt['candidate_ids']):
        index,row=mapping[key];point=row['configuration']['u_position'];assert row['split']=='FIT' and len(point)==3
        assert all(type(x) in (int,float) and math.isfinite(x) for x in point)
        item={'next12_position':pos,'source_index':index,'candidate_id':key,'house_id':row['house_id'],'hub_position':point}
        ledger.append(item)
        if any(old['house_id']==row['house_id'] and math.dist(old['configuration']['u_position'],point)<1 for old in chosen):
            item['status']='SAME_PHYSICAL_HUB_NOT_ANOTHER_ATTEMPT';continue
        if len(chosen)==3:item['status']='NOT_SELECTED_THREE_HUB_BUDGET';continue
        item['status']='PREDECLARED_SELECTED';chosen.append(row);indices.append(index);positions.append(pos)
    assert len(chosen)==3,'NEED_THREE_DISTINCT_PHYSICAL_HUBS'
    return indices,positions,ledger
def adapted_prepare_source(source,positions):
    assert t.sha(t.BE/'prepare.py')==t.private.HASHES['prepare.py']
    old="cfg['split_sha256']=sha(split)"
    added="\n    cfg.update(budget_transport='clock_batching_fresh_only_v1',fresh_only=True,resume_allowed=False,comparison='unpaired_actual_yield_and_wall_only',cohort_membership=None,next12_positions="+repr(positions)+")"
    assert source.count(old)==1;value=source.replace(old,old+added)
    old2='for p in code:lock[str(p.resolve())]=sha(p)';new2='code += extra_code\n    '+old2
    assert value.count(old2)==1;value=value.replace(old2,new2)
    assert value.replace(new2,old2).replace(old+added,old)==source
    return value
def wrapper(entry):
    assert entry in ('worker_main','run_main')
    return "import importlib.util\nfrom pathlib import Path\nHERE=Path(__file__).resolve().parent\ns=importlib.util.spec_from_file_location('batch08_clock_transport',HERE.parent/'budget_batched_transport_v2/transport.py')\nt=importlib.util.module_from_spec(s);s.loader.exec_module(t)\nif __name__=='__main__':t."+entry+"(HERE)\n"
def prepare():
    indices,positions,ledger=selection(SNAPSHOT)
    assert indices==[0,24,48] and positions==[0,1,2],'FIXED_NEW_HUB_NEXT12_SELECTION'
    t.base('prepare.py');assert t.sha(t.CLOCK)==t.CLOCK_SHA
    module=types.ModuleType('batch08_private_original_prepare');module.__file__=str(t.BE/'prepare.py')
    exec(compile(adapted_prepare_source((t.BE/'prepare.py').read_text(),positions),str(HERE/'prepare.py'),'exec'),module.__dict__)
    selection_file=HERE/'SELECTION.json'
    with selection_file.open('x') as f:json.dump({'source_snapshot':str(SNAPSHOT),'indices':indices,'next12_positions':positions,
        'ledger':ledger,'same_house_hubs_correlated':True,'old_cohort_members_replaced':False,'gpu_operations':0,'scientific_pass':False},f,indent=2)
    module.extra_code=[*HERE.glob('*.py'),HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',selection_file,SNAPSHOT/'NEXT12.json',SNAPSHOT/'SOURCE_LOCK.json',
        t.ORIGINAL,t.ORIGINAL.parent/'prepare.py',t.ORIGINAL.parent/'SHA256SUMS',
        t.CLOCK,t.CLOCK.parent/'SHA256SUMS',t.CLOCK.parent/'SPEC_ZH.md',
        SNAPSHOT.parent/'POSTPROCESS_SHA256SUMS',SNAPSHOT.parent/'acceptance_v1/result.json']
    batch=t.BE/'batch_08';batch.mkdir(exist_ok=False)
    for name,entry in [('worker.py','worker_main'),('run.py','run_main')]:
        with (batch/name).open('x') as f:f.write(wrapper(entry))
    module.prepare(SNAPSHOT,'batch_08',indices,2,'winding_v1')
    cfg=t.check_inputs(batch,worker=True)
    assert cfg['source_selection_indices']==indices and cfg['next12_positions']==positions
    result={'status':'BATCH08_CPU_PREPARED_WAITING_MAIN_AGENT_LAUNCH_APPROVAL','gpu_operations':0,
        'indices':indices,'next12_positions':positions,'candidates':len(cfg['candidates']),
        'physical_hubs':3,'houses':len({r['house_id'] for r in cfg['candidates']}),
        'input_lock_entries':len(json.loads((batch/'run_v1/INPUT_LOCK.json').read_text())),
        'source_root_allowed_by_original_prepare_without_scope_patch':True,
        'supervisor_seconds':3900,'factory_seconds':3600,'total_actions':60000,
        'original_stricter_disk_guard_bytes':7*1024**3,'old_cohort_members_replaced':False,
        'launch_approved_by_this_CPU_node':False,'scientific_pass':False}
    with (HERE/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':prepare()
