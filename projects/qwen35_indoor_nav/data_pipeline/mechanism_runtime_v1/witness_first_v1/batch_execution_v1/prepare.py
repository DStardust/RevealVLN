"""Main-agent frozen batch admission; no simulator/GPU work."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
WF=HERE.parent
ROOT=WF.parents[4]
def sha(path):
    path=Path(path).resolve(strict=True);assert path.is_relative_to(ROOT)
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def prepare(snapshot,name,indices,gpu,variant):
    snapshot=Path(snapshot).resolve(strict=True)
    assert snapshot.is_relative_to(WF) and name.startswith('batch_') and '/' not in name
    assert gpu in (1,2) and variant in ('lr_v2','winding_v1')
    draft=json.loads((snapshot/'CONFIG_DRAFT.json').read_text())
    assert draft['runtime_allowed'] is False and draft['training_allowed'] is False
    lock=json.loads((snapshot/'SOURCE_LOCK.json').read_text())
    for p,h in lock.items():assert sha(p)==h,p
    selected=[draft['candidates'][i] for i in indices]
    assert 1<=len(selected)<=3 and len(set(indices))==len(indices)
    assert len({(r['house_id'],tuple(r['configuration']['u_position'])) for r in selected})==len(selected)
    method_path={'lr_v2':WF/'batch_plan_v1/balanced_v2.py','winding_v1':WF/'winding_balance_v1/method.py'}[variant]
    spec=importlib.util.spec_from_file_location('batch_admission_method',method_path)
    method=importlib.util.module_from_spec(spec);spec.loader.exec_module(method)
    from core_bridge import Compiler
    for row in selected:
        assert row['split']=='FIT' and row['component_provenance']
        for p,h in row['assets'].items():assert sha(p)==h,p
        row['runtime_allowed']=True;row['executable']=True
        comp=Compiler({k:(v['mpcat40'],v['room']) for k,v in row['roles'].items()},row['tasks'],row['expected_eligible'])
        method.BalancedFactory(None,comp,None,lambda *args:None,'task_A','task_B',
            context={'house_id':row['house_id'],'asset_config':row['assets']},components=row['components'],balance=row['balance'])
    cfg=dict(node='Q35N_WITNESS_BANK_'+name.upper(),runtime_allowed=True,executable=True,training_allowed=False,
        candidates=selected,factory_variant=variant,gpu_device=gpu,
        gpu_uuid={1:'GPU-734a5268-31fe-6452-105b-36cd08c3d9c8',2:'GPU-be1b30d0-517b-b079-871b-de195d35a1a2'}[gpu],
        control_type='completed_subgoal_revisit_placement_not_event_free_detour',
        budget=dict(total_actions=60000,total_seconds=3600,discovery_actions=15000,discovery_seconds=1000,
                    certification_actions=20000,certification_seconds=1500),supervision_wall_seconds=3900,
        scientific_pass=False,source_snapshot=str(snapshot),source_selection_indices=indices)
    split=WF.parents[2]/'data_pipeline/ordinary_scale_v1/SPLIT_FREEZE.json'
    split_data=json.loads(split.read_text())
    assert all(r['house_id'] in split_data['FIT'] for r in selected)
    cfg['split_sha256']=sha(split)
    lock[str(split.resolve())]=sha(split)
    batch=HERE/name
    assert batch.is_dir() and {p.name for p in batch.iterdir()}=={'worker.py','run.py'}
    out=batch/'run_v1';out.mkdir(exist_ok=False)
    save(out/'EXECUTION_CONFIG.json',cfg)
    code=[*HERE.glob('*.py'),*batch.glob('*.py'),HERE/'SPEC_ZH.md',out/'EXECUTION_CONFIG.json',snapshot/'CONFIG_DRAFT.json',
          WF/'short_revisit_v2/worker.py',WF/'assembly_v1/worker.py',WF/'short_revisit_v2/method.py',
          WF/'revisit_v1/method.py',WF/'assembly_v1/method.py',WF/'budget_and_trace/adapters.py',
          WF.parent/'compact_loop_v2/run.py',WF.parent/'feedback_generation_v1/store.py',
          WF.parent/'exporter.py',WF.parent/'loader.py']
    code += [method_path, WF.parent/'core_bridge.py',WF.parent/'habitat_backend.py',WF.parent/'guard.py',
             WF.parent/'runtime_journal.py',WF.parent/'SHA256SUMS',
             WF.parent.parent/'mechanism_factory_v2/compiler.py',WF.parent.parent/'mechanism_factory_v2/factory.py',
             WF.parent.parent/'mechanism_factory_v2/planning.py',WF.parent.parent/'mechanism_factory_v2/SHA256SUMS',
             WF/'budget_and_trace/SHA256SUMS',WF.parent/'feedback_diagnosis_v1/budget_optimization_cpu/budget.py',
             WF.parent/'feedback_diagnosis_v1/budget_optimization_cpu/SHA256SUMS']
    for p in code:lock[str(p.resolve())]=sha(p)
    save(out/'INPUT_LOCK.json',lock)
    print(json.dumps(dict(prepared=True,batch=name,gpu=gpu,candidates=len(selected),locked_files=len(lock))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--name',required=True)
    p.add_argument('--indices',type=int,nargs='+',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--variant',required=True);a=p.parse_args();prepare(a.snapshot,a.name,a.indices,a.gpu,a.variant)
