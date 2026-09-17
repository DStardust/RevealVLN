import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'budget_and_trace'),str(RUNTIME)]
from method import WitnessFactory
from core_bridge import Compiler,FreezeLedger,Reject,BudgetExceeded,digest
from adapters import Journal,durable_budget,PartialTraceRunner
from habitat_backend import HabitatBackend
from exporter import export_family
from loader import FamilyLoader

def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def store_class():
    path=RUNTIME/'feedback_generation_v1/store.py'
    spec=importlib.util.spec_from_file_location('witness_scoped_store',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.FEEDBACK_ROOT=HERE
    return module.TrackedContentStore
def main():
    out=HERE/'run_v1'
    for p,h in json.loads((out/'INPUT_LOCK.json').read_text()).items():assert sha(p)==h,p
    cfg=json.loads((out/'EXECUTION_CONFIG.json').read_text());assert cfg['runtime_allowed'] and not cfg['training_allowed']
    (out/'bundles').mkdir();started=time.monotonic();backend=None;store=None;error=None;results=[]
    counts=dict(traces=0,actual_actions=0,collisions=0,complete_traces=0)
    def progress(extra=None):
        value=dict(counts,physical_families=sum(x.get('physical_certified',False) for x in results),training_ready_families=0,**(extra or {}))
        tmp=out/'PROGRESS.pending';tmp.write_text(json.dumps(value,indent=2));tmp.replace(out/'PROGRESS.json')
    try:
        with store_class()(out/'content',6*1024**3) as store,Journal(out/'journal',cfg) as journal:
            budget=durable_budget(journal,cfg['budget'],clock=time.monotonic)
            ledger=FreezeLedger(persist=lambda s:journal.append('freeze',s))
            for row in cfg['candidates']:
                budget.check_time();folder=out/'bundles'/row['candidate_id'];folder.mkdir();(folder/'traces').mkdir()
                for p,h in row['assets'].items():assert sha(p)==h,p
                backend=HabitatBackend(row['scene_glb'],cfg['gpu_device'],row['roles'],store,dict(cfg,scene_glb=row['scene_glb']))
                save(folder/'SEMANTIC_INVENTORY.json',dict(objects=backend.objects,eligible=backend.eligible))
                assert backend.eligible==row['expected_eligible'],'ROLE_IDENTITY'
                compiler=Compiler(backend.compiler_roles,row['tasks'],backend.eligible)
                index=0;certifying=False;retained=[]
                def emit(kind,value):
                    nonlocal index
                    if kind=='trace':
                        path=folder/'traces'/('%06d.json'%index);save(path,value)
                        journal.append('trace_saved',dict(bundle=row['candidate_id'],index=index,sha256=sha(path),complete=value['complete']))
                        index+=1;counts['traces']+=1;counts['complete_traces']+=int(value['complete']);counts['collisions']+=value['collisions']
                        if certifying and value['complete']:retained.append(value)
                        progress(dict(current_bundle=row['candidate_id']))
                    else:
                        journal.append(kind,dict(bundle=row['candidate_id'],value=value))
                        if kind=='action_completed':counts['actual_actions']+=1
                factory=WitnessFactory(backend,compiler,budget,emit,'task_A','task_B',context={'house_id':row['house_id'],'asset_config':row['assets']},components=row['components'])
                factory.runner=PartialTraceRunner(backend,compiler,budget,emit)
                result=dict(candidate_id=row['candidate_id'],house_id=row['house_id'],physical_certified=False,training_admission=False)
                try:
                    budget.start_bundle(row['candidate_id'],'discovery')
                    candidate=factory.discover([row['configuration']],ledger,row['candidate_id'])
                    save(folder/'FROZEN_CANDIDATE.json',candidate)
                    budget.finish_phase();budget.start_bundle(row['candidate_id'],'certification')
                    certifying=True;certificate=factory.replay_seeds(candidate);certifying=False
                    assert len(retained)==27
                    keys=[(h,c) for h in candidate['histories'] for c in candidate['continuations']]
                    grids=[{k:t for k,t in zip(keys,retained[i*9:(i+1)*9])} for i in range(3)]
                    export_family(folder/'export_v4',compiler,candidate,grids[0],out/'content',family_id=row['candidate_id'],split='FIT',
                        provenance={'kind':'witness_first_balanced_assembly_v1','source':row['component_provenance']},repeated_traces=grids[1:])
                    loader=FamilyLoader(folder/'export_v4',compiler)
                    save(folder/'READBACK.json',loader.validate_supervision_contract())
                    for prefix in loader.prefix_index:
                        for record in loader.prefix_records(prefix):loader.policy_payload(record)
                    for cell in loader.cells:
                        for record in loader.action_stream(cell['cell_id']):loader.policy_payload(record)
                    save(folder/'CERTIFICATE.json',certificate);ledger.certify(row['candidate_id'],True)
                    action_counts=[dict(__import__('collections').Counter(v)) for v in candidate['histories'].values()]
                    assert all(x==action_counts[0] for x in action_counts)
                    save(folder/'COUNT_SHORTCUT_CONTROL.json',dict(exact_history_action_counts=action_counts,pass_counts=True,
                        common_current_window=True,query_policy_leakage=False,training_admission=False))
                    result.update(status='PHYSICAL_REPLAY_CERTIFIED_PENDING_FULL_QUALITY',physical_certified=True,exact_action_counts_matched=True)
                except (Reject,BudgetExceeded) as exc:
                    certifying=False;result.update(status='RESOURCE_CENSORED' if isinstance(exc,BudgetExceeded) else 'REJECTED',reason=str(exc))
                    rec=ledger.records.get(row['candidate_id'])
                    if rec and rec['status']=='discovering':ledger.reject_discovery(row['candidate_id'],str(exc),resource_censored=isinstance(exc,BudgetExceeded))
                    elif rec and rec['status']=='frozen':ledger.certify(row['candidate_id'],False,str(exc))
                finally:
                    if budget.snapshot()['active'] is not None:budget.finish_phase()
                save(folder/'result.json',result);results.append(result);backend.close();backend=None;progress()
            save(out/'BUDGET_FINAL.json',budget.snapshot());save(out/'FREEZE_LEDGER.json',ledger.records)
        assert store.final_audit and store.final_audit['audit_pass'],'FINAL_STORE_AUDIT'
    except BaseException as exc:error=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend:backend.close()
        if store is not None:save(out/'STORE_CLOSE_AUDIT.json',store.final_audit)
    result=dict(node=cfg['node'],bundles=results,error=error,counts=counts,wall_seconds=time.monotonic()-started,
        new_physical_families=sum(r['physical_certified'] for r in results),new_train_ready_families=0,training_started=False,scientific_pass=False)
    save(out/'result.json',result);print(json.dumps(result),flush=True)
    if error:raise SystemExit(1)
if __name__=='__main__':main()
