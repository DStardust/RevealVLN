import json
import math
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parent
sys.path[:0]=[str(HERE),str(RUNTIME)]
from prepare import sha
from feedback import FeedbackFactory
from core_bridge import Compiler,BudgetLedger,FreezeLedger,BudgetExceeded,Reject
from habitat_backend import HabitatBackend
from runtime_journal import Journal
from store import TrackedContentStore
from exporter import export_family
from loader import FamilyLoader

def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def progress(out,x):
    p=out/'PROGRESS.json';t=out/'PROGRESS.pending'
    with t.open('w') as f:json.dump(x,f,indent=2)
    t.replace(p)
def main():
    out=HERE/'run_v1'
    for p,h in json.loads((out/'INPUT_LOCK.json').read_text()).items():assert sha(Path(p))==h,p
    cfg=json.loads((out/'EXECUTION_CONFIG.json').read_text());assert cfg['runtime_allowed'] and not cfg['training_allowed']
    (out/'bundles').mkdir()
    results=[];backend=None;error=None;budget=None;store=None;counts={'traces':0,'actual_actions':0,'collisions':0,'complete_traces':0}
    started=time.monotonic()
    try:
        with TrackedContentStore(out/'content',6*1024**3) as store,Journal(out/'journal',cfg) as journal:
            budget=BudgetLedger(cfg['budget'],persist=lambda s:journal.append('budget',s))
            ledger=FreezeLedger(persist=lambda s:journal.append('freeze',s))
            for row in cfg['candidates']:
                budget.check_time()
                folder=out/'bundles'/row['candidate_id'];folder.mkdir();(folder/'traces').mkdir()
                for p,h in row['assets'].items():assert sha(Path(p))==h,p
                backend=HabitatBackend(row['scene_glb'],cfg['gpu_device'],row['roles'],store,
                    dict(cfg,scene_glb=row['scene_glb']))
                save(folder/'SEMANTIC_INVENTORY.json',dict(objects=backend.objects,eligible=backend.eligible))
                assert backend.eligible==row['expected_eligible'],'RUNTIME_ROLE_IDENTITY'
                centers=[backend.objects[v[0]]['center'] for v in backend.eligible.values()]
                positions=[p for p in row['source_positions'] if (backend.snap_position(p) is not None and math.dist(p,backend.snap_position(p))<=1e-5)]
                positions=sorted(positions,key=lambda p:(sum(math.dist(p,c) for c in centers),p))[:4]
                configs=[dict(u_position=p,yaw_bin=y,public_tail='LRLRLRLR') for p in positions for y in (0,6,12,18)]
                save(folder/'FROZEN_GEOMETRY_CONFIGS.json',configs)
                compiler=Compiler(backend.compiler_roles,row['tasks'],backend.eligible)
                trace_index=0;retained=[];certifying=False
                def emit(kind,value):
                    nonlocal trace_index
                    if kind=='trace':
                        path=folder/'traces'/('%06d.json'%trace_index);save(path,value)
                        journal.append('trace_saved',dict(bundle=row['candidate_id'],index=trace_index,sha256=sha(path),complete=value['complete']))
                        trace_index+=1;counts['traces']+=1
                        counts['complete_traces']+=int(value['complete']);counts['collisions']+=value['collisions']
                        if certifying:retained.append(value)
                    else:
                        journal.append(kind,dict(bundle=row['candidate_id'],value=value))
                        if kind=='action_completed':counts['actual_actions']+=1
                    if kind=='trace':progress(out,dict(counts,current_bundle=row['candidate_id'],physical_families=sum(r.get('physical_certified',False) for r in results),training_ready_families=0))
                factory=FeedbackFactory(backend,compiler,budget,emit,'task_A','task_B',
                    context={'house_id':row['house_id'],'asset_config':row['assets']})
                result=dict(candidate_id=row['candidate_id'],house_id=row['house_id'],physical_certified=False,training_admission=False)
                try:
                    budget.start_bundle(row['candidate_id'],'discovery')
                    if not configs:raise Reject('NO_SOURCE_COORDINATE_ON_NAVMESH')
                    candidate=factory.discover(configs,ledger,row['candidate_id'])
                    save(folder/'FROZEN_CANDIDATE.json',candidate)
                    budget.finish_phase();budget.start_bundle(row['candidate_id'],'certification')
                    certifying=True;certificate=factory.replay_seeds(candidate);certifying=False
                    assert len(retained)==27
                    keys=[(h,c) for h in candidate['histories'] for c in candidate['continuations']]
                    grids=[{key:tr for key,tr in zip(keys,retained[i*9:(i+1)*9])} for i in range(3)]
                    export_family(folder/'export_v4',compiler,candidate,grids[0],out/'content',
                        family_id=row['candidate_id']+'_feedback_v1',split='FIT',
                        provenance={'kind':'actual_feedback_discovery','semantic_candidate_sha256':row['semantic_fingerprint_sha256']},
                        repeated_traces=grids[1:])
                    loader=FamilyLoader(folder/'export_v4',compiler)
                    readback=loader.validate_supervision_contract();save(folder/'READBACK.json',readback)
                    for prefix in loader.prefix_index:
                        for record in loader.prefix_records(prefix):loader.policy_payload(record)
                    for cell in loader.cells:
                        for record in loader.action_stream(cell['cell_id']):loader.policy_payload(record)
                    save(folder/'CERTIFICATE.json',certificate)
                    ledger.certify(row['candidate_id'],True)
                    result.update(status='PHYSICAL_REPLAY_CERTIFIED_PENDING_SHORTCUT_AUDIT',physical_certified=True,
                                  training_quality_pass=None,scientific_pass=False)
                except (Reject,BudgetExceeded) as exc:
                    certifying=False
                    result.update(status='RESOURCE_CENSORED' if isinstance(exc,BudgetExceeded) else 'REJECTED',reason=str(exc))
                    rec=ledger.records.get(row['candidate_id'])
                    if rec and rec['status']=='discovering':
                        ledger.reject_discovery(row['candidate_id'],str(exc),resource_censored=isinstance(exc,BudgetExceeded))
                    elif rec and rec['status']=='frozen':ledger.certify(row['candidate_id'],False,str(exc))
                finally:
                    if budget.snapshot()['active'] is not None:budget.finish_phase()
                result['trace_count']=trace_index;save(folder/'result.json',result);results.append(result)
                backend.close();backend=None
                progress(out,dict(counts,closed_bundles=len(results),physical_families=sum(r['physical_certified'] for r in results),training_ready_families=0))
            disk_audit=store.full_audit();save(out/'STORE_AUDIT.json',disk_audit)
            assert disk_audit['audit_pass'],'STORE_FINAL_AUDIT_FAILED'
            save(out/'BUDGET_FINAL.json',budget.snapshot());save(out/'FREEZE_LEDGER.json',ledger.records)
    except BaseException as exc:error=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend:backend.close()
        if budget and not (out/'BUDGET_FINAL.json').exists():save(out/'BUDGET_FINAL.json',budget.snapshot())
        if store is not None:
            save(out/'STORE_CLOSE_AUDIT.json',store.final_audit)
            if not store.final_audit or not store.final_audit.get('audit_pass'):
                error=error or {'error':'STORE_CLOSE_AUDIT_FAILED'}
    result=dict(node=cfg['node'],bundles=results,error=error,counts=counts,
                wall_seconds=time.monotonic()-started,new_physical_families=sum(r['physical_certified'] for r in results),
                new_train_ready_families=0,training_started=False,scientific_pass=False)
    save(out/'result.json',result);print(json.dumps(result),flush=True)
    if error:raise SystemExit(1)
if __name__=='__main__':main()
