"""Fresh bounded physical run. Reuses sealed compiler/exporter/runner read-only."""
import json
import sys
import time
import traceback

from common import HERE, OUT, RUNTIME, WF, load, save, sha, verify_inputs
sys.path.insert(0, str(RUNTIME))
from core_bridge import Compiler, FreezeLedger, Reject, BudgetExceeded
from habitat_backend import HabitatBackend
from exporter import export_family
from loader import FamilyLoader
from method import DiversityFactory


def main():
    cfg = verify_inputs()
    assert not any((OUT/p).exists() for p in ('content','bundles','journal','result.json'))
    adapter = load('diversity_runtime_adapter',WF/'budget_and_trace/adapters.py')
    clock = load('diversity_clock_ledger',WF/'budget_clock_batching_cpu_v1/ledger.py')
    stores = load('diversity_scoped_content_store',RUNTIME/'feedback_generation_v1/store.py')
    stores.FEEDBACK_ROOT = HERE
    (OUT/'bundles').mkdir()
    results = []
    counts = dict(traces=0,actual_actions=0,collisions=0,complete_traces=0)
    backend = store = budget = ledger = None
    error = None
    started = time.monotonic()
    def progress(**extra):
        value=dict(counts,physical_families=sum(x.get('physical_certified',False) for x in results),
                   training_ready_families=0,unix=time.time(),**extra)
        temporary=OUT/'PROGRESS.pending';temporary.write_text(json.dumps(value));temporary.replace(OUT/'PROGRESS.json')
    try:
        with stores.TrackedContentStore(OUT/'content',6*1024**3) as store, adapter.Journal(OUT/'journal',cfg) as journal:
            budget=clock.ClockBatchingBudgetLedger(cfg['budget'],clock=time.monotonic,
                                                  persist=lambda s:journal.append('budget',s))
            ledger=FreezeLedger(persist=lambda s:journal.append('freeze',s))
            for row in cfg['candidates']:
                aid=row['candidate_id'];folder=OUT/'bundles'/aid;folder.mkdir();(folder/'traces').mkdir()
                retained=[];index=0;certifying=False
                result=dict(candidate_id=aid,house_id=row['house_id'],physical_certified=False,
                            training_admission=False,status='NOT_STARTED')
                try:
                    budget.start_bundle(aid,'discovery')
                    backend=HabitatBackend(row['scene_glb'],cfg['gpu_device'],row['roles'],store,dict(cfg,scene_glb=row['scene_glb']))
                    save(folder/'SEMANTIC_INVENTORY.json',dict(objects=backend.objects,eligible=backend.eligible))
                    assert backend.eligible==row['expected_eligible']
                    compiler=Compiler(backend.compiler_roles,row['tasks'],backend.eligible)
                    def emit(kind,value):
                        nonlocal index
                        if kind=='trace':
                            path=folder/'traces'/('%06d.json'%index);save(path,value)
                            journal.append('trace_saved',dict(bundle=aid,index=index,sha256=sha(path),complete=value['complete']))
                            index+=1;counts['traces']+=1;counts['complete_traces']+=int(value['complete']);counts['collisions']+=value['collisions']
                            if certifying and value['complete']:retained.append(value)
                            progress(current_bundle=aid,phase='certification' if certifying else 'discovery')
                        else:
                            journal.append(kind,dict(bundle=aid,value=value))
                            if kind=='action_completed':counts['actual_actions']+=1
                    factory=DiversityFactory(backend,compiler,budget,emit,'task_A','task_B',
                        context=dict(house_id=row['house_id'],asset_config=row['assets']),source=row['source_candidate'])
                    factory.runner=adapter.PartialTraceRunner(backend,compiler,budget,emit)
                    candidate=factory.discover(row['configurations'],ledger,aid)
                    save(folder/'FROZEN_CANDIDATE.json',candidate)
                    budget.finish_phase();budget.start_bundle(aid,'certification');certifying=True
                    certificate=factory.replay_seeds(candidate);certifying=False
                    assert len(retained)==27
                    keys=[(h,c) for h in candidate['histories'] for c in candidate['continuations']]
                    grids=[dict(zip(keys,retained[i*9:(i+1)*9])) for i in range(3)]
                    export_family(folder/'export_v4',compiler,candidate,grids[0],OUT/'content',family_id=aid,
                                  split='candidate_fit_pool',repeated_traces=grids[1:],
                                  provenance=dict(kind='diversity_v1_real_forward_common_tail',
                                                  source_family=row['source_family_id'],source_manifest=row['source_manifest'],
                                                  source_manifest_sha256=row['source_manifest_sha256'],
                                                  language_variant=row['language_variant'],new_house=False))
                    loader=FamilyLoader(folder/'export_v4',compiler)
                    save(folder/'READBACK.json',loader.validate_supervision_contract())
                    save(folder/'CERTIFICATE.json',certificate)
                    ledger.certify(aid,True)
                    result.update(status='PHYSICAL_CERTIFIED_PENDING_STRONG_AUDIT',physical_certified=True)
                except (Reject,BudgetExceeded) as exc:
                    certifying=False
                    result.update(status='RESOURCE_CENSORED' if isinstance(exc,BudgetExceeded) else 'REJECTED',reason=str(exc))
                    state=ledger.records.get(aid,{})
                    if state.get('status')=='discovering':ledger.reject_discovery(aid,str(exc),resource_censored=isinstance(exc,BudgetExceeded))
                    elif state.get('status')=='frozen':ledger.certify(aid,False,str(exc))
                finally:
                    if backend is not None:backend.close();backend=None
                    if budget.snapshot()['active'] is not None:budget.finish_phase()
                save(folder/'result.json',result);results.append(result);progress(last_result=result)
            save(OUT/'BUDGET_FINAL.json',budget.snapshot());save(OUT/'FREEZE_LEDGER.json',ledger.records)
        assert store.final_audit and store.final_audit['audit_pass'],'STORE_CLOSED_AUDIT'
    except BaseException as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend is not None:backend.close()
        if store is not None:save(OUT/'STORE_CLOSE_AUDIT.json',store.final_audit)
    result=dict(node=cfg['node'],bundles=results,error=error,counts=counts,wall_seconds=time.monotonic()-started,
                new_physical_families=sum(r['physical_certified'] for r in results),
                training_started=False,scientific_pass=False)
    save(OUT/'result.json',result);print(json.dumps(result),flush=True)
    if error:raise SystemExit(1)


if __name__=='__main__': main()
