"""Bounded cross-house real component collection; programs are proposals only."""
import collections
import json
from pathlib import Path
import sys
import time
import traceback

HERE=Path('/mnt/data_nas/deeprobotics/daiyang/vla/projects/qwen35_indoor_nav/data_pipeline/mechanism_runtime_v1/witness_first_v1/new_hub_scout_v1')
sys.path.insert(0,str(HERE))
from new_hub_scoped_common_v1 import (RUNTIME,ROOT,Compiler,compiler,digest,BudgetExceeded,Journal,durable_budget,
    PartialTraceRunner,HabitatBackend,bank,sha,save,store_class,registry,select_hubs,
    ScoutFeedbackRunner,compact_actions,runtime_config)

def main():
    out=HERE/'run_v1';cfg=runtime_config()
    assert json.loads((out/'EXECUTION_CONFIG.json').read_text())==cfg
    (out/'houses').mkdir();backend=None;store=None;budget=None;error=None;results=[]
    started=time.monotonic();counts=collections.Counter()
    def progress(extra=None):
        value=dict(counts,closed_houses=len(results),new_physical_families=0,new_train_ready_families=0,**(extra or {}))
        temp=out/'PROGRESS.pending'
        with temp.open('w') as f:json.dump(value,f,indent=2)
        temp.replace(out/'PROGRESS.json')
    try:
        with store_class()(out/'content',6*1024**3) as store,Journal(out/'journal',cfg) as journal:
            budget=durable_budget(journal,cfg['budget'],clock=time.monotonic)
            for candidate in cfg['candidates']:
                budget.check_time();house=candidate['house_id'];folder=out/'houses'/house
                folder.mkdir();(folder/'traces').mkdir();(folder/'partial_traces').mkdir()
                result=dict(house_id=house,status='STARTED',training_admission=False,physical_certified=False)
                house_start_actions=counts['actual_actions']
                trace_index=0;records=[];current={};all_hub_summaries=[]
                def emit(kind,value):
                    nonlocal trace_index
                    if kind=='trace':
                        complete=compiler.complete(value) and bool(value['actions']) and all(a in ('F','L','R') for a in value['actions'])
                        sub='traces' if complete else 'partial_traces'
                        path=folder/sub/f'{trace_index:06d}.json';save(path,value)
                        journal.append('trace_saved',dict(house_id=house,index=trace_index,path=str(path.relative_to(out)),
                            sha256=sha(path),bank_eligible_complete_motion=complete,context=current))
                        trace_index+=1;counts['saved_traces']+=1;counts['complete_motion_traces']+=int(complete)
                        counts['partial_or_zero_motion_traces']+=int(not complete);counts['trace_collisions']+=value['collisions']
                        if complete:
                            loop=bank.compat.closed_pose(value['observations'][0]['pose'],value['observations'][-1]['pose'])
                            record=dict(id=f'SCOUT:{house}:{trace_index-1:06d}',trace_ref=str(path.relative_to(ROOT)),
                                house_id=house,scene_fingerprint=digest(candidate['assets']),split='FIT',trace=value,
                                closed_loop=loop,roles=roles)
                            records.append(record)
                            with (folder/'BANK_RECORDS.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
                            counts['stored_closed_loops']+=int(loop)
                        progress(dict(current_house=house,current_context=current))
                    else:
                        journal.append(kind,dict(house_id=house,context=current,value=value))
                        if kind=='action_completed':counts['actual_actions']+=1
                try:
                    budget.start_bundle(house,'discovery')
                    for path,expected in candidate['assets'].items():assert sha(path)==expected,path
                    backend=HabitatBackend(candidate['scene_glb'],2,candidate['roles'],store,dict(cfg,scene_glb=candidate['scene_glb']))
                    assert backend.eligible==candidate['expected_eligible'],'ORIGINAL_PERMISSION_ROLE_IDENTITY'
                    save(folder/'SEMANTIC_INVENTORY.json',dict(objects=backend.objects,permission_roles=backend.eligible))
                    roles=registry(backend.objects);assert roles,'NO_VOCABULARY_GROUPS'
                    rolemap={bank.role_key(row['signature']):tuple(row['signature'][:2]) for row in roles}
                    eligible={bank.role_key(row['signature']):row['eligible_ids'] for row in roles}
                    first=next(iter(rolemap))
                    comp=Compiler(rolemap,{'probe':dict(anchor=first,terminal=first,instruction='Scout role witness collection')},eligible)
                    hubs,geometry=select_hubs(backend,roles,candidate['source_positions'],budget)
                    assert counts['actual_actions']==house_start_actions,'ACTION_BEFORE_CONFIG_FREEZE'
                    save(folder/'FROZEN_HUB_CONFIGS.json',dict(hubs=hubs,roles=roles,geometry=geometry,
                        saved_before_any_house_action=True))
                    before_actions=counts['actual_actions']
                    runner=PartialTraceRunner(backend,comp,budget,emit);feedback=ScoutFeedbackRunner(runner)
                    for hub_index,hub in enumerate(hubs):
                        records=[];current=dict(hub=hub_index,phase='public_tail')
                        runner.run(hub['position'],0,list(hub['public_tail']))
                        for group_index,group in enumerate(hub['groups']):
                            for target_index,target in enumerate(group['targets']):
                                budget.check_time()
                                current=dict(hub=hub_index,group=group_index,signature=group['role']['signature'],target=target_index,phase='outbound_feedback')
                                counts['outbound_attempts']+=1
                                outbound=feedback.navigate(hub['position'],0,target['target'],target['center'],max_actions=140)
                                if not comp.complete(outbound):
                                    journal.append('outbound_rejected',dict(house_id=house,context=current,reason=outbound.get('failure_reason'),complete=False));continue
                                counts['outbound_complete']+=1
                                actions=compact_actions(outbound['actions'])
                                if len(actions)>504:
                                    journal.append('compact_rejected',dict(house_id=house,context=current,reason='HISTORY_LENGTH',actions=len(actions)));continue
                                current=dict(current,phase='actual_compact_loop')
                                counts['compact_attempts']+=1
                                compact=runner.run(hub['position'],0,actions)
                                if comp.complete(compact) and bank.compat.closed_pose(compact['observations'][0]['pose'],compact['observations'][-1]['pose']):
                                    counts['compact_complete_and_closed']+=1
                        proposed=bank.propose(records,max_programs=32)
                        save(folder/f'HUB_{hub_index:02d}_PROGRAMS.json',proposed)
                        all_hub_summaries.append(dict(hub=hub_index,stored_records=len(records),proposed=len(proposed['proposals']),
                            enumerated_programs=proposed['enumerated_programs'],physical_families=0))
                        records=[]
                    result.update(status='SCOUT_COMPONENT_BANK_COMPLETE',hubs=all_hub_summaries,
                        actual_actions=counts['actual_actions']-before_actions,selected_hubs=len(hubs),trace_count=trace_index)
                except BudgetExceeded as exc:
                    # Preserve partial records; do not continue the censored trace or relabel it a failure.
                    if records:
                        proposed=bank.propose(records,max_programs=32)
                        save(folder/'CENSORED_HUB_PROGRAMS.json',proposed)
                    result.update(status='RESOURCE_CENSORED',reason=str(exc),trace_count=trace_index,hubs=all_hub_summaries,
                        actual_actions=counts['actual_actions']-house_start_actions)
                except BaseException as exc:
                    # Unknown backend/observation failures are not negative labels.
                    journal.append('scout_unknown_failure',dict(house_id=house,context=current,error=repr(exc),
                        training_admission=False,actual_actions_so_far=counts['actual_actions']-house_start_actions))
                    raise
                finally:
                    if budget.snapshot()['active'] is not None:budget.finish_phase()
                    if backend:backend.close();backend=None
                save(folder/'result.json',result);results.append(result);progress()
            save(out/'BUDGET_FINAL.json',budget.snapshot())
        assert store.final_audit and store.final_audit['audit_pass'],'FINAL_STORE_AUDIT'
    except BaseException as exc:error=dict(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if backend:backend.close()
        if store is not None:save(out/'STORE_CLOSE_AUDIT.json',store.final_audit)
        if budget is not None and not (out/'BUDGET_FINAL.json').exists():save(out/'BUDGET_FINAL.json',budget.snapshot())
    result=dict(status='SCOUT_CLOSED' if error is None else 'SCOUT_ENGINEERING_FAILURE',houses=results,error=error,
        counts=dict(counts),wall_seconds=time.monotonic()-started,new_physical_families=0,new_train_ready_families=0,
        rates=dict(outbound_not_complete_including_censored=(1-counts['outbound_complete']/counts['outbound_attempts']) if counts['outbound_attempts'] else None,
            compact_not_complete_or_not_closed_including_censored=(1-counts['compact_complete_and_closed']/counts['compact_attempts']) if counts['compact_attempts'] else None),
        training_started=False,scientific_pass=False)
    save(out/'result.json',result);print(json.dumps(result),flush=True)
    if error:raise SystemExit(1)

if __name__=='__main__':main()
