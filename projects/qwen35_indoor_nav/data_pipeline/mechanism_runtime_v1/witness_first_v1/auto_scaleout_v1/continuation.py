"""Explicit main-authorized migration of queued, never-started programs only."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('scaleout_continuation_planner',HERE/'planner.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
MAPPINGS=[(1,old,old+17) for old in range(203,212)]+[(2,old,old+136) for old in range(104,112)]
AUTH=p.WF.parents[2]/'authorizations/DATA_PRODUCTION_SCALEOUT_V1.json'
def check_unstarted(run):
    assert run.resolve()==run and run.is_dir()
    assert {x.name for x in run.iterdir()}=={'EXECUTION_CONFIG.json','INPUT_LOCK.json'},'OLD_RUN_NOT_PROVEN_NEVER_STARTED'
    cfg=p.read(run/'EXECUTION_CONFIG.json');lock=p.read(run/'INPUT_LOCK.json')
    assert lock[str(run/'EXECUTION_CONFIG.json')]==p.sha(run/'EXECUTION_CONFIG.json')
    return cfg,lock
def freeze():
    out=HERE/'continuation_queue_v1';out.mkdir(exist_ok=False)
    assert AUTH.is_file(),'MAIN_SCALEOUT_AUTHORIZATION_REQUIRED'
    bindings={str(AUTH):p.sha(AUTH)};batches=[];checks=[]
    for gpu,old_number,new_number in MAPPINGS:
        assert old_number not in (103,202)
        old=p.BE/('batch_'+str(old_number))/'run_v1';cfg,source_lock=check_unstarted(old)
        assert cfg['gpu_device']==gpu and len(cfg['candidates'])==3
        snapshot=Path(cfg['source_snapshot']);draft=p.read(snapshot/'CONFIG_DRAFT.json')
        original_rows=[draft['candidates'][i] for i in cfg['source_selection_indices']]
        for a,b in zip(original_rows,cfg['candidates']):
            aa=json.loads(json.dumps(a));bb=json.loads(json.dumps(b))
            for r in (aa,bb):r.pop('runtime_allowed',None);r.pop('executable',None)
            assert aa==bb,'MIGRATION_MUST_PRESERVE_ENTIRE_ORIGINAL_CANDIDATE'
        assert len(original_rows)==3 and cfg['source_selection_indices']==[0,1,2]
        new=p.BE/('batch_'+str(new_number));assert not new.exists(),'NEW_OUTPUT_MUST_NOT_EXIST'
        for path in (old/'EXECUTION_CONFIG.json',old/'INPUT_LOCK.json',snapshot/'CONFIG_DRAFT.json',snapshot/'SOURCE_LOCK.json'):
            bindings[str(path)]=p.sha(path)
        checks.append({'old_run_root':str(old),'observed_filenames':['EXECUTION_CONFIG.json','INPUT_LOCK.json'],
            'status':'NEVER_STARTED_AT_FREEZE_RUNTIME_MUST_RECHECK','input_lock_sha256':p.sha(old/'INPUT_LOCK.json')})
        batches.append({'id':new.name,'gpu':gpu,'prepared_snapshot':str(snapshot),'source_snapshot':str(snapshot),
            'source_indices':[0,1,2],'candidate_ids':[r['candidate_id'] for r in original_rows],
            'previous_run_root':str(old),'runtime_recheck_unstarted_required':True,
            'unchanged_original_candidate_and_language':True,'retry_of_attempted_candidate':False})
    terminal_proofs=[]
    for name in ('batch_202','batch_103'):
        run=p.BE/name/'run_v1';terminal=p.read(run/'SUPERVISOR_RESULT.json')
        assert terminal['cleanup_complete'] is True,'PREVIOUS_LANE_NOT_CLEANLY_TERMINAL'
        proof={'run_root':str(run),'supervisor_result':terminal}
        for path in (run/'SUPERVISOR_RESULT.json',run/'LAUNCH_RESULT.json'):
            bindings[str(path)]=p.sha(path)
        terminal_proofs.append(proof)
    queue={'schema_version':'q35n.explicit_unstarted_migration.v1','batches':batches,'batch_count':len(batches),
        'candidate_count':3*len(batches),'gpu_devices':[1,2],'each_lane_wall_seconds':43200,
        'authorization_path':str(AUTH),'failed_partial_103_202_excluded':True,'runtime_allowed':False,'scientific_pass':False}
    p.save(out/'QUEUE.json',queue);p.save(out/'UNSTARTED_EVIDENCE.json',checks)
    p.save(out/'PREVIOUS_LANE_TERMINALS.json',terminal_proofs)
    for gpu in (1,2):
        selected=[b for b in batches if b['gpu']==gpu]
        path=out/('GPU_'+str(gpu)+'_QUEUE.json')
        p.save(path,dict(queue,batches=selected,batch_count=len(selected),candidate_count=3*len(selected),gpu_devices=[gpu]))
        bindings[str(path)]=p.sha(path)
    for path in [*HERE.glob('*.py'),HERE/'SPEC_ZH.md',HERE/'SHA256SUMS',out/'QUEUE.json',out/'UNSTARTED_EVIDENCE.json',out/'PREVIOUS_LANE_TERMINALS.json',p.ORIGINAL]:
        bindings[str(path)]=p.sha(path)
    p.save(out/'SOURCE_LOCK.json',bindings)
    print(json.dumps({'migration_batches':17,'unchanged_candidates':51,'gpu1_batches':9,'gpu2_batches':8,
        'no_physical_retry':True,'gpu_operations':0},indent=2))
if __name__=='__main__':freeze()
