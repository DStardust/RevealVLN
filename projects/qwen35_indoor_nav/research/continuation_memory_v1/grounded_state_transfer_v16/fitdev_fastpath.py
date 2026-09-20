"""Authorized FIT16 training and DEV2 diagnostic path; never collects or scores TEST."""
import argparse
import collections
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
FITDEV_SCOPE='FIT16_TRAIN_DEV2_DIAGNOSTIC'


def load_json(path):
    return json.loads(Path(path).read_text())


def process_rss_kib(pgid):
    result=subprocess.run(['ps','-eo','pgid=,rss='],text=True,capture_output=True,check=True)
    return sum(int(row.split()[1]) for row in result.stdout.splitlines() if row.split() and int(row.split()[0])==pgid)


def gpu_snapshot(index):
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
    rows=[]
    for line in result.stdout.splitlines():
        fields=[x.strip() for x in line.split(',')]
        rows.append(dict(index=int(fields[0]),uuid=fields[1],name=fields[2],memory_total_mib=int(fields[3]),
            memory_used_mib=int(fields[4]),memory_free_mib=int(fields[5]),utilization_percent=int(fields[6])))
    return next(row for row in rows if row['index']==index)


def local_gpu_hours(run,c):
    path=run/'RESOURCE_SESSIONS.jsonl'
    return sum(row['wall_seconds'] for row in c.records(path))/3600 if path.exists() else 0.0


def execute(run,name,argv,config,c,gpu):
    attempts=run/'attempts';attempts.mkdir(exist_ok=True)
    number=len(list(attempts.glob(name+'_*')))+1;folder=attempts/f'{name}_{number:03d}';folder.mkdir()
    env=os.environ.copy();env['V16_ASSET_LINE_ROOT']=config['asset_line_root'];env['PYTHONUNBUFFERED']='1'
    if gpu:
        env['CUDA_VISIBLE_DEVICES']=str(config['gpu']);env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
        env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
    else:env['CUDA_VISIBLE_DEVICES']=''
    before=gpu_snapshot(config['gpu']) if gpu else None
    c.write(folder/'COMMAND.json',dict(argv=list(map(str,argv)),gpu=before,cwd=str(ROOT),started_unix=time.time()),True)
    began=time.monotonic();peak=0;code=None;timed_out=False
    with (folder/'stdout.log').open('x') as output:
        proc=subprocess.Popen(list(map(str,argv)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            peak=max(peak,process_rss_kib(proc.pid));elapsed=time.monotonic()-began
            c.write(folder/'RESOURCES.json',dict(elapsed_seconds=elapsed,peak_process_group_rss_kib=peak,unix=time.time()))
            if elapsed>config['max_session_hours']*3600+300:
                timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=90)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=30)
                break
            time.sleep(10)
        code=proc.wait()
    row=dict(stage=name,attempt=number,wall_seconds=time.monotonic()-began,returncode=code,timed_out=timed_out,
        peak_process_group_rss_kib=peak,gpu_session=gpu,gpu_before=before,gpu_after=gpu_snapshot(config['gpu']) if gpu else None)
    c.write(folder/'EXIT.json',row,True)
    if gpu:c.append(run/'RESOURCE_SESSIONS.jsonl',row)
    if code or timed_out:
        tail=(folder/'stdout.log').read_text(errors='replace')[-12000:]
        raise RuntimeError(f'STAGE_PROCESS_FAILED:{name}:{number}:{code}:{tail}')
    return row


def run_until_complete(run,name,script,complete,config,c):
    infra_failures=0
    while not complete():
        if local_gpu_hours(run,c)>=config['gpu_session_hours']:raise RuntimeError('RESOURCE_TOTAL_BUDGET')
        try:execute(run,name,[config['torch_python'],'-I','-B',HERE/script,run],config,c,True)
        except RuntimeError as error:
            transient=any(key in str(error) for key in ('CUDA out of memory','SIMULATOR_EOF','ConnectionResetError'))
            if not transient or infra_failures>=config['infra_retries']:raise
            infra_failures+=1;c.append(run/'INFRA_RETRIES.jsonl',dict(stage=name,retry=infra_failures,error=repr(error),score_based=False,unix=time.time()))


def prepare_inputs(run,config,c,digest):
    source=Path(config['source_run']).resolve();families=[];files=[]
    for path in sorted(source.glob('HOUSE_*.json')):
        row=c.read(path);files.append(dict(path=str(path),sha256=c.sha(path),house=row['house'],families=len(row['families'])))
        families.extend(row['families'])
        selected=[family for family in row['families'] if family['split'] in ('FIT','DEV')]
        if selected:c.immutable(run/path.name,dict(row,families=selected))
    by_split=collections.Counter(f['split'] for f in families)
    expected={k:config['inventory_counts'][k] for k in ('FIT','DEV','TEST')}
    if dict(by_split)!=expected:raise ValueError('CERTIFIED_INVENTORY_CHANGED:'+str(by_split))
    fit=sorted(f['family_id'] for f in families if f['split']=='FIT');dev=sorted(f['family_id'] for f in families if f['split']=='DEV')
    test=sorted(f['family_id'] for f in families if f['split']=='TEST')
    if fit!=sorted(config['authorized_training_families']) or dev!=sorted(config['authorized_diagnostic_families']) or test!=sorted(config['frozen_test_families']):
        raise ValueError('FROZEN_FAMILY_IDENTITY_CHANGED')
    reused=[dict(family_id=f['family_id'],house=f['house'],split=f['split'],canonical_sha256=digest(f),
        source_house_record=next(row['path'] for row in files if row['house']==f['house']),recollected=False,newly_certified=False) for f in families if f['split'] in ('FIT','DEV')]
    c.immutable(run/'INPUT_REUSE.json',dict(scope=FITDEV_SCOPE,source_run=str(source),source_house_files=files,
        project_inventory=dict(FIT=fit,DEV=dev,TEST=test,missing_TEST_slots=2),consumed_families=reused,
        consumed_counts=dict(FIT=len(fit),DEV=len(dev),TEST=0),family_values_preserved=True,new_collection=0))
    c.immutable(run/'OFFICIAL_TEST_INVENTORY.json',dict(status='PRESERVED_NOT_SCORED',certified_family_ids=test,
        missing_slots=2,planned_families=8,planned_conditions=80,planned_rollouts=720,method_scores_read=False))


def preflight(run,config,c,lock):
    paths=[Path(config[k]).resolve() for k in ('asset_line_root','source_run','torch_python','sim_python','standalone_python','checkpoint')]
    if not all(p.exists() for p in paths):raise FileNotFoundError([str(p) for p in paths if not p.exists()])
    if c.sha(config['checkpoint'])!=config['checkpoint_sha256']:raise ValueError('WRONG_CHECKPOINT')
    if c.sha(HERE.parents[2]/'sft_acceptance/ordinary_sync_recovery_v1/model.py')!=config['model_source_sha256']:raise ValueError('WRONG_MODEL_IMPLEMENTATION')
    gpu=gpu_snapshot(config['gpu'])
    if gpu['uuid']!=config['gpu_uuid']:raise ValueError('GPU_INDEX_UUID_MISMATCH')
    if gpu['memory_free_mib']<config['min_free_gpu_gib']*1024:raise RuntimeError('GPU_MEMORY_BUDGET')
    head=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
    c.immutable(run/'PREFLIGHT.json',dict(status='FITDEV_INPUTS_AND_RUNTIME_VERIFIED',worktree=str(ROOT),execution_head=head,
        reviewed_commit=config['reviewed_commit'],asset_line_root=config['asset_line_root'],paths=list(map(str,paths)),gpu=gpu,
        source_lock_sha256=c.sha(run/'SOURCE_LOCK.json'),qwen_loaded=False,navigation_started=False,new_collection_started=False))


def train_summary(run,c):
    rows=[]
    for path in sorted((run/'train').glob('*_*/RESULT.json')):
        result=c.read(path);logs=[]
        for log in sorted(path.parent.glob('attempt_*/STEPS.jsonl')):logs.extend(c.records(log))
        result=dict(result,logged_updates=len(logs),first_loss=logs[0]['loss'] if logs else None,last_loss=logs[-1]['loss'] if logs else None,
            finite_losses=all(__import__('math').isfinite(row['loss']) for row in logs),positive_gradient_steps=sum(row['gradient_norm']>0 for row in logs))
        rows.append(result)
    if len(rows)!=9 or any(row['updates']!=1200 or row['logged_updates']<1200 for row in rows):raise ValueError('NINE_MODEL_COMPLETION')
    initial={seed:{row['initial_state_sha256'] for row in rows if row['seed']==seed} for seed in config_seeds(run,c)}
    if any(len(values)!=1 for values in initial.values()):raise ValueError('SAME_SEED_INITIALIZATION_MISMATCH')
    value=dict(models=rows,completed_models=9,planned_models=9,updates_per_model=1200,total_scheduled_updates=10800,
        base_optimizer_updates=0,same_seed_three_arm_initialization={str(k):next(iter(v)) for k,v in initial.items()},test_scores_read=False)
    c.immutable(run/'TRAIN_SUMMARY.json',value);return value


def config_seeds(run,c):
    return c.read(run/'PROTOCOL.json')['seeds']


def rollout_summary(run,c,reg,admitted):
    rows=[]
    for slot in reg['slots']:
        condition=reg['conditions'][slot['condition']];session=admitted.get(slot['condition']);status='NOT_RUN';label='UNKNOWN';error=''
        if session:
            path=session/'rollouts'/f"{slot['rank']:04d}"/'ROLLOUT.json'
            if path.exists():
                record=c.read(path);label=record['task_result'].get('safe_v16_label','UNKNOWN');status=label if label in ('PASS','FAIL') else 'UNKNOWN'
            else:status='ERROR';error='SEALED_GROUP_MISSING_ROLLOUT'
        rows.append(dict(rank=slot['rank'],condition=slot['condition'],family_id=condition['family_id'],house=condition['house'],
            history_id=condition['history_id'],task_id=condition['task_id'],endpoint=condition['endpoint'],model=slot['model'],arm=slot['arm'],seed=slot['seed'],status=status,label=label,error=error))
    csv_path=run/'DEV_ROLLOUTS.csv';fields=list(rows[0])
    temporary=csv_path.with_name(csv_path.name+'.tmp.'+str(os.getpid()))
    with temporary.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows);stream.flush();os.fsync(stream.fileno())
    if csv_path.exists():
        if csv_path.read_bytes()!=temporary.read_bytes():raise ValueError('DEV_ROLLOUT_SUMMARY_CHANGED')
        temporary.unlink()
    else:os.link(temporary,csv_path);temporary.unlink()
    counts=collections.Counter((row['endpoint'],row['status']) for row in rows)
    by_arm={arm:dict(collections.Counter(row['status'] for row in rows if row['arm']==arm)) for arm in ('B1','B2','Ours')}
    value=dict(scope='DEV_DIAGNOSTIC',families=2,conditions=20,planned_slots=180,completed_slots=sum(row['status']!='NOT_RUN' for row in rows),
        main_planned=144,control_planned=36,status_counts=[dict(endpoint=k[0],status=k[1],count=v) for k,v in sorted(counts.items())],
        by_arm=by_arm,unknown_error_not_run=[row for row in rows if row['status'] in ('UNKNOWN','ERROR','NOT_RUN')],
        independent_house_count=1,interpretation='DEV development evidence; 180 rollouts are not 180 independent houses and are not formal TEST.')
    c.immutable(run/'DEV_DIAGNOSIS.json',value);return rows,value


def seal(run,c,train,dev):
    report=(f"# V16 FIT/DEV 训练与诊断\n\n九个固定模型均完成 1200 次更新；底模更新为 0。\n\n"
        f"DEV 注册表为 20 条条件、180 个槽位，其中主任务 144、task_T 控制 36；已完成 {dev['completed_slots']} 个槽位。\n\n"
        "本节点未采集新物理候选、未接纳新族、未启动正式 TEST。DEV 结果仅作开发诊断。\n")
    path=run/'REPORT_ZH.md';payload=report.encode();temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('REPORT_CHANGED')
    else:
        with temporary.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path);temporary.unlink()
    selected=[run/name for name in ('PROTOCOL.json','SOURCE_LOCK.json','REVIEW_RECEIPT.json','INPUT_REUSE.json','RAW_DATA_AUDIT.json',
        'BASE_TRAINING_SCENE_AUDIT.json','EVALUATION_REGISTRY.json','TRAIN_SUMMARY.json','DEV_ROLLOUTS.csv','DEV_DIAGNOSIS.json',
        'EVENT_STATE_DIAGNOSIS.json','QUERY_DIAGNOSIS.json','MEMORY_INTERVENTIONS.json','FAILURE_LOCALIZATION.json','REPORT_ZH.md')]
    selected.extend(sorted((run/'train').glob('*_*/RESULT.json')));selected.extend(sorted((run/'train').glob('*_*/FINAL.pt')))
    selected.append(run/'features/FEATURE_RESULT.json')
    manifest=[dict(path=str(path.relative_to(run)),bytes=path.stat().st_size,sha256=c.sha(path)) for path in selected]
    c.immutable(run/'OUTPUT_MANIFEST.json',dict(version='v16_fitdev_training_unblock_v1',files=manifest))
    c.write(run/'STATUS.json',dict(status='FITDEV_TRAIN_AND_DEV_DIAGNOSTICS_COMPLETE',models='9/9',updates_per_model=1200,
        DEV_main='144/144 planned',DEV_control='36/36 planned',DEV_completed=dev['completed_slots'],
        official_TEST_evaluations_started_by_this_run=0,new_candidate_collections_started_by_this_run=0,
        new_family_admissions=0,gpu_session_hours=local_gpu_hours(run,c),finished_unix=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--run-id',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=load_json(args.config)
    if args.run_id!=config['run_id'] or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',args.run_id):raise ValueError('RUN_ID')
    os.environ['V16_ASSET_LINE_ROOT']=config['asset_line_root'];sys.path.insert(0,str(HERE))
    import v16_common as common
    from build_data import build
    from evaluate_continuations import registry,admitted
    run=HERE/'runs'/args.run_id
    if run.exists() and not args.resume:raise FileExistsError('USE_RESUME_FOR_EXISTING_RUN')
    run.mkdir(parents=True,exist_ok=True)
    with (run/'RUN.lock').open('a') as lock_file:
        fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            common.immutable(run/'PROTOCOL.json',config)
            source=common.source_lock();source.update(reviewed_commit=config['reviewed_commit'],asset_line_root=config['asset_line_root'])
            common.immutable(run/'SOURCE_LOCK.json',source);common.verify_lock(source)
            common.immutable(run/'REVIEW_RECEIPT.json',dict(reviewed_commit=config['reviewed_commit'],
                offline_errata_review='ACCEPTED_FOR_ENGINEERING_PROGRESSION',cpu_fixes_review='ACCEPTED_WITHIN_REPORTED_AND_INSPECTED_COVERAGE',
                authorized_new_run=args.run_id,authorized_scope=FITDEV_SCOPE,qwen_feature_loading_allowed=True,lightweight_training_allowed=True,
                authorized_final_models=9,updates_per_model=1200,authorized_dev_rollout_slots=180,new_candidate_collection_allowed=False,
                official_test_evaluation_allowed=False,modify_legacy_runs_allowed=False,per_stage_reapproval_required=False,actual_execution_started_by_reviewer=False))
            prepare_inputs(run,config,common,common.digest);preflight(run,config,common,source)
            if not (run/'FITDEV_SCOPE_TEST_RESULT.json').exists():
                result=execute(run,'scope_tests',[config['torch_python'],'-I','-B',HERE/'test_fitdev_scope.py'],config,common,False)
                common.immutable(run/'FITDEV_SCOPE_TEST_RESULT.json',dict(status='PASS',returncode=result['returncode']))
            if not (run/'DATA.json').exists():build(run,scope=FITDEV_SCOPE)
            if not (run/'RAW_DATA_AUDIT.json').exists():execute(run,'raw_audit',[config['torch_python'],'-I','-B',HERE/'audit_data.py',run],config,common,False)
            reg=registry(run)
            run_until_complete(run,'features','extract_features.py',lambda:(run/'features/FEATURE_RESULT.json').exists(),config,common)
            run_until_complete(run,'train','train.py',lambda:(run/'train/RESULT.json').exists(),config,common)
            train=train_summary(run,common)
            common.immutable(run/'FINAL_MODEL_MANIFEST.json',dict(models=[dict(tag=row['tag'],checkpoint=str((run/'train'/row['tag']/'FINAL.pt').relative_to(run)),
                checkpoint_sha256=row['checkpoint_sha256'],updates=row['updates']) for row in train['models']],frozen_before_dev=True,test_scores_read=False))
            run_until_complete(run,'evaluate','evaluate_continuations.py',lambda:len(admitted(run,registry(run)))==len(reg['conditions']),config,common)
            groups=admitted(run,reg)
            if not (run/'EVENT_STATE_DIAGNOSIS.json').exists():execute(run,'diagnose',[config['torch_python'],'-I','-B',HERE/'diagnose.py',run],config,common,False)
            _,dev=rollout_summary(run,common,reg,groups);seal(run,common,train,dev)
        except BaseException as error:
            text=repr(error);status='BLOCKED_RESOURCE' if any(x in text for x in ('RESOURCE_','GPU_MEMORY','No space left','CUDA out of memory')) else 'ERROR'
            common.append(run/'FAILURES.jsonl',dict(status=status,error=text,traceback=traceback.format_exc(),unix=time.time()))
            common.write(run/'STATUS.json',dict(status=status,error=text,gpu_session_hours=local_gpu_hours(run,common),unix=time.time()))
            raise


if __name__=='__main__':main()
