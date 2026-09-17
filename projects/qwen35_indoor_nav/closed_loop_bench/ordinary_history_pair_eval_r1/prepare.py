"""Preseal development inputs now; bind one accepted final training arm only later."""
import ast
import copy
import json
from pathlib import Path
import runpy
import sys
import time
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
BASE=HERE.parent/'ordinary_expanded_dev_after_single_v1'
TRAIN=LINE/'sft_acceptance/ordinary_history8_paired_train_r1'
CASES={'control_recent2':'ordinary_history2_dev_r1','treatment_prefix8':'ordinary_history8_dev_r1'}
r=runpy.run_path(str(TRAIN/'runtime.py'))
sha=r['sha'];save=r['save']
def read(p):return json.loads(Path(p).read_text())
def freeze():
    test=read(HERE/'CPU_TEST_RESULT.json')
    assert test['passed'] and test['status']=='PASS_ACTUAL_WINDOW_AND_AUDIT_CPU'
    for path,digest in test['sources'].items():assert sha(Path(path))==digest,path
    orchestration=read(HERE/'CPU_ORCHESTRATION_RESULT.json')
    assert orchestration['status']=='PASS_ORCHESTRATION_CPU_ONLY'
    for path,digest in orchestration['sources'].items():assert sha(Path(path))==digest,path
    tp=read(TRAIN/'PROTOCOL.json')
    base=read(BASE/'PROTOCOL.json')
    assert base['episode_count']==100 and base['comparison_batch_size_forced']==1
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[HERE/'CPU_TEST_RESULT.json',HERE/'CPU_ORCHESTRATION_RESULT.json',TRAIN/'PROTOCOL.json',TRAIN/'MAIN_AGENT_APPROVAL.json',LINE/'reviews/Q35N_HISTORY_PAIR_R1_CHECKPOINTS/first200.py']
    for arm,name in CASES.items():
        case=HERE.parent/name
        assert not (case/'PENDING_SEAL.json').exists()
        for filename in ('EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json'):
            with (case/filename).open('xb') as stream:stream.write((BASE/filename).read_bytes())
            assert sha(case/filename)==sha(BASE/filename)
        protocol=copy.deepcopy(base)
        protocol.update(id='Q35N_HISTORY_PAIR_'+arm.upper()+'_DEV_R1',history_arm=arm,
            checkpoint=str(TRAIN/arm/'run_001/checkpoint_000004000.pt'),checkpoint_sha256=None,
            checkpoint_role='fixed_paired_'+arm+'_4000',checkpoint_updates=4000,
            training_protocol_sha256=sha(TRAIN/'PROTOCOL.json'),sample_index_sha256=tp['sample_index_sha256'],
            wall_seconds=7200,checkpoint_selection='fixed 4000, no intermediate selection',
            main_criterion='INTERNAL_DEV only: SR strict up, SPL nondecrease, nDTW delta >= -0.01',
            ordinary_only=True,special_data_used=False,controller_enabled=False,
            rgb_history='recent2' if arm=='control_recent2' else 'causal prefix-quantile8 left-black padding',
            diagnostic_forward_decisions=32,full_val_unseen_not_authorized_by_this_protocol=True)
        save(case/'PROTOCOL_TEMPLATE.json',protocol,True)
        save(case/'CPU_TEST_RESULT.json',dict(passed=True,shared_result_sha256=sha(HERE/'CPU_TEST_RESULT.json'),
            arm=arm,actual_window_audited=True,physics_and_metrics_unchanged=True,zero_gpu_actions=True),True)
        casepaths=list(case.glob('*.py'))+list(case.glob('*.json'))
        save(case/'PENDING_SEAL.json',dict(status='AWAIT_FIXED_PAIRED_4000_ACCEPTANCE',unix=time.time(),
            files={str(p):sha(p) for p in paths+casepaths}),True)
    save(HERE/'PREPARATION_RESULT.json',dict(status='PREFROZEN_BOTH_INTERNAL100',unix=time.time(),
        arms=CASES,checkpoints_not_yet_bound=True,training_updates=0,gpu_launches=0),True)
    print('PREFROZEN_TWO_INTERNAL100; NO GPU ACTIONS',flush=True)
def bind(arm):
    case=HERE.parent/CASES[arm]
    assert not (case/'PROTOCOL.json').exists() and not (case/'SOURCE_LOCK.json').exists()
    pending=read(case/'PENDING_SEAL.json')
    for path,digest in pending['files'].items():assert sha(Path(path))==digest,path
    tp=read(TRAIN/'PROTOCOL.json')
    final=read(TRAIN/arm/'run_001/RESULT.json')
    accepted=read(TRAIN/arm/'ACCEPTANCE.json')
    launch=read(TRAIN/arm/'LAUNCH_RESULT.json')
    assert final['status']=='COMPLETE' and final['updates']==4000 and final['global_decisions']==384000
    assert accepted['status']=='PASS_FINAL_TRAINING_ONLY' and accepted['updates']==4000 and accepted['arm']==arm
    assert launch['status']=='COMPLETE' and all(not x['pids'] for x in launch['gpu_after'])
    protocol=read(case/'PROTOCOL_TEMPLATE.json');checkpoint=Path(protocol['checkpoint'])
    receipt=read(Path(str(checkpoint)+'.json'))
    assert str(checkpoint)==final['latest_checkpoint']==accepted['checkpoint']
    assert sha(checkpoint)==receipt['sha256']==accepted['checkpoint_sha256']
    assert receipt['cursor']==dict(epoch=1,position=0,updates=4000,decisions=128000)
    assert receipt['global_decisions']==384000 and receipt['arm']==arm
    protocol['checkpoint_sha256']=receipt['sha256']
    files=dict(read(BASE/'SOURCE_LOCK.json')['files'])
    files.update(tp['code_sha256']);files.update(pending['files'])
    extra=[checkpoint,Path(str(checkpoint)+'.json'),TRAIN/arm/'ACCEPTANCE.json',
        TRAIN/arm/'LAUNCH_RESULT.json',TRAIN/arm/'run_001/RESULT.json',case/'PENDING_SEAL.json',
        HERE.parent/'ordinary_stop_calibration_fit_v2/reuse.py',
        HERE.parent/'ordinary_stop_calibration_fit_v2/tree_size.py',
        HERE.parent/'ordinary_stop_calibration_fit_v1/reuse.py']
    for path in extra:files[str(path)]=sha(path)
    for path,digest in files.items():assert sha(Path(path))==digest,path
    # Compare actual safe launcher pre-GPU prefix using in-memory protocol-independent source.
    src=runpy.run_path(str(case/'reuse.py'))['source']('launch.py')
    fn=next(x for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name=='main')
    prefix=[]
    for node in fn.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lease' for t in node.targets):break
        prefix.append(node)
    assert len(prefix)==3 and not (case/'run_001').exists()
    save(case/'PROTOCOL.json',protocol,True);files[str(case/'PROTOCOL.json')]=sha(case/'PROTOCOL.json')
    save(case/'SOURCE_LOCK.json',dict(files=files,scope='same full INTERNAL_DEV100, fixed paired4000 '+arm),True)
    namespace=runpy.run_path(str(case/'launch.py'),run_name='CPU_LAUNCHER_PREFIX')
    exec(compile(ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[])),'SAFE_LAUNCHER_PREFIX','exec'),namespace)
    save(case/'MAIN_REVIEW.json',dict(status='PASS_FOR_FIXED_PAIRED4000_INTERNAL100',unix=time.time(),
        source_lock_sha256=sha(case/'SOURCE_LOCK.json'),checkpoint_sha256=receipt['sha256'],
        arm=arm,automatic_retry=False,other_arm_training_may_continue=True,gpu=1,wall_seconds=7200,
        no_borrowed_holders=True,no_training_mutation=True,full1839_separate_gate=True),True)
    print('BOUND_FIXED_PAIRED4000:'+arm,flush=True)
if __name__=='__main__':
    if sys.argv[1:]==['freeze']:freeze()
    else:
        assert len(sys.argv)==3 and sys.argv[1]=='bind' and sys.argv[2] in CASES
        bind(sys.argv[2])
