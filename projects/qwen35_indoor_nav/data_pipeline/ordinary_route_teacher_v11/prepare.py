"""Freeze new route teacher only; no training or new policy rollout."""
import copy,importlib.util,json,os,runpy,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_recovery_v2'
FIT=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
def read(p):return json.loads(Path(p).read_text())
def main():
    assert not (HERE/'PROTOCOL.json').exists()
    c=runpy.run_path(str(HERE/'common.py'));sha=c['sha'];save=c['write']
    py=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
    tests=[]
    for name in ('tests.py','test_route.py'):
        r=subprocess.run([str(py),'-I','-S','-B',str(HERE/name)],cwd=ROOT,capture_output=True,text=True,timeout=60)
        tests.append(dict(name=name,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        assert r.returncode==0,r.stderr
    launch=runpy.run_path(str(HERE/'launch.py'),run_name='CPU_IMPORT_ONLY')
    assert launch['OUT']==HERE/'run_001' and callable(launch['_scan'].tree_size)
    assert not (HERE/'run_001').exists()
    previous=read(LINE/'reviews/Q35N_ORDINARY_ACTION_ALIGNED_HISTORY_V10_POSTAUDIT/FINAL_RESULT.json')
    assert previous['status']=='PASS' and not previous['navigation']['positive_development_signal']
    assert previous['navigation']['evaluation_owned_processes_cleaned']
    for p,d in read(OLD/'SOURCE_LOCK.json')['files'].items():assert sha(p)==d,p
    assert read(OLD/'run_001/RESULT.json')['data_gate']
    split=read(LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001/SPLIT.json')
    p=copy.deepcopy(read(OLD/'PROTOCOL.json'))
    assert set(p['houses'])<=set(split['FIT']) and set(p['houses']).isdisjoint(split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
    p.update(id='Q35N_ORDINARY_ROUTE_TEACHER_DATA_V11',teacher_target='next uncompleted ordered reference waypoint, then final goal',
      teacher_stop_distance_at_most=.35,teacher_stop_requires_all_ordered_waypoints=True,
      ordered_waypoint_reach_distance_at_most=.35,route_progress_source='only original actual pre-action history',
      original_policy_input_window='last2_RGB_last8_executed',policy_inputs_changed=False,
      original_source_rollouts_reused=True,new_independent_routes=0,automatic_training=False,training_allowed=False,
      automatic_retry=False,wall_seconds=1200,explicit_primitive_budget=32000,gpu_memory_gib=8,cpu_memory_gib=32,output_gib=2)
    p.pop('teacher_stop_distance_strictly_less_than',None)
    save(HERE/'PROTOCOL.json',p,True)
    save(HERE/'CPU_TEST_RESULT.json',dict(passed=True,tests=tests,actual_launcher_import_passed=True,gpu_actions=0),True)
    files=dict(read(OLD/'SOURCE_LOCK.json')['files'])
    paths=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.json',HERE/'CPU_TEST_RESULT.json',HERE/'PLAN_ZH.md',
      OLD/'SOURCE_LOCK.json',OLD/'run_001/DATA_SEAL.json',OLD/'run_001/RESULT.json',
      LINE/'reviews/Q35N_TEACHER_ROUTE_ALIGNMENT_READONLY_V1/RESULT.json',
      LINE/'reviews/Q35N_ORDINARY_ACTION_ALIGNED_HISTORY_V10_POSTAUDIT/FINAL_RESULT.json',
      LINE/'data_pipeline/ordinary_scale_v1/audit.py']
    for f in paths:files[str(f)]=sha(f)
    for f,d in files.items():assert sha(f)==d,f
    save(HERE/'SOURCE_LOCK.json',dict(files=files,scope='same64FIT exact replay; changed offline ordered-route teacher only'),True)
    save(HERE/'MAIN_AGENT_APPROVAL.json',dict(scope='USER_CONTINUE_UNTIL_FIRST_POSITIVE_DATA_GATE_BEFORE_TRAINING',
      source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),protocol_sha256=sha(HERE/'PROTOCOL.json'),
      gpu1_empty_only=True,no_holder_release=True,training_allowed=False,automatic_retry=False,unix=time.time()),True)
    print('ORDERED_ROUTE_TEACHER_CPU_GATE_FROZEN_NO_TRAINING')
if __name__=='__main__':main()
