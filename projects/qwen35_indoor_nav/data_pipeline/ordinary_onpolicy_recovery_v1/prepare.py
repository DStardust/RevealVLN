"""CPU-only admission of the fixed 64 FIT replay-and-teacher diagnostic."""
import copy,importlib.util,json,os,runpy,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
FIT=LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2'
def load():
    s=importlib.util.spec_from_file_location('recovery_prepare_io',HERE/'common.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text())
def main():
    assert not (HERE/'PROTOCOL.json').exists(),'ALREADY_FROZEN'
    c=load();old=read(FIT/'PROTOCOL.json');result=read(FIT/'run_001/RESULT.json')
    assert old['split']=='FIT' and result['completed']==64 and result['trace_audit_passed']
    assert result['environment_actions']==7225
    assert old['checkpoint_sha256']=='c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775'
    split=read(LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001/SPLIT.json')
    assert set(old['houses'])<=set(split['FIT'])
    assert set(old['houses']).isdisjoint(split['INTERNAL_DEV']+split['INTERNAL_CONFIRM'])
    py=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
    tested=subprocess.run([str(py),'-I','-S','-B',str(HERE/'tests.py')],capture_output=True,text=True,timeout=60)
    assert tested.returncode==0,tested.stderr
    # Complete stdlib launcher import catches missing transitive local dependencies.
    launch=runpy.run_path(str(HERE/'launch.py'),run_name='CPU_IMPORT_ONLY')
    assert launch['OUT']==HERE/'run_001' and callable(launch['_scan'].tree_size)
    assert not (HERE/'run_001').exists()
    p=copy.deepcopy(old);p.update(id='Q35N_ORDINARY_ONPOLICY_RECOVERY_DATA_V1',lanes=1,
        wall_seconds=1200,gpu_memory_gib=8,cpu_memory_gib=32,output_gib=2,
        main_criterion='64 exact double replays;>=1000 unique legal advice in>=8FIT houses;>=100 disagreements in>=8houses',
        teacher_stop_distance_strictly_less_than=3.,teacher_motion_goal_radius=.35,teacher_reset_each_query=True,
        explicit_primitive_budget=32000,new_model_rollouts=False,teacher_target='geometric goal navigation at original model-visited states',
        automatic_training=False,automatic_retry=False,training_allowed=False)
    c.write(HERE/'PROTOCOL.json',p,True)
    c.write(HERE/'CPU_TEST_RESULT.json',dict(passed=True,output=tested.stdout+tested.stderr,
        actual_launcher_import_passed=True,no_GPU_or_simulator_created=True,all_FIT=True),True)
    files=dict(read(FIT/'SOURCE_LOCK.json')['files'])
    sourcefiles=[FIT/'SOURCE_LOCK.json',FIT/'run_001/RESULT.json',FIT/'run_001/LAUNCH_RESULT.json',
        LINE/'reviews/Q35N_ORDINARY_R2R_ADAPT_V5_TRANSPORT_R1/RESULT.json',
        LINE/'reviews/Q35N_ORDINARY_R2R_ADAPT_V5_TRANSPORT_R1/REPORT_ZH.md',
        LINE/'data_pipeline/ordinary_expansion_v1/runtime_v2/INPUT_LOCK.json',
        LINE/'data_pipeline/ordinary_expansion_v1/runtime_gpu3_recovery_v1/INPUT_LOCK.json',
        LINE/'runtime/q35n_habitat_v017_g0r/src/habitat-sim/habitat_sim/nav/greedy_geodesic_follower.py',
        LINE/'.envs/q35n_habitat_v017_g0r/lib/python3.10/site-packages/habitat_sim/nav/greedy_geodesic_follower.py']
    for lane in (FIT/'run_001/lanes').iterdir():
        if not lane.is_dir():continue
        for name in ('POLICY_STEPS.jsonl','STEPS_PRIVILEGED.jsonl','INTERFACE.jsonl'):
            sourcefiles.append(lane/name)
        sourcefiles.extend(lane.glob('episode_*.json'))
    for h in old['houses']:
        scene=ROOT/f'third_party/ETP-R1/data/scene_datasets/mp3d/{h}'
        sourcefiles.append(scene/(h+'.glb'))
        sourcefiles.extend(scene.glob('*.navmesh'))
    for f in sourcefiles+list(HERE.glob('*.py'))+[HERE/'PLAN_ZH.md',HERE/'PROTOCOL.json',HERE/'CPU_TEST_RESULT.json']:
        files[str(f)]=c.sha(f)
    for f,d in files.items():assert c.sha(f)==d,f
    c.write(HERE/'SOURCE_LOCK.json',dict(files=files,scope='fixed64FIT replay and one-step teacher labels only'),True)
    c.write(HERE/'MAIN_AGENT_APPROVAL.json',dict(scope='USER_20260913_CONTINUE_UNTIL_FIRST_POSITIVE_OFFPOLICY_SUPERVISION_DIAGNOSTIC',
        source_lock_sha256=c.sha(HERE/'SOURCE_LOCK.json'),protocol_sha256=c.sha(HERE/'PROTOCOL.json'),
        gpu1_empty_only=True,no_holder_release=True,training_allowed=False,automatic_retry=False,unix=time.time()),True)
    print('FROZEN_CPU_READY_NO_TRAINING')
if __name__=='__main__':main()
