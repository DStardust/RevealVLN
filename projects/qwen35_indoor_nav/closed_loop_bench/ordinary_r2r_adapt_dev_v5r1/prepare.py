"""Transport-only revision after zero-action missing-module startup failure."""
import ast,hashlib,importlib.util,json,os,runpy,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
OLD=HERE.parent/'ordinary_r2r_adapt_dev_v5'
REVIEW=LINE/'reviews/Q35N_ORDINARY_R2R_ADAPT_V5_TRANSPORT_R1'
PREVIOUS=LINE/'reviews/Q35N_ORDINARY_R2R_ADAPT_V5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    assert not (HERE/'SOURCE_LOCK.json').exists()
    failure=read(PREVIOUS/'WORKFLOW_RESULT.json')
    assert failure['status']=='FAILED_OR_BLOCKED'
    assert not (OLD/'run_001').exists() and not (OLD/'gpu1_eval.lock').exists(),'NOT_ZERO_ACTION_FAILURE'
    assert 'tree_size.py' in (PREVIOUS/'eval_launcher.log').read_text()
    assert not (OLD/'tree_size.py').exists()
    assert read(PREVIOUS/'FINAL_CHECKPOINT_ACCEPTANCE.json')['status']=='PASS'
    previous=read(OLD/'SOURCE_LOCK.json')
    for path,digest in previous['files'].items():assert sha(path)==digest,path
    for name in ('PROTOCOL.json','EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json'):
        with (HERE/name).open('xb') as f:f.write((OLD/name).read_bytes())
        assert sha(HERE/name)==sha(OLD/name)
    for name in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py','reuse.py'):
        assert (HERE/name).read_bytes()==(OLD/name).read_bytes(),'SCIENTIFIC_CODE_CHANGED:'+name
    s=importlib.util.spec_from_file_location('transport_r1_source',HERE/'reuse.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    for name in ('common.py','evaluate.py','executor.py','path_metrics.py','official_distance.py','aggregate.py','launch.py'):ast.parse(m.source(name))
    # Import the complete launcher, not only AST parsing: catches the original missing dependency.
    launch=runpy.run_path(str(HERE/'launch.py'),run_name='CPU_IMPORT_TEST')
    assert launch['OUT']==HERE/'run_001' and callable(launch['_scan'].tree_size)
    size,misses=launch['_scan'].tree_size(HERE);assert size>0 and misses==0
    assert not (HERE/'run_001').exists(),'CPU_SMOKE_STARTED_GPU_RUN'
    save(HERE/'CPU_TEST_RESULT.json',dict(passed=True,launcher_actual_import_passed=True,
        tree_scan_passed=True,policy_simulator_metrics_protocol_inputs_byte_identical=True,gpu_actions=0))
    save(HERE/'TRANSPORT_REVISION.json',dict(id='Q35N_R2R_ADAPT_DEV_V5_TRANSPORT_R1',unix=time.time(),
        predecessor=str(OLD),failure_sha256=sha(PREVIOUS/'WORKFLOW_RESULT.json'),
        failure_log_sha256=sha(PREVIOUS/'eval_launcher.log'),predecessor_navigation_actions=0,
        predecessor_gpu_worker_started=False,only_change='supply already-tested missing tree_size.py dependency',
        same_checkpoint=True,same_100_inputs=True,same_metrics=True,same_budget=True,
        automatic_retry=False,no_training=True))
    files=dict(previous['files'])
    for f in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[PREVIOUS/'WORKFLOW_RESULT.json',PREVIOUS/'eval_launcher.log',
        PREVIOUS/'FINAL_CHECKPOINT_ACCEPTANCE.json',LINE/'closed_loop_bench/ordinary_full_epoch_dev_v4/tree_size.py']:
        files[str(f)]=sha(f)
    save(HERE/'SOURCE_LOCK.json',dict(files=files,scope='transport dependency completion only; same fixed R2R8000 and100inputs'))
    save(HERE/'MAIN_REVIEW.json',dict(status='PASS_FOR_ONE_TRANSPORT_REVISION',source_lock_sha256=sha(HERE/'SOURCE_LOCK.json'),gpu_actions=0,no_training=True))
    print('FROZEN_TRANSPORT_ONLY_ACTUAL_LAUNCHER_IMPORT_PASS')
if __name__=='__main__':main()
