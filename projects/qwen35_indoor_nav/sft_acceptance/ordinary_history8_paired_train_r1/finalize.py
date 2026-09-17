"""Freeze explicit process-memory revision; scientific and resource gates unchanged."""
import ast
import copy
import json
from pathlib import Path
import runpy
import subprocess
import time
HERE=Path(__file__).resolve().parent
P=HERE.parent/'ordinary_history8_paired_train_v1'
LINE=HERE.parents[1];ROOT=LINE.parents[1]
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
QPY=LINE/'.envs/q35n_qwen_g2_v1/bin/python3'
def main():
    r=runpy.run_path(str(HERE/'runtime.py'))
    sha=r['sha'];save=r['save']
    read=lambda path:json.loads(path.read_text())
    assert not (HERE/'PROTOCOL.json').exists() and not (HERE/'lease_v1').exists()
    test=read(HERE/'CPU_TEST_RESULT.json')
    assert test['status']=='PASS_CPU_THREAD_TRANSPORT_ONLY' and test['no_gpu_initialized']
    for path,digest in test['sources'].items():assert sha(Path(path))==digest,path
    old=read(P/'PROTOCOL.json')
    for path,digest in old['code_sha256'].items():assert sha(Path(path))==digest,path
    fail=read(P/'control_recent2/LAUNCH_RESULT.json')
    assert fail['status']=='FAILED' and fail['error']=="AssertionError('TOTAL_CPU_RSS_BUDGET')"
    closed=read(P/'control_recent2/run_001/RESULT.json')
    assert closed['updates']==4 and closed['global_decisions']==384
    assert read(P/'lease_v1/LEASE_RESULT.json')['holders_restored']
    assert all(x['restored'] for x in read(P/'lease_v1/RESTORATION.json')['holders'])
    # Source-level equality of optimizer/update objective; only input transport changed.
    for name in ('runtime.py','model.py','loss.py','batching_fixed4.py','accept.py','lease.py'):
        assert ast.dump(ast.parse((HERE/name).read_text()))==ast.dump(ast.parse((P/name).read_text())),name
    def main_ast(path):
        return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    def update_loop(path):
        return next(n for n in ast.walk(main_ast(path)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    assert ast.dump(update_loop(HERE/'train.py'))==ast.dump(update_loop(P/'train.py'))
    assert 'num_workers=2' not in (HERE/'train.py').read_text()
    assert 'total_rss<64*1024**3' in (HERE/'supervise.py').read_text()
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    files=dict(old['code_sha256'])
    evidence=[P/'PROTOCOL.json',P/'RUNBOOK.json',P/'MAIN_AGENT_APPROVAL.json',
              P/'control_recent2/LAUNCH_RESULT.json',P/'control_recent2/FAILURE.json',
              P/'control_recent2/run_001/RESULT.json',P/'control_recent2/run_001/checkpoint_000000004.pt',
              P/'control_recent2/run_001/checkpoint_000000004.pt.json',
              P/'lease_v1/LEASE_RESULT.json',P/'lease_v1/RESTORATION.json']
    for path in evidence+list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[HERE/'CPU_TEST_RESULT.json']:
        files[str(path)]=sha(path)
    protocol=copy.deepcopy(old)
    protocol.update(id='ORDINARY_HISTORY8_PAIRED_TRAIN_R1',created_unix=time.time(),code_sha256=files,
        revision_parent=str(P/'PROTOCOL.json'),revision_parent_sha256=sha(P/'PROTOCOL.json'),
        selected_samples_path=str(P/'SELECTED_SAMPLES.jsonl'),
        previous_failed_run_cost=dict(training_updates=4,training_decisions=384,
            startup_forward_decisions=12,startup_backward_calls=3,wall_seconds=fail['wall_seconds']))
    protocol['data'].update(num_workers_per_rank=0,prefetch_factor=None,prefetch_threads_per_rank=1,
        queue_capacity=2,producer_inflight_batches_max=1,pinned_memory=True,pinning_device='current rank')
    save(HERE/'PROTOCOL.json',protocol,True)
    rb=read(P/'RUNBOOK.json')
    checker=runpy.run_path(str(HERE.parent/'ordinary_sync_recovery_v1/lease_run.py'),run_name='READONLY_PREFLIGHT')
    identities=[]
    for holder in rb['holders']:
        actual=checker['holder_identity'](holder)
        holder.update(expected_pid=actual['pid'],expected_start=actual['starttime_ticks'])
        identities.append(actual)
    rb.update(name=protocol['id'],code_sha256=files,steps=[])
    for arm in protocol['arms']:
        rb['steps'].append(dict(name=arm,argv=[str(PY),'-I','-S','-B','-u',str(HERE/'supervise.py'),'--arm',arm]))
        rb['steps'].append(dict(name=arm+'_accept',argv=[str(QPY),'-I','-B',str(HERE/'accept.py'),'--arm',arm],env_extra={'CUDA_VISIBLE_DEVICES':''}))
    save(HERE/'RUNBOOK.json',rb,True)
    result=dict(decision='PASS_FOR_THREAD_TRANSPORT_RETRY',unix=time.time(),
        protocol_sha256=sha(HERE/'PROTOCOL.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),
        cpu_gate_sha256=sha(HERE/'CPU_TEST_RESULT.json'),holder_identities=identities,
        code_files=len(files),scientific_loop_ast_unchanged=True,old_failure_retained=True,
        budget_per_arm_seconds=18000,max_policy_updates_per_arm=4000,total_cpu_rss_limit_gib=64,
        gpu_indices=[3,4,5],no_special_data=True,no_production_restart=True,no_paid_api=True,
        old_failed_weights_not_resumed=True,navigation_gain_claimed=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).splitlines())
    save(HERE/'MAIN_AGENT_APPROVAL.json',result,True)
    print(json.dumps(dict(status='FROZEN_R1_READY',code_files=len(files),holder_identities=identities)),flush=True)
if __name__=='__main__':main()
