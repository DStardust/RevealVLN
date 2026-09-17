"""Freeze exactly one output-preservation intervention before its first GPU batch."""
import ast,copy,hashlib,json,runpy,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_fp32_master_v8r1'
REVIEW=LINE/'reviews/Q35N_ORDINARY_POLICY_PRESERVATION_V9'
PRIOR=LINE/'reviews/Q35N_ORDINARY_FP32_MASTER_V8_TRANSPORT_R1'
CASE=LINE/'closed_loop_bench/ordinary_policy_preservation_dev_v9'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    assert not (HERE/'PROTOCOL_FILESTORE.json').exists()
    test=read(HERE/'CPU_TEST_RESULT.json');assert test['status']=='PASS' and test['count']==15 and not test['cuda_initialized']
    for n,d in test['source_sha256'].items():assert sha(HERE/n)==d,n
    ipc=read(HERE/'IPC_TEST_RESULT.json');assert ipc['status']=='PASS' and ipc['actual_unix_socket_bind'] and ipc['all_rows_exact']
    assert ipc['cpu_tensor_rows_transferred']==64 and ipc['max_expected_worker_socket_path_bytes']<108
    tmp=LINE/'.t9';assert tmp.resolve()==tmp and tmp.is_dir()
    assert not list(tmp.rglob('.nfs*')) and not any(p.is_socket() for p in tmp.rglob('*'))
    old=read(OLD/'PROTOCOL_FILESTORE.json')
    for n,d in old['code_sha256'].items():assert sha(OLD/n)==d,n
    assert sha(old['resume_from']['path'])==old['resume_from']['sha256']
    prior=read(PRIOR/'RESULT.json');assert not prior['positive_development_signal'] and prior['after']['result']['sr']==.15
    lease=read(OLD/'lease_v1/LEASE_RESULT.json');assert lease['holders_restored'] and lease['external_processes_stopped']==0
    diagnostic=read(REVIEW/'FIT_BEHAVIOR_DIAGNOSTIC.json')
    assert diagnostic['eligible_new_inputs']==4983 and diagnostic['training_updates']==0 and diagnostic['simulator_actions']==0
    for n in ('control.py','data.py','model.py','train_filestore.py','supervise_filestore.py','launcher.py','lease_run.py'):
        assert (HERE/n).read_bytes()==(OLD/n).read_bytes(),n
    a=runpy.run_path(str(OLD/'reuse.py'));b=runpy.run_path(str(HERE/'reuse.py'))
    for n in ('train_filestore.py','model.py','control.py','supervise_filestore.py','launcher.py','lease_run.py'):
        x=b['source'](n);ast.parse(x)
        if n not in ('train_filestore.py','supervise_filestore.py'):assert x==a['source'](n),n
    source=b['source']('train_filestore.py')
    assert "loss = ((ce + kl) * batch['weights']).sum() * world / weight_sum_global" in source
    assert "cursor['updates'] + protocol['optimizer_step_offset']" in source
    for n in ('PLAN.json','BUILD_AUDIT.json','PRECISION_BRIDGE_AUDIT.json'):
        with (HERE/n).open('xb') as f:f.write((OLD/n).read_bytes())
    assert sha(HERE/'PLAN.json')==old['sampling_plan_sha256']
    for folder in (HERE,REVIEW,CASE):
        for f in folder.glob('*.py'):ast.parse(f.read_text())
    transport=dict(status='PASS',unix=time.time(),temporary_directory=str(tmp),ipc_test=ipc,
        same_original_data_plan_and_initial_checkpoint=True,one_scientific_change='fixed_source_forward_KL_lambda1_T1',
        prior_scientific_failure_preserved=True,automatic_retry=False,
        ipc_cleanup_warning='NFS finalizer EBUSY after successful 64-row transfer; prefreeze confirms no .nfs/socket remains')
    save(HERE/'TRANSPORT_AUDIT.json',transport)
    save(REVIEW/'CPU_PREFLIGHT.json',dict(status='PASS',transport=transport,tests=test,
        fit_diagnostic_sha256=sha(REVIEW/'FIT_BEHAVIOR_DIAGNOSTIC.json'),reference_is_fixed_best4000=True,
        gpu_runtime_parity_still_required=True,navigation_gain_verified=False))
    code={str(OLD/n):d for n,d in old['code_sha256'].items()}
    paths=list(HERE.glob('*.py'))+[HERE/n for n in ('PLAN_ZH.md','PLAN.json','BUILD_AUDIT.json','PRECISION_BRIDGE_AUDIT.json','TRANSPORT_AUDIT.json','CPU_TEST_RESULT.json','IPC_TEST_RESULT.json')]
    paths += [OLD/'PROTOCOL_FILESTORE.json',OLD/'lease_v1/LEASE_RESULT.json',PRIOR/'RESULT.json',REVIEW/'FIT_BEHAVIOR_DIAGNOSTIC.json']
    for f in paths:code[f.name if f.parent==HERE else str(f)]=sha(f)
    p=copy.deepcopy(old);p.update(id='Q35N_ORDINARY_POLICY_PRESERVATION_V9',code_sha256=code,
        accounting=dict(initial_charged_decisions=0,deadline_unix=time.time()+2400,legacy_decisions_in_this_stage=0),
        loss='weighted CE + weighted KL(fixed_best4000 || student); lambda1 T1; category sum and global weight normalization',
        reference_checkpoint=copy.deepcopy(old['resume_from']),reference_kl_lambda=1.0,reference_temperature=1.0,
        expected_teacher_forward_decisions=99047,expected_total_policy_forward_decisions=198094,
        segment_note='single fixed-reference KL objective change vs V8R1; known ordinary engineering',
        temporary_directory=str(tmp),same_recipe_control_sha256=prior['after']['result']['checkpoint_sha256'])
    p.pop('transport_only_revision',None);p['budget']['wall_seconds']=2400
    p['budget']['max_teacher_forward_decisions']=99179;p['budget']['max_total_policy_forward_decisions']=198358
    save(HERE/'PROTOCOL_FILESTORE.json',p)
    book=read(OLD/'RUNBOOK.json');book.update(name=p['id'],code_sha256=code,lease_wall_seconds=2700)
    step=book['steps'][0];step['name']='fixed_reference_kl1000';step['argv'][-1]=str(HERE/'supervise_filestore.py')
    for k,v in list(step['env_extra'].items()):
        if isinstance(v,str):step['env_extra'][k]=v.replace('/cache/ordinary_fp32_master_v8r1/','/cache/ordinary_policy_preservation_v9/')
    cache=LINE/'runtime/cache/ordinary_policy_preservation_v9'
    for n in ('hf','xdg','xdg/torch/kernels','torch','cuda'):(cache/n).mkdir(parents=True,exist_ok=True)
    for k in ('TMPDIR','TMP','TEMP'):step['env_extra'][k]=str(tmp)
    save(HERE/'RUNBOOK.json',book)
    save(HERE/'MAIN_AGENT_APPROVAL.json',dict(unix=time.time(),scope='USER_CONTINUE_UNTIL_FIRST_POSITIVE_BOUNDED_KNOWN_OUTPUT_PRESERVATION',
        protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),
        new_updates_cap=1000,wall_seconds=2400,gpu_scope=[3,4,5],restore_exact_holders=True,
        extra_teacher_forward_decisions=99047,automatic_retry=False,scientific_change='one_KL_objective_term',
        reference_fixed=True,no_architecture_or_UAD_novelty_claim=True))
    subprocess.run([str(PY),'-I','-S','-B',str(CASE/'prepare.py'),'freeze'],cwd=ROOT,check=True)
    print('FIXED_REFERENCE_KL_ONE_FACTOR_FROZEN')
if __name__=='__main__':main()
