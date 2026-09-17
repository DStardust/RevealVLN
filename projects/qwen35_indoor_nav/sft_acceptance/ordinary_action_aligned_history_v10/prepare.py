"""Freeze one aligned-input continuation after all CPU and transport checks."""
import ast,copy,hashlib,json,runpy,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_fp32_master_v8r1'
PRIOR=HERE.parent/'ordinary_policy_preservation_v9'
REVIEW=LINE/'reviews/Q35N_ORDINARY_ACTION_ALIGNED_HISTORY_V10'
CASE=LINE/'closed_loop_bench/ordinary_action_aligned_history_dev_v10'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    assert not (HERE/'PROTOCOL_FILESTORE.json').exists()
    ipc=read(HERE/'IPC_TEST_RESULT.json');cpu=read(HERE/'CPU_TEST_RESULT.json');view=read(HERE/'VIEW_AUDIT.json')
    assert cpu['status']==ipc['status']==view['status']==read(HERE/'AGGREGATE_CPU_TEST_RESULT.json')['status']=='PASS'
    assert ipc['actual_unix_socket_bind'] and ipc['all_rows_exact'] and ipc['cpu_tensor_rows_transferred']==64
    assert ipc['max_expected_worker_socket_path_bytes']<108 and cpu['actual_training_samples']>=60
    tmp=LINE/'.t10';assert tmp.resolve()==tmp and tmp.is_dir()
    assert not list(tmp.rglob('.nfs*')) and not any(p.is_socket() for p in tmp.rglob('*'))
    old=read(OLD/'PROTOCOL_FILESTORE.json')
    for n,d in old['code_sha256'].items():assert sha(OLD/n)==d,n
    assert sha(old['resume_from']['path'])==old['resume_from']['sha256']
    for folder in (OLD,PRIOR):
        lease=read(folder/'lease_v1/LEASE_RESULT.json')
        assert lease['holders_restored'] and lease['external_processes_stopped']==0 and lease['error'] is None
    previous=read(LINE/'reviews/Q35N_ORDINARY_POLICY_PRESERVATION_V9/WORKFLOW_RESULT.json')
    assert previous['status']=='COMPLETE' and previous['positive_development_signal'] is False
    a=runpy.run_path(str(OLD/'reuse.py'));b=runpy.run_path(str(HERE/'reuse.py'))
    for n in ('train_filestore.py','model.py','control.py','supervise_filestore.py','launcher.py','lease_run.py'):
        assert (HERE/n).read_bytes()==(OLD/n).read_bytes(),n
        source=b['source'](n);ast.parse(source)
        normalized=source.replace('Q35N_ORDINARY_ACTION_ALIGNED_HISTORY_V10','Q35N_ORDINARY_FP32_MASTER_V8_TRANSPORT_R1').replace('triton_ordinary_action_aligned_history_v10','triton_ordinary_fp32_master_v8r1').replace('inductor_ordinary_action_aligned_history_v10','inductor_ordinary_fp32_master_v8r1')
        assert normalized==a['source'](n),n
        assert 'reference_kl' not in source
    for n in ('PLAN.json','BUILD_AUDIT.json','PRECISION_BRIDGE_AUDIT.json'):
        with (HERE/n).open('xb') as f:f.write((OLD/n).read_bytes())
    save(REVIEW/'CPU_PREFLIGHT.json',dict(status='PASS',cpu=cpu,view_audit=view,ipc=ipc,
      scientific_factor='old RGB at max(0,t-8), same current RGB and eight executed actions',previous_negative=previous,
      same_model_optimizer_initialization_plan_labels_weights_lr=True,reference_kl_enabled=False,
      cleanup_note='IPC may emit NFS finalizer EBUSY; after process exit no .nfs/socket remains'))
    code={str(OLD/n):d for n,d in old['code_sha256'].items()}
    for f in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+list(HERE.glob('*.md'))+[
      OLD/'PROTOCOL_FILESTORE.json',PRIOR/'lease_v1/LEASE_RESULT.json',
      LINE/'reviews/Q35N_ORDINARY_POLICY_PRESERVATION_V9/WORKFLOW_RESULT.json',
      CASE.parent/'ordinary_fp32_master_dev_v8r1/reuse.py']:
        code[f.name if f.parent==HERE else str(f)]=sha(f)
    p=copy.deepcopy(old);p.update(id='Q35N_ORDINARY_ACTION_ALIGNED_HISTORY_V10',code_sha256=code,
      accounting=dict(initial_charged_decisions=0,deadline_unix=time.time()+1800,legacy_decisions_in_this_stage=0),
      temporary_directory=str(tmp),frame_stride=8,history_implementation='history_r1.py',
      input_view_sha256=sha(HERE/'RECOVERY_VIEW.json'),reference_kl_enabled=False,
      scientific_revision='only old RGB temporal position; no added encoded images or trainable parameters')
    p.pop('transport_only_revision',None)
    save(HERE/'PROTOCOL_FILESTORE.json',p)
    book=read(OLD/'RUNBOOK.json');book.update(name=p['id'],code_sha256=code,lease_wall_seconds=2100)
    step=book['steps'][0];step['name']='action_aligned1000';step['argv'][-1]=str(HERE/'supervise_filestore.py')
    for k,v in list(step['env_extra'].items()):
        if isinstance(v,str):step['env_extra'][k]=v.replace('/cache/ordinary_fp32_master_v8r1/','/cache/ordinary_action_aligned_history_v10/')
    cache=LINE/'runtime/cache/ordinary_action_aligned_history_v10'
    for n in ('hf','xdg','xdg/torch/kernels','torch','cuda'):(cache/n).mkdir(parents=True,exist_ok=True)
    for k in ('TMPDIR','TMP','TEMP'):step['env_extra'][k]=str(tmp)
    save(HERE/'RUNBOOK.json',book)
    save(HERE/'MAIN_AGENT_APPROVAL.json',dict(unix=time.time(),scope='USER_CONTINUE_UNTIL_FIRST_POSITIVE_BOUNDED_ORDINARY_ENGINEERING',
      protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),new_updates_cap=1000,
      wall_seconds=1800,gpu_scope=[3,4,5],restore_exact_holders=True,automatic_retry=False,frame_stride=8,
      special_data_untouched=True,frozen_sources_and_input_locks_unchanged=True))
    subprocess.run([str(PY),'-I','-S','-B',str(CASE/'prepare.py'),'freeze'],cwd=ROOT,check=True)
    print('ONE_ACTION_ALIGNED_INPUT_FACTOR_FROZEN')
if __name__=='__main__':main()
