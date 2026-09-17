"""Same V8 numerical recipe; short tested Unix temp transport only."""
import ast,copy,hashlib,json,os,runpy,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_fp32_master_v8'
REVIEW=LINE/'reviews/Q35N_ORDINARY_FP32_MASTER_V8_TRANSPORT_R1'
PRIOR=LINE/'reviews/Q35N_ORDINARY_FP32_MASTER_V8'
CASE=LINE/'closed_loop_bench/ordinary_fp32_master_dev_v8r1'
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
    ipc=read(HERE/'IPC_TEST_RESULT.json')
    assert ipc['status']=='PASS' and ipc['actual_unix_socket_bind'] and ipc['all_rows_exact']
    assert ipc['cpu_tensor_rows_transferred']==64 and ipc['max_expected_worker_socket_path_bytes']<108
    tmp=LINE/'.t8r1';assert tmp.resolve()==tmp and tmp.is_dir()
    assert not list(tmp.rglob('.nfs*')) and not any(p.is_socket() for p in tmp.rglob('*'))
    old=read(OLD/'PROTOCOL_FILESTORE.json')
    for n,d in old['code_sha256'].items():assert sha(OLD/n)==d,n
    assert sha(old['resume_from']['path'])==old['resume_from']['sha256']
    lease=read(OLD/'lease_v1/LEASE_RESULT.json');assert lease['holders_restored'] and lease['external_processes_stopped']==0
    assert read(PRIOR/'WORKFLOW_RESULT.json')['status']=='FAILED_OR_BLOCKED'
    log=(OLD/'formal/attempt_001/train.log').read_text()
    assert 'AF_UNIX path too long' in log and not (OLD/'formal/attempt_001/PROGRESS.json').exists()
    assert not list((OLD/'formal/attempt_001').glob('checkpoint_*.pt'))
    events=[]
    for line in log.splitlines():
        if line.startswith('{'):
            try:x=json.loads(line)
            except ValueError:continue
            if x.get('event')=='MASTER_PRECISION_READY':events.append(x)
    assert len(events)==3 and {e['rank'] for e in events}=={0,1,2}
    for n in ('control.py','data.py','model.py','train_filestore.py','supervise_filestore.py','launcher.py','lease_run.py'):
        assert (HERE/n).read_bytes()==(OLD/n).read_bytes(),n
    a=runpy.run_path(str(OLD/'reuse.py'),run_name='OLD_SCIENTIFIC_SOURCE')
    b=runpy.run_path(str(HERE/'reuse.py'),run_name='NEW_SCIENTIFIC_SOURCE')
    for n in ('train_filestore.py','model.py','control.py','supervise_filestore.py','launcher.py','lease_run.py'):
        x=b['source'](n);ast.parse(x)
        normalized=x.replace('Q35N_ORDINARY_FP32_MASTER_V8_TRANSPORT_R1','Q35N_ORDINARY_FP32_MASTER_V8')
        normalized=normalized.replace('triton_ordinary_fp32_master_v8r1','triton_ordinary_fp32_master_v8').replace('inductor_ordinary_fp32_master_v8r1','inductor_ordinary_fp32_master_v8')
        assert normalized==a['source'](n),n
    for n in ('PLAN.json','BUILD_AUDIT.json','PRECISION_BRIDGE_AUDIT.json'):
        with (HERE/n).open('xb') as f:f.write((OLD/n).read_bytes())
    for folder in (HERE,REVIEW,CASE):
        for f in folder.glob('*.py'):ast.parse(f.read_text())
    transport=dict(status='PASS',unix=time.time(),old_failure='AF_UNIX path too long before any DataLoader batch',
      no_successful_prior_training_updates=True,old_master_precision_runtime_checks=events,
      same_scientific_sources_after_namespace_normalization=True,same_initial_checkpoint=True,same_plan=True,
      temporary_directory=str(tmp),ipc_test=ipc,
      cpu_test_cleanup_warning='NFS finalizer EBUSY warnings; after exit only two empty test directories, no .nfs/socket/worker remained',
      old_failure_and_resource_receipts_preserved=True,automatic_retry=False)
    save(HERE/'TRANSPORT_AUDIT.json',transport)
    save(REVIEW/'CPU_PREFLIGHT.json',dict(status='PASS',transport=transport,original_bridge=read(HERE/'PRECISION_BRIDGE_AUDIT.json')))
    code={str(OLD/n):d for n,d in old['code_sha256'].items()}
    for f in list(HERE.glob('*.py'))+[HERE/'PLAN_ZH.md',HERE/'PLAN.json',HERE/'BUILD_AUDIT.json',HERE/'PRECISION_BRIDGE_AUDIT.json',
      HERE/'TRANSPORT_AUDIT.json',HERE/'IPC_TEST_RESULT.json',OLD/'PROTOCOL_FILESTORE.json',
      OLD/'lease_v1/LEASE_RESULT.json',PRIOR/'WORKFLOW_RESULT.json',PRIOR/'OWNED_TRANSPORT_STOP.json']:
        code[f.name if f.parent==HERE else str(f)]=sha(f)
    p=copy.deepcopy(old);p.update(id='Q35N_ORDINARY_FP32_MASTER_V8_TRANSPORT_R1',code_sha256=code,
      accounting=dict(initial_charged_decisions=0,deadline_unix=time.time()+1800,legacy_decisions_in_this_stage=0),
      temporary_directory=str(tmp),transport_only_revision='tested short Unix path after zero-batch IPC startup failure')
    save(HERE/'PROTOCOL_FILESTORE.json',p)
    book=read(OLD/'RUNBOOK.json');book.update(name=p['id'],code_sha256=code)
    step=book['steps'][0];step['argv'][-1]=str(HERE/'supervise_filestore.py')
    for k,v in list(step['env_extra'].items()):
        if isinstance(v,str):step['env_extra'][k]=v.replace('/cache/ordinary_fp32_master_v8/','/cache/ordinary_fp32_master_v8r1/')
    cache=LINE/'runtime/cache/ordinary_fp32_master_v8r1'
    for n in ('hf','xdg','xdg/torch/kernels','torch','cuda'):(cache/n).mkdir(parents=True,exist_ok=True)
    for k in ('TMPDIR','TMP','TEMP'):step['env_extra'][k]=str(tmp)
    save(HERE/'RUNBOOK.json',book)
    save(HERE/'MAIN_AGENT_APPROVAL.json',dict(unix=time.time(),scope='USER_CONTINUE_UNTIL_FIRST_POSITIVE_SAME_RECIPE_SHORT_UNIX_TRANSPORT',
      protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),new_updates_cap=1000,
      wall_seconds=1800,gpu_scope=[3,4,5],restore_exact_holders=True,automatic_retry=False,scientific_recipe_unchanged=True))
    subprocess.run([str(PY),'-I','-S','-B',str(CASE/'prepare.py'),'freeze'],cwd=ROOT,check=True)
    print('SHORT_UNIX_TRANSPORT_AND_UNCHANGED_FP32_RECIPE_FROZEN')
if __name__=='__main__':main()
