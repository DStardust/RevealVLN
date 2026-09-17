"""Final admission of one data-only route-teacher continuation after CPU acceptance."""
import ast,copy,hashlib,json,runpy,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_fp32_master_v8r1'
DATA=LINE/'data_pipeline/ordinary_route_teacher_v11/run_001'
REVIEW=LINE/'reviews/Q35N_ORDINARY_ROUTE_TEACHER_V11'
CASE=LINE/'closed_loop_bench/ordinary_route_teacher_dev_v11'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    assert not (HERE/'PROTOCOL_FILESTORE.json').exists()
    ipc=read(HERE/'IPC_TEST_RESULT.json');cpu=read(HERE/'CPU_TEST_RESULT.json');build=read(HERE/'BUILD_AUDIT.json')
    assert ipc['status']==cpu['status']==build['status']=='PASS'
    assert ipc['actual_unix_socket_bind'] and ipc['all_rows_exact'] and ipc['cpu_tensor_rows_transferred']==64
    tmp=LINE/'.t11';assert tmp.resolve()==tmp and tmp.is_dir() and ipc['max_expected_worker_socket_path_bytes']<108
    assert not list(tmp.rglob('.nfs*')) and not any(p.is_socket() for p in tmp.rglob('*'))
    assert read(DATA/'RESULT.json')['data_gate'] and read(DATA/'RESULT.json')['ordered_route_teacher']
    dl=read(DATA/'LAUNCH_RESULT.json')
    assert dl['status']=='COMPLETE' and not dl['cleanup']['remaining'] and dl['foreign_processes_signaled']==[]
    old=read(OLD/'PROTOCOL_FILESTORE.json')
    for n,d in old['code_sha256'].items():assert sha(OLD/n)==d,n
    assert read(OLD/'lease_v1/LEASE_RESULT.json')['holders_restored']
    a=runpy.run_path(str(OLD/'reuse.py'));b=runpy.run_path(str(HERE/'reuse.py'))
    for n in ('train_filestore.py','model.py','control.py','supervise_filestore.py','launcher.py','lease_run.py'):
        assert (HERE/n).read_bytes()==(OLD/n).read_bytes(),n
        source=b['source'](n);ast.parse(source)
        normalized=source.replace('Q35N_ORDINARY_ROUTE_TEACHER_V11','Q35N_ORDINARY_FP32_MASTER_V8_TRANSPORT_R1').replace("HERE / 'SAMPLE_INDEX.jsonl'","HERE.parent / 'ordinary_onpolicy_adapt_v6/SAMPLE_INDEX.jsonl'").replace('triton_ordinary_route_teacher_v11','triton_ordinary_fp32_master_v8r1').replace('inductor_ordinary_route_teacher_v11','inductor_ordinary_fp32_master_v8r1')
        assert normalized==a['source'](n),n
        assert 'reference_kl' not in source
    for folder in (HERE,REVIEW,CASE):
        for f in folder.glob('*.py'):ast.parse(f.read_text())
    with (HERE/'PRECISION_BRIDGE_AUDIT.json').open('xb') as f:f.write((OLD/'PRECISION_BRIDGE_AUDIT.json').read_bytes())
    save(REVIEW/'CPU_PREFLIGHT.json',dict(status='PASS',build=build,cpu=cpu,ipc=ipc,new_data=read(DATA/'RESULT.json'),
      same_model_optimizer_values_rng_lr=True,input_window='last2_RGB_last8_executed',
      data_availability_and_target_changes_explicit=True,cleanup_note='IPC NFS EBUSY finalizer warnings; after exit no live socket/.nfs remains'))
    code={str(OLD/n):d for n,d in old['code_sha256'].items()}
    for p,d in read(DATA.parent/'SOURCE_LOCK.json')['files'].items():
        if p in code:assert code[p]==d,p
        code[p]=d
    for p,d in read(DATA/'DATA_SEAL.json')['files'].items():code[p]=d
    for f in list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+list(HERE.glob('*.md'))+[
      OLD/'PROTOCOL_FILESTORE.json',DATA.parent/'SOURCE_LOCK.json',DATA/'DATA_SEAL.json',DATA/'RESULT.json',DATA/'LAUNCH_RESULT.json',
      Path(build['initial_path']),Path(build['initial_path']+'.json')]:
        code[f.name if f.parent==HERE else str(f)]=sha(f)
    for n,d in code.items():assert sha(HERE/n)==d,n
    p=copy.deepcopy(old);p.update(read(HERE/'DATA_BINDING.json'))
    p.update(id='Q35N_ORDINARY_ROUTE_TEACHER_V11',code_sha256=code,
      accounting=dict(initial_charged_decisions=0,deadline_unix=time.time()+1800,legacy_decisions_in_this_stage=0),
      temporary_directory=str(tmp),teacher_target='ordered_reference_waypoints',
      resume_from=dict(path=build['initial_path'],sha256=build['initial_sha256'],receipt_sha256=build['initial_receipt_sha256'],updates=0),
      current_segment_planned_decisions=build['planned_decisions'],max_global_batch_decisions=build['max_global_batch_decisions'],
      expected_final_cursor=dict(epoch=1,position=0,updates=1000,decisions=build['rank_planned_decisions'][0]),
      expected_final_global_decisions=build['planned_decisions'],reference_kl_enabled=False,frame_stride=1,
      segment_note='same1000 slots; ordinary89172 exact; route advice9807, unavailable68 removed; same weighting rule',
      resume_semantics='best4k FP32 values/moments/RNG identical; only sample-index metadata rebound')
    p.pop('transport_only_revision',None);p['budget'].update(wall_seconds=1800,max_updates=1001,max_decisions=build['planned_decisions']+build['max_global_batch_decisions'])
    save(HERE/'PROTOCOL_FILESTORE.json',p)
    book=read(OLD/'RUNBOOK.json');book.update(name=p['id'],code_sha256=code,lease_wall_seconds=2100)
    step=book['steps'][0];step['name']='route_teacher1000';step['argv'][-1]=str(HERE/'supervise_filestore.py')
    for k,v in list(step['env_extra'].items()):
        if isinstance(v,str):step['env_extra'][k]=v.replace('/cache/ordinary_fp32_master_v8r1/','/cache/ordinary_route_teacher_v11/')
    cache=LINE/'runtime/cache/ordinary_route_teacher_v11'
    for n in ('hf','xdg','xdg/torch/kernels','torch','cuda'):(cache/n).mkdir(parents=True,exist_ok=True)
    for k in ('TMPDIR','TMP','TEMP'):step['env_extra'][k]=str(tmp)
    save(HERE/'RUNBOOK.json',book)
    save(HERE/'MAIN_AGENT_APPROVAL.json',dict(unix=time.time(),scope='USER_CONTINUE_UNTIL_FIRST_POSITIVE_BOUNDED_ROUTE_TEACHER_ORDINARY_ENGINEERING',
      protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),new_updates_cap=1000,
      wall_seconds=1800,gpu_scope=[3,4,5],restore_exact_holders=True,automatic_retry=False,new_data_gate_passed=True,
      special_data_untouched=True,frozen_sources_and_input_locks_unchanged=True))
    subprocess.run([str(PY),'-I','-S','-B',str(CASE/'prepare.py'),'freeze'],cwd=ROOT,check=True)
    print('ROUTE_TEACHER_DATA_ONLY_INTERVENTION_FROZEN')
if __name__=='__main__':main()
