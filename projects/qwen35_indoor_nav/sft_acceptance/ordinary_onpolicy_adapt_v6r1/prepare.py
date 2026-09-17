"""One versioned transport continuation after V6 wall-budget truncation."""
import ast,copy,hashlib,importlib.util,json,os,runpy,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_adapt_v6'
REVIEW=LINE/'reviews/Q35N_ORDINARY_ONPOLICY_ADAPT_V6_TRANSPORT_R1'
PRIOR=LINE/'reviews/Q35N_ORDINARY_ONPOLICY_ADAPT_V6'
CASE=LINE/'closed_loop_bench/ordinary_onpolicy_adapt_dev_v6r1'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (HERE/'PROTOCOL_FILESTORE.json').exists()
    import torch
    assert not torch.cuda.is_initialized()
    p=read(OLD/'PROTOCOL_FILESTORE.json');final=read(OLD/'formal/attempt_001/RESULT.json')
    lease=read(OLD/'lease_v1/LEASE_RESULT.json')
    assert final['stop']==['BUDGET:wall_seconds'] and final['cursor']['updates']==950
    assert lease['execute_returned'] and lease['holders_restored'] and lease['error'] is None
    assert read(PRIOR/'WORKFLOW_RESULT.json')['status']=='FAILED_OR_BLOCKED'
    assert not (LINE/'closed_loop_bench/ordinary_onpolicy_adapt_dev_v6/run_001').exists()
    for name,d in p['code_sha256'].items():assert sha(OLD/name)==d,name
    plan=read(OLD/'PLAN.json');counts=[sum(map(len,r[:950])) for r in plan]
    ckpt=Path(final['latest_checkpoint']);rec=read(str(ckpt)+'.json')
    assert sha(ckpt)==rec['sha256']=='a00a6006bdea3ec0d30b79cfad538deaf44242f9c2e972c1f43207ca43433cda'
    state=torch.load(ckpt,map_location='cpu',weights_only=True)
    assert state['cursor']==rec['cursor']==final['cursor']==dict(epoch=0,position=950,updates=950,decisions=counts[0])
    assert state['global_decisions']==state['charged_compute_decisions']==sum(counts)==94032
    assert state['binding']==dict(protocol_sha256=sha(OLD/'PROTOCOL_FILESTORE.json'),sample_index_sha256=p['sample_index_sha256'])
    assert all(torch.isfinite(v).all() for v in state['trainable'].values())
    for v in state['optimizer']['state'].values():
        assert int(v['step'])==4950 and all(torch.isfinite(x).all() for x in v.values() if isinstance(x,torch.Tensor))
    assert all(k in state for k in ('torch_rng','cuda_rng','python_rng'))
    for name in ('PLAN.json','BUILD_AUDIT.json'):
        with (HERE/name).open('xb') as f:f.write((OLD/name).read_bytes())
    with (REVIEW/'FIRST_CHECKPOINT_ACCEPTANCE.json').open('xb') as f:f.write((PRIOR/'FIRST_CHECKPOINT_ACCEPTANCE.json').read_bytes())
    assert read(REVIEW/'FIRST_CHECKPOINT_ACCEPTANCE.json')['status']=='PASS'
    r=runpy.run_path(str(HERE/'reuse.py'),run_name='CPU_SOURCE_TEST')
    for name in ('data.py','model.py','control.py','train_filestore.py','supervise_filestore.py','launcher.py','lease_run.py'):
        if name!='data.py':ast.parse(r['source'](name))
    for directory in (HERE,REVIEW,CASE):
        for f in directory.glob('*.py'):ast.parse(f.read_text())
    data=runpy.run_path(str(HERE/'data.py'),run_name='CPU_DATA_TEST')
    rows,report=data['load_rows']();s=data['SampleStore'](rows)
    assert set(s.get(37114,0))=={'instruction','images','executed','target'}
    audit=dict(status='PASS',source_checkpoint_sha256=rec['sha256'],source_cursor=state['cursor'],
      source_optimizer_step=4950,rank_decisions_before=counts,original_attempt_decisions=94032,
      remaining_updates=50,remaining_decisions=5015,fixed_total_updates=1000,fixed_total_decisions=99047,
      same_plan=True,optimizer_moments_and_rng_preserved=True,no_reexecuted_prefix=True,
      old_950_wall_truncation_preserved=True,original_200_acceptance_sha256=sha(PRIOR/'FIRST_CHECKPOINT_ACCEPTANCE.json'))
    save(HERE/'RESUME_AUDIT.json',audit)
    save(REVIEW/'RESUME_CHECKPOINT_ACCEPTANCE.json',dict(audit,unix=time.time(),gpu_actions=0))
    code={str(OLD/name):d for name,d in p['code_sha256'].items()}
    for f in list(HERE.glob('*.py'))+[HERE/'PLAN_ZH.md',HERE/'PLAN.json',HERE/'BUILD_AUDIT.json',HERE/'RESUME_AUDIT.json',
      OLD/'PROTOCOL_FILESTORE.json',OLD/'formal/attempt_001/RESULT.json',OLD/'lease_v1/LEASE_RESULT.json',
      PRIOR/'WORKFLOW_RESULT.json',PRIOR/'FIRST_CHECKPOINT_ACCEPTANCE.json']:
        code[f.name if f.parent==HERE else str(f)]=sha(f)
    p.update(id='Q35N_ORDINARY_ONPOLICY_ADAPT_V6_TRANSPORT_R1',code_sha256=code,
      resume_from=dict(path=str(ckpt),sha256=rec['sha256'],receipt_sha256=sha(str(ckpt)+'.json'),updates=950),
      accounting=dict(initial_charged_decisions=94032,deadline_unix=time.time()+900,legacy_decisions_in_this_stage=0),
      current_segment_start_updates=950,current_segment_planned_decisions=5015,current_segment_max_new_updates=50,
      resume_semantics='same frozen mixed index and plan; exact original950 cursor/moments/RNG to fixed1000',
      segment_note='transport continuation only; no scientific recipe change; preserve V6 wall truncation')
    p['budget']['wall_seconds']=900
    save(HERE/'PROTOCOL_FILESTORE.json',p)
    book=read(OLD/'RUNBOOK.json');book.update(name=p['id'],lease_wall_seconds=1050,code_sha256=code)
    step=book['steps'][0];step['argv'][-1]=str(HERE/'supervise_filestore.py');step['name']='resume_exact_remaining_50'
    for k,v in step['env_extra'].items():
        if isinstance(v,str):step['env_extra'][k]=v.replace('/cache/ordinary_onpolicy_adapt_v6/','/cache/ordinary_onpolicy_adapt_v6r1/')
    cache=LINE/'runtime/cache/ordinary_onpolicy_adapt_v6r1'
    for n in ('hf','xdg','xdg/torch/kernels','torch','cuda'):(cache/n).mkdir(parents=True,exist_ok=True)
    save(HERE/'RUNBOOK.json',book)
    save(HERE/'MAIN_AGENT_APPROVAL.json',dict(unix=time.time(),scope='USER_CONTINUE_UNTIL_FIRST_POSITIVE_TRANSPORT_ONLY_FINISH_FIXED1000',
      protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),
      new_updates_cap=50,remaining_decisions=5015,wall_seconds=900,restore_exact_holders=True,
      gpu_scope=[3,4,5],automatic_retry=False,no_new_method_claim=True))
    subprocess.run([str(PY),'-I','-S','-B',str(CASE/'prepare.py'),'freeze'],cwd=ROOT,check=True)
    assert not torch.cuda.is_initialized()
    print(json.dumps(audit))
if __name__=='__main__':main()
