"""Preserve exact V6 data/plan/start; promote only two BF16 master groups."""
import ast,copy,hashlib,json,os,runpy,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_onpolicy_adapt_v6'
REVIEW=LINE/'reviews/Q35N_ORDINARY_FP32_MASTER_V8'
CASE=LINE/'closed_loop_bench/ordinary_fp32_master_dev_v8'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (HERE/'PROTOCOL_FILESTORE.json').exists()
    import torch
    assert not torch.cuda.is_initialized()
    oldp=read(OLD/'PROTOCOL_FILESTORE.json')
    for name,d in oldp['code_sha256'].items():assert sha(OLD/name)==d,name
    initial=Path(oldp['resume_from']['path']);assert sha(initial)==oldp['resume_from']['sha256']
    source=torch.load(initial,map_location='cpu',weights_only=True);state=copy.deepcopy(source)
    assert state['cursor']==dict(epoch=0,position=0,updates=0,decisions=0)
    assert state['global_decisions']==state['charged_compute_decisions']==0
    changes=[]
    for name in ('action_query','exec_embed.weight'):
        x=source['trainable'][name];assert x.dtype==torch.bfloat16
        state['trainable'][name]=x.float()
        assert torch.equal(state['trainable'][name].to(torch.bfloat16),x)
        slots=[i for i,v in state['optimizer']['state'].items() if v['exp_avg'].shape==x.shape]
        assert len(slots)==1
        for field in ('exp_avg','exp_avg_sq'):
            v=source['optimizer']['state'][slots[0]][field]
            assert v.dtype==torch.bfloat16
            state['optimizer']['state'][slots[0]][field]=v.float()
            assert torch.equal(state['optimizer']['state'][slots[0]][field].double(),v.double())
        changes.append(dict(name=name,optimizer_slot=slots[0],shape=list(x.shape),old_dtype='bfloat16',new_dtype='float32'))
    for name,x in state['trainable'].items():
        assert x.dtype==torch.float32 and torch.isfinite(x).all()
        assert torch.equal(x.double(),source['trainable'][name].double())
    for i,slot in state['optimizer']['state'].items():
        assert int(slot['step'])==4000
        assert slot['exp_avg'].dtype==slot['exp_avg_sq'].dtype==torch.float32
        for field,v in slot.items():
            old=source['optimizer']['state'][i][field]
            if isinstance(v,torch.Tensor):assert torch.equal(v.double(),old.double())
            else:assert v==old
    assert state['optimizer']['param_groups']==source['optimizer']['param_groups']
    assert torch.equal(state['torch_rng'],source['torch_rng']) and torch.equal(state['cuda_rng'],source['cuda_rng'])
    assert state['python_rng']==source['python_rng']
    # Exhaustive 3-action lookup plus query injection: same effective BF16 model inputs.
    indices=torch.tensor([0,1,2,2,1,0],dtype=torch.long)
    x=torch.nn.functional.embedding(indices,source['trainable']['exec_embed.weight'])
    y=torch.nn.functional.embedding(indices,state['trainable']['exec_embed.weight']).to(torch.bfloat16)
    assert torch.equal(x,y)
    assert torch.equal(state['trainable']['action_query'].to(torch.bfloat16),source['trainable']['action_query'])
    state['binding']=dict(sample_index_sha256=oldp['sample_index_sha256'],source_checkpoint_sha256=sha(initial),
      bridge='same_initial_values_and_moments_two_FP32_master_groups')
    checkpoint=HERE/'initial_fp32_master_from_best4000.pt'
    with checkpoint.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    back=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,back['trainable'][k]) for k,v in state['trainable'].items())
    save(HERE/(checkpoint.name+'.json'),dict(sha256=sha(checkpoint),cursor=state['cursor'],global_decisions=0,
      charged_compute_decisions=0,source_initial_sha256=sha(initial),optimizer_reset=False,
      optimizer_step_offset=4000,precision_change_only=changes))
    for n in ('PLAN.json','BUILD_AUDIT.json'):
        with (HERE/n).open('xb') as f:f.write((OLD/n).read_bytes())
    plan=read(HERE/'PLAN.json');assert all(len(x)==1000 for x in plan) and sum(len(b) for r in plan for b in r)==99047
    reuse=runpy.run_path(str(HERE/'reuse.py'),run_name='CPU_SOURCE')
    for n in ('train_filestore.py','supervise_filestore.py','model.py','control.py','launcher.py','lease_run.py'):ast.parse(reuse['source'](n))
    assert reuse['source']('model.py')==reuse['parent'].source('model.py')
    raw=reuse['source']('model.py')
    assert "p.exec_embed(exec_index[:, 2]).to(emb.dtype)" in raw and "p.action_query.to(emb.dtype)" in raw
    for folder in (HERE,REVIEW,CASE):
        for f in folder.glob('*.py'):ast.parse(f.read_text())
    audit=dict(status='PASS',unix=time.time(),source_initial_sha256=sha(initial),source_best_sha256=oldp['source_checkpoint']['sha256'],
      initial_parameter_values_exact=True,optimizer_moment_values_exact=True,rng_exact=True,
      all_exec_and_query_bf16_injections_exact=True,changes=changes,actual_model_and_forward_unchanged=True,
      plan_sha256=sha(HERE/'PLAN.json'),same_V6_1000_update_order=True,source_optimizer_step=4000,
      cpu_optimizer_steps=0,gpu_actions=0,navigation_gain_verified=False)
    save(HERE/'PRECISION_BRIDGE_AUDIT.json',audit);save(REVIEW/'CPU_PREFLIGHT.json',audit)
    code={str(OLD/n):d for n,d in oldp['code_sha256'].items()}
    for f in list(HERE.glob('*.py'))+[HERE/'PLAN_ZH.md',HERE/'PLAN.json',HERE/'BUILD_AUDIT.json',HERE/'PRECISION_BRIDGE_AUDIT.json',
      OLD/'PROTOCOL_FILESTORE.json',initial,Path(str(initial)+'.json'),
      LINE/'reviews/Q35N_BF16_ACTION_PARAMETER_READONLY_V1/RESULT.json',
      LINE/'reviews/Q35N_ORDINARY_ONPOLICY_ADAPT_V6_EVAL_R3/RESULT.json']:
        code[f.name if f.parent==HERE else str(f)]=sha(f)
    p=copy.deepcopy(oldp)
    p.update(id='Q35N_ORDINARY_FP32_MASTER_V8',code_sha256=code,
      resume_from=dict(path=str(checkpoint),sha256=sha(checkpoint),receipt_sha256=sha(str(checkpoint)+'.json'),updates=0),
      accounting=dict(initial_charged_decisions=0,deadline_unix=time.time()+1800,legacy_decisions_in_this_stage=0),
      resume_semantics='same V6 start/plan/moment values, only action query and exec embedding FP32 master precision',
      segment_note='single numerical-precision factor vs V6; ordinary engineering not UAD novelty',
      action_master_precision='FP32',same_recipe_control_sha256='d5fd559cb78038c806ca0c13bf97241f7bd4bd3d7326b7bd02779916fce8dc0a')
    save(HERE/'PROTOCOL_FILESTORE.json',p)
    book=read(OLD/'RUNBOOK.json');book.update(name=p['id'],code_sha256=code,lease_wall_seconds=2100)
    step=book['steps'][0];step['argv'][-1]=str(HERE/'supervise_filestore.py');step['name']='fp32_master_1000'
    for k,v in list(step['env_extra'].items()):
        if isinstance(v,str):step['env_extra'][k]=v.replace('/cache/ordinary_onpolicy_adapt_v6/','/cache/ordinary_fp32_master_v8/')
    cache=LINE/'runtime/cache/ordinary_fp32_master_v8'
    for n in ('hf','xdg','xdg/torch/kernels','torch','cuda','tmp'):(cache/n).mkdir(parents=True,exist_ok=True)
    for k in ('TMPDIR','TMP','TEMP'):step['env_extra'][k]=str(cache/'tmp')
    save(HERE/'RUNBOOK.json',book)
    save(HERE/'MAIN_AGENT_APPROVAL.json',dict(unix=time.time(),scope='USER_CONTINUE_UNTIL_FIRST_POSITIVE_TARGETED_NUMERICAL_REPAIR',
      protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),
      new_updates_cap=1000,wall_seconds=1800,restore_exact_holders=True,gpu_scope=[3,4,5],
      temporary_directory_transport='project NAS; full system disk untouched',automatic_retry=False,no_new_method_claim=True))
    subprocess.run([str(PY),'-I','-S','-B',str(CASE/'prepare.py'),'freeze'],cwd=ROOT,check=True)
    assert not torch.cuda.is_initialized();print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
