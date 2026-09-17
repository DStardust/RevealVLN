"""CPU-only exact mixed-plan counters, unchanged sources and preserved optimizer."""
import hashlib,json,math,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
TRAIN=LINE/'sft_acceptance/ordinary_onpolicy_fp32_master_v8r1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check(n):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    assert not torch.cuda.is_initialized()
    p=read(TRAIN/'PROTOCOL_FILESTORE.json');plan=read(TRAIN/'PLAN.json')
    for name,d in p['code_sha256'].items():assert sha(TRAIN/name)==d,name
    for key in ('source_checkpoint','resume_from'):
        assert sha(p[key]['path'])==p[key]['sha256']
    checkpoint=TRAIN/f'formal/attempt_001/checkpoint_{n:09d}.pt'
    rec=read(str(checkpoint)+'.json');assert sha(checkpoint)==rec['sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    before=torch.load(p['resume_from']['path'],map_location='cpu',weights_only=True)
    counts=[sum(map(len,r[:n])) for r in plan];total=sum(counts)
    # Update1000 is saved immediately before deterministic epoch-boundary normalization.
    expected=dict(epoch=0,position=n,updates=n,decisions=counts[0])
    assert state['cursor']==rec['cursor']==expected
    assert state['global_decisions']==state['charged_compute_decisions']==rec['global_decisions']==total
    assert state['binding']==dict(protocol_sha256=sha(TRAIN/'PROTOCOL_FILESTORE.json'),sample_index_sha256=p['sample_index_sha256'])
    assert set(state['trainable'])==set(before['trainable'])
    assert all(v.dtype==torch.float32 for v in state['trainable'].values())
    for s in state['optimizer']['state'].values():assert s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32
    changed=[]
    for k,v in state['trainable'].items():
        assert torch.isfinite(v).all()
        if not torch.equal(v,before['trainable'][k]):changed.append(k)
    assert changed and len(state['optimizer']['state'])==len(before['optimizer']['state'])
    for v in state['optimizer']['state'].values():
        assert int(v['step'])==4000+n
        assert all(torch.isfinite(x).all() for x in v.values() if isinstance(x,torch.Tensor))
    warmup=int(.03*31059);lr=5e-6*.5*(1+math.cos(math.pi*(4000+n-1-warmup)/(31059-warmup)))
    assert all(math.isclose(g['lr'],lr,rel_tol=1e-12,abs_tol=1e-15) for g in state['optimizer']['param_groups'])
    history=[json.loads(x) for x in (TRAIN/'formal/attempt_001/PROGRESS.jsonl').read_text().splitlines() if x.strip()]
    selected=[x for x in history if x['cursor']['updates']<=n]
    assert selected and all(0<x['cursor']['updates']<=n and x['resume_updates']==0 for x in selected)
    assert all(math.isfinite(x['metrics']['mean_ce']) and math.isfinite(x['metrics']['grad_norm']) and x['metrics']['grad_norm']>0 for x in selected)
    precision=[json.loads(s) for s in (TRAIN/'formal/attempt_001/train.log').read_text().splitlines() if s.startswith('{') and 'MASTER_PRECISION_READY' in s]
    assert len(precision)==3 and {s['rank'] for s in precision}=={0,1,2}
    assert all(s['all_trainable_float32'] and s['initial_bf16_injections_equal'] for s in precision)
    resume=[]
    for line in (TRAIN/'formal/attempt_001/train.log').read_text().splitlines():
        if line.startswith('{'):
            try:v=json.loads(line)
            except ValueError:continue
            if v.get('event')=='RESUME_READY':resume.append(v)
    assert len(resume)==3 and {x['rank'] for x in resume}=={0,1,2}
    assert all(x['cursor']==dict(epoch=0,position=0,updates=0,decisions=0) and x['charged_base']==0 for x in resume)
    if n==1000:
        final=read(TRAIN/'formal/attempt_001/RESULT.json')
        assert final['status']=='EPOCHS_COMPLETED' and final['stop']==[]
        assert final['cursor']==p['expected_final_cursor']
        assert final['global_decisions']==final['charged_compute_decisions']==p['expected_final_global_decisions']==total
        lease=read(TRAIN/'lease_v1/LEASE_RESULT.json')
        assert lease['execute_returned'] and lease['holders_restored'] and lease['error'] is None
    locks=LINE/'data_pipeline/ordinary_expansion_v1'
    assert sha(locks/'runtime_v2/INPUT_LOCK.json')=='c693ff433c3a02557c5d125400ce77d877b396a4d26544ffd7420fba1189509f'
    assert sha(locks/'runtime_gpu3_recovery_v1/INPUT_LOCK.json')=='fcb9eb183ccf48b66e6a532b55c23550b96c4c76312cdc42a309bae73aedf6f4'
    action_changes={}
    for name in ('action_query','exec_embed.weight'):
        v=state['trainable'][name];old=before['trainable'][name]
        action_changes[name]=dict(elements=v.numel(),stored_fp32_changed=int((v!=old).sum()),
            effective_bf16_changed=int((v.to(torch.bfloat16)!=old.to(torch.bfloat16)).sum()))
    result=dict(status='PASS',unix=time.time(),checkpoint_sha256=rec['sha256'],cursor=state['cursor'],action_parameter_changes=action_changes,
      stage_updates=n,optimizer_step=4000+n,new_decisions=total,rank_decisions=counts,
      parameter_tensors_changed=len(changed),all_state_finite=True,learning_rate=lr,
      original_optimizer_values_preserved=True,action_master_precision='FP32',navigation_gain_verified=False)
    name='FIRST' if n==200 else 'FINAL'
    with (HERE/(name+'_CHECKPOINT_ACCEPTANCE.json')).open('x') as f:json.dump(result,f,indent=2)
    assert not torch.cuda.is_initialized();print(json.dumps(result))
