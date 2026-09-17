"""Fixed endpoint: finite moments, exact counters and original LR; CPU only."""
import hashlib,json,math,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
TRAIN=LINE/'sft_acceptance/ordinary_r2r_adapt_v5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    p=read(TRAIN/'PROTOCOL_FILESTORE.json');final=read(TRAIN/'formal/attempt_001/RESULT.json')
    for name,digest in p['code_sha256'].items():assert sha(TRAIN/name)==digest,name
    assert final['cursor']==p['expected_final_cursor'] and final['stop']==['BUDGET:max_updates']
    assert final['global_decisions']==final['charged_compute_decisions']==p['expected_final_global_decisions']==736713
    ck=Path(final['latest_checkpoint']);assert ck==TRAIN/'formal/attempt_001/checkpoint_000008000.pt'
    receipt=read(str(ck)+'.json');assert sha(ck)==receipt['sha256']
    state=torch.load(ck,map_location='cpu',weights_only=True)
    assert state['cursor']==receipt['cursor']==p['expected_final_cursor']
    assert state['global_decisions']==state['charged_compute_decisions']==736713
    assert state['binding']==dict(protocol_sha256=sha(TRAIN/'PROTOCOL_FILESTORE.json'),sample_index_sha256=p['sample_index_sha256'])
    assert len(state['trainable'])==len(state['optimizer']['state'])==28
    assert all(torch.isfinite(v).all() for v in state['trainable'].values())
    for slot in state['optimizer']['state'].values():
        assert int(slot['step'])==8000
        assert all(torch.isfinite(v).all() for v in slot.values() if isinstance(v,torch.Tensor))
    total=31059;warmup=int(.03*total);lr=5e-5*.5*(1+math.cos(math.pi*(7999-warmup)/(total-warmup)))
    assert all(math.isclose(g['lr'],lr,rel_tol=1e-12,abs_tol=1e-15) for g in state['optimizer']['param_groups'])
    source=Path(p['resume_from']['path']);assert sha(source)==p['resume_from']['sha256']
    assert read(HERE/'FIRST_CHECKPOINT_ACCEPTANCE.json')['status']=='PASS'
    lease=read(TRAIN/'lease_v1/LEASE_RESULT.json')
    assert lease['execute_returned'] and lease['holders_restored'] and lease['error'] is None
    result=dict(status='PASS',unix=time.time(),checkpoint_sha256=receipt['sha256'],cursor=state['cursor'],
        new_decisions=396001,branch_decisions=736713,optimizer_step=8000,all_state_finite=True,
        learning_rate=lr,source_locks_unchanged=True,navigation_gain_verified=False)
    with (HERE/'FINAL_CHECKPOINT_ACCEPTANCE.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
