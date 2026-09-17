"""CPU-only full-epoch cursor, optimizer, finite-state and source acceptance."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
TRAIN=LINE/'sft_acceptance/ordinary_expanded_full_epoch_v4'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    assert not torch.cuda.is_initialized()
    p=read(TRAIN/'PROTOCOL_FILESTORE.json');r=read(TRAIN/'formal/attempt_001/RESULT.json')
    for name,digest in p['code_sha256'].items():assert sha(TRAIN/name)==digest,name
    assert r['status']=='EPOCHS_COMPLETED' and r['stop']==[] and r['cursor']==p['expected_final_cursor']
    checkpoint=TRAIN/'formal/attempt_001/checkpoint_000031059.pt'
    receipt=read(str(checkpoint)+'.json');assert sha(checkpoint)==receipt['sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['cursor']==receipt['cursor']==p['expected_final_cursor']
    assert state['global_decisions']==state['charged_compute_decisions']==p['expected_epoch_unique_decisions']==2650347
    assert state['binding']==dict(protocol_sha256=sha(TRAIN/'PROTOCOL_FILESTORE.json'),sample_index_sha256=p['sample_index_sha256'])
    assert all(torch.isfinite(v).all() for v in state['trainable'].values())
    for slot in state['optimizer']['state'].values():
        assert int(slot['step'])==31059
        assert all(torch.isfinite(v).all() for v in slot.values() if isinstance(v,torch.Tensor))
    warmup=int(.03*31059);lr=5e-5*.5*(1+math.cos(math.pi*(31058-warmup)/(31059-warmup)))
    assert all(math.isclose(g['lr'],lr,rel_tol=1e-12,abs_tol=1e-15) for g in state['optimizer']['param_groups'])
    assert sha(p['resume_from']['path'])==p['resume_from']['sha256']
    result=dict(status='PASS',unix=time.time(),checkpoint_sha256=receipt['sha256'],cursor=state['cursor'],
         cumulative_unique_decisions=2650347,new_decisions=1967696,optimizer_step=31059,
         all_state_finite=True,learning_rate=lr,source_locks_unchanged=True,navigation_gain_verified=False)
    with (HERE/'FINAL_CHECKPOINT_ACCEPTANCE.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    assert not torch.cuda.is_initialized();print(json.dumps(result))


if __name__=='__main__':main()
