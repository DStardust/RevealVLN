"""Read-only CPU numerical acceptance of stage200; does not select a policy."""
import json
import os
from pathlib import Path
import runpy
import sys
import time
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
TRAIN=LINE/'sft_acceptance/ordinary_history8_paired_train_r1'
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert len(sys.argv)==2 and sys.argv[1] in ('control_recent2','treatment_prefix8')
    arm=sys.argv[1]
    r=runpy.run_path(str(TRAIN/'runtime.py'))
    read=lambda p:json.loads(p.read_text())
    protocol=read(TRAIN/'PROTOCOL.json')
    for path,digest in protocol['code_sha256'].items():assert r['sha'](Path(path))==digest,path
    import torch
    assert not torch.cuda.is_initialized()
    out=TRAIN/arm/'run_001';checkpoint=out/'checkpoint_000000200.pt'
    receipt=read(Path(str(checkpoint)+'.json'))
    assert r['sha'](checkpoint)==receipt['sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['binding']==dict(protocol_sha256=r['sha'](TRAIN/'PROTOCOL.json'),sample_index_sha256=protocol['sample_index_sha256'],arm=arm)
    assert state['cursor']==dict(epoch=0,position=200,updates=200,decisions=6400)
    assert state['global_decisions']==state['charged_compute_decisions']==19200
    assert len(state['trainable'])==len(state['optimizer']['state'])==28
    for row in state['optimizer']['state'].values():
        assert int(row['step'])==200 and row['exp_avg'].dtype==row['exp_avg_sq'].dtype==torch.float32
        assert torch.isfinite(row['exp_avg']).all() and torch.isfinite(row['exp_avg_sq']).all()
    original=torch.load(r['BRIDGE'],map_location='cpu',weights_only=True)['trainable']
    changes={}
    for name,value in state['trainable'].items():
        assert value.dtype==torch.float32 and torch.isfinite(value).all() and value.shape==original[name].shape
        changes[name]=dict(coordinates_changed=int((value!=original[name]).sum()),coordinates=value.numel(),
            l2_change=float(torch.linalg.vector_norm(value-original[name])),
            bf16_visible_changes=int((value.bfloat16()!=original[name].bfloat16()).sum()))
    assert any(x['coordinates_changed'] for x in changes.values())
    initial=[read(out/f'INITIAL_RANK{i}.json') for i in range(3)]
    assert len({x['trainable_fingerprint'] for x in initial})==1 and all(x['optimizer_reset'] for x in initial)
    workers=[read(out/f'WORKERS_RANK{i}.json') for i in range(3)]
    assert all(x['workers']==[] and x['process_workers']==0 and x['queued_batches_max']==2 for x in workers)
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_FIRST200_NUMERICAL_ONLY',unix=time.time(),arm=arm,checkpoint_sha256=receipt['sha256'],
        checkpoint=str(checkpoint),updates=200,global_decisions=19200,trainable_count=28,
        actual_parameter_changes=changes,initial_ranks_equal=True,data_process_workers=0,
        final_checkpoint_selection_unchanged=4000,navigation_gain_verified=False)
    r['save'](HERE/(arm+'_FIRST200.json'),result,True)
    print(json.dumps({k:v for k,v in result.items() if k!='actual_parameter_changes'}),flush=True)
if __name__=='__main__':main()

