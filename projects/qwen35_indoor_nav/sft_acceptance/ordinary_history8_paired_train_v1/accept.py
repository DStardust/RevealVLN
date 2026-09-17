"""CPU final checkpoint acceptance; this is not navigation evaluation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('paired_accept_runtime',HERE/'runtime.py')
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['control_recent2','treatment_prefix8'],required=True);a=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    assert not torch.cuda.is_initialized()
    p=json.loads((HERE/'PROTOCOL.json').read_text());out=HERE/a.arm/'run_001'
    for path,digest in p['code_sha256'].items():assert r.sha(Path(path))==digest,path
    result=json.loads((out/'RESULT.json').read_text())
    launch=json.loads((HERE/a.arm/'LAUNCH_RESULT.json').read_text())
    assert launch['status']=='COMPLETE' and all(not x['pids'] for x in launch['gpu_after'])
    assert result['status']=='COMPLETE' and result['updates']==4000 and result['global_decisions']==384000
    checkpoint=out/'checkpoint_000004000.pt';assert result['latest_checkpoint']==str(checkpoint)
    receipt=json.loads(Path(str(checkpoint)+'.json').read_text());assert receipt['sha256']==r.sha(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['cursor']==dict(epoch=1,position=0,updates=4000,decisions=128000)
    assert state['global_decisions']==state['charged_compute_decisions']==384000
    assert state['binding']==dict(protocol_sha256=r.sha(HERE/'PROTOCOL.json'),sample_index_sha256=p['sample_index_sha256'],arm=a.arm)
    assert len(state['trainable'])==len(state['optimizer']['state'])==28
    assert all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in state['trainable'].values())
    for row in state['optimizer']['state'].values():
        assert int(row['step'])==4000 and row['exp_avg'].dtype==row['exp_avg_sq'].dtype==torch.float32
        assert torch.isfinite(row['exp_avg']).all() and torch.isfinite(row['exp_avg_sq']).all()
    ranks=[json.loads((out/f'FINAL_RANK{i}.json').read_text()) for i in range(3)]
    assert len({x['trainable_fingerprint'] for x in ranks})==1
    assert all(x['data_workers_joined'] and x['all_parameters_finite'] and x['optimizer_steps']==[4000] for x in ranks)
    progress=[json.loads(x) for x in (out/'PROGRESS.jsonl').read_text().splitlines()]
    assert progress[0]['updates']==1 and progress[-1]['updates']==4000
    assert all(x['decisions']==x['updates']*96 and x['arm']==a.arm for x in progress)
    r.save(HERE/a.arm/'ACCEPTANCE.json',dict(status='PASS_FINAL_TRAINING_ONLY',unix=time.time(),arm=a.arm,
        checkpoint=str(checkpoint),checkpoint_sha256=receipt['sha256'],updates=4000,decisions=384000,
        all_trainable_fp32_finite=True,all_ranks_equal=True,cleanup_complete=True,navigation_result=None),True)
    print(json.dumps(dict(status='PASS_FINAL_TRAINING_ONLY',arm=a.arm,updates=4000)),flush=True)
if __name__=='__main__':main()

