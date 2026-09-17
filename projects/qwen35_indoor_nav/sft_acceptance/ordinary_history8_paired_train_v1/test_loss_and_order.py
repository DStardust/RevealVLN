"""CPU exact gradient algebra and actual frozen paired data-order audit."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);torch.manual_seed(1209)
    fn=runpy.run_path(str(HERE/'loss.py'))['local_micro_loss']
    x=torch.randn(96,13,dtype=torch.float64)
    y=torch.arange(96)%4
    weights=torch.where(torch.arange(96)%5==0,3.2,1.).double()
    initial=torch.randn(13,4,dtype=torch.float64)
    baseline=initial.clone().requires_grad_()
    reference=(torch.nn.functional.cross_entropy(x@baseline,y,reduction='none')*weights).sum()/weights.sum()
    reference.backward()
    tests=[]
    for size in (4,8,16,32):
        gradients=[]
        for rank in range(3):
            parameter=initial.clone().requires_grad_()
            for start in range(rank*32,(rank+1)*32,size):
                stop=start+size
                fn(x[start:stop]@parameter,y[start:stop],weights[start:stop],float(weights.sum()),3).backward()
            gradients.append(parameter.grad)
        actual=torch.stack(gradients).mean(0)
        error=float((actual-baseline.grad).abs().max())
        assert error<1e-12
        tests.append(dict(microbatch=size,max_abs_error=error))
    report=json.loads((HERE/'PREPARATION_RESULT.json').read_text())
    assert not report['runtime_allowed'] and report['ordinary_only'] and report['fit_houses']==51
    assert sha(HERE/'PLAN.json')==report['plan_sha256']
    assert sha(HERE/'DECISION_ORDER.json')==report['order_sha256']
    assert sha(HERE/'SELECTED_SAMPLES.jsonl')==report['selected_samples_sha256']
    plan=json.loads((HERE/'PLAN.json').read_text());order=json.loads((HERE/'DECISION_ORDER.json').read_text())
    assert [i for step in range(4000) for rank in range(3) for i in plan[rank][step]]==order
    assert len(set(order))==len(order)==384000
    samples=[json.loads(line) for line in (HERE/'SELECTED_SAMPLES.jsonl').read_text().splitlines()]
    assert [s['source_sample_id'] for s in samples]==order
    assert all(len(s['entry'])==5 and 0<=s['entry'][0]<37114 for s in samples)
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_CPU_ACCUMULATION_AND_ORDER_ONLY',unix=time.time(),gradient_tests=tests,
                selected_decisions=384000,global_batch=96,rank_batch=32,
                actual_gpu_accumulation_parity_tested=False,training_started=False,
                gpu_launches=0,training_updates=0)
    with (HERE/'CPU_LOSS_AND_ORDER_RESULT.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

