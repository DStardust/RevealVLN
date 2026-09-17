import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import traceback
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as q

OUT=Path(__file__).resolve().parent;LINE=OUT.parents[1]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
e=module('efficiency_helpers',OUT.parent/'efficiency_run_v1/worker.py');e.h.OUT=OUT
k=module('triangular_kernel',OUT/'kernel.py')
def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)

def main():
    lock=json.loads((OUT/'LOCK.json').read_text())
    for name,digest in lock['code'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    row=json.loads((OUT/'SUBSET.json').read_text())['rows'][0]
    record=e.old.load_record(row)
    policy=e.old.build();policy.eval()
    initial=torch.load(OUT.parent/'v1/checkpoints/initial.pt',map_location='cpu',weights_only=True)['trainable']
    e.old.load_trainable(policy,initial)
    original=q.torch_chunk_gated_delta_rule
    baseline_warm=e.one_chunk(policy,record)
    raw=[e.one_chunk(policy,record) for _ in range(3)]
    try:
        q.torch_chunk_gated_delta_rule=k.triangular_chunk_gated_delta_rule
        fast_warm=e.one_chunk(policy,record)
        fast=[e.one_chunk(policy,record) for _ in range(3)]
    finally:q.torch_chunk_gated_delta_rule=original
    noise=max(e.h.compare(raw[0]['gradient'],x['gradient'])['relative_L2'] for x in raw[1:])
    gradient=[e.h.compare(raw[0]['gradient'],x['gradient']) for x in fast]
    delta=max(float((raw[0]['logits']-x['logits']).abs().max()) for x in fast)
    limit=.02+.01*float(raw[0]['logits'].abs().max())
    passed=noise<=.10 and delta<=limit and all(x['cosine']>=.98 and x['relative_L2']<=max(.15,3*noise) for x in gradient)
    raw_m=statistics.median(x['seconds'] for x in raw);fast_m=statistics.median(x['seconds'] for x in fast)
    result=dict(correctness_pass=passed,reference_same_card_gradient_noise=noise,gradient_checks=gradient,
        logit_max_abs_error=delta,logit_limit=limit,reference_logit=raw[0]['logits'].tolist(),fast_logit=fast[0]['logits'].tolist(),
        baseline_warm_seconds=baseline_warm['seconds'],fast_warm_seconds=fast_warm['seconds'],
        reference_seconds=[x['seconds'] for x in raw],triangular_seconds=[x['seconds'] for x in fast],
        reference_median_seconds=raw_m,triangular_median_seconds=fast_m,speedup=raw_m/fast_m,
        recommended_for_next_version=passed and fast_m<=.9*raw_m,optimizer_updates=0,navigation_episodes=0,
        scientific_pass=False,peak_cuda_allocated=torch.cuda.max_memory_allocated(),forward_tokens=policy.forward_tokens)
    save('BENCHMARK.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as ex:save('FAILURE.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
