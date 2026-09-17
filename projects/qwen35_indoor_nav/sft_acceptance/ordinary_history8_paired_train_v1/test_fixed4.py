"""Final fixed4 plan and assembled trainer validation; prior GPU comparison remains FAIL."""
import ast
import json
from pathlib import Path
import runpy
import time
HERE=Path(__file__).resolve().parent
def main():
    r=runpy.run_path(str(HERE/'runtime.py'));b=runpy.run_path(str(HERE/'batching_fixed4.py'))
    raw=[json.loads(x) for x in (HERE/'SELECTED_SAMPLES.jsonl').read_text().splitlines()]
    samples=[dict(t=v['entry'][1],est=v['entry'][4]) for v in raw]
    order=[];largest=0
    for start in range(0,384000,32):
        groups=b['partition'](samples[start:start+32],start)
        assert len(groups)==8 and all(len(g)==4 for g in groups)
        for group in groups:
            order.extend(group);largest=max(largest,4*max(b['history8_est'](samples[i]) for i in group))
    assert order==list(range(384000)) and largest==4740
    for name in ('train_fixed4.py','supervise_fixed4.py'):
        m=runpy.run_path(str(HERE/name),run_name='CPU_IMPORT_ONLY')
        assert m['HERE']==HERE and callable(m['main'])
        assert 'batching_fixed4.py' in m['_source'] if name=='train_fixed4.py' else 'train_fixed4.py' in m['_source']
        ast.parse(m['_source'])
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    prior=json.loads((HERE/'PACKING_AND_ENTRY_RESULT.json').read_text())
    assert prior['status']=='PASS_PACKING_AND_FINAL_ENTRY'
    assert all(x['actual_tokens']<=x['history8_estimate'] for x in prior['actual_encoding_checks'])
    result=dict(status='PASS_FIXED4_CPU',unix=time.time(),microbatch=4,accumulation=8,global_batch=96,
        all_384000_order_exact=True,maximum_padded_tokens=4740,
        prior_actual_encoding_receipt_sha256=r['sha'](HERE/'PACKING_AND_ENTRY_RESULT.json'),
        prior_actual_dataloader_receipt_sha256=r['sha'](HERE/'CPU_TRAINING_ENTRY_RESULT.json'),
        gpu_fixed4_accumulation_gate_separate=True,training_started=False,
        sources={str(p):r['sha'](p) for p in HERE.glob('*.py')})
    with (HERE/'FIXED4_CPU_RESULT.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)
if __name__=='__main__':main()

