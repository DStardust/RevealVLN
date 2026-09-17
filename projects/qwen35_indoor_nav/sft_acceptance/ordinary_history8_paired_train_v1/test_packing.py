"""Audit fixed 384k partitions and actual longest-token encodings before training."""
import ast
import collections
import json
import os
from pathlib import Path
import runpy
import time
HERE=Path(__file__).resolve().parent
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    from transformers import AutoProcessor
    runtime=runpy.run_path(str(HERE/'runtime.py'));packing=runpy.run_path(str(HERE/'batching.py'))
    selected=[json.loads(x) for x in (HERE/'SELECTED_SAMPLES.jsonl').read_text().splitlines()]
    samples=[dict(record_idx=x['entry'][0],t=x['entry'][1],target=x['entry'][2],weight=x['entry'][3],est=x['entry'][4]) for x in selected]
    sizes=collections.Counter();flattened=[];maximum=0
    for base in range(0,384000,32):
        batches=packing['partition'](samples[base:base+32],base)
        for batch in batches:
            sizes[len(batch)]+=1;flattened.extend(batch)
            maximum=max(maximum,len(batch)*max(packing['history8_est'](samples[i]) for i in batch))
    assert flattened==list(range(384000)) and maximum<=6144 and set(sizes)=={4,8}
    data=runtime['ordinary']();rows,_=data.load_rows()
    model=runtime['load']('packing_model',HERE/'model.py')
    processor=AutoProcessor.from_pretrained(model.MODEL,local_files_only=True,trust_remote_code=False)
    ids=json.loads((runtime['LINE']/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json').read_text())['ids']
    processor.tokenizer.add_special_tokens({'additional_special_tokens':list(ids)})
    config=json.loads((model.MODEL/'config.json').read_text())
    collate=model.make_collate(processor.tokenizer.pad_token_id,{a:ids[model.EXEC_TOKENS[a]] for a in model.ACTIONS[:-1]},ids['<NAV_ACTION_QUERY>'],config['image_token_id'])
    picks=sorted(set(list(range(8))+sorted(range(384000),key=lambda i:packing['history8_est'](samples[i]),reverse=True)[:8]+[next(i for i,x in enumerate(samples) if x['t']==0)]))
    checked=[]
    for arm,store in runtime['stores'](rows).items():
        ds=model.DecisionDataset(samples,store,processor)
        for idx in picks:
            batch=collate([ds[idx]])
            actual=batch['input_ids'].shape[1];estimate=packing['history8_est'](samples[idx])
            assert actual<=estimate,(arm,idx,actual,estimate)
            if arm=='treatment_prefix8':assert actual==estimate,(actual,estimate)
            checked.append(dict(arm=arm,source_sample_id=selected[idx]['source_sample_id'],actual_tokens=actual,history8_estimate=estimate))
    # The changed training/transport entries are imported and compiled again after edits.
    for name in ('train.py','supervise.py','lease.py','accept.py'):
        module=runpy.run_path(str(HERE/name),run_name='CPU_IMPORT_ONLY')
        assert module['HERE']==HERE and callable(module['main'])
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_PACKING_AND_FINAL_ENTRY',unix=time.time(),all_384000_order_exact=True,
        microbatch_counts=dict(sizes),max_estimated_padded_tokens=maximum,actual_encoding_checks=checked,
        prior_cpu_loader_receipt_sha256=runtime['sha'](HERE/'CPU_TRAINING_ENTRY_RESULT.json'),
        prior_transport_regression_sha256=runtime['sha'](HERE/'TRANSPORT_R1_REGRESSION.json'),
        gpu_worst_batch_startup_check_required=True,gpu_launches=0,training_updates=0,
        source_hashes={str(p):runtime['sha'](p) for p in HERE.glob('*.py')})
    with (HERE/'PACKING_AND_ENTRY_RESULT.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','actual_encoding_checks')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

