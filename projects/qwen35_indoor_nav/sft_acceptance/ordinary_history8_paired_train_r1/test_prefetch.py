"""CPU-only lifecycle, exact real tensor equivalence and revision scope regression."""
import ast
import itertools
import json
import os
from pathlib import Path
import runpy
import threading
import time
HERE=Path(__file__).resolve().parent
P=HERE.parent/'ordinary_history8_paired_train_v1'
LINE=HERE.parents[1]

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=runpy.run_path(str(HERE/'runtime.py'))
    Prefetch=runpy.run_path(str(HERE/'prefetch.py'))['OrderedPrefetch']
    checks=[]
    before={t.ident for t in threading.enumerate()}
    p=Prefetch(lambda:iter(range(100)))
    assert list(p)==list(range(100)) and not p.is_alive() and p.max_queued<=2
    checks.append('exact_100_order_exhaustion_join_queue_bound')
    p=Prefetch(lambda:iter(()))
    assert list(p)==[] and not p.is_alive()
    checks.append('empty_iterator_joined')
    def failure():
        yield 11
        raise ValueError('TEST_VISIBLE_PRODUCER_ERROR')
    p=iter(Prefetch(failure))
    assert next(p)==11
    try:next(p)
    except ValueError as exc:assert str(exc)=='TEST_VISIBLE_PRODUCER_ERROR'
    else:raise AssertionError('producer exception lost')
    assert not p.is_alive()
    checks.append('producer_exception_propagated_and_joined')
    p=iter(Prefetch(lambda:itertools.count()))
    assert next(p)==0
    p.close();p.close()
    assert not p.is_alive() and p.queue.empty()
    checks.append('early_close_full_queue_joined_idempotent')
    p=Prefetch(lambda:iter([1]))
    p.close()
    try:iter(p)
    except RuntimeError:pass
    else:raise AssertionError('closed before start accepted')
    checks.append('closed_before_start_rejected')
    p=Prefetch(lambda:iter([1]),timeout=.001)
    p.started=True  # Simulate an already-exited producer that delivered no message.
    p.thread.start();p.thread.join()
    while not p.queue.empty():p.queue.get_nowait()
    try:next(p)
    except TimeoutError:pass
    else:raise AssertionError('empty queue timeout lost')
    checks.append('consumer_timeout_visible')
    assert {t.ident for t in threading.enumerate()}==before

    import torch
    torch.set_num_threads(2)
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    data=r['ordinary']();rows,_=data.load_rows()
    model=r['load']('r1_cpu_model',HERE/'model.py')
    processor=AutoProcessor.from_pretrained(model.MODEL,local_files_only=True,trust_remote_code=False)
    ids=json.loads((LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json').read_text())['ids']
    processor.tokenizer.add_special_tokens({'additional_special_tokens':list(ids)})
    assert all(processor.tokenizer.convert_tokens_to_ids(k)==v for k,v in ids.items())
    image_id=json.loads((model.MODEL/'config.json').read_text())['image_token_id']
    collate=model.make_collate(processor.tokenizer.pad_token_id,{a:ids[model.EXEC_TOKENS[a]] for a in model.ACTIONS[:-1]},ids['<NAV_ACTION_QUERY>'],image_id)
    selected=[json.loads(x) for x in (P/'SELECTED_SAMPLES.jsonl').read_text().splitlines()]
    packing=runpy.run_path(str(HERE/'batching_fixed4.py'))
    all_samples=[dict(record_idx=x['entry'][0],t=x['entry'][1],target=x['entry'][2],weight=x['entry'][3],est=x['entry'][4]) for x in selected]
    longest=sorted(range(len(all_samples)),key=lambda i:packing['history8_est'](all_samples[i]),reverse=True)[:4]
    indices=list(range(96))+longest
    samples=[all_samples[i] for i in indices]
    summaries={}
    for arm in ('control_recent2','treatment_prefix8'):
        ds=model.DecisionDataset(samples,r['stores'](rows)[arm],processor)
        order=[list(range(i,i+4)) for i in range(0,len(samples),4)]
        reference=[collate([ds[i] for i in batch]) for batch in order]
        ds2=model.DecisionDataset(samples,r['stores'](rows)[arm],processor)
        def stream():
            for batch in order:yield collate([ds2[i] for i in batch])
        got=iter(Prefetch(stream))
        seen=0
        try:
            for index,batch in enumerate(got):
                target=reference[index]
                assert set(batch)==set(target)
                assert all(torch.equal(batch[k],target[k]) for k in batch)
                assert all(v.device.type=='cpu' for v in batch.values())
                assert batch['targets'].tolist()==[samples[i]['target'] for i in order[index]]
                seen+=len(batch['targets'])
        finally:got.close()
        assert seen==100 and not got.is_alive()
        summaries[arm]=dict(actual_decisions=seen,batches=len(reference),all_tensor_fields_bitwise_equal=True,
                            thread_joined=True,queue_max=got.max_queued,process_workers=0)
        del reference,ds,ds2
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    train=runpy.run_path(str(HERE/'train.py'),run_name='CPU_IMPORT_ONLY')
    assert train['HERE']==HERE and train['lr_at'](119)==5e-5 and train['lr_at'](4000)==0
    supervisor=runpy.run_path(str(HERE/'supervise.py'),run_name='CPU_IMPORT_ONLY')
    assert supervisor['HERE']==HERE
    lease=runpy.run_path(str(HERE/'lease.py'),run_name='CPU_IMPORT_ONLY')
    assert lease['HERE']==HERE
    assert not torch.cuda.is_initialized()
    old=json.loads((P/'PROTOCOL.json').read_text())
    assert all(r['sha'](Path(path))==digest for path,digest in old['code_sha256'].items())
    result=dict(status='PASS_CPU_THREAD_TRANSPORT_ONLY',unix=time.time(),lifecycle_checks=checks,
        actual_tensor_equivalence=summaries,first_96_plus_longest4_source_ids=[selected[i]['source_sample_id'] for i in indices],
        old_sources_unchanged=True,no_gpu_initialized=True,training_updates=0,
        sources={str(p):r['sha'](p) for p in HERE.glob('*.py')})
    r['save'](HERE/'CPU_TEST_RESULT.json',result,True)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','first_96_plus_longest4_source_ids')}),flush=True)
if __name__=='__main__':main()

