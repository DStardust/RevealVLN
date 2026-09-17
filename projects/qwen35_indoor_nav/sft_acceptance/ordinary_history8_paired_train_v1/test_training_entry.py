"""Actual CPU DataLoader/processor transport and training-source checks."""
import ast
import itertools
import json
import math
import os
from pathlib import Path
import runpy
import socket
import tempfile
import time
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert os.environ.get('TMPDIR')==str(LINE/'.th8p')
    import torch
    torch.set_num_threads(2)
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    runtime=runpy.run_path(str(HERE/'runtime.py'))
    data=runtime['ordinary']();rows,_=data.load_rows()
    model=runtime['load']('paired_model_cpu',HERE/'model.py')
    processor=AutoProcessor.from_pretrained(model.MODEL,local_files_only=True,trust_remote_code=False)
    ids=json.loads((LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json').read_text())['ids']
    processor.tokenizer.add_special_tokens({'additional_special_tokens':list(ids)})
    assert all(processor.tokenizer.convert_tokens_to_ids(k)==v for k,v in ids.items())
    image_id=json.loads((model.MODEL/'config.json').read_text())['image_token_id']
    collate=model.make_collate(processor.tokenizer.pad_token_id,{a:ids[model.EXEC_TOKENS[a]] for a in model.ACTIONS[:-1]},ids['<NAV_ACTION_QUERY>'],image_id)
    with (HERE/'SELECTED_SAMPLES.jsonl').open() as stream:
        selected=[json.loads(x) for x in itertools.islice(stream,16)]
    samples=[dict(record_idx=x['entry'][0],t=x['entry'][1],target=x['entry'][2],weight=x['entry'][3]) for x in selected]
    address=str(LINE/'.th8p'/'paired_probe_socket')
    assert len(address.encode())<108 and not Path(address).exists()
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:sock.bind(address);sock.listen(1)
    finally:sock.close()
    assert Path(address).is_socket();Path(address).unlink()
    checks={}
    for arm,store in runtime['stores'](rows).items():
        ds=model.DecisionDataset(samples,store,processor)
        batches=(list(range(i,i+8)) for i in (0,8))
        loader=torch.utils.data.DataLoader(ds,batch_sampler=batches,num_workers=2,prefetch_factor=2,collate_fn=collate,timeout=90)
        iterator=iter(loader);got=[]
        try:
            for batch in iterator:
                expected=samples[len(got):len(got)+8]
                assert batch['targets'].tolist()==[x['target'] for x in expected]
                assert torch.equal(batch['weights'],torch.tensor([x['weight'] for x in expected]))
                assert batch['action_index'].shape==(8,2)
                frames=64 if arm=='treatment_prefix8' else sum(1 if x['t']==0 else 2 for x in expected)
                assert batch['image_grid_thw'].shape[0]==frames
                assert all(x.device.type=='cpu' for x in batch.values())
                got.extend(batch['targets'].tolist())
        finally:iterator._shutdown_workers()
        assert len(got)==16 and all(not w.is_alive() for w in iterator._workers)
        checks[arm]=dict(actual_rows=16,workers=2,workers_joined=True)
    train=runpy.run_path(str(HERE/'train.py'),run_name='CPU_IMPORT_ONLY')
    assert math.isclose(train['lr_at'](119),5e-5) and train['lr_at'](0)>0
    assert train['lr_at'](4000)==0 and all(train['lr_at'](t)>=train['lr_at'](t+1) for t in range(120,4000))
    supervisor=runpy.run_path(str(HERE/'supervise.py'),run_name='CPU_IMPORT_ONLY')
    assert supervisor['HERE']==HERE and supervisor['ROOT']==LINE.parents[1]
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_CPU_TRAINING_ENTRY',unix=time.time(),actual_transport=checks,
        actual_unix_bind=True,tempdir=tempfile.gettempdir(),all_cpu_only=True,
        actual_model_gradient_accumulation_gate_separate=True,new_training_updates=0,
        first_selected_source_ids=[x['source_sample_id'] for x in selected],
        sources={str(p):runtime['sha'](p) for p in HERE.glob('*.py')})
    with (HERE/'CPU_TRAINING_ENTRY_RESULT.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)
if __name__=='__main__':main()

