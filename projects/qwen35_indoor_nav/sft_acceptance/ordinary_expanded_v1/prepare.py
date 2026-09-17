"""CPU-only index, warm-start conversion and frozen admission; never releases GPUs."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_sync_recovery_v1'
CKPT=OLD/'formal/attempt_001/checkpoint_000051301.pt'
CKPT_SHA='4d60cdb09a909121856e438547df5c6e1e9ea35fc40a8f257302bf563af4979e'
def load(name):
    s=importlib.util.spec_from_file_location('prepare_expanded_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,obj):
    with (HERE/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())


def index():
    from transformers import AutoProcessor
    d=load('data');rows,report=d.load_rows()
    processor=AutoProcessor.from_pretrained(LINE/'runtime/models/Qwen3.5-2B_15852e8',local_files_only=True,trust_remote_code=False)
    result=d.build_sample_index(rows,processor,3.2,HERE/'SAMPLE_INDEX.jsonl')
    save('SAMPLE_INDEX_RESULT.json',dict(result,snapshot=report))
    print(json.dumps(result),flush=True)


def freeze():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_ONLY_PREPARATION'
    assert sha(CKPT)==CKPT_SHA and read(str(CKPT)+'.json')['sha256']==CKPT_SHA
    parent=read(OLD/'PROTOCOL_FILESTORE.json')
    tests=subprocess.run([str(ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'test_expanded.py')],capture_output=True,text=True,timeout=60)
    save('CPU_TEST_RESULT.json',dict(passed=tests.returncode==0,output=tests.stdout+tests.stderr));assert tests.returncode==0
    r=load('reuse');d=load('data');rows,report=d.load_rows();si=read(HERE/'SAMPLE_INDEX_RESULT.json')
    samples=d.load_sample_index(HERE/'SAMPLE_INDEX.jsonl',si['sha256'],2650347)
    plan=d.plan_epoch_batches(samples,6144,1209,0,3)
    assert len({len(x) for x in plan})==1
    maxbatch=max(sum(len(plan[rank][i]) for rank in range(3)) for i in range(len(plan[0])))
    exposure={};pilot=0
    for rank in range(3):
        for batch in plan[rank][:4000]:
            for idx in batch:
                source=rows[samples[idx]['record_idx']]['source'];exposure[source]=exposure.get(source,0)+1;pilot+=1
    assert pilot<350000 and len(plan[0])>4000
    # New pool => fresh cursor and fresh Adam moments. This is explicitly a warm-start,
    # not falsely labeled exact resume. Reuse only the learned parameter values.
    state=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert state['binding']['protocol_sha256']==sha(OLD/'PROTOCOL_FILESTORE.json')
    assert state['binding']['sample_index_sha256']==parent['sample_index_sha256']
    assert state['cursor']==read(str(CKPT)+'.json')['cursor']
    assert all(torch.isfinite(t).all() for t in state['trainable'].values())
    prior_cursor=copy.deepcopy(state['cursor'])
    state['optimizer']['state']={}
    for group in state['optimizer']['param_groups']:group['lr']=5e-5
    state['binding']=dict(sample_index_sha256=si['sha256'],warm_start_source_sha256=CKPT_SHA,stage='NEW_DATA_STAGE_ZERO')
    state['cursor']=dict(epoch=0,position=0,updates=0,decisions=0)
    state['global_decisions']=0;state['charged_compute_decisions']=0
    torch.manual_seed(1209);random.seed(1209)
    state['torch_rng']=torch.get_rng_state();state['python_rng']=random.getstate()
    initial=HERE/'initial_from_51301.pt'
    with initial.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    back=torch.load(initial,map_location='cpu',weights_only=True)
    assert all(torch.equal(back['trainable'][k],v) for k,v in state['trainable'].items())
    assert back['optimizer']['state']=={} and back['cursor']['updates']==0
    save(initial.name+'.json',dict(sha256=sha(initial),cursor=back['cursor'],global_decisions=0,charged_compute_decisions=0,
         unix=time.time(),source_checkpoint_sha256=CKPT_SHA,source_cursor=prior_cursor,optimizer_reset=True))
    code={p.name:sha(p) for p in HERE.glob('*.py')}
    code['PLAN_ZH.md']=sha(HERE/'PLAN_ZH.md')
    code.update({str(r.OLD/name):digest for name,digest in r.HASHES.items()})
    # Bind dependencies inherited from old data/model adapters too.
    ref=HERE.parent/'ordinary_baseline_v2'
    for p in (ref/'data.py',ref/'runner.py',LINE/'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1/TOKENIZER_ADDITION.json'):code[str(p)]=sha(p)
    protocol=copy.deepcopy(parent)
    protocol.update(id='Q35N_ORDINARY_EXPANDED_V1',created='2026-09-12',seed=1209,epochs=1,
        snapshot=dict(training_index_sha256=report['training_index_sha256'],decisions_per_epoch=2650347,
                      path=str(LINE/'data_pipeline/ordinary_expansion_v1/training_snapshot_20260912_v1/run_001')),
        sample_index_sha256=si['sha256'],code_sha256=code,max_global_batch_decisions=maxbatch,
        resume_from=dict(path=str(initial),sha256=sha(initial),receipt_sha256=sha(str(initial)+'.json'),updates=0),
        warm_start_source=dict(path=str(CKPT),sha256=CKPT_SHA,updates=51301,optimizer_reset=True,data_cursor_reset=True),
        accounting=dict(initial_charged_decisions=0,deadline_unix=time.time()+3600,legacy_decisions_in_this_stage=0),
        resume_semantics='weights-only warm start 51301 -> explicit stage0; exact same-stage checkpoint resume only',
        segment_note='Standard mixed-data BC pilot; no special data, no new model mechanism; no automatic long training',
        pilot_exposure_by_source=exposure,planned_pilot_decisions=pilot,
        full_epoch_updates=len(plan[0]),automatic_retry=False,automatic_long_training=False)
    protocol['optimizer']['learning_rate']=5e-5
    protocol['budget'].update(wall_seconds=3600,max_updates=4000,max_decisions=350000)
    for key in ('gpu2_note','parent_protocol_sha256'):protocol.pop(key,None)
    save('PROTOCOL_FILESTORE.json',protocol)
    runbook=read(OLD/'RUNBOOK_FILESTORE.json')
    runbook.update(name=protocol['id'],out='lease_v1',lease_wall_seconds=3900,code_sha256=code)
    runbook.pop('gpu2_note',None)
    caches=LINE/'runtime/cache/ordinary_expanded_v1'
    env=dict(HF_HOME=str(caches/'hf'),XDG_CACHE_HOME=str(caches/'xdg'),TORCH_HOME=str(caches/'torch'),
             CUDA_CACHE_PATH=str(caches/'cuda'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             HF_HUB_DISABLE_TELEMETRY='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    runbook['steps']=[dict(name='expanded_pilot_train',argv=[str(ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),
                          '-I','-S','-B','-u',str(HERE/'supervise_filestore.py')],env_extra=env)]
    save('RUNBOOK.json',runbook)
    save('PREFLIGHT.json',dict(status='CPU_READY',snapshot=report,sample_index=si['sha256'],
         checkpoint_parameter_reload_exact=True,optimizer_reset=True,new_stage_cursor=back['cursor'],
         full_epoch_updates=len(plan[0]),max_global_batch_decisions=maxbatch,pilot_decisions=pilot,pilot_by_source=exposure,
         live_GPU_training_verified=False,navigation_gain_verified=False))
    save('MAIN_AGENT_APPROVAL.json',dict(scope='USER_20260912_RESEARCH_NEXT_STEP_AND_EXECUTE_BOUNDED_ORDINARY_PILOT',
         protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),
         source_snapshot_seal_sha256=report['snapshot_seal_sha256'],old_source_files_preserved=True,
         no_new_method_experiment=True,automatic_retry=False,automatic_long_training=False,unix=time.time()))
    print(json.dumps(dict(status='FROZEN',pilot_decisions=pilot,by_source=exposure,deadline=protocol['accounting']['deadline_unix'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['index','freeze']);args=p.parse_args()
    index() if args.phase=='index' else freeze()
