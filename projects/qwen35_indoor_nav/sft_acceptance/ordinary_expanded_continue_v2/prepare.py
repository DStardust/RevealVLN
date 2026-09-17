"""Read-only checkpoint/plan audit and finite continuation registration, CPU only."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_expanded_v1'
CKPT=OLD/'formal/attempt_001/checkpoint_000004000.pt'
CKPT_SHA='c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775'
INDEX=OLD/'SAMPLE_INDEX.jsonl'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,obj):
    with (HERE/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def load(name):
    s=importlib.util.spec_from_file_location('continue_prepare_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_ONLY_PREPARATION'
    assert not (HERE/'PROTOCOL_FILESTORE.json').exists(),'ALREADY_FROZEN'
    import torch
    assert not torch.cuda.is_initialized()
    parent=read(OLD/'PROTOCOL_FILESTORE.json')
    for name,digest in parent['code_sha256'].items():assert sha(OLD/name)==digest,name
    assert sha(CKPT)==CKPT_SHA and read(str(CKPT)+'.json')['sha256']==CKPT_SHA
    state=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert state['binding']['protocol_sha256']==sha(OLD/'PROTOCOL_FILESTORE.json')
    assert state['binding']['sample_index_sha256']==parent['sample_index_sha256']
    assert state['cursor']==read(str(CKPT)+'.json')['cursor']
    assert state['cursor']['epoch']==0 and state['cursor']['position']==state['cursor']['updates']==4000
    assert state['global_decisions']==state['charged_compute_decisions']==340712
    assert all(torch.isfinite(t).all() for t in state['trainable'].values())
    opt=state['optimizer'];assert len(opt['state'])==len(state['trainable']) and len(opt['state'])>0
    for value in opt['state'].values():
        assert int(value['step'])==4000
        for k in ('exp_avg','exp_avg_sq'):assert torch.isfinite(value[k]).all()
    assert all(k in state for k in ('torch_rng','cuda_rng','python_rng'))
    test=subprocess.run([str(ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'tests.py')],capture_output=True,text=True,timeout=60)
    save('CPU_TEST_RESULT.json',dict(passed=test.returncode==0,output=test.stderr));assert test.returncode==0,test.stderr
    d=load('data');control=load('control');rows,report=d.load_rows()
    samples=d.load_sample_index(INDEX,parent['sample_index_sha256'],2650347)
    plan=d.plan_epoch_batches(samples,6144,1209,0,3)
    assert all(len(p)==31059 for p in plan)
    old_counts=[control.processed_by_rank([plan],state['cursor'],r) for r in range(3)]
    assert sum(old_counts)==340712 and old_counts[0]==state['cursor']['decisions']
    before=set();after=set();exposure={}
    for rank in range(3):
        for batch in plan[rank][:4000]:before.update(batch)
        for batch in plan[rank][4000:8000]:
            for idx in batch:
                assert idx not in after,'DUPLICATE_NEW_DECISION';after.add(idx)
                source=rows[samples[idx]['record_idx']]['source'];exposure[source]=exposure.get(source,0)+1
    assert len(before)==340712 and before.isdisjoint(after) and 0<len(after)<360000
    next_counts=[sum(len(b) for b in plan[r][4000:8000]) for r in range(3)]
    assert sum(next_counts)==len(after)
    save('RESUME_AUDIT.json',dict(status='PASS',source_checkpoint_sha256=CKPT_SHA,source_protocol_sha256=sha(OLD/'PROTOCOL_FILESTORE.json'),
        checkpoint_not_rewritten=True,optimizer_moments_preserved=True,optimizer_step=4000,optimizer_tensors=len(opt['state']),
        cursor=state['cursor'],rank_decisions_before=old_counts,next_rank_decisions=next_counts,
        next_segment_decisions=len(after),next_segment_by_source=exposure,
        previous_and_next_conditioned_decision_indices_disjoint=True,full_epoch_updates=31059,
        seed=1209,learning_rate_schedule_continues_at_step=4000,all_training_data_FIT=True,
        snapshot_sha256=report['training_index_sha256'],sample_index_sha256=parent['sample_index_sha256'],
        gpu_initialized=torch.cuda.is_initialized()))
    code={str(OLD/name):digest for name,digest in parent['code_sha256'].items()}
    for p in list(HERE.glob('*.py'))+[HERE/'PLAN_ZH.md',OLD/'reuse.py',OLD/'PROTOCOL_FILESTORE.json',HERE/'RESUME_AUDIT.json']:
        code[p.name if p.parent==HERE else str(p)]=sha(p)
    protocol=copy.deepcopy(parent)
    protocol.update(id='Q35N_ORDINARY_EXPANDED_CONTINUE_V2',created='2026-09-12',code_sha256=code,
        resume_from=dict(path=str(CKPT),sha256=CKPT_SHA,receipt_sha256=sha(str(CKPT)+'.json'),updates=4000),
        accounting=dict(initial_charged_decisions=340712,deadline_unix=time.time()+4200,legacy_decisions_in_this_stage=340712),
        resume_semantics='same data plan / optimizer moments / cursor preserved; original rank0 RNG restoration convention retained',
        segment_note='Continue positions [4000,8000); no controller, special data, optimizer reset or automatic retry',
        current_segment_start_updates=4000,current_segment_max_new_updates=4000,
        current_segment_planned_decisions=len(after),current_segment_by_source=exposure,
        automatic_retry=False,automatic_long_training=False)
    protocol['budget'].update(wall_seconds=4200,max_decisions=700712,max_updates=8000)
    # The inherited first-stage provenance remains available, but does not describe this continuation.
    for key in ('pilot_exposure_by_source','planned_pilot_decisions','warm_start_source'):protocol.pop(key,None)
    protocol['initial_stage_protocol']=str(OLD/'PROTOCOL_FILESTORE.json')
    save('PROTOCOL_FILESTORE.json',protocol)
    runbook=copy.deepcopy(read(OLD/'RUNBOOK.json'))
    runbook.update(name=protocol['id'],lease_wall_seconds=4500,code_sha256=code)
    step=runbook['steps'][0];step['name']='expanded_continuation_train'
    step['argv'][-1]=str(HERE/'supervise_filestore.py')
    for key,value in list(step['env_extra'].items()):
        if isinstance(value,str) and '/cache/ordinary_expanded_v1/' in value:
            step['env_extra'][key]=value.replace('/cache/ordinary_expanded_v1/','/cache/ordinary_expanded_continue_v2/')
    cache=LINE/'runtime/cache/ordinary_expanded_continue_v2'
    for name in ('hf','xdg','xdg/torch/kernels','torch','cuda'):(cache/name).mkdir(parents=True,exist_ok=True)
    save('RUNBOOK.json',runbook)
    save('MAIN_AGENT_APPROVAL.json',dict(scope='USER_20260912_CONTINUE_TOWARD_POSITIVE_OUTCOME_BOUNDED_STANDARD_BC',
        protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),
        resume_audit_sha256=sha(HERE/'RESUME_AUDIT.json'),new_updates_cap=4000,new_decisions_cap=360000,
        no_new_method_experiment=True,automatic_retry=False,automatic_long_training=False,unix=time.time()))
    assert sha(CKPT)==CKPT_SHA and not torch.cuda.is_initialized()
    print(json.dumps(dict(status='FROZEN',next_decisions=len(after),by_source=exposure,deadline=protocol['accounting']['deadline_unix']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
