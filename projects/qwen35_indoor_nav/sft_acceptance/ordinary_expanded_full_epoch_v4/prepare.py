"""Audit remaining unique decisions and freeze one full-epoch continuation."""
import collections
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_expanded_continue_v2'
FIRST=HERE.parent/'ordinary_expanded_v1'
CKPT=OLD/'formal/attempt_001/checkpoint_000008000.pt'
CKPT_SHA='1c5b5aa89e15caeb3d2f29f328280bdce7eb8240ce659cd8bffda865afe06f98'
INDEX=FIRST/'SAMPLE_INDEX.jsonl'
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,x):
    with (HERE/name).open('x') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def load(name):
    s=importlib.util.spec_from_file_location('full_epoch_prepare_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU_ONLY_PREPARATION'
    assert not (HERE/'PROTOCOL_FILESTORE.json').exists(),'ALREADY_FROZEN'
    closed=LINE/'reviews/Q35N_ORDINARY_STOP_CALIBRATION_V1/FINAL_DECISION.json'
    prior=read(closed)
    assert prior['status']=='COMPLETE' and not prior['positive_development_signal'],'PREVIOUS_GATE_MUST_BE_CLOSED'
    import torch
    assert not torch.cuda.is_initialized()
    parent=read(OLD/'PROTOCOL_FILESTORE.json')
    for name,digest in parent['code_sha256'].items():assert sha(OLD/name)==digest,name
    assert sha(CKPT)==CKPT_SHA==read(Path(str(CKPT)+'.json'))['sha256']
    state=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert state['binding']['protocol_sha256']==sha(OLD/'PROTOCOL_FILESTORE.json')
    assert state['binding']['sample_index_sha256']==parent['sample_index_sha256']
    assert state['cursor']==dict(epoch=0,position=8000,updates=8000,decisions=227411)
    assert state['global_decisions']==state['charged_compute_decisions']==682651
    assert all(torch.isfinite(x).all() for x in state['trainable'].values())
    assert len(state['optimizer']['state'])==len(state['trainable'])>0
    for values in state['optimizer']['state'].values():
        assert int(values['step'])==8000
        assert all(torch.isfinite(values[key]).all() for key in ('exp_avg','exp_avg_sq'))
    test=subprocess.run([str(PY),'-I','-S','-B',str(HERE/'tests.py')],capture_output=True,text=True,timeout=60)
    save('CPU_TEST_RESULT.json',dict(passed=test.returncode==0,output=test.stdout+test.stderr));assert test.returncode==0,test.stderr
    data=load('data');control=load('control');rows,report=data.load_rows()
    samples=data.load_sample_index(INDEX,parent['sample_index_sha256'],2650347)
    plan=data.plan_epoch_batches(samples,6144,1209,0,3)
    assert all(len(rank)==31059 for rank in plan)
    before=set();remaining=set();by_source=collections.Counter();classes=collections.Counter()
    old_counts=[control.processed_by_rank([plan],state['cursor'],rank) for rank in range(3)]
    assert sum(old_counts)==682651 and old_counts[0]==227411
    for rank in plan:
        for batch in rank[:8000]:before.update(batch)
        for batch in rank[8000:]:
            for index in batch:
                assert index not in remaining,'DUPLICATE_REMAINING_DECISION'
                remaining.add(index);s=samples[index]
                by_source[rows[s['record_idx']]['source']]+=1;classes[s['target']]+=1
    assert len(before)==682651 and before.isdisjoint(remaining)
    total=len(before)+len(remaining);dropped=2650347-total
    assert 0<=dropped<154 and len(remaining)>1900000
    final_counts=[sum(len(batch) for batch in rank) for rank in plan]
    assert sum(final_counts)==total
    save('RESUME_AUDIT.json',dict(status='PASS',source_checkpoint_sha256=CKPT_SHA,
         source_protocol_sha256=sha(OLD/'PROTOCOL_FILESTORE.json'),optimizer_step=8000,
         optimizer_moments_preserved=True,cursor=state['cursor'],rank_decisions_before=old_counts,
         final_rank_decisions=final_counts,next_segment_decisions=len(remaining),next_segment_by_source=dict(by_source),
         next_segment_action_counts=dict(classes),previous_and_next_conditioned_indices_disjoint=True,
         epoch_planned_unique_decisions=total,epoch_dropped_tail_decisions=dropped,
         expected_final_cursor=dict(epoch=1,position=0,updates=31059,decisions=final_counts[0]),
         full_epoch_updates=31059,seed=1209,sample_index_sha256=parent['sample_index_sha256'],
         snapshot_sha256=report['training_index_sha256'],old_source_checkpoint_preserved=True,gpu_initialized=False))
    code={str(OLD/name):digest for name,digest in parent['code_sha256'].items()}
    for path in list(HERE.glob('*.py'))+[HERE/'PLAN_ZH.md',HERE/'RESUME_AUDIT.json',OLD/'reuse.py',OLD/'PROTOCOL_FILESTORE.json',closed]:
        code[path.name if path.parent==HERE else str(path)]=sha(path)
    protocol=copy.deepcopy(parent)
    protocol.update(id='Q35N_ORDINARY_EXPANDED_FULL_EPOCH_V4',created='2026-09-13',code_sha256=code,
         resume_from=dict(path=str(CKPT),sha256=CKPT_SHA,receipt_sha256=sha(Path(str(CKPT)+'.json')),updates=8000),
         accounting=dict(initial_charged_decisions=682651,deadline_unix=time.time()+14400,legacy_decisions_in_this_stage=682651),
         current_segment_start_updates=8000,current_segment_max_new_updates=23059,
         current_segment_planned_decisions=len(remaining),current_segment_by_source=dict(by_source),
         segment_note='Finish original single epoch; unchanged optimizer/schedule/loss/model/data; no controllers',
         expected_epoch_unique_decisions=total,expected_final_cursor=read(HERE/'RESUME_AUDIT.json')['expected_final_cursor'],
         low_lr_failed_branch_extra_compute=341939,automatic_retry=False,automatic_long_training=False,
         automatic_next_epoch=False,source_checkpoint_is_not_adopted_navigation_model=True)
    assert protocol['optimizer']==parent['optimizer'] and protocol['optimizer']['learning_rate']==5e-5
    assert protocol['epochs']==1
    # One update of headroom lets the existing epoch-boundary code complete.
    # It cannot start a second epoch because epochs stays exactly one.
    protocol['budget'].update(wall_seconds=14400,max_updates=31060,max_decisions=2650347+154)
    save('PROTOCOL_FILESTORE.json',protocol)
    runbook=copy.deepcopy(read(OLD/'RUNBOOK.json'))
    runbook.update(name=protocol['id'],lease_wall_seconds=14700,code_sha256=code)
    step=runbook['steps'][0];step['name']='complete_original_single_epoch';step['argv'][-1]=str(HERE/'supervise_filestore.py')
    for key,value in list(step['env_extra'].items()):
        if isinstance(value,str):step['env_extra'][key]=value.replace('/cache/ordinary_expanded_continue_v2/','/cache/ordinary_expanded_full_epoch_v4/')
    cache=LINE/'runtime/cache/ordinary_expanded_full_epoch_v4'
    for name in ('hf','xdg','xdg/torch/kernels','torch','cuda'):(cache/name).mkdir(parents=True,exist_ok=True)
    save('RUNBOOK.json',runbook)
    save('MAIN_AGENT_APPROVAL.json',dict(scope='USER_CONTINUE_WORK_TOWARD_POSITIVE_OUTCOME_COMPLETE_ONE_FROZEN_ORDINARY_EPOCH',
         protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),
         resume_audit_sha256=sha(HERE/'RESUME_AUDIT.json'),preceding_gate_closed_sha256=sha(closed),
         new_updates_cap=23059,new_decisions_planned=len(remaining),wall_seconds_cap=14400,
         no_new_method_experiment=True,no_automatic_retry=True,no_automatic_next_epoch=True,
         preserve_best_4000_checkpoint=True,unix=time.time()))
    assert sha(CKPT)==CKPT_SHA and not torch.cuda.is_initialized()
    print(json.dumps(dict(status='FROZEN',next_decisions=len(remaining),epoch_decisions=total,dropped=dropped,
         final_cursor=protocol['expected_final_cursor'],wall_seconds_cap=14400),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
