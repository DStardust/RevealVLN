"""CPU-only sampling/optimizer audit and bounded admission; no GPU signals."""
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
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,obj):
    with (HERE/name).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (HERE/'PROTOCOL_FILESTORE.json').exists(),'ALREADY_FROZEN'
    import torch
    assert not torch.cuda.is_initialized()
    parent=read(OLD/'PROTOCOL_FILESTORE.json')
    for name,digest in parent['code_sha256'].items():assert sha(OLD/name)==digest,name
    assert sha(CKPT)==CKPT_SHA==read(str(CKPT)+'.json')['sha256']
    state=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert state['binding']['protocol_sha256']==sha(OLD/'PROTOCOL_FILESTORE.json')
    assert state['binding']['sample_index_sha256']==parent['sample_index_sha256']
    assert state['cursor']==dict(epoch=0,position=4000,updates=4000,decisions=113585)
    assert state['global_decisions']==state['charged_compute_decisions']==340712
    assert all(torch.isfinite(v).all() for v in state['trainable'].values())
    assert len(state['optimizer']['state'])==len(state['trainable'])==28
    for slot in state['optimizer']['state'].values():
        assert int(slot['step'])==4000
        assert all(torch.isfinite(v).all() for v in slot.values() if isinstance(v,torch.Tensor))
    test=subprocess.run([str(ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'tests.py')],capture_output=True,text=True,timeout=60)
    save('CPU_TEST_RESULT.json',dict(passed=test.returncode==0,output=test.stdout+test.stderr));assert test.returncode==0
    d=module('original_r2r_sampling',OLD/'data.py');rows,report=d.load_rows()
    samples=d.load_sample_index(OLD/'SAMPLE_INDEX.jsonl',parent['sample_index_sha256'],2650347)
    original=d.plan_epoch_batches(samples,6144,1209,0,3)
    prefix=[r[:4000] for r in original]
    seen={i for r in prefix for b in r for i in b};assert len(seen)==340712
    eligible_rows={i for i,r in enumerate(rows) if r['source']=='R2R'}
    assert len(eligible_rows)==8700
    assert all(rows[i]['split']=='FIT' and rows[i]['instruction_provenance']=='official_human_instruction' for i in eligible_rows)
    eligible=[i for i,x in enumerate(samples) if x['record_idx'] in eligible_rows and i not in seen]
    assert len(eligible)==464548
    packed=d.plan_epoch_batches([samples[i] for i in eligible],6144,1209,0,3)
    mapped=[[[eligible[i] for i in b] for b in r] for r in packed]
    assert all(len(r)==4691 for r in mapped)
    plan=[prefix[r]+mapped[r] for r in range(3)]
    # Whole stitched plan is unique; dropped tail is disclosed separately.
    flat=[i for r in plan for b in r for i in b]
    assert len(set(flat))==len(flat),'DUPLICATE_STITCHED_PLAN_INDEX'
    next_ids=[i for r in plan for b in r[4000:8000] for i in b]
    assert len(next_ids)==396001 and set(next_ids).isdisjoint(seen)
    assert all(rows[samples[i]['record_idx']]['source']=='R2R' for i in next_ids)
    old_counts=[sum(map(len,r[:4000])) for r in plan]
    new_counts=[sum(map(len,r[4000:8000])) for r in plan]
    assert old_counts==[113585,113149,113978] and new_counts==[131911,132201,131889]
    assert all(max(sum(samples[i]['est'] for i in b) for b in r)<=6144 for r in plan)
    save('PLAN.json',plan)
    audit=dict(status='PASS',source_checkpoint_sha256=CKPT_SHA,checkpoint_not_rewritten=True,
        optimizer_moments_preserved=True,optimizer_step=4000,cursor=state['cursor'],
        rank_decisions_before=old_counts,next_rank_decisions=new_counts,next_segment_decisions=396001,
        next_segment_by_source={'R2R':396001},previous_and_next_conditioned_decision_indices_disjoint=True,
        r2r_records=8700,r2r_physical_routes=len({rows[i]['physical_source_route_sha256'] for i in eligible_rows}),
        r2r_houses=51,r2r_remaining_candidate_decisions=464548,
        unexecuted_eligible_after_segment=464548-396001,packing_dropped_tail_decisions=464548-sum(len(b) for r in mapped for b in r),
        all_training_data_FIT=True,sample_index_sha256=parent['sample_index_sha256'],sampling_plan_sha256=sha(HERE/'PLAN.json'),
        prefix_plan_unchanged=True,new_training_plan_updates=8691,lr_reference_total_updates=31059,
        source_filter_not_proof_of_bad_synthetic_labels=True,gpu_initialized=False)
    save('RESUME_AUDIT.json',audit)
    code={str(OLD/name):digest for name,digest in parent['code_sha256'].items()}
    deps=[OLD/'data.py',OLD/'reuse.py',OLD/'PROTOCOL_FILESTORE.json',HERE.parent/'ordinary_expanded_continue_v2/reuse.py',
          LINE/'reviews/Q35N_ORDINARY_FULL_EPOCH_V4/WORKFLOW_RESULT.json',
          LINE/'reviews/Q35N_ORDINARY_FULL_EPOCH_V4/REPORT_ZH.md']
    for p in list(HERE.glob('*.py'))+[HERE/'PLAN.json',HERE/'PLAN_ZH.md',HERE/'RESUME_AUDIT.json']+deps:
        code[p.name if p.parent==HERE else str(p)]=sha(p)
    protocol=copy.deepcopy(parent)
    protocol.update(id='Q35N_ORDINARY_R2R_ADAPT_V5',created='2026-09-13',code_sha256=code,
        resume_from=dict(path=str(CKPT),sha256=CKPT_SHA,receipt_sha256=sha(str(CKPT)+'.json'),updates=4000),
        accounting=dict(initial_charged_decisions=340712,deadline_unix=time.time()+4200,legacy_decisions_in_this_stage=340712),
        resume_semantics='exact original weights/optimizer/RNG/counters at4000; new explicitly registered R2R-only suffix sampling',
        segment_note='4000 new updates from R2R only; original mixed prefix not reexecuted; fixed original LR clock',
        current_segment_start_updates=4000,current_segment_max_new_updates=4000,
        current_segment_planned_decisions=396001,current_segment_by_source={'R2R':396001},
        sampling_plan_sha256=sha(HERE/'PLAN.json'),lr_reference_total_updates=31059,
        full_epoch_updates=8691,training_plan_is_full_original_epoch=False,
        expected_final_cursor=dict(epoch=0,position=8000,updates=8000,decisions=245496),
        expected_final_global_decisions=736713,automatic_retry=False,automatic_long_training=False)
    protocol['budget'].update(wall_seconds=4200,max_decisions=760712,max_updates=8000)
    protocol['max_global_batch_decisions']=max(sum(len(plan[r][i]) for r in range(3)) for i in range(8691))
    for k in ('pilot_exposure_by_source','planned_pilot_decisions','warm_start_source'):protocol.pop(k,None)
    save('PROTOCOL_FILESTORE.json',protocol)
    runbook=copy.deepcopy(read(OLD/'RUNBOOK.json'))
    runbook.update(name=protocol['id'],lease_wall_seconds=4500,code_sha256=code)
    step=runbook['steps'][0];step['name']='r2r_source_adaptation';step['argv'][-1]=str(HERE/'supervise_filestore.py')
    for key,value in list(step['env_extra'].items()):
        if isinstance(value,str):step['env_extra'][key]=value.replace('/cache/ordinary_expanded_v1/','/cache/ordinary_r2r_adapt_v5/')
    cache=LINE/'runtime/cache/ordinary_r2r_adapt_v5'
    for name in ('hf','xdg','xdg/torch/kernels','torch','cuda'):(cache/name).mkdir(parents=True,exist_ok=True)
    save('RUNBOOK.json',runbook)
    save('MAIN_AGENT_APPROVAL.json',dict(scope='USER_20260913_CONTINUE_UNTIL_FIRST_POSITIVE_BOUNDED_R2R_SOURCE_CONTROL',
        protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),
        no_new_method_claim=True,new_updates_cap=4000,new_decisions_cap=420000,automatic_retry=False,
        gpu_scope=[3,4,5],restore_exact_holders=True,old_results_and_input_locks_preserved=True,unix=time.time()))
    assert sha(CKPT)==CKPT_SHA and not torch.cuda.is_initialized()
    print(json.dumps(dict(status='FROZEN',new_decisions=396001,total=736713,deadline=protocol['accounting']['deadline_unix'])),flush=True)
if __name__=='__main__':main()
