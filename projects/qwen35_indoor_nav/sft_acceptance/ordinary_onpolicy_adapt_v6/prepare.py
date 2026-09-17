"""Build an audited mixed view, optimizer-preserving stage bridge and finite plan."""
import collections,copy,hashlib,importlib.util,json,os,random,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
OLD=HERE.parent/'ordinary_expanded_v1'
DATA=LINE/'data_pipeline/ordinary_onpolicy_recovery_v2/run_001'
CKPT=OLD/'formal/attempt_001/checkpoint_000004000.pt'
CKPT_SHA='c30a0936ce85723ebca8fa95d52e67453200af708e0c237c6654b583ff7f9775'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def save(name,value):
    with (HERE/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (HERE/'PROTOCOL_FILESTORE.json').exists()
    import torch
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    parent=read(OLD/'PROTOCOL_FILESTORE.json')
    for name,digest in parent['code_sha256'].items():assert sha(OLD/name)==digest,name
    assert read(DATA/'RESULT.json')['data_gate'] and read(DATA/'PIXEL_AUDIT.json')['passed']
    for p,d in read(DATA/'DATA_SEAL.json')['files'].items():assert sha(p)==d,p
    old_data=module('bridge_original_data',OLD/'data.py');rows,old_report=old_data.load_rows()
    original_rows=copy.deepcopy(rows)
    samples=old_data.load_sample_index(OLD/'SAMPLE_INDEX.jsonl',parent['sample_index_sha256'],2650347)
    contracts=module('bridge_causal_keys',DATA.parent/'contracts.py')
    payloads={x['record_id']:x for x in map(json.loads,(DATA/'POLICY_INPUTS.jsonl').read_text().splitlines())}
    targets={x['record_id']:x for x in read(DATA/'TRAINING_TARGETS.json')}
    assert len(payloads)==len(targets)==5365
    texts={x['instruction'] for x in payloads.values()};text_hashes={hashlib.sha256(x.encode()).hexdigest() for x in texts}
    source_eps=read(LINE/'closed_loop_bench/ordinary_stop_calibration_fit_v2/EPISODES_PRIVILEGED.json')
    source_ids={str(e['episode_id']) for e in source_eps}
    overlaps={};conflicts={};matched_records=0;checked_decisions=0;matched_source_files={}
    # Audit all potentially matching R2R instruction texts, not only episode-ID aliases.
    for r in rows:
        if r['source']!='R2R':continue
        if r.get('instruction_sha256') not in text_hashes and r.get('instruction_sha256') and Path(r['policy_file']).stem.removeprefix('policy_') not in source_ids:continue
        source=ROOT/r['sourceRoot'];po=read(source/r['policy_file'])
        if po['instruction'] not in texts:continue
        su=read(source/r['supervision_file']);matched_records+=1;checked_decisions+=len(su['actions'])
        for field in ('policy','supervision'):
            path=source/r[field+'_file'];assert sha(path)==r[field+'_sha256']
            matched_source_files[str(path)]=sha(path)
        for t,a in enumerate(su['actions']):
            value=contracts.payload(po['instruction'],[Path(x).stem for x in po['rgb_sequence'][max(0,t-1):t+1]],su['actions'][max(0,t-8):t])
            key=contracts.key(value)
            if key in targets:
                evidence=dict(old_target=contracts.ACTIONS.index(a),new_target=targets[key]['target'],old_record_id=r['record_id'],old_step=t)
                overlaps[key]=evidence
                if evidence['old_target']!=evidence['new_target']:conflicts[key]=evidence
    eligible=sorted(set(targets)-set(overlaps))
    assert len(eligible)>=1000 and len({h for key in eligible for h in targets[key]['scene_groups']})>=8
    save('SOURCE_RECONCILIATION.json',dict(matched_old_records=matched_records,old_decisions_checked=checked_decisions,
        old_new_input_overlap=overlaps,conflicting_advice=conflicts,
        policy='remove ALL already-existing causal inputs from NEW advice; retain old reference labels',
        eligible_new_advice=eligible,old_sources_sha256=matched_source_files,old_datasets_unchanged=True))
    processor=AutoProcessor.from_pretrained(LINE/'runtime/models/Qwen3.5-2B_15852e8',local_files_only=True,trust_remote_code=False)
    token_cache={}
    new_entries=[]
    for key in eligible:
        value=payloads[key];target=targets[key]['target'];n=len(value['rgb_sha256']);hist=value['executed_actions']
        cache_key=value['instruction'],n
        if cache_key not in token_cache:
            text=processor.apply_chat_template([{'role':'user','content':[
                *[{'type':'image'} for _ in range(n)],{'type':'text','text':value['instruction']}]}],
                tokenize=False,add_generation_prompt=True)
            token_cache[cache_key]=len(processor.tokenizer(text)['input_ids'])+63*n
        est=token_cache[cache_key]+len(hist)+1
        weight=3.2 if not hist or contracts.ACTIONS[target]!=hist[-1] else 1.0
        ri=len(rows)
        rows.append(dict(record_id='recovery_'+key,advice_record_id=key,target=target,decisions=1,
            split='FIT',scene_groups=targets[key]['scene_groups'],source='AUDITED_ONPOLICY_GOAL_TEACHER',
            quality_tier='ACTUAL_VISITED_STATE_ONE_STEP_TEACHER_CHECKED'))
        entry=[ri,0,target,weight,est];new_entries.append(entry)
        samples.append(dict(record_idx=ri,t=0,target=target,weight=weight,est=est))
    save('TRAINING_ROWS.json',rows)
    index=HERE/'SAMPLE_INDEX.jsonl'
    with index.open('xb') as dst,(OLD/'SAMPLE_INDEX.jsonl').open('rb') as src:
        for block in iter(lambda:src.read(2**20),b''):dst.write(block)
        for entry in new_entries:dst.write((json.dumps(entry)+'\n').encode())
        dst.flush();os.fsync(dst.fileno())
    original_plan=old_data.plan_epoch_batches(samples[:2650347],6144,1209,0,3)
    already={i for rank in original_plan for batch in rank[:4000] for i in batch}
    assert len(already)==340712
    r2r_rows={i for i,r in enumerate(original_rows) if r['source']=='R2R'}
    pool=[i for i,x in enumerate(samples[:2650347]) if x['record_idx'] in r2r_rows and i not in already]
    assert len(pool)==464548
    rng=random.Random(1209);ordinary=rng.sample(pool,100000);recovery=[]
    while len(recovery)<11111:
        cycle=list(range(2650347,len(samples)));rng.shuffle(cycle);recovery.extend(cycle)
    recovery=recovery[:11111];occurrences=ordinary+recovery
    packed=old_data.plan_epoch_batches([samples[i] for i in occurrences],6144,1209,0,3)
    assert all(len(r)>1000 for r in packed)
    plan=[[[occurrences[i] for i in batch] for batch in rank[:1000]] for rank in packed]
    flat=[i for rank in plan for batch in rank for i in batch]
    counts=[sum(map(len,rank)) for rank in plan];planned=sum(counts)
    old_ids=[i for i in flat if i<2650347];new_ids=[i for i in flat if i>=2650347]
    assert len(old_ids)==len(set(old_ids)) and set(old_ids).isdisjoint(already)
    assert .08<len(new_ids)/planned<.12 and set(new_ids)<=set(range(2650347,len(samples)))
    assert all(sum(samples[i]['est'] for i in batch)<=6144 for rank in plan for batch in rank)
    maxbatch=max(sum(len(plan[r][j]) for r in range(3)) for j in range(1000))
    save('PLAN.json',plan)
    assert sha(CKPT)==CKPT_SHA==read(str(CKPT)+'.json')['sha256']
    state=torch.load(CKPT,map_location='cpu',weights_only=True)
    assert state['cursor']==dict(epoch=0,position=4000,updates=4000,decisions=113585)
    assert state['binding']['protocol_sha256']==sha(OLD/'PROTOCOL_FILESTORE.json')
    assert state['binding']['sample_index_sha256']==parent['sample_index_sha256']
    assert len(state['trainable'])==len(state['optimizer']['state'])==28
    for slot in state['optimizer']['state'].values():
        assert int(slot['step'])==4000
        assert all(torch.isfinite(v).all() for v in slot.values() if isinstance(v,torch.Tensor))
    assert all(torch.isfinite(v).all() for v in state['trainable'].values())
    source_cursor=copy.deepcopy(state['cursor']);state['cursor']=dict(epoch=0,position=0,updates=0,decisions=0)
    state['binding']=dict(sample_index_sha256=sha(index),source_checkpoint_sha256=CKPT_SHA,stage='ONPOLICY_NEW_DATA_STAGE_ZERO_OPTIMIZER_PRESERVED')
    state['global_decisions']=0;state['charged_compute_decisions']=0
    initial=HERE/'initial_from_4000_optimizer_preserved.pt'
    with initial.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    back=torch.load(initial,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,back['trainable'][key]) for key,v in state['trainable'].items())
    assert all(torch.equal(v,back['optimizer']['state'][key][field]) for key,slot in state['optimizer']['state'].items() for field,v in slot.items() if isinstance(v,torch.Tensor))
    save(initial.name+'.json',dict(sha256=sha(initial),cursor=back['cursor'],global_decisions=0,charged_compute_decisions=0,
        source_checkpoint_sha256=CKPT_SHA,source_cursor=source_cursor,optimizer_reset=False,optimizer_step_offset=4000,unix=time.time()))
    tested=subprocess.run([str(ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),'-I','-S','-B',str(HERE/'tests.py')],capture_output=True,text=True,timeout=60)
    save('CPU_TEST_RESULT.json',dict(passed=tested.returncode==0,output=tested.stdout+tested.stderr));assert tested.returncode==0
    audit=dict(status='PASS',source_checkpoint_sha256=CKPT_SHA,initial_parameter_reload_exact=True,optimizer_moments_preserved=True,
        source_optimizer_step=4000,new_stage_cursor=back['cursor'],recovery_unique_inputs=len(eligible),
        existing_input_overlap_excluded=len(overlaps),contradictory_advice_excluded=len(conflicts),
        planned_updates=1000,planned_decisions=planned,rank_planned_decisions=counts,
        ordinary_draws=len(old_ids),ordinary_unique=len(set(old_ids)),recovery_draws=len(new_ids),recovery_unique_drawn=len(set(new_ids)),
        recovery_decision_fraction=len(new_ids)/planned,all_data_FIT=True,old_prefix_not_reexecuted=True,
        lr_reference_total_updates=31059,optimizer_step_offset=4000,peak_lr=5e-6,automatic_training_restart=False)
    save('BUILD_AUDIT.json',audit)
    code={str(OLD/name):digest for name,digest in parent['code_sha256'].items()}
    deps=[OLD/'data.py',OLD/'reuse.py',OLD/'PROTOCOL_FILESTORE.json',HERE.parent/'ordinary_expanded_continue_v2/reuse.py',
        DATA/'DATA_SEAL.json',DATA/'RESULT.json',DATA.parent/'contracts.py']
    for p in list(HERE.glob('*.py'))+[HERE/'PLAN_ZH.md',HERE/'PLAN.json',HERE/'TRAINING_ROWS.json',HERE/'SOURCE_RECONCILIATION.json',HERE/'BUILD_AUDIT.json']+deps:
        code[p.name if p.parent==HERE else str(p)]=sha(p)
    p=copy.deepcopy(parent)
    p.update(id='Q35N_ORDINARY_ONPOLICY_ADAPT_V6',created='2026-09-13',code_sha256=code,
        snapshot=dict(path=str(HERE/'TRAINING_ROWS.json'),training_index_sha256=sha(HERE/'TRAINING_ROWS.json'),decisions_per_epoch=len(samples)),
        sample_index_sha256=sha(index),sampling_plan_sha256=sha(HERE/'PLAN.json'),recovery_unique_inputs=len(eligible),
        resume_from=dict(path=str(initial),sha256=sha(initial),receipt_sha256=sha(str(initial)+'.json'),updates=0),
        accounting=dict(initial_charged_decisions=0,deadline_unix=time.time()+1800,legacy_decisions_in_this_stage=0),
        optimizer_step_offset=4000,lr_reference_total_updates=31059,current_segment_start_updates=0,
        current_segment_planned_decisions=planned,current_segment_max_new_updates=1000,max_global_batch_decisions=maxbatch,
        full_epoch_updates=1000,training_plan_is_full_original_epoch=False,automatic_retry=False,automatic_long_training=False,
        expected_final_cursor=dict(epoch=1,position=0,updates=1000,decisions=counts[0]),expected_final_global_decisions=planned,
        source_checkpoint=dict(path=str(CKPT),sha256=CKPT_SHA,optimizer_updates=4000),
        resume_semantics='new-index stage0 cursor with original weights/moments/RNG retained; optimizer clock offset4000',
        segment_note='fixed1000 updates, about10% audited onpolicy recovery; ordinary base engineering, not method novelty')
    p['optimizer']['learning_rate']=5e-6;p['budget'].update(wall_seconds=1800,max_updates=1001,max_decisions=planned+maxbatch)
    for key in ('pilot_exposure_by_source','planned_pilot_decisions','warm_start_source'):p.pop(key,None)
    save('PROTOCOL_FILESTORE.json',p)
    book=copy.deepcopy(read(OLD/'RUNBOOK.json'));book.update(name=p['id'],lease_wall_seconds=2100,code_sha256=code)
    step=book['steps'][0];step['name']='onpolicy_short_adaptation';step['argv'][-1]=str(HERE/'supervise_filestore.py')
    for key,value in list(step['env_extra'].items()):
        if isinstance(value,str):step['env_extra'][key]=value.replace('/cache/ordinary_expanded_v1/','/cache/ordinary_onpolicy_adapt_v6/')
    cache=LINE/'runtime/cache/ordinary_onpolicy_adapt_v6'
    for name in ('hf','xdg','xdg/torch/kernels','torch','cuda'):(cache/name).mkdir(parents=True,exist_ok=True)
    save('RUNBOOK.json',book)
    save('MAIN_AGENT_APPROVAL.json',dict(scope='USER_20260913_CONTINUE_UNTIL_FIRST_POSITIVE_BOUNDED_VISITED_STATE_CORRECTION',
        protocol_sha256=sha(HERE/'PROTOCOL_FILESTORE.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),
        data_gate_passed=True,new_updates_cap=1000,wall_seconds=1800,no_new_method_claim=True,
        gpu_scope=[3,4,5],restore_exact_holders=True,automatic_retry=False,automatic_long_training=False,unix=time.time()))
    assert sha(CKPT)==CKPT_SHA and not torch.cuda.is_initialized()
    print(json.dumps(audit,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
