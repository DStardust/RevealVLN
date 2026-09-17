"""Read-only source audit and exclusive creation of the preregistration."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from collections import defaultdict

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
ROOT = LINE.parents[1]
DATA = LINE / 'data_pipeline/ordinary_pilot_v1'
G2 = LINE / 'reviews/Q35N_G2_INTERFACE_ACCEPTANCE_V1'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(name, obj):
    with (OUT/name).open('x') as f: json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)

def main():
    assert not (OUT/'EXPERIMENT_SPEC.json').exists()
    checks=[]
    for d in (DATA,G2):
        r=subprocess.run(['sha256sum','--check','--quiet','SHA256SUMS'],cwd=d,capture_output=True,text=True)
        checks.append(dict(path=str(d.relative_to(ROOT)),manifest_sha256=sha(d/'SHA256SUMS'),returncode=r.returncode,output=r.stdout+r.stderr))
    save('SEALED_SOURCE_CHECKS.json',checks)
    assert all(x['returncode']==0 for x in checks)
    rows=[json.loads(x) for x in (DATA/'TRAINING_INDEX.jsonl').read_text().splitlines()]
    groups=defaultdict(list); locks={}; all_pixels=set()
    for row in rows:
        pp=DATA/row['policy_file']; sp=DATA/row['supervision_file']
        p=json.loads(pp.read_text()); s=json.loads(sp.read_text())
        assert set(p)=={'instruction','rgb_sequence'}
        assert len(p['rgb_sequence'])==len(s['actions'])==len(s['frames'])
        assert s['actions'][-1]=='STOP' and all(a in ['move_forward','turn_left','turn_right'] for a in s['actions'][:-1])
        for path in (pp,sp):
            assert path.resolve().is_relative_to(ROOT)
            locks[str(path.relative_to(ROOT))]=sha(path)
        for ref,frame in zip(p['rgb_sequence'],s['frames']):
            path=(DATA/row['rgb_reference_root']/ref).resolve()
            assert path.is_relative_to(DATA) and path.stem==frame['rgb_sha256'] and path.is_file()
            all_pixels.add(str(path))
        key=(s['scene_id'],s['physical_source_route_sha256'])
        groups[key].append(dict(row,source_episode_id=int(s['source_episode_id']),physical_route_sha256=key[1],instruction_episode_id=int(pp.stem.split('_')[-1]),decisions=len(s['actions'])))
    assert len(rows)==298 and len(groups)==99
    houses=json.loads((DATA/'SPLIT_FREEZE.json').read_text())['fit_pilot']
    split={'train':[],'seen_house_route_dev':[],'closed_loop':[]}
    for house in houses:
        routes=sorted([v for k,v in groups.items() if k[0]==house],key=lambda v:v[0]['source_episode_id'])
        for i,aliases in enumerate(routes):
            aliases.sort(key=lambda x:x['instruction_episode_id'])
            split['train' if i<15 else 'seen_house_route_dev'].extend(aliases)
            if 15<=i<17: split['closed_loop'].append(aliases[0])
    assert len({x['job_id'] for x in split['train']})==75
    assert len({x['job_id'] for x in split['seen_house_route_dev']})==24 and len(split['closed_loop'])==10
    assert not ({x['physical_route_sha256'] for x in split['train']}&{x['physical_route_sha256'] for x in split['seen_house_route_dev']})
    save('SPLIT.json',split)
    for path in [DATA/'TRAINING_INDEX.jsonl',DATA/'SPLIT_FREEZE.json',DATA/'JOBS.json',DATA/'worker.py',G2/'probe.py',G2/'TOKENIZER_ADDITION.json',G2/'ENVIRONMENT_ACCEPTANCE.json',ROOT/'AGENTS.md',LINE/'AGENTS.md',LINE/'MAINLINE_FREEZE_V3.md',LINE/'03_DATA_CONTRACT.md',LINE/'04_VALIDATION_PLAN.md',LINE/'05_ISOLATION_AND_HANDOFF.md',LINE/'authorizations/PARALLEL_SFT_AND_MECHANISM_V1.json']:
        locks[str(path.relative_to(ROOT))]=sha(path)
    for name in ['MODEL_FILE_LOCK.json','MODEL_SUPPLEMENT_LOCK.json']:
        for rec in json.loads((G2/name).read_text()):
            path=LINE/'runtime/models/Qwen3.5-2B_15852e8'/rec['file']
            assert sha(path)==rec['sha256']
            locks[str(path.relative_to(ROOT))]=rec['sha256']
    save('SOURCE_LOCK.json',locks)
    save('DATA_AUDIT.json',dict(data_integrity_pass=True,instruction_records=len(rows),physical_routes=len(groups),train_records=len(split['train']),dev_records=len(split['seen_house_route_dev']),referenced_png_files=len(all_pixels),official_val_test_read=False,mechanism_results_read=False,scope='seen-house route holdout; no unseen-house claim',pixel_integrity='All referenced PNGs covered by verified sealed source manifest; pixel decode checked in new loader'))
    spec=dict(experiment_id='Q35N_ORDINARY_ACTION_SFT_ACCEPTANCE_V1',created_unix=time.time(),scientific_pass=False,
        seed=1109,model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',split_sha256=sha(OUT/'SPLIT.json'),source_lock_sha256=sha(OUT/'SOURCE_LOCK.json'),
        algorithm=dict(loss='real_action_cross_entropy_only',actions=['move_forward','turn_left','turn_right','STOP'],memory_slots=8,hidden=2048,window_frames=2,history_actions=8,lora_rank=8,lora_alpha=16,lora_dropout=0.0,lora_targets=['q_proj','v_proj'],vision_frozen=True,base_dtype='bfloat16',writer_head_dtype='float32',memory_dtype='bfloat16',implicit_cache=False,reader=False,collision_input=False),
        sensors=dict(rgb_resolution=[224,224],hfov_degrees=90,sensor_height_m=1.25,agent_height_m=1.5,agent_radius_m=0.1,forward_m=0.25,turn_degrees=15,physics=False,sliding=False),
        training=dict(optimizer='AdamW',learning_rate=0.0001,betas=[0.9,0.999],eps=1e-8,weight_decay=0.01,gradient_clip_norm=1.0,schedule='constant; no warmup',batch_size=1,tbptt=4,gradient_accumulation_chunks=8,gradient_normalization='sum action losses divided by actual decision count in the eight chunks',max_optimizer_updates=400,absolute_update_limit_including_smoke=500,smoke_optimizer_updates=0,token_budget=8000000,max_sequence_tokens=512,overflow='fail; never silently truncate',sample_order='seed1109 Random.shuffle over all train instruction records, repeated epochs',memory_reset='zero at every episode/instruction; detach after each <=4-step chunk; carry within same episode across optimizer updates',checkpoint_selection='initial and terminal update400 only; never dev best',engineering_restart='new amendment; preserve all prior costs and updates; no automatic retries'),
        offline=dict(records='every original instruction of the fixed 24 routes, all decisions including STOP',timing=['before_update0','after_terminal'],metrics=['decision_micro_CE','decision_micro_accuracy','route_macro_CE','route_macro_accuracy','4x4_target_row_prediction_column_confusion','STOP_TP_FP_FN_TN'],memory='full causal episode recurrence; zero reset between instructions'),
        closed_loop=dict(selection='first two numerical source_episode_id route-dev routes per house; smallest original instruction_episode_id',episodes_total_max=20,motion_actions_max=512,stop_decisions_max=1,policy='argmax four logits; no heuristic STOP or follower',metrics='stop geodesic and Euclidean goal distance; ever within 3m and stopped within 3m separate; collisions; geodesic nDTW against source reference_path with radius3m; limit and service failure',official_metrics=False,interface_check='initial RGB exact pixel hash and initial pose agreement in simulator; transport policy payload RGB only; failures do not authorize replacement episodes',fallback='if interface fails retain evidence and offline only; no efficacy claims'),
        gates=dict(training_interface='finite real CE; next-step action CE gives finite nonzero >1e-12 previous-memory, writer, write_query and LoRA B gradients; vision frozen; whitelist negative tests; checkpoint reload logits max_abs<=1e-5; finite effective parameter updates; source integrity and split isolation',offline_learning_signal='after decision_micro_CE < before CE, descriptive only; report accuracy and STOP even if worse',scientific_pass=False),
        resources=dict(physical_gpu_excluded=[3],single_gpu=True,max_gpu_gib=28,max_worker_ram_gib=48,max_all_gpu_seconds=21600,max_new_output_bytes=20*1024**3,new_downloads=0,all_forward_tokens_budget=16000000,oom='stop and report; no automatic reduced-config retry'),
        authorized_scope='ordinary SFT and bounded engineering navigation acceptance; no complete paper experiment, no mechanism labels or auxiliary losses')
    save('EXPERIMENT_SPEC.json',spec)
    save('PREREGISTRATION_LOCK.json',{'EXPERIMENT_SPEC.json':sha(OUT/'EXPERIMENT_SPEC.json'),'SPLIT.json':sha(OUT/'SPLIT.json'),'created_before_any_model_result':True})
    print(json.dumps({'routes':99,'train_records':len(split['train']),'dev_records':len(split['seen_house_route_dev']),'spec_sha256':sha(OUT/'EXPERIMENT_SPEC.json')}))

if __name__=='__main__':main()
