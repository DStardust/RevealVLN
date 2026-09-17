"""Freeze a fully specified training pair only after actual fixed4 and CPU gates pass."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import time
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1];ROOT=LINE.parents[1]
PY=ROOT/'.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'
QPY=LINE/'.envs/q35n_qwen_g2_v1/bin/python3'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text())
def save(name,value):
    with (HERE/name).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False)
def main():
    assert not (HERE/'PROTOCOL.json').exists() and not (HERE/'lease_v1').exists()
    d=LINE/'reviews/Q35N_HISTORY8_FIXED4_ACCUMULATION_V1'
    result=read(d/'RESULT.json');launch=read(d/'LAUNCH_RESULT.json')
    assert result['status']=='PASS_INTERFACE_ONLY' and result['recommended_microbatch']==4
    assert launch['status']=='COMPLETE' and launch['cleanup']['exit_code']==0 and not launch['gpu_after']['processes']
    assert read(HERE/'FIXED4_CPU_RESULT.json')['status']=='PASS_FIXED4_CPU'
    for path,digest in read(HERE/'FIXED4_CPU_RESULT.json')['sources'].items():assert sha(Path(path))==digest,path
    assert read(HERE/'CPU_LOSS_AND_ORDER_RESULT.json')['status']=='PASS_CPU_ACCUMULATION_AND_ORDER_ONLY'
    assert read(HERE/'CPU_TRAINING_ENTRY_RESULT.json')['status']=='PASS_CPU_TRAINING_ENTRY'
    assert read(HERE/'TRANSPORT_R1_REGRESSION.json')['status']=='PASS_TRANSPORT_R1_REGRESSION'
    r=runpy.run_path(str(HERE/'runtime.py'));assert sha(r['BRIDGE'])==r['BRIDGE_SHA']
    # Check the actually used inherited source chain, weights and frozen evidence.
    files=dict(read(d/'SOURCE_LOCK.json')['files'])
    for path,digest in files.items():assert sha(Path(path))==digest,path
    for path in list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+list(HERE.glob('*.json'))+[HERE/'SELECTED_SAMPLES.jsonl',d/'RESULT.json',d/'LAUNCH_RESULT.json']:
        files[str(path)]=sha(path)
    files[str(HERE.parent/'ordinary_sync_recovery_v1/lease_run.py')]=sha(HERE.parent/'ordinary_sync_recovery_v1/lease_run.py')
    holders=read(HERE.parent/'ordinary_route_teacher_v11/RUNBOOK.json')['holders']
    check=runpy.run_path(str(HERE.parent/'ordinary_sync_recovery_v1/lease_run.py'),run_name='READONLY_HOLDER_HELPER')
    identities=[]
    for holder in holders:
        identity=check['holder_identity'](holder)
        assert identity['gpu_uuid']==holder['gpu_uuid']
        holder['expected_pid']=identity['pid'];holder['expected_start']=identity['starttime_ticks']
        identities.append(identity)
    prepared=read(HERE/'PREPARATION_RESULT.json')
    protocol=dict(id='ORDINARY_HISTORY8_PAIRED_TRAIN_V1',created_unix=time.time(),runtime_allowed=True,
        arms=['control_recent2','treatment_prefix8'],seed=1209,updates_per_arm=4000,
        decisions_per_arm=384000,world_size=3,global_batch=96,rank_batch=32,microbatch=4,accumulation=8,
        max_padded_tokens=6144,observed_max_padded_tokens=4740,wall_seconds_per_arm=18000,
        rank_memory_limit_gib=26,total_cpu_rss_limit_gib=64,output_limit_gib_per_arm=2,
        snapshot_sha256=prepared['ordinary_snapshot_sha256'],sample_index_sha256=prepared['selected_samples_sha256'],
        original_sample_index_sha256=prepared['original_index_sha256'],sampling_plan_sha256=prepared['plan_sha256'],
        source_checkpoint=str(r['BRIDGE']),source_checkpoint_sha256=r['BRIDGE_SHA'],
        optimizer=dict(reset_moments=True,learning_rate=5e-5,weight_decay=.01,betas=[.9,.999],eps=1e-8,
                       gradient_clip=1.,warmup_steps=120,total_schedule_steps=4000,schedule='cosine_to_zero'),
        data=dict(only_ordinary=True,special_decisions=0,recovery64_decisions=0,fit_houses=51,
                  num_workers_per_rank=2,prefetch_factor=2),
        startup_diagnostic=dict(forward_decisions_per_arm=12,backward_calls_per_arm=3,optimizer_steps=0,
                                input='four longest-token actual selected ordinary samples per rank'),
        fixed_checkpoint_selection=4000,navigation_goal=dict(split='full R2R-CE v1-3 val_unseen',episodes=1839,sr_min=.4),
        development_gate=dict(episodes=100,scope='INTERNAL_DEV_ONLY',sr_delta_gt=0,spl_delta_gte=0,ndtw_delta_gte=-.01),
        code_sha256=files,automatic_retry=False,automatic_long_extension=False)
    save('PROTOCOL.json',protocol)
    steps=[]
    for arm in protocol['arms']:
        steps.append(dict(name=arm,argv=[str(PY),'-I','-S','-B','-u',str(HERE/'supervise_fixed4.py'),'--arm',arm]))
        steps.append(dict(name=arm+'_accept',argv=[str(QPY),'-I','-B',str(HERE/'accept.py'),'--arm',arm],env_extra={'CUDA_VISIBLE_DEVICES':''}))
    runbook=dict(name=protocol['id'],runbook_sha256_self='SELF_EXCLUDED',out='lease_v1',
        lease_wall_seconds=36720,holders=holders,steps=steps,code_sha256=files)
    save('RUNBOOK.json',runbook)
    save('MAIN_AGENT_APPROVAL.json',dict(decision='PASS_FOR_FIXED_ORDINARY_HISTORY_PAIR_TRAINING',
        unix=time.time(),protocol_sha256=sha(HERE/'PROTOCOL.json'),runbook_sha256=sha(HERE/'RUNBOOK.json'),
        gpu_accumulation_gate_sha256=sha(d/'RESULT.json'),gpu_cleanup_sha256=sha(d/'LAUNCH_RESULT.json'),
        cpu_gate_sha256=sha(HERE/'FIXED4_CPU_RESULT.json'),holder_identities=identities,
        budget_per_arm_seconds=18000,max_policy_updates_per_arm=4000,source_files=len(files),
        no_special_data=True,no_production_restart=True,no_paid_api=True,old_failures_unchanged=True,
        navigation_gain_claimed=False))
    print(json.dumps(dict(status='FROZEN_READY',source_files=len(files),identities=identities)),flush=True)
if __name__=='__main__':main()

