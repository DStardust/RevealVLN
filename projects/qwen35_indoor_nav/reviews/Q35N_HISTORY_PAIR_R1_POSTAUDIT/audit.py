"""CPU-only repeated official metrics and full causal-input trace verification."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import runpy
import sys
import time
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
SOURCE=LINE/'reviews/Q35N_ORDINARY_FP32_MASTER_V8R1_POSTAUDIT_R1/audit.py'
SOURCE_SHA='0d2b431adf72fd5caf23b5040ab4d7bf2e5f5d67b324708f9105652b3f5d65e6'
TRAIN=LINE/'sft_acceptance/ordinary_history8_paired_train_r1'
SHARED=LINE/'closed_loop_bench/ordinary_history_pair_eval_r1'
BASE=LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1'
CASES={'control_recent2':'ordinary_history2_dev_r1','treatment_prefix8':'ordinary_history8_dev_r1'}
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert len(sys.argv)==2 and sys.argv[1] in CASES
    arm=sys.argv[1];case=SHARED.parent/CASES[arm]
    assert sha(SOURCE)==SOURCE_SHA
    m=runpy.run_path(str(SOURCE),run_name='READ_ONLY_METRIC_FUNCTIONS')
    compare=runpy.run_path(str(SHARED/'compare.py'),run_name='READ_ONLY_COMPARISON')
    history=runpy.run_path(str(SHARED/'history_audit.py'),run_name='READ_ONLY_HISTORY')
    p=read(case/'PROTOCOL.json');base=read(BASE/'PROTOCOL.json')
    fixed=['seed','environment_seed','episode_count','house_count','houses','house_counts','max_steps','rgb_size','hfov',
        'camera_height','agent_height','agent_radius','forward_m','turn_deg','allow_sliding','success_distance',
        'greedy','optimizer_updates','metric_backend','ndtw_backend','lanes','gt_path','comparison_batch_size_forced']
    assert all(p[k]==base[k] for k in fixed)
    for name in ('EPISODES_PRIVILEGED.json','GEOMETRY_PREFLIGHT.json','PARITY_FIXTURES.json'):
        assert sha(case/name)==sha(BASE/name)
    lock=read(case/'SOURCE_LOCK.json')
    for name,digest in lock['files'].items():assert sha(name)==digest,name
    accepted=read(TRAIN/arm/'ACCEPTANCE.json')
    assert accepted['status']=='PASS_FINAL_TRAINING_ONLY' and accepted['updates']==4000
    assert p['training_protocol_sha256']==sha(TRAIN/'PROTOCOL.json') and p['history_arm']==arm
    assert sha(p['checkpoint'])==p['checkpoint_sha256']==accepted['checkpoint_sha256']
    loaded=read(case/'run_001/MODEL_LOADED.json');inference=read(case/'run_001/INFERENCE_RESULT.json')
    assert loaded['checkpoint_sha256']==p['checkpoint_sha256']
    assert loaded['trainable_sha256']==inference['trainable_sha256'] and inference['trainable_unchanged']
    launch=read(case/'run_001/LAUNCH_RESULT.json')
    assert launch['status']=='COMPLETE' and launch['returncode']==0 and not launch['cleanup']['remaining']
    assert not launch['foreign_processes_signaled'] and not launch['borrowed_holders']
    assert not launch['gpus_after'][1]['contexts']
    old_rows,old_metrics,old_actions=m['aggregate'](BASE)
    new_rows,metrics,actions=m['aggregate'](case)
    audited=0
    for folder in sorted((case/'run_001/lanes').glob('lane_*')):
        observed={row['index']:[row['rgb_sha256']] for row in records(folder/'INTERFACE.jsonl')}
        for policy,step in itertools.zip_longest(records(folder/'POLICY_STEPS.jsonl'),records(folder/'STEPS_PRIVILEGED.jsonl')):
            assert policy is not None and step is not None
            assert policy['index']==step['index'] and policy['step']==step['step'] and policy['action']==step['action']
            history['validate'](arm,policy,step,observed[step['index']])
            observed[step['index']].append(step['rgb_sha256']);audited+=1
    assert audited==actions==inference['total_actions']
    paired=compare['comparison'](compare['episodes'](case),compare['episodes'](BASE))
    result=dict(status='PASS_AUDIT_NOT_SCIENTIFIC_PASS',unix=time.time(),arm=arm,scope='complete INTERNAL_DEV100 only',
        baseline=old_metrics,after=metrics,versus_old_best=paired,full_val1839_sr=None,sr40_goal_achieved=False,
        source_files=len(lock['files']),source_lock_sha256=sha(case/'SOURCE_LOCK.json'),
        checkpoint_sha256=p['checkpoint_sha256'],official_metric_replayed_episodes=200,
        actual_causal_history_audited_actions=audited,eval_wall_seconds=launch['wall_seconds'],
        additional_training_updates=0,additional_simulator_actions=0,recorded_distances_not_new_oracle_queries=True,
        other_arm_training_not_touched=True,evaluation_gpu_cleanup_verified=True,source_sha256=sha(__file__))
    with (HERE/(arm+'_RESULT.json')).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

