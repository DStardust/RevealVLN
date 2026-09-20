"""Freeze all evaluation conditions before reading any new trained model scores."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
LINE=HERE.parents[2]
sys.path.insert(0,str(LINE/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c


def main():
    train=c.read(HERE/'TRAIN_PROTOCOL.json');data=c.read(HERE/'DATA.json')
    conditions=[dict(family_id=f['family_id'],house=f['house'],history_id=h,task_id=t)
        for f in data['families'] if f['split']=='check' for h in ('H_A','H_B') for t in ('task_A','task_B')]
    models=[f'{size}_{arm}_{seed}' for size in train['data_sizes'] for seed in train['seeds'] for arm in train['arms']]
    rollouts=[]
    for i,condition in enumerate(conditions):
        order=models[i%len(models):]+models[:i%len(models)]
        rollouts.extend(dict(condition=i,model=tag) for tag in order)
    assert len(conditions)==12 and len(models)==18 and len(rollouts)==216
    dependencies=[*HERE.glob('*.py'),HERE/'TRAIN_PROTOCOL.json',HERE/'DATA.json',
        LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json',
        LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15/common.py',
        LINE/'data_pipeline/mechanism_runtime_v1/legal_history_v15/content_store.py',
        LINE/'data_pipeline/mechanism_runtime_v1/habitat_backend.py',
        LINE/'data_pipeline/mechanism_runtime_v1/guard.py',
        LINE/'data_pipeline/mechanism_runtime_v1/feedback_generation_v1/store.py',
        LINE/'data_pipeline/mechanism_factory_v2/compiler.py',
        LINE/'closed_loop_bench/ordinary_memory_transfer_v10/evaluate.py',
        LINE/'closed_loop_bench/ordinary_memory_transfer_v10/transport.py']
    hashes=dict(train['source_hashes'])
    hashes.update({str(p.relative_to(LINE)):c.sha(p) for p in dependencies})
    protocol=dict(version='Q35N_V15_LIVE_CONTINUATION_1',gpu=1,gpu_uuid=train['gpu_uuid'],
        conditions=conditions,models=models,rollouts=rollouts,source_hashes=hashes,
        raw_config='data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008/CONFIG.json',
        raw_run='data_pipeline/mechanism_runtime_v1/legal_history_v15/raw_run_008',
        max_total_decisions=500,prefix_and_STOP_count_in_budget=True,environment_seed=0,
        actual_prefix_replay=True,interior_state_assignments=0,native_STOP_preserved=True,
        prefill='Live frozen Qwen causal features measured on first physical replay of each condition; shared between matched heads only after raw input identity checks. Cutoff and all autonomous decisions use fresh Qwen forwards.',
        primary='Frozen SEE2 compiler PASS / all 12 assigned conditions for each model; UNKNOWN is separately counted and never called FAIL.',
        collision_rule='Frozen compiler considers collision traces UNKNOWN. Preserve this restriction and report full-denominator pass lower bound, FAIL and UNKNOWN; no collider filtering or metric rewriting.',
        comparisons='Each size and seed: Ours against B1 and strong B2, input/native action prefix before first actual method action difference.',
        generalization_scope='One new CHECK house, three correlated families; fixed real-history continuation only, not full R2R SR.',
        checks_not_read_for_selection=True,all_models_evaluated_regardless_of_score=True,
        terminal_only_and_neutral_controls='Present in training/offline diagnostics; not claimed independently evaluated in this 216-rollout continuation screen.',
        head_updates=0,base_updates=0,model_results_used_to_select_protocol=False)
    c.write(HERE/'CONTINUATION_PROTOCOL.json',protocol,True)
    print(dict(conditions=len(conditions),models=len(models),rollouts=len(rollouts)))


if __name__=='__main__':main()
