"""Read-only SFT diagnosis; write a separate, time-scoped evidence summary."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
SFT = LINE/'sft_acceptance/v1'
REC = SFT/'recovery_r1'
DATA = LINE/'data_pipeline/ordinary_pilot_v1'
ACTIONS = ['move_forward', 'turn_left', 'turn_right', 'STOP']


def read(path):
    path = path.resolve()
    assert path.is_relative_to(LINE)
    return json.loads(path.read_text())


def main():
    split = read(SFT/'SPLIT.json')
    source_stats, previous_correct, dev_count = {}, 0, 0
    for name in ('train', 'seen_house_route_dev'):
        counts = collections.Counter()
        for row in split[name]:
            actions = read(DATA/row['supervision_file'])['actions']
            counts.update(actions)
            if name == 'seen_house_route_dev':
                previous_correct += sum(a == ('move_forward' if t == 0 else actions[t-1]) for t, a in enumerate(actions))
                dev_count += len(actions)
        source_stats[name] = {'instructions': len(split[name]), 'routes': len({r['job_id'] for r in split[name]}),
                              'decisions': sum(counts.values()), 'class_counts': dict(counts)}
    # Snapshot once; evaluation may append after this read. Training rows are final.
    raw = (REC/'MODEL_STEP_LEDGER.jsonl').read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith(b'\n')]
    train = [r for r in rows if r['stage'] == 'train']
    dev = [r for r in rows if r['stage'] == 'after']
    counts = collections.Counter(r['target'] for r in train)
    assert len(dev) == dev_count == 3864
    prior_ce = -sum(math.log(counts[r['target']]/len(train)) for r in dev)/len(dev)
    before, after = read(SFT/'OFFLINE_BEFORE.json'), read(REC/'OFFLINE_AFTER.json')
    cm = after['confusion_target_rows_prediction_columns']
    assert sum(map(sum, cm)) == len(dev)
    updates = [json.loads(line) for line in (REC/'UPDATE_LEDGER.jsonl').read_text().splitlines()]
    closed = [json.loads(line) for line in (REC/'EPISODE_LEDGER.jsonl').read_text().splitlines()]
    completed = [r for r in closed if r['event'] == 'complete']
    result = {'scope': 'CPU_DIAGNOSIS_AND_REPAIR_PLAN_NOT_NEW_TRAINING', 'source_statistics': source_stats,
       'actual_train_decisions': len(train), 'actual_train_action_counts': {ACTIONS[k]: v for k, v in counts.items()},
       'equivalent_full_train_passes': len(train)/source_stats['train']['decisions'],
       'train_routes_seen': len({r['job_id'] for r in train}), 'train_instructions_seen': len({r['policy_file'] for r in train}),
       'before_CE': before['decision_micro_CE'], 'after_CE': after['decision_micro_CE'],
       'after_accuracy': after['decision_micro_accuracy'], 'train_frequency_prior_dev_CE': prior_ce,
       'always_forward_dev_accuracy': sum(cm[0])/len(dev), 'copy_previous_teacher_forced_dev_accuracy': previous_correct/dev_count,
       'after_predicted_forward_fraction': sum(r[0] for r in cm)/len(dev),
       'after_balanced_class_recall': sum(cm[i][i]/sum(cm[i]) for i in range(4))/4,
       'after_STOP_recall': cm[3][3]/sum(cm[3]), 'after_left_recall': cm[1][1]/sum(cm[1]),
       'optimizer_updates_in_terminal': len(updates), 'prior_discarded_engineering_update': 1,
       'clipped_update_count': sum(r['grad_norm_before_clip'] > 1 for r in updates),
       'grad_norm_median': statistics.median(r['grad_norm_before_clip'] for r in updates),
       'grad_norm_max': max(r['grad_norm_before_clip'] for r in updates),
       'median_update_wall_seconds': statistics.median(b['unix']-a['unix'] for a,b in zip(updates, updates[1:])),
       'closed_after_completed_at_snapshot': len(completed),
       'closed_after_stopped_within3m_at_snapshot': sum(r['stopped_within_3m'] for r in completed),
       'closed_loop_snapshot_not_final_adjudication': True,
       'new_GPU_operations': 0, 'new_optimizer_updates': 0, 'scientific_pass': False,
       'hypotheses': {'single_action_bias': 'OBSERVED', 'one_pass_low_exposure': 'OBSERVED',
          'class_imbalance_is_sole_cause': 'UNTESTED', 'memory_scale_instability': 'UNTESTED',
          'more_epochs_alone_will_fix_navigation': 'UNTESTED', 'input_or_target_misalignment': 'NOT_FOUND_IN_STATIC_PATH_NOT_RUNTIME_EXCLUDED'},
       'snapshot_sha256': hashlib.sha256(raw).hexdigest(),
       'protected_source_files_sha256': {str(p.relative_to(LINE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [SFT/'SPLIT.json', SFT/'EXPERIMENT_SPEC.json', SFT/'OFFLINE_BEFORE.json', REC/'OFFLINE_AFTER.json',
             REC/'policy.py', REC/'run_sft.py', REC/'UPDATE_LEDGER.jsonl', REC/'RELOAD_CHECKS.jsonl']}}
    with (HERE/'EVIDENCE.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
