"""Complete fixed controls and paired fork contrasts; no checkpoint selection."""
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]/'closed_loop_bench/ordinary_cycle_pair_recovery_v5'))
import common as c


def main():
    original = c.read(HERE/'run_001/RESULT.json')
    strong = c.read(HERE/'strong_state_run_001/RESULT.json')
    verified = c.read(HERE/'REVIEW.json')
    assert verified['all_fixed_checkpoints_and_branch_actions_read_back']
    assert set(strong['runs']) == {f'B2_MLP_{seed}' for seed in (1209, 1210, 1211)}
    runs = dict(original['runs'], **strong['runs'])
    for seed in (1209, 1210, 1211):
        key = f'B2_MLP_{seed}'
        measured = runs[key]
        folder = HERE/'strong_state_run_001'
        assert measured['checkpoint_branch_actions_read_back']
        assert c.sha(folder/f'{key}_MEMORY.pt') == measured['checkpoint_sha256']
        left = c.records(folder/f'{key}_STEPS.jsonl')
        right = c.records(HERE/'run_001'/f'Ours_{seed}_STEPS.jsonl')
        assert len(left) == len(right) == 600
        assert all(all(a[k] == b[k] for k in ('step', 'family_index', 'ordinary_indices')) for a, b in zip(left, right))
    def key(row):
        return row['family_id']+'::'+row['task_id']
    comparisons = {}
    for baseline in ('N0', 'B1', 'B2', 'B2_MLP'):
        comparisons[baseline] = {}
        for seed in (1209, 1210, 1211):
            ours = {key(r): r for r in runs[f'Ours_{seed}']['branch_actions']['rows'] if r['split'] == 'check'}
            other = {key(r): r for r in runs[f'{baseline}_{seed}']['branch_actions']['rows'] if r['split'] == 'check'}
            assert ours.keys() == other.keys() and len(ours) == 24
            wins = [k for k in ours if ours[k]['both_correct'] and not other[k]['both_correct']]
            losses = [k for k in ours if not ours[k]['both_correct'] and other[k]['both_correct']]
            comparisons[baseline][str(seed)] = dict(wins=wins, losses=losses, denominator=24,
                retained_correct=[k for k in ours if ours[k]['both_correct'] and other[k]['both_correct']],
                delta_correct=len(wins)-len(losses), delta_rate=(len(wins)-len(losses))/24)
    summaries = {key: dict(branches=v['branch_actions']['summaries'], ordinary=v['ordinary']['summaries'])
                 for key, v in runs.items()}
    arms = ('N0', 'B1', 'B2', 'B2_MLP', 'Ours')
    totals = {arm: {split: {metric: sum(runs[f'{arm}_{seed}']['branch_actions']['summaries'][split][metric]
              for seed in (1209, 1210, 1211)) for metric in ('pairs', 'both_correct', 'both_correct_wrong', 'both_correct_sham', 'individual_correct')}
              for split in ('fit', 'check')} for arm in arms}
    evidence = dict(status='DEBUG_MEMORY_LEARNABILITY_SIGNAL_NO_SUPERVISION_INCREMENT', runs=summaries,
        totals=totals, ours_against_controls=comparisons,
        same_runtime_initialization_and_sample_schedule=True,
        main_optimizer_updates=7200, additional_exact_state_updates=1800,
        main_cpu_wall_seconds=verified['cpu_wall_seconds'], additional_cpu_wall_seconds=strong['cpu_wall_seconds'],
        cpu_jobs_overlapped=True, exclusive_device_latency_measured=False,
        gpu_hours=0, new_qwen_forwards=0, base_updates=0,
        state_head_parameters=66180, outcome_head_parameters=66305,
        positive_scope='FIT fork actor learning improved and depends on supplied history memory. This is a debug learnability result shared with B1; not superiority of crossed supervision.',
        method_increment_supported=False, closed_loop_gain_measured=False, publication_ready=False,
        source_training_admission=False, data_exposure='12 previously exposed CHECK families in2houses,24 task/fork units repeated across3seeds.72 rows are not72independent cases.',
        next_action='Preserve all results, complete already frozen V10 navigation; do not tune another Ours variant on this CHECK pool. A stronger claim requires physically admitted independent families and actual closed-loop benefits beyond B1/B2.',
        original_review_sha256=c.sha(HERE/'REVIEW.json'), strong_result_sha256=c.sha(HERE/'strong_state_run_001/RESULT.json'),
        source_sha256=c.sha(Path(__file__)))
    c.write(HERE/'STRONG_STATE_REVIEW.json', evidence, True)
    print(dict(status=evidence['status'], totals=totals,
        ours_minus_controls={a:[v['delta_correct'] for v in values.values()] for a, values in comparisons.items()}))


if __name__ == '__main__':
    main()
