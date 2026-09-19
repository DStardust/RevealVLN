"""Read back all final actors and recompute actual fork actions, with every control."""
import os
from pathlib import Path
import statistics
import sys
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train
c = train.c


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    run = HERE/'run_001'
    config = c.read(HERE/'PROTOCOL.json')
    result = c.read(run/'RESULT.json')
    launch = c.read(run/'LAUNCH_RESULT.json')
    assert launch['status'] == 'COMPLETE'
    assert set(result['runs']) == {f'{arm}_{seed}' for arm in config['arms'] for seed in config['seeds']}
    data = c.read(HERE.parent/'multifamily_v7/DATA.json')
    cache_path = HERE.parent/'multifamily_v7/run_001/FEATURES.pt'
    assert c.sha(cache_path) == c.read(cache_path.parent/'FEATURE_RESULT.json')['file_sha256']
    cache = {k: v.float() for k, v in torch.load(cache_path, map_location='cpu', weights_only=True).items()}
    admission = c.read(HERE.parent/'cost_teacher_v13/ACTOR_ADMISSION.json')
    prior = c.read(HERE.parent/'cost_teacher_v13/run_001/RESULT.json')
    rows = {}
    for key, measured in result['runs'].items():
        assert measured['updates'] == config['steps_per_arm']
        initial = torch.load(run/f"INITIAL_{measured['seed']}.pt", map_location='cpu', weights_only=True)
        checkpoint = run/f'{key}_MEMORY.pt'
        final = torch.load(checkpoint, map_location='cpu', weights_only=True)
        net = train.models.MemoryPolicy(2048, len(data['query_vocabulary']), 8, 64, .99,
                                       no_memory=measured['arm'] == 'N0')
        net.load_state_dict(initial)
        assert c.model_identity(net)['sha256'] == measured['initial_state_sha256']
        net.load_state_dict(final)
        assert c.model_identity(net)['sha256'] == measured['final_state_sha256']
        changes = {}
        for name, value in final.items():
            assert bool(torch.isfinite(value).all())
            delta = value-initial[name]
            changes[name] = dict(changed_elements=int(torch.count_nonzero(delta)),
                                 l2=float(delta.norm()), max_abs=float(delta.abs().max()))
            assert bool(changes[name]['changed_elements']) == measured['parameter_changed'][name]
        for name in ('action.0.weight', 'action.2.weight'):
            assert changes[name]['changed_elements'] > 0
        for name in ('writer.weight', 'recurrent.weight'):
            assert (changes[name]['changed_elements'] > 0) == (measured['arm'] != 'N0')
        reloaded = train.branch_metrics.evaluate(net, cache, data, admission)
        assert reloaded == measured['branch_actions'], 'CHECKPOINT_BRANCH_READBACK_DIFFERS:'+key
        gradient = c.read(run/f'{key}_GRADIENT.json')
        if measured['arm'] in ('B2', 'Ours'):
            assert gradient['critical_writer_auxiliary_gradient_norm'] > 0
        assert (gradient['action_memory_gradient_norm'] > 0) == (measured['arm'] != 'N0')
        assert sum(1 for _ in (run/f'{key}_STEPS.jsonl').open()) == config['steps_per_arm']
        by_house = {}
        for house in sorted({r['house'] for r in reloaded['rows']}):
            selected = [r for r in reloaded['rows'] if r['house'] == house]
            by_house[house] = dict(pairs=len(selected), both_correct=sum(r['both_correct'] for r in selected),
                wrong_history_both_correct=sum(r['both_correct_wrong'] for r in selected),
                same_state_sham_both_correct=sum(bool(r['both_correct_sham']) for r in selected))
        rows[key] = dict(branches=reloaded['summaries'], by_house=by_house,
            ordinary=measured['ordinary']['summaries'], gradient=gradient,
            checkpoint_sha256=c.sha(checkpoint), parameter_deltas=changes,
            prior_unbalanced_fork_teacher_branches=prior['runs'][key]['branch_actions']['summaries'] if key in prior['runs'] else None)
    deltas = [rows[f'Ours_{seed}']['branches']['check']['both_correct']-
              rows[f'B2_{seed}']['branches']['check']['both_correct'] for seed in config['seeds']]
    b1_deltas = [rows[f'Ours_{seed}']['branches']['check']['both_correct']-
                 rows[f'B1_{seed}']['branches']['check']['both_correct'] for seed in config['seeds']]
    signal = all(all(d >= 0 for d in values) and statistics.mean(values) > 0 for values in (deltas, b1_deltas))
    means = {arm: {split: {field: statistics.mean(rows[f'{arm}_{seed}']['branches'][split][field]
             for seed in config['seeds']) for field in ('both_correct', 'both_correct_wrong', 'both_correct_sham', 'individual_correct')}
             for split in ('fit', 'check')} for arm in config['arms']}
    review = dict(status='MATCHED_FORK_BALANCED_ACTOR_PILOT_COMPLETE', all_fixed_checkpoints_and_branch_actions_read_back=True,
        protocol_sha256=c.sha(HERE/'PROTOCOL.json'), source_sha256=c.sha(Path(__file__)),
        actor_admission_sha256=c.sha(HERE.parent/'cost_teacher_v13/ACTOR_ADMISSION.json'),
        runs=rows, means=means, ours_minus_b2_check_both_correct=deltas, ours_minus_b1_check_both_correct=b1_deltas,
        preregistered_exploratory_signal=signal, optimizer_updates=result['optimizer_updates'],
        cpu_wall_seconds=launch['cpu_wall_seconds'], gpu_hours=0, base_updates=0, new_qwen_forwards=0,
        closed_loop_sr='NOT_MEASURED', adopted=False, publication_ready=False,
        limitations='24 exposed CHECK forks in12 correlated families and2houses, repeated across3seeds. All50 cases compare real minimum-tested-cost expert labels, not logically necessary actions or global shortest paths; they are real logged teacher forks, not independently executed policy continuations. Original debug certificate limitations remain. This is not formal data admission, independent generalization, navigation success or a publication claim.')
    c.write(HERE/'REVIEW.json', review, True)
    print({k: v for k, v in review.items() if k != 'runs'})


if __name__ == '__main__':
    main()
