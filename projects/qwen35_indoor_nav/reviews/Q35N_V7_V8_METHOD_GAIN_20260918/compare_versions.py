"""Read-only comparison of complete versions; never selects or resumes a run."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def complete(version):
    folder = LINE / 'closed_loop_bench' / f'ordinary_memory_transfer_v{version}'
    result = read(folder / 'RESULT.json')
    assert result['status'] == 'VALID_COMPLETE' and result['metric_and_log_recomputed']
    pairs = {}
    for session in (folder / 'sessions').glob('session_*'):
        seals = [read(path) for path in session.glob('STATE_SEAL_*.json')]
        covered = {rank for seal in seals if seal['unchanged'] for rank in seal['quartet_ranks']}
        for path in session.glob('pairs/pair_*/QUARTET.json'):
            row = read(path)
            assert row['rank'] in covered and row['valid_behavioral_quartet']
            assert row['rank'] not in pairs
            pairs[row['rank']] = (row, path.parent)
    assert set(pairs) == set(range(100))
    return result, pairs


def costs_and_stops(pairs):
    result = {}
    for arm in ('A', 'B', 'C', 'D'):
        kinds = {key: [] for key in ('success', 'entered_then_far_stop',
            'entered_without_stop', 'never_entered_far_stop', 'never_entered_without_stop')}
        latency = []
        repeats = []
        for rank in sorted(pairs):
            row, folder = pairs[rank]
            episode = row['episodes'][arm]
            entered = min(episode['distances']) < 3
            kind = 'success' if episode['success'] else (
                'entered_then_far_stop' if entered and episode['stopped'] else
                'entered_without_stop' if entered else
                'never_entered_far_stop' if episode['stopped'] else 'never_entered_without_stop')
            kinds[kind].append(episode['episode_id'])
            steps = records(folder / arm / 'POLICY_STEPS.jsonl')
            latency.extend(d['preprocess_seconds'] + d['inference_seconds'] + d['controller_seconds'] for d in steps)
            count = sum(d['repeated_input'] for d in steps)
            repeats.append(dict(episode_id=episode['episode_id'], repeated_short_window=count,
                decisions=len(steps), fraction=count / len(steps)))
        assert sum(map(len, kinds.values())) == 100
        latency.sort()
        result[arm] = dict(stop_types={k: dict(count=len(v), episode_ids=v) for k, v in kinds.items()},
            policy_compute_p50_seconds=statistics.median(latency),
            policy_compute_p95_seconds=latency[math.ceil(.95 * len(latency)) - 1],
            latency_scope='Preprocessing + model inference + memory/action selection only; excludes simulator, queue, fingerprint/log I/O. Shared-machine measurement, not exclusive deployment latency.',
            short_window_repeats_per_episode=repeats,
            repeat_scope='Original instruction/RGB/executed-action window; learned memory is excluded. These repeats do not imply identical full memory-policy inputs.')
    return result


def main():
    old, before = complete(7)
    new, after = complete(8)
    native = []
    fields = ('raw', 'processed', 'native_action', 'executed_action', 'logits')
    for rank in range(100):
        a, ap = before[rank]
        b, bp = after[rank]
        assert a['episode_id'] == b['episode_id'] and a['index'] == b['index']
        x = records(ap / 'A/POLICY_STEPS.jsonl')
        y = records(bp / 'A/POLICY_STEPS.jsonl')
        first = {}
        for field in fields:
            for step, (left, right) in enumerate(zip(x, y), 1):
                if left[field] != right[field]:
                    first[field] = step
                    break
        e, f = a['episodes']['A'], b['episodes']['A']
        native.append(dict(rank=rank, episode_id=a['episode_id'], first_difference=first,
            same_decision_count=len(x) == len(y),
            same_physical_trajectory=e['positions'] == f['positions'],
            same_distances=e['distances'] == f['distances'],
            success=[e['success'], f['success']],
            same_terminal=e['stopped'] == f['stopped'] and e['success'] == f['success']))
    output = dict(status='COMPLETE_VERSIONS_READ_ONLY_COMPARISON',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        identical_native_inputs_actions_logits_and_trajectory=sum(not r['first_difference'] and r['same_decision_count']
            and r['same_physical_trajectory'] and r['same_distances'] and r['same_terminal'] for r in native),
        same_native_action_sequences=sum(r['same_decision_count'] and 'executed_action' not in r['first_difference'] for r in native),
        same_native_physical_trajectories=sum(r['same_physical_trajectory'] for r in native),
        native_success_changes=[r['episode_id'] for r in native if r['success'][0] != r['success'][1]],
        native_episode_denominator=100, native_trace_comparison=native,
        old_arms=old['arms'], repaired_arms=new['arms'],
        old_costs_and_stops=costs_and_stops(before), repaired_costs_and_stops=costs_and_stops(after),
        repaired_within_session_comparisons=new['comparisons'],
        interpretation='V7-to-V8 is a development iteration across sessions, not a same-process causal ablation. Each version retains its own fresh native control and complete four-arm denominator. No seed or checkpoint is selected by this report.')
    with (HERE / 'VERSION_COMPARISON.json').open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('Identical native inputs/actions/logits/trajectory:', output['identical_native_inputs_actions_logits_and_trajectory'], '/100')
    print('Same native action sequences:', output['same_native_action_sequences'], '/100')


if __name__ == '__main__':
    main()
