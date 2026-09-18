"""Read-only comparison of complete versions; never selects or resumes a run."""
import hashlib
import json
from pathlib import Path

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
            same_terminal=e['stopped'] == f['stopped'] and e['success'] == f['success']))
    output = dict(status='COMPLETE_VERSIONS_READ_ONLY_COMPARISON',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        same_native_traces=sum(not r['first_difference'] and r['same_decision_count']
            and r['same_physical_trajectory'] and r['same_distances'] and r['same_terminal'] for r in native),
        native_episode_denominator=100, native_trace_comparison=native,
        old_arms=old['arms'], repaired_arms=new['arms'],
        repaired_within_session_comparisons=new['comparisons'],
        interpretation='V7-to-V8 is a development iteration across sessions, not a same-process causal ablation. Each version retains its own fresh native control and complete four-arm denominator. No seed or checkpoint is selected by this report.')
    with (HERE / 'VERSION_COMPARISON.json').open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('Native traces identical across versions:', output['same_native_traces'], '/100')


if __name__ == '__main__':
    main()
