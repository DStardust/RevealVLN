"""Existing native trajectories across sessions; no replay or kernel attribution."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review
c = review.c


def main():
    seeds = c.read(HERE/'PROTOCOL.json')['method_seeds']
    groups = {seed: review.committed(seed) for seed in seeds}
    assert all(len(rows) == 100 for rows in groups.values())
    comparisons = {}
    for seed in seeds[1:]:
        rows = []
        for rank in range(100):
            reference, current = groups[seeds[0]][rank], groups[seed][rank]
            assert reference['episode_id'] == current['episode_id']
            paths = [Path(row['path']).parent/'A/POLICY_STEPS.jsonl' for row in (reference, current)]
            streams = [c.records(path) for path in paths]
            maximum = 0.
            first_numeric = first_divergence = None
            matched = 0
            for a, b in zip(*streams):
                assert a['executed_action'] == a['native_action']
                assert b['executed_action'] == b['native_action']
                comparison = c.prefix_compare(a, b)
                if not comparison['input_prefix_matched']:
                    first_divergence = dict(kind='INPUT_DIFFERS_WITH_SAME_PRIOR_ACTIONS', step=a['step'],
                        reference=a, current=b)
                    break
                matched += 1
                delta = comparison['max_logit_delta']
                maximum = max(maximum, delta)
                if delta and first_numeric is None:
                    first_numeric = dict(step=a['step'], max_logit_delta=delta,
                        reference_logits=a['logits'], current_logits=b['logits'],
                        same_native_action=a['native_action'] == b['native_action'])
                if a['native_action'] != b['native_action']:
                    first_divergence = dict(kind='NATIVE_ACTION_DIFFERS_WITH_SAME_INPUT', step=a['step'],
                        reference=a, current=b)
                    break
            if first_divergence is None and len(streams[0]) != len(streams[1]):
                first_divergence = dict(kind='TERMINATION_DIFFERS_WITH_MATCHED_PREFIX',
                                       lengths=[len(x) for x in streams])
            same_geometry = reference['episodes']['A']['positions'] == current['episodes']['A']['positions']
            rows.append(dict(rank=rank, episode_id=reference['episode_id'],
                matched_prefix_decisions=matched, max_logit_delta_before_first_behavior_or_input_difference=maximum,
                first_numeric_difference=first_numeric, first_divergence=first_divergence,
                same_native_geometry=same_geometry, source_paths=[str(path.relative_to(HERE)) for path in paths]))
        comparisons[str(seed)] = dict(reference_seed=seeds[0], current_seed=seed, episodes=100,
            first_divergence_counts={kind: sum(row['first_divergence'] is not None and row['first_divergence']['kind'] == kind for row in rows)
                for kind in ('INPUT_DIFFERS_WITH_SAME_PRIOR_ACTIONS', 'NATIVE_ACTION_DIFFERS_WITH_SAME_INPUT', 'TERMINATION_DIFFERS_WITH_MATCHED_PREFIX')},
            episodes_with_numeric_difference=sum(row['first_numeric_difference'] is not None for row in rows),
            same_complete_native_action_and_input_sequence=sum(row['first_divergence'] is None for row in rows),
            rows=rows)
    c.write(HERE/'CROSS_SESSION_NATIVE_AUDIT.json', dict(status='EXISTING_NATIVE_SESSION_DIFFERENCES_RECORDED',
        comparisons=comparisons, source_sha256=c.sha(Path(__file__)), gpu_calls=0, optimizer_updates=0,
        scope='Native A across separately loaded model sessions. All within-group causal prefix audits remain separate. Session differences are not attributed solely to learned-head seed variation.',
        attribution='A same-input native-action flip records a numerical runtime difference, not proof of an FLA/Triton cause. No new model replay, parameter change, kernel scan, or selected trajectory replacement.'), True)
    print({seed: {k: v for k, v in value.items() if k != 'rows'} for seed, value in comparisons.items()})


if __name__ == '__main__':
    main()
