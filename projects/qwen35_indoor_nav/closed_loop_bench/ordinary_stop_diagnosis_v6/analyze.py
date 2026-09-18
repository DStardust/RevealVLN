"""Read-only STOP diagnosis aligned to the state before each actual decision."""
import collections
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
V5 = HERE.parent / 'ordinary_cycle_pair_recovery_v5'


def auc(rows):
    """Weighted ranking AUC; ties receive half credit, no threshold selection."""
    groups = collections.defaultdict(lambda: [0., 0.])
    for score, label, weight in rows:
        groups[score][label] += weight
    positive = sum(x[1] for x in groups.values())
    negative = sum(x[0] for x in groups.values())
    if not positive or not negative:
        return None
    below = area = 0.
    for score in sorted(groups):
        neg, pos = groups[score]
        area += pos * (below + .5 * neg)
        below += neg
    return area / (positive * negative)


def decision_labels(episode):
    # The last state after the final action has no further decision opportunity.
    assert len(episode['distances']) == episode['steps'] + 1
    return [int(distance < 3.) for distance in episode['distances'][:-1]]


def main():
    pairs = sorted((V5 / 'sessions').glob('session_*/pairs/pair_*/PAIR.json'))
    assert len(pairs) == 100
    cases, rankings, unique_rankings, balanced_rankings = [], [], [], []
    by_house = collections.defaultdict(list)
    key_labels = collections.defaultdict(set)
    near_margins, far_margins = [], []
    for path in pairs:
        pair = json.loads(path.read_text())
        e = pair['episodes']['A']
        decisions = [json.loads(line) for line in (path.parent / 'A/POLICY_STEPS.jsonl').read_text().splitlines()]
        labels = decision_labels(e)
        assert len(decisions) == len(labels) == e['steps']
        near = [i for i, value in enumerate(labels) if value]
        seen = set()
        for i, (d, label) in enumerate(zip(decisions, labels)):
            row = (d['stop_margin'], label, 1.)
            rankings.append(row)
            balanced_rankings.append((row[0], label, 1. / len(decisions)))
            by_house[e['house']].append(row)
            key = d['raw']['input_key']
            key_labels[key].add(label)
            if key not in seen:
                unique_rankings.append(row)
                seen.add(key)
            (near_margins if label else far_margins).append(row[0])
            assert (d['executed_action'] == 'STOP') == (e['stopped'] and i == len(decisions)-1)
        cases.append(dict(episode_id=e['episode_id'], house=e['house'], success=e['success'],
            decisions=len(decisions), stopped=e['stopped'], terminal_distance=e['distances'][-1],
            decision_opportunities_in_range=len(near), first_in_range_decision=near[0]+1 if near else None,
            last_in_range_decision=near[-1]+1 if near else None,
            missed_legal_stop=bool(near and not e['success']),
            max_margin_while_in_range=max((decisions[i]['stop_margin'] for i in near), default=None),
            repeats_while_in_range=sum(decisions[i]['repeated_input'] for i in near),
            source_pair=str(path.relative_to(V5))))
    result = dict(status='OFFLINE_DIAGNOSIS_COMPLETE', source='V5 native A only', episodes=100,
        sr=sum(x['success'] for x in cases)/100,
        near_at_decision=sum(x['decision_opportunities_in_range']>0 for x in cases),
        missed_stop=sum(x['missed_legal_stop'] for x in cases),
        far_stop=sum(x['stopped'] and x['terminal_distance']>=3. for x in cases),
        never_entered_at_decision=sum(x['decision_opportunities_in_range']==0 for x in cases),
        near_decisions=len(near_margins), far_decisions=len(far_margins),
        near_margin_median=statistics.median(near_margins), far_margin_median=statistics.median(far_margins),
        stop_margin_auc=auc(rankings), episode_balanced_auc=auc(balanced_rankings),
        within_episode_unique_input_auc=auc(unique_rankings), by_house_auc={h:auc(rows) for h,rows in by_house.items()},
        raw_keys_with_conflicting_range_labels=sum(len(v)>1 for v in key_labels.values()),
        existing_trace_oracle_stop_ceiling=sum(x['decision_opportunities_in_range']>0 for x in cases)/100,
        ceiling_scope='Privileged earlier truncation of recorded native trajectories only; not a deployable SR or an upper bound after changing motion/suppressing old STOP',
        thresholds_searched=0, model_updates=0, new_environment_decisions=0, gpu_hours=0,
        exposure='Already exposed INTERNAL_DEV; these labels are diagnostic, not new independent training/test data',
        cases=cases)
    with (HERE/'RESULT.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print({k:v for k,v in result.items() if k!='cases'})


if __name__ == '__main__':
    main()
