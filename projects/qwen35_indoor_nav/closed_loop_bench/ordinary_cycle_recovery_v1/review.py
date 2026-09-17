"""Independent online-input audit and paired development comparison."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run_001'
BASE = HERE.parent / 'ordinary_expanded_dev_after_single_v1'
ACTIONS = ('move_forward', 'turn_left', 'turn_right', 'STOP')


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def traces(run):
    result = {}
    for lane in sorted((run / 'lanes').glob('lane_*')):
        initial = {row['index']: row for row in records(lane / 'INTERFACE.jsonl')}
        states = {(row['index'], row['step']): row for row in records(lane / 'STEPS_PRIVILEGED.jsonl')}
        for row in records(lane / 'POLICY_STEPS.jsonl'):
            index = row['index']
            if index not in result:
                result[index] = dict(initial=initial[index]['rgb_sha256'], policies=[], states=[],
                                     result=read(lane / f'episode_{index:02d}.json'))
            result[index]['policies'].append(row)
            result[index]['states'].append(states[index, row['step']])
    return result


def audit(index, old, new, instruction):
    images = [new['initial']]
    old_images = [old['initial']]
    executed = []
    old_executed = []
    counts = {}
    first_override = None
    mismatch = None
    overrides = repeated = prefix_steps = 0
    max_logit_delta = 0.
    for offset, (policy, state) in enumerate(zip(new['policies'], new['states'])):
        assert policy['step'] == state['step'] == offset + 1
        assert policy['action'] == state['action']
        logits = policy['logits']
        native = max(range(4), key=lambda i: logits[i])
        signature = [instruction, images[-2:], executed[-8:]]
        key = hashlib.sha256(json.dumps(signature, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        seen = key in counts
        attempts = counts.setdefault(key, [0, 0, 0])
        chosen = native
        if seen and native != 3:
            chosen = sorted(range(3), key=lambda i: (attempts[i], -logits[i], i))[0]
        changed = chosen != native
        assert policy['input_key'] == key
        assert policy['native_action'] == ACTIONS[native]
        assert policy['action'] == ACTIONS[chosen]
        assert policy['cycle_override'] == changed and policy['repeated_input'] == seen
        if first_override is None:
            prefix_steps += 1
            if offset >= len(old['policies']):
                mismatch = mismatch or dict(step=offset + 1, reason='BASELINE_ALREADY_TERMINATED')
            else:
                before = old['policies'][offset]
                if images[-2:] != old_images[-2:] or executed[-8:] != old_executed[-8:]:
                    mismatch = mismatch or dict(step=offset + 1, reason='OBSERVATION_OR_HISTORY_MISMATCH')
                elif policy['native_action'] != before['action']:
                    mismatch = mismatch or dict(step=offset + 1, reason='NATIVE_ACTION_MISMATCH')
                max_logit_delta = max(max_logit_delta, max(abs(a - b) for a, b in zip(logits, before['logits'])))
                old_images.append(old['states'][offset]['rgb_sha256'])
                old_executed.append(before['action'])
        if changed and first_override is None:
            first_override = offset + 1
        if chosen != 3:
            attempts[chosen] += 1
        overrides += changed
        repeated += seen
        images.append(state['rgb_sha256'])
        executed.append(policy['action'])
    assert len(new['policies']) == len(new['states']) == new['result']['steps']
    return dict(index=index, episode_id=new['result']['episode_id'], first_override=first_override,
                overrides=overrides, repeated_inputs=repeated, prefix_checked_steps=prefix_steps,
                prefix_mismatch=mismatch, prefix_max_abs_logit_delta=max_logit_delta,
                baseline_success=old['result']['success'], recovery_success=new['result']['success'])


def main():
    before = read(BASE / 'run_001/RESULT.json')
    after = read(RUN / 'RESULT.json')
    assert before['completed'] == after['completed'] == 100
    assert before['checkpoint_sha256'] == after['checkpoint_sha256']
    assert read(HERE / 'EPISODES_PRIVILEGED.json') == read(BASE / 'EPISODES_PRIVILEGED.json')
    episodes = read(HERE / 'EPISODES_PRIVILEGED.json')
    old = traces(BASE / 'run_001')
    new = traces(RUN)
    assert set(old) == set(new) == set(range(100))
    rows = [audit(i, old[i], new[i], episodes[i]['instruction']['instruction_text']) for i in range(100)]
    wins = [r['episode_id'] for r in rows if not r['baseline_success'] and r['recovery_success']]
    losses = [r['episode_id'] for r in rows if r['baseline_success'] and not r['recovery_success']]
    prefixes_match = all(r['prefix_mismatch'] is None for r in rows)
    thresholds = dict(sr=after['sr'] > before['sr'], spl=after['spl'] >= before['spl'],
                      ndtw=after['ndtw'] >= before['ndtw'] - .01)
    result = dict(status='COMPLETE', role='Engineering development diagnostic; no novelty or full benchmark claim',
        online_input_and_action_audit_passed=True, prefixes_match=prefixes_match,
        prefix_max_abs_logit_delta=max(r['prefix_max_abs_logit_delta'] for r in rows),
        baseline={k: before[k] for k in ('sr', 'spl', 'ndtw', 'osr', 'collisions', 'environment_actions')},
        recovery={k: after[k] for k in ('sr', 'spl', 'ndtw', 'osr', 'collisions', 'environment_actions')},
        metric_delta={k: after[k] - before[k] for k in ('sr', 'spl', 'ndtw', 'osr', 'collisions', 'environment_actions')},
        wins=wins, losses=losses, retained_successes=sum(bool(r['baseline_success'] and r['recovery_success']) for r in rows),
        affected_episodes=sum(r['first_override'] is not None for r in rows),
        overrides=sum(r['overrides'] for r in rows), repeated_inputs=sum(r['repeated_inputs'] for r in rows),
        thresholds=thresholds, engineering_candidate_pass=prefixes_match and all(thresholds.values()),
        recovery_seconds=sum(p['cycle_seconds'] for item in new.values() for p in item['policies']),
        by_house={house: dict(baseline=before['by_house'][house], recovery=after['by_house'][house]) for house in before['by_house']},
        episodes=rows)
    with (RUN / 'REVIEW.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('episodes', 'by_house')}))


if __name__ == '__main__':
    main()
