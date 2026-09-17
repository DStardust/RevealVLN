"""Measure exact deterministic input cycles in all 100 previously exposed DEV episodes.

Use only RGB hashes and actual executed actions to identify policy inputs.
Collision/goal/outcome fields are separately reported diagnostic evidence.
No parameters, labels, or episode selection are changed.
"""
from collections import defaultdict
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parents[1]
RUN = LINE/'closed_loop_bench/ordinary_expanded_dev_after_single_v1/run_001'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    episodes = []
    for lane in sorted((RUN/'lanes').glob('lane_*')):
        initial = {x['index']:x for x in read(lane/'INTERFACE.jsonl')}
        policy = defaultdict(list)
        states = {}
        for x in read(lane/'POLICY_STEPS.jsonl'):
            policy[x['index']].append(x)
        for x in read(lane/'STEPS_PRIVILEGED.jsonl'):
            states[x['index'],x['step']] = x
        for index, steps in policy.items():
            frame_history = [initial[index]['rgb_sha256']]
            executed = []
            seen = {}
            repeated = []
            previous = None
            consecutive_same_input = 0
            longest_same_input = 0
            stationary_forwards = 0
            for t,x in enumerate(steps,1):
                assert x['step']==t
                key = (tuple(frame_history[-2:]),tuple(executed[-8:]))
                # Instruction is constant within each episode; never join episodes.
                if key in seen:
                    old = seen[key]
                    repeated.append(dict(step=t,first_step=old['step'],action=x['action'],
                        same_action=x['action']==old['action'],max_logit_difference=max(abs(a-b) for a,b in zip(x['logits'],old['logits']))))
                else:
                    seen[key] = x
                consecutive_same_input = consecutive_same_input+1 if key==previous else 1
                longest_same_input = max(longest_same_input,consecutive_same_input)
                previous = key
                state = states[index,t]
                assert state['action']==x['action']
                stationary_forwards += x['action']=='move_forward' and frame_history[-1]==state['rgb_sha256']
                frame_history.append(state['rgb_sha256']);executed.append(x['action'])
            outcome = json.loads((lane/('episode_%02d.json'%index)).read_text())
            episodes.append(dict(index=index,steps=len(steps),success=outcome['success'],
                repeated_input_decisions=len(repeated),longest_identical_input_run=longest_same_input,
                stationary_rgb_forward_actions=stationary_forwards,
                repeated_actions_all_identical=all(x['same_action'] for x in repeated),
                max_repeat_logit_difference=max((x['max_logit_difference'] for x in repeated),default=0.),
                first_repeat=repeated[0] if repeated else None))
    assert len(episodes)==100 and {e['index'] for e in episodes}==set(range(100))
    summary=dict(episodes=100,total_actions=sum(e['steps'] for e in episodes),
        episodes_with_repeated_inputs=sum(e['repeated_input_decisions']>0 for e in episodes),
        failures_with_repeated_inputs=sum(e['repeated_input_decisions']>0 and not e['success'] for e in episodes),
        repeated_input_decisions=sum(e['repeated_input_decisions'] for e in episodes),
        stationary_rgb_forward_actions=sum(e['stationary_rgb_forward_actions'] for e in episodes),
        episodes_with_50_identical_inputs=sum(e['longest_identical_input_run']>=50 for e in episodes),
        max_repeat_logit_difference=max(e['max_repeat_logit_difference'] for e in episodes),
        all_repeated_actions_identical=all(e['repeated_actions_all_identical'] for e in episodes))
    result=dict(scope='All 100 already exposed best4k INTERNAL_DEV episodes; read-only trace analysis',
        source=str(RUN.relative_to(LINE)),summary=summary,episodes=sorted(episodes,key=lambda e:e['index']),
        limitation='Exact input cycles establish policy repetition on these traces, not the success of a recovery intervention.')
    (HERE/'REPEATED_INPUTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
