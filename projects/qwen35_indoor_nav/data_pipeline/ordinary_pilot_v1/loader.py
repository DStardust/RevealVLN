"""Only this causal projection is allowed to supply policy input, never whole records."""


def decision_example(policy_record, supervision_record, t):
    frames = policy_record['rgb_sequence']
    actions = supervision_record['actions']
    if len(frames) != len(actions) or not 0 <= t < len(actions):
        raise ValueError('LENGTH_OR_TIME_MISMATCH')
    allowed = {'move_forward', 'turn_left', 'turn_right', 'STOP'}
    if any(a not in allowed for a in actions) or actions[-1] != 'STOP' or 'STOP' in actions[:-1]:
        raise ValueError('INVALID_ACTIONS')
    return {'policy_input':{'instruction':policy_record['instruction'],
                            'rgb_history':frames[:t+1],
                            'executed_actions':actions[:t]},
            'target_action':actions[t]}
