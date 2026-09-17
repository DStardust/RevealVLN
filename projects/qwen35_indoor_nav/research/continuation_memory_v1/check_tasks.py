"""Compositional exact-state targets for the existing SEE2 task checker, CPU only.

This does not implement room arrival or change the frozen checker semantics.
"""


def exact_state_targets(compiler, trace, task_id, cutoff):
    if type(cutoff) is not int or not 0 <= cutoff < len(trace['observations']):
        raise ValueError('CAUSAL_CUTOFF_REQUIRED')
    prefix = dict(trace, observations=trace['observations'][:cutoff+1], actions=trace['actions'][:cutoff])
    task = compiler.tasks[task_id]
    events = compiler.atoms(prefix['observations'])
    original = compiler.m2(prefix, task_id)
    seen = False
    rows = []
    for t, event in enumerate(events):
        known = original[t]['truth_known']
        before = seen
        seen = seen or bool(event[task['anchor']])
        terminal = bool(event[task['terminal']])
        ready = before and terminal
        if known and ready != (original[t]['state'] == 'READY_TO_STOP'):
            raise ValueError('EXACT_STATE_DISAGREES_WITH_FROZEN_CHECKER')
        rows.append(dict(step=t, truth_known=known, loss_mask=int(known),
            anchor_seen_strictly_before=before if known else None,
            anchor_seen_through_current=seen if known else None,
            terminal_witness_now=terminal if known else None,
            ordered_ready_to_stop=ready if known else None))
    return rows
