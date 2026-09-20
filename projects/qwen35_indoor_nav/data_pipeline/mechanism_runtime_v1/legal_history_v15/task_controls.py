"""History-independent terminal control; the ordered SEE2 checker is unchanged."""


def terminal_only(compiler,trace):
    if not compiler.complete(trace):return 'unknown'
    return 'pass' if (trace['actions'][-1:] == ['S'] and
        compiler.atoms(trace['observations'])[-1]['terminal']) else 'fail'


def neutral_span(compiler,trace,span):
    lo,hi=span
    assert 0<=lo<hi<len(trace['observations'])
    events=compiler.atoms(trace['observations'])
    roles={t['anchor'] for t in compiler.tasks.values()}|{'terminal'}
    witnesses=[dict(step=t,role=r,instances=events[t][r]) for t in range(lo,hi+1)
               for r in sorted(roles) if events[t][r]]
    return dict(complete=compiler.complete(trace),motion_steps=hi-lo,
        forward_steps=trace['actions'][lo:hi].count('F'),witnesses=witnesses,
        pass_control=compiler.complete(trace) and not witnesses and 'F' in trace['actions'][lo:hi])
