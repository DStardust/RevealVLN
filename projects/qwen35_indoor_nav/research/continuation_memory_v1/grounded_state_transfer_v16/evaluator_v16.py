"""Prospective collision-free SEE2 endpoint; old compiler remains read-only."""
from v16_common import LINE, load
legacy = load('v16_legacy_compiler', LINE/'data_pipeline/mechanism_factory_v2/compiler.py')

def state_sequence(compiler, observations, task_id):
    events = compiler.atoms(observations)
    seen = task_id == 'task_T'
    result = []
    for event in events:
        terminal = bool(event['terminal'])
        anchor = False if task_id == 'task_T' else bool(event[compiler.tasks[task_id]['anchor']])
        result.append([int(seen), int(seen or anchor), int(terminal), int(seen and terminal)])
        seen = seen or anchor
    return result

def query_context(compiler, trace, task_id, cutoff):
    events = compiler.atoms(trace['observations'])
    return dict(stop_at_cutoff=len(events)-1==cutoff, stops=trace['actions'][-1:]==['S'],
                suffix_anchor_before_final=task_id=='task_T' or any(bool(e[compiler.tasks[task_id]['anchor']]) for e in events[cutoff+1:-1]),
                terminal_at_final=bool(events[-1]['terminal']))

def query_contexts(compiler, trace, task_id):
    """Equivalent to querying each cutoff, with one immutable event scan."""
    events=compiler.atoms(trace['observations']);last=len(events)-1
    future_anchor=task_id=='task_T';result=[None]*len(trace['actions'])
    for t in range(last,-1,-1):
        if t<len(result):
            result[t]=dict(stop_at_cutoff=t==last,stops=trace['actions'][-1:]==['S'],
                suffix_anchor_before_final=future_anchor,terminal_at_final=bool(events[-1]['terminal']))
        if t<last and task_id!='task_T':future_anchor=future_anchor or bool(events[t][compiler.tasks[task_id]['anchor']])
    return result

def compose(state, query):
    value = state[3] if query['stop_at_cutoff'] else (state[1] or query['suffix_anchor_before_final'])
    return bool(value and query['stops'] and (query['stop_at_cutoff'] or query['terminal_at_final']))

def evaluate(compiler, trace, task_id, cutoff):
    collision = trace.get('collisions')
    structural = type(collision) is int and collision >= 0 and legacy.complete(dict(trace, collisions=0))
    actions = trace.get('actions', [])
    active_stop = bool(actions[-1:] == ['S'])
    within_budget = len(actions) <= 500
    old = ('unknown' if not legacy.complete(trace) else
           ('pass' if active_stop and bool(compiler.atoms(trace['observations'])[-1]['terminal']) else 'fail')) if task_id=='task_T' else compiler.evaluate(trace, task_id)
    states = state_sequence(compiler, trace['observations'], task_id) if structural else []
    ready = [bool(z[3]) for z in states]
    safe = ('UNKNOWN' if not structural else
            'PASS' if active_stop and within_budget and collision==0 and ready[-1] else 'FAIL')
    return dict(legacy_v15_label=old.upper(), safe_v16_label=safe, structure_complete=structural,
                collisions=collision, total_decisions=len(actions), stopped=active_stop,
                legal_stop=structural and active_stop and within_budget,
                budget_exhausted=len(actions)==500 and not active_stop,
                event_order_satisfied=ready[-1] if ready else None,
                forced_prefix_ready=any(ready[:cutoff]),
                ready_at_takeover=ready[cutoff] if len(ready)>cutoff else None,
                ready_after_autonomous_motion=any(ready[cutoff+1:]),
                ready_at_active_stop=ready[-1] if ready and active_stop else False,
                budget_penalized_cost=len(actions)/500 if safe=='PASS' else 1.)
