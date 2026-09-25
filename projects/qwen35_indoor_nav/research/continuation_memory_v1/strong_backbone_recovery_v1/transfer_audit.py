"""Behavioral prefix audit; finite floating-point differences remain visible."""
import math


def argmax(values):
    if len(values) != 4 or not all(math.isfinite(x) for x in values):
        raise ValueError('NONFINITE_OR_INVALID_ACTION_LOGITS')
    return max(range(4), key=lambda i: values[i])


def audit_pair(native, method, native_result, method_result, zero=False):
    actions = [[r for r in rows if r['event'] == 'action'] for rows in (native, method)]
    first = next((i for i, (a,b) in enumerate(zip(*actions)) if a['executed_action'] != b['executed_action']), None)
    if first is None and len(actions[0]) != len(actions[1]):
        raise ValueError('TERMINATION_WITHOUT_ACTION_INTERVENTION')
    limit = len(actions[0]) if first is None else first
    for i in range(min(len(actions[0]),len(actions[1]),limit+1)):
        a,b=actions[0][i],actions[1][i]
        if a['before_rgb_sha256'] != b['before_rgb_sha256']:
            raise ValueError(f'RAW_PREFIX_DIVERGED_AT_{i}')
        if i<limit and a['after_rgb_sha256'] != b['after_rgb_sha256']:
            raise ValueError(f'TRANSITION_DIVERGED_AT_{i}')
    resets=[[r for r in rows if r['event']=='reset'] for rows in (native,method)]
    if resets[0]!=resets[1]:raise ValueError('INITIAL_STATE_DIVERGED')
    queries=[{r['environment_step']:r for r in rows if r['event']=='generation' and r['environment_step']<=limit} for rows in (native,method)]
    if set(queries[0])!=set(queries[1]):raise ValueError('QUERY_CADENCE_DIVERGED_BEFORE_INTERVENTION')
    max_delta=0.;max_relative=0.;minimum_margin=None;count=0
    for step,a in queries[0].items():
        b=queries[1][step]
        if a['input']!=b['input']:raise ValueError(f'PROCESSED_PREFIX_DIVERGED_AT_{step}')
        x,y=a['native_logits'],b['native_logits']
        if argmax(x)!=argmax(y):raise ValueError(f'NATIVE_ARGMAX_FLIP_AT_{step}')
        delta=max(abs(i-j) for i,j in zip(x,y));max_delta=max(max_delta,delta)
        max_relative=max(max_relative,delta/max(1e-12,max(abs(i) for i in x)))
        ordered=sorted(x,reverse=True);margin=ordered[0]-ordered[1]
        minimum_margin=margin if minimum_margin is None else min(minimum_margin,margin)
        count+=1
    if first is None and native_result!=method_result:raise ValueError('TERMINAL_DIVERGED_WITHOUT_INTERVENTION')
    if zero and (first is not None or max_delta !=0 or
            [r['generated_ids'] for r in queries[0].values()] != [r['generated_ids'] for r in queries[1].values()]):
        raise ValueError('ZERO_RESIDUAL_IDENTITY_FAILED')
    return dict(input_prefix_matched=True,action_prefix_matched=True,logits_bitwise_equal=max_delta==0,
        max_logit_delta=max_delta,max_relative_logit_delta=max_relative,minimum_native_margin=minimum_margin,
        argmax_flip_count=0,compared_queries=count,first_executed_override_step=first,
        full_trajectory_matched=first is None)

