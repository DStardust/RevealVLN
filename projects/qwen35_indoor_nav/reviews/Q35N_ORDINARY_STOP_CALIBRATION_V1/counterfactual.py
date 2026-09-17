"""Exact causal prefix evaluation for nonnegative STOP-only logit offsets.

The deployed decision function takes four logits and one frozen scalar only.
Distances/poses below are offline scoring inputs, never decision inputs.
"""
import math
import statistics

ACTIONS = ('move_forward', 'turn_left', 'turn_right', 'STOP')
GRID = (0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2.)


def choose(logits, bias):
    if len(logits) != 4 or not all(math.isfinite(x) for x in logits):
        raise ValueError('FOUR_FINITE_LOGITS_REQUIRED')
    if not math.isfinite(bias) or bias < 0:
        raise ValueError('NONNEGATIVE_FINITE_BIAS_REQUIRED')
    values = list(logits)
    values[3] += bias
    if not math.isfinite(values[3]):
        raise ValueError('ADJUSTED_LOGIT_OVERFLOW')
    return ACTIONS[max(range(4), key=lambda i: values[i])]


def ndtw(positions, reference):
    unique = [x for i, x in enumerate(positions) if i == 0 or x != positions[i-1]]
    assert unique and reference
    previous = [0.] + [math.inf] * len(reference)
    for a in unique:
        row = [math.inf] * (len(reference)+1)
        for j, b in enumerate(reference, 1):
            row[j] = math.dist(a, b) + min(previous[j], previous[j-1], row[j-1])
        previous = row
    return math.exp(-previous[-1]/(3*len(reference)))


def prefix(episode, decisions, reference, bias):
    assert len(decisions) == episode['steps']
    assert len(episode['positions']) == len(episode['distances']) == len(decisions)+1
    assert 0 < len(decisions) <= 500 and (episode['stopped'] or len(decisions) == 500)
    cutoff = None
    for j, d in enumerate(decisions):
        assert d['step'] == j+1 and d['index'] == episode['index']
        assert d['action'] == choose(d['logits'], 0), 'RAW_ACTION_NOT_ARGMAX'
        assert d['action'] != 'STOP' or j == len(decisions)-1
        if cutoff is None and choose(d['logits'], bias) == 'STOP':
            cutoff = j
    if cutoff is None:
        assert not episode['stopped']
        positions = episode['positions']
        distances = episode['distances']
        steps = len(decisions)
    else:
        # Decision j consumes observation at position j. Its counterfactual STOP
        # does NOT execute the recorded movement from j to j+1.
        positions = episode['positions'][:cutoff+1] + [episode['positions'][cutoff]]
        distances = episode['distances'][:cutoff+1] + [episode['distances'][cutoff]]
        steps = cutoff+1
    success = float(cutoff is not None and distances[-1] < 3.)
    path = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
    spl = success * distances[0] / max(distances[0], path)
    return dict(index=episode['index'], episode_id=episode['episode_id'], house=episode['house'],
                success=success, spl=spl, ndtw=ndtw(positions, reference), steps=steps,
                stopped=cutoff is not None, navigation_error_m=distances[-1],
                oracle_success=float(min(distances)<3.), path_length_m=path,
                newly_truncated=steps<episode['steps'], first_stop_step=cutoff+1 if cutoff is not None else None)


def stats(rows):
    return dict(n=len(rows), successes=int(sum(r['success'] for r in rows)),
                sr=statistics.mean(r['success'] for r in rows),
                spl=statistics.mean(r['spl'] for r in rows),
                ndtw=statistics.mean(r['ndtw'] for r in rows))


def assess(before, after):
    assert len(before) == len(after) and before
    pairs = list(zip(before, after))
    assert all((a['index'], a['episode_id'], a['house']) ==
               (b['index'], b['episode_id'], b['house']) for a, b in pairs)
    delta = {k: statistics.mean(b[k]-a[k] for a, b in pairs) for k in ('success','spl','ndtw')}
    houses = sorted({a['house'] for a, b in pairs})
    gains = {h: sum(b['success']-a['success'] for a, b in pairs if a['house']==h) for h in houses}
    loo = {h: statistics.mean(b['success']-a['success'] for a, b in pairs if a['house']!=h) for h in houses}
    wins = sum(not a['success'] and bool(b['success']) for a, b in pairs)
    losses = sum(bool(a['success']) and not b['success'] for a, b in pairs)
    gate = (wins-losses >= 2 and delta['spl'] >= 0 and delta['ndtw'] >= -.01
            and sum(x>0 for x in gains.values()) >= 2 and all(x>=0 for x in loo.values()))
    return dict(delta=delta, wins=wins, losses=losses, by_house_net_success=gains,
                leave_one_house_out_sr_delta=loo, fit_gate=gate)


def select(candidates):
    assert tuple(c['bias'] for c in candidates) == GRID
    eligible = [c for c in candidates if c['comparison']['fit_gate']]
    return min(eligible, key=lambda c: (-c['metrics']['sr'], -c['metrics']['spl'], c['bias'])) if eligible else None
