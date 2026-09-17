"""Train-count-only bounded weighting; no tensor/device dependencies."""
import math
ACTIONS = ['move_forward', 'turn_left', 'turn_right', 'STOP']

def make_weights(counts):
    assert set(counts) == set(ACTIONS)
    assert all(isinstance(counts[a], int) and counts[a] > 0 for a in ACTIONS)
    peak = max(counts.values())
    raw = [min(4., math.sqrt(peak / counts[a])) for a in ACTIONS]
    mean = sum(counts[a]*w for a,w in zip(ACTIONS,raw))/sum(counts.values())
    values = [w/mean for w in raw]
    assert all(math.isfinite(w) and w > 0 for w in values)
    return dict(actions=ACTIONS,counts=counts,raw=raw,normalizer=mean,values=values,
                formula='min(4,sqrt(max_train_count/count)); normalize train-frequency mean to 1',
                denominator='global sum of target weights per optimizer update',
                source='same fixed 10 train routes, not sampled schedule or dev')
