"""Known prefix-quantile history sampling; no learned memory or privileged input."""
import hashlib
import runpy
from pathlib import Path

FRAMES = 8
RGB_BYTES = 224 * 224 * 3
BLACK = bytes(RGB_BYTES)
LINE = Path(__file__).resolve().parents[2]
COMMON = LINE / 'closed_loop_bench/r2r_ce_tiny_v1/common.py'

def indices(t):
    if type(t) is not int or t < 0:
        raise ValueError('DECISION_TIME')
    length = max(FRAMES, t + 1)
    padding = length - t - 1
    # Equivalent to np.linspace(0, length-1, 7, endpoint=False, dtype=int)
    selected = [(j * (length - 1)) // (FRAMES - 1) - padding for j in range(FRAMES - 1)]
    return [None if x < 0 else x for x in selected] + [t]

def choose(prefix, t, black):
    if len(prefix) <= t:
        raise ValueError('MISSING_CURRENT_OBSERVATION')
    return [black if i is None else prefix[i] for i in indices(t)]

_base = runpy.run_path(str(COMMON))
class PrefixWindow(_base['Window']):
    def __init__(self):
        super().__init__()
        self.frames = []
        self.decision_t = None

    def receive(self, payload, executed=None):
        if executed is not None and payload.get('done') is not True and len(self.frames) >= 500:
            raise ValueError('ONLINE_OBSERVATION_BUDGET')
        alive = super().receive(payload, executed)
        if not alive:
            return False
        latest = self.images[-1]
        if executed is None:
            self.frames = [latest]
            self.decision_t = 0
        else:
            self.frames.append(latest)
            self.decision_t += 1
        self.images = choose(self.frames, self.decision_t, BLACK)
        return True

    def input_audit(self):
        return dict(input_frame_indices=indices(self.decision_t),
                    input_rgb_sha256=[hashlib.sha256(x).hexdigest() for x in self.images],
                    history_rule='navila_prefix_quantiles_8',
                    padding_mask=[i is None for i in indices(self.decision_t)])

