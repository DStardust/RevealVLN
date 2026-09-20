"""One selector for training diagnostics, rollout and evidence reconstruction."""
import math

ACTIONS = ('move_forward', 'turn_left', 'turn_right', 'STOP')

def index(values):
    values = list(values)
    if len(values) != 4 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError('FOUR_FINITE_METHOD_LOGITS_REQUIRED')
    return max(range(4), key=lambda i: (float(values[i]), -i))

def select(native, method):
    a, b = index(native), index(method)
    def margin(values):
        ordered = sorted(map(float, values), reverse=True)
        return ordered[0]-ordered[1]
    return dict(native_action=ACTIONS[a], method_action=ACTIONS[b], executed_action=ACTIONS[b],
                native_margin=margin(native), method_margin=margin(method),
                native_stop_method_continue=a==3 and b!=3,
                legacy_guard_shadow_action=ACTIONS[3 if a==3 else b], override=a!=b)

def tensor_indices(logits):
    """Differentiation is through losses, never through this discrete selector."""
    import torch
    if logits.shape[-1] != 4 or not bool(torch.isfinite(logits).all()):
        raise ValueError('FOUR_FINITE_METHOD_LOGITS_REQUIRED')
    return logits.argmax(-1)
