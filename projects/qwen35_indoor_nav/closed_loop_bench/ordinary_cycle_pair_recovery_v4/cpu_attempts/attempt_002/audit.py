"""CPU checks for the requested V4 contract; no model or simulator is started."""
import hashlib
import json
import math

TENSOR_FIELDS = (
    'input_ids', 'attention_mask', 'mm_token_type_ids', 'image_grid_thw',
    'pixel_values', 'exec_index', 'action_index',
)
ACTIONS = ('move_forward', 'turn_left', 'turn_right', 'STOP')


class NumericInvalid(ValueError):
    def __init__(self, reason, control=None, recovery=None):
        self.evidence = dict(status='NUMERIC_OR_TRANSPORT_INVALID', reason=reason,
                             control=control, recovery=recovery)
        super().__init__(reason)


def raw_key(original_input):
    payload = [original_input['instruction'], original_input['rgb_sha256'],
               original_input['executed']]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def check_decision(row):
    original = row['original_input']
    if row['input_key'] != raw_key(original):
        raise NumericInvalid('RAW_KEY_MISMATCH', recovery=row)
    tensors = row['processed_sha256']
    if set(tensors) != set(TENSOR_FIELDS):
        raise NumericInvalid('PROCESSED_TENSOR_COVERAGE', recovery=row)
    if any(not isinstance(v, str) or len(v) != 64 or
           any(c not in '0123456789abcdef' for c in v) for v in tensors.values()):
        raise NumericInvalid('INVALID_TENSOR_HASH', recovery=row)
    values = row['logits']
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise NumericInvalid('NONFINITE_OR_MISSING_LOGITS', recovery=row)
    native = ACTIONS[max(range(4), key=values.__getitem__)]
    if row['native_action'] != native or row['executed_action'] not in ACTIONS:
        raise NumericInvalid('ACTION_RECORD_INVALID', recovery=row)
    if row['override'] != (row['executed_action'] != native):
        raise NumericInvalid('OVERRIDE_RECORD_INVALID', recovery=row)
    if native == 'STOP' and row['override']:
        raise NumericInvalid('STOP_OVERRIDDEN', recovery=row)


def check_prefix_decision(control, recovery):
    """Includes the first actual override's original input and native logits."""
    check_decision(control)
    check_decision(recovery)
    for field in ('step', 'original_input', 'input_key', 'processed_sha256',
                  'logits', 'native_action'):
        if control[field] != recovery[field]:
            raise NumericInvalid('PREFIX_MISMATCH:' + field, control, recovery)
    if control['override'] or control['executed_action'] != control['native_action']:
        raise NumericInvalid('CONTROL_NOT_NATIVE', control, recovery)
    if not recovery['override'] and control['executed_action'] != recovery['executed_action']:
        raise NumericInvalid('PREFIX_EXECUTED_ACTION_MISMATCH', control, recovery)


def check_pair(control, recovery, control_terminal, recovery_terminal):
    for offset, row in enumerate(recovery):
        if offset >= len(control):
            raise NumericInvalid('CONTROL_ENDED_BEFORE_MATCHED_PREFIX', recovery=row)
        if row['step'] != offset + 1 or control[offset]['step'] != offset + 1:
            raise NumericInvalid('NONCONTIGUOUS_PREFIX', control[offset], row)
        check_prefix_decision(control[offset], row)
        if row['override']:
            return dict(checked_decisions=offset + 1, first_override=offset + 1)
    if not recovery or len(control) != len(recovery):
        raise NumericInvalid('NO_OVERRIDE_TRAJECTORY_LENGTH_MISMATCH')
    if control_terminal is None or control_terminal != recovery_terminal:
        raise NumericInvalid('NO_OVERRIDE_TERMINAL_MISMATCH', control_terminal, recovery_terminal)
    return dict(checked_decisions=len(recovery), first_override=None)


def require_complete(launcher_status, indices, prefixes_exact, parameters_unchanged):
    if launcher_status != 'COMPLETE' or sorted(indices) != list(range(200)):
        raise ValueError('INCOMPLETE_DEV_CASE_NO_ADMITTED_METRICS')
    if not prefixes_exact or not parameters_unchanged:
        raise NumericInvalid('CORRECTNESS_GATE_NOT_PASSED')


def require_owned(initial, current, members, launcher_pid):
    """Reject a cleanup request before any signal unless its receipt still binds."""
    pid = initial['pid']
    if initial['ppid'] != launcher_pid or initial['pgid'] != pid or initial['sid'] != pid:
        raise ValueError('NOT_CREATED_AS_OWN_SESSION')
    if current is None or any(current[k] != initial[k] for k in
                              ('pid', 'start_ticks', 'ppid', 'pgid', 'sid')):
        raise ValueError('OWN_PID_IDENTITY_NOT_VERIFIABLE')
    if not members or any(m['pgid'] != pid or m['sid'] != pid for m in members):
        raise ValueError('FOREIGN_PROCESS_GROUP_MEMBER')


def blocked_result(reason):
    return dict(status='RESOURCE_NOT_RESERVED', reason=reason, formal_starts=0,
                completed_episodes=0, completed_pairs=0, environment_decisions=0,
                optimizer_updates=0, gpu_wall_seconds=0, task_execution_success=False,
                evaluation_valid=False, intervention_benefit='unknown',
                scientific_generalization_established=False,
                native=None, recovery=None, delta_sr=None, engineering_candidate_pass=None,
                all_owned_gpu_jobs_closed=True, external_processes_signaled=[])
