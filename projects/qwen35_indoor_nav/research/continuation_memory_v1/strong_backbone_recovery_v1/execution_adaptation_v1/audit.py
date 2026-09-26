"""Audit all-token generation prefixes, including a causal EOS intervention."""
import math
import struct


def _check(condition, reason):
    if not condition:
        raise ValueError(reason)


def _argmax(values):
    _check(len(values) == 4 and all(math.isfinite(value) for value in values), 'NONFINITE_OR_INVALID_ACTION_LOGITS')
    return max(range(4), key=lambda index: values[index])


def _trace(rows, action_token_ids, eos_token_id):
    mapping = {token: action for action, token in enumerate(action_token_ids)}
    _check(len(mapping) == 4 and eos_token_id not in mapping, 'INVALID_TOKEN_MAPPING')
    reset = None
    generations, actions = [], []
    pending = []
    current_rgb = None
    stopped = False
    current_query = None
    for event in rows:
        kind = event['event']
        if kind == 'reset':
            _check(reset is None and not generations and not actions, 'MULTIPLE_OR_LATE_RESET')
            reset = dict(rgb_sha256=event['rgb_sha256'], instruction_sha256=event['instruction_sha256'])
            current_rgb = reset['rgb_sha256']
        elif kind == 'memory_write':
            _check(not stopped, 'STOP_CREATED_MEMORY_OBSERVATION')
        elif kind == 'generation':
            _check(reset is not None and not stopped and not pending, 'GENERATION_OUTSIDE_EXECUTION_BOUNDARY')
            start = event['environment_step']
            _check(start == len(actions), 'GENERATION_STEP_MISMATCH')
            _check(event['input']['rgb_sha256'] == current_rgb, 'QUERY_RAW_INPUT_MISMATCH')
            for field in ('inputs', 'images', 'depths', 'poses', 'intrinsics'):
                _check(set(('shape', 'dtype', 'sha256')) <= set(event['input'][field]), 'INCOMPLETE_PROCESSED_INPUT_EVIDENCE')
            _check('time_ids' in event['input'], 'MISSING_TIME_IDS')
            header = event['header_ids']
            generated = event['generated_ids']
            _check(header and generated[:len(header)] == header, 'ASSISTANT_HEADER_MISMATCH')
            body = generated[len(header):]
            _check(body and body[-1] == eos_token_id and eos_token_id not in body[:-1], 'UNSUPPORTED_EOS_SEQUENCE')
            _check(len(body) <= 5 and all(token in mapping for token in body[:-1]), 'UNSUPPORTED_GENERATED_TOKEN_OR_FIFTH_ACTION')
            tokens = event['tokens']
            _check(len(tokens) == len(body), 'MISSING_TOKEN_DECISION_EVIDENCE')
            memory_hashes = set()
            for offset, (record, generated_token) in enumerate(zip(tokens, body)):
                _check(record['offset'] == offset, 'TOKEN_OFFSET_MISMATCH')
                _check(record['generated_token'] == record['method_token'] == generated_token, 'ACTUAL_GENERATION_ALIGNMENT_FAILED')
                _check(record['native_token'] in mapping or record['native_token'] == eos_token_id, 'UNSUPPORTED_NATIVE_FULL_VOCABULARY_TOKEN')
                _check(record['native_action'] == _argmax(record['native_logits'])
                       and record['method_action'] == _argmax(record['method_logits']), 'FOUR_CLASS_ACTION_LOG_MISMATCH')
                # Full-vocabulary EOS is legitimate even when the best four-class
                # action is movement. Equal four-class scores may also tie by
                # vocabulary index rather than this action-list order.
                for field, logits_name in (('native_token', 'native_logits'), ('method_token', 'method_logits')):
                    if record[field] in mapping:
                        _check(record[logits_name][mapping[record[field]]] == max(record[logits_name]), 'FULL_ACTION_TOKEN_NOT_MAXIMAL')
                _check(record['query_writes'] == start + 1, 'FUTURE_MEMORY_IN_GENERATED_CHUNK')
                memory_hashes.add(record['query_memory_sha256'])
            _check(len(memory_hashes) == 1, 'MEMORY_CHANGED_DURING_GENERATION')
            parsed = [mapping[token] for token in body[:-1]]
            _check(event['parsed_actions'] == parsed and event['empty_action_fallback'] == (not parsed), 'PARSED_ACTIONS_MISMATCH')
            pending = list(enumerate(parsed)) if parsed else [(None, 0)]
            current_query = start
            generations.append(event)
        elif kind == 'action':
            _check(reset is not None and not stopped and bool(pending), 'EXECUTED_ACTION_WITHOUT_GENERATED_TOKEN')
            offset, action = pending.pop(0)
            _check(event['step'] == len(actions) + 1 <= 500, 'ACTION_STEP_OR_BUDGET_MISMATCH')
            _check(event['query_environment_step'] == current_query and event['token_offset'] == offset,
                   'EXECUTED_TOKEN_OFFSET_MISMATCH')
            _check(event['executed_action'] == action and event['before_rgb_sha256'] == current_rgb,
                   'EXECUTION_OR_RGB_TRANSPORT_MISMATCH')
            current_rgb = event['after_rgb_sha256']
            actions.append(event)
            stopped = action == 0
    _check(reset is not None and bool(actions), 'EMPTY_OR_INCOMPLETE_TRACE')
    _check(stopped or len(actions) == 500, 'TRACE_NOT_TERMINATED')
    return reset, generations, actions


def audit_pair(native, method, native_result, method_result, zero=False, *, action_token_ids, eos_token_id):
    """Compare through the first GENERATED difference; report physical change separately.

    An EOS/action change can legitimately alter later query cadence before
    physical actions diverge. Consequently queries after that generated-token
    intervention are not falsely compared as an untouched numeric prefix.
    """
    a_reset, a_queries, a_actions = _trace(native, action_token_ids, eos_token_id)
    b_reset, b_queries, b_actions = _trace(method, action_token_ids, eos_token_id)
    _check(native_result['steps'] == len(a_actions) and method_result['steps'] == len(b_actions), 'TERMINAL_STEP_COUNT_MISMATCH')
    _check(a_reset == b_reset, 'INITIAL_RGB_OR_INSTRUCTION_DIVERGED')
    first_executed = next((i for i, (a, b) in enumerate(zip(a_actions, b_actions))
                           if a['executed_action'] != b['executed_action']), None)
    if first_executed is None and len(a_actions) != len(b_actions):
        first_executed = min(len(a_actions), len(b_actions))
    limit = min(len(a_actions), len(b_actions)) if first_executed is None else first_executed
    for i, (a, b) in enumerate(zip(a_actions, b_actions)):
        if i > limit:
            break
        _check(a['before_rgb_sha256'] == b['before_rgb_sha256'], f'RAW_ACTION_PREFIX_DIVERGED_AT_{i}')
        if i < limit:
            for key in ('after_rgb_sha256', 'distance', 'collision'):
                _check(a[key] == b[key], f'PHYSICAL_TRANSITION_DIVERGED_AT_{i}:{key}')
    maximum = relative = 0.0
    bitwise = True
    minimum_margin = None
    compared_tokens = compared_queries = 0
    first_generated = None
    for query_index, (a, b) in enumerate(zip(a_queries, b_queries)):
        start = a['environment_step']
        _check(start == b['environment_step'], f'QUERY_CADENCE_DIVERGED_BEFORE_TOKEN_INTERVENTION_AT_{query_index}')
        _check(a['header_ids'] == b['header_ids'] and a['input'] == b['input'], f'PROCESSED_QUERY_PREFIX_DIVERGED_AT_{start}')
        compared_queries += 1
        for offset, (x, y) in enumerate(zip(a['tokens'], b['tokens'])):
            left, right = x['native_logits'], y['native_logits']
            _check(_argmax(left) == _argmax(right), f'NATIVE_FOUR_CLASS_ARGMAX_FLIP_AT_{start}:{offset}')
            _check(x['native_token'] == y['native_token'], f'NATIVE_FULL_VOCABULARY_FLIP_AT_{start}:{offset}')
            delta = max(abs(i - j) for i, j in zip(left, right))
            maximum = max(maximum, delta)
            relative = max(relative, delta / max(1e-12, max(abs(value) for value in left)))
            bitwise = bitwise and struct.pack('<4f', *left) == struct.pack('<4f', *right)
            margin = sorted(left, reverse=True)[0] - sorted(left, reverse=True)[1]
            minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
            compared_tokens += 1
            if x['generated_token'] != y['generated_token']:
                first_generated = dict(query_index=query_index, query_environment_step=start, token_offset=offset,
                    reference_token=x['generated_token'], method_token=y['generated_token'],
                    reference_is_eos=x['generated_token'] == eos_token_id, method_is_eos=y['generated_token'] == eos_token_id)
                break
        if first_generated is not None:
            break
        _check(len(a['tokens']) == len(b['tokens']), 'GENERATED_LENGTH_CHANGED_WITHOUT_TOKEN_INTERVENTION')
    if first_generated is None:
        _check(len(a_queries) == len(b_queries), 'QUERY_COUNT_CHANGED_WITHOUT_TOKEN_INTERVENTION')
        _check(first_executed is None and len(a_actions) == len(b_actions), 'EXECUTION_CHANGED_WITHOUT_TOKEN_INTERVENTION')
        _check(native_result == method_result, 'TERMINAL_DIVERGED_WITHOUT_TOKEN_INTERVENTION')
    if zero:
        _check(first_generated is None, 'ZERO_RESIDUAL_GENERATED_TOKEN_CHANGED')
    full_physical = first_executed is None and len(a_actions) == len(b_actions) and native_result == method_result
    return dict(input_prefix_matched=True, action_prefix_matched=True,
        logits_bitwise_equal=bitwise, max_logit_delta=maximum, max_relative_logit_delta=relative,
        minimum_native_four_class_margin=minimum_margin, argmax_flip_count=0, full_vocab_argmax_flip_count=0,
        compared_queries=compared_queries, compared_token_decisions=compared_tokens,
        first_generated_difference=first_generated, first_executed_override_step=first_executed,
        first_executed_difference=None if first_executed is None else dict(environment_step=first_executed,
            reference_action=a_actions[first_executed]['executed_action'] if first_executed < len(a_actions) else None,
            method_action=b_actions[first_executed]['executed_action'] if first_executed < len(b_actions) else None),
        full_trajectory_matched=first_generated is None, full_physical_trajectory_matched=full_physical,
        prefix_boundary='FIRST_ACTUAL_GENERATED_TOKEN_DIFFERENCE_INCLUDING_EOS')
