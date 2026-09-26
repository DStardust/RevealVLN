"""CPU admission of captured action tokens; no model or simulator execution."""
from collections import Counter, defaultdict
import hashlib
import math
import struct


def _check(condition, reason):
    if not condition:
        raise ValueError(reason)


def _matrix_hash(rows, count, width, name):
    _check(len(rows) == count, name + '_ROW_COUNT')
    digest = hashlib.sha256(str((count, width, 'float32')).encode())
    for row in rows:
        _check(len(row) == width, name + '_WIDTH')
        values = [float(value) for value in row]
        _check(all(math.isfinite(value) for value in values), name + '_NONFINITE')
        try:
            digest.update(struct.pack('<' + 'f' * width, *values))
        except (OverflowError, struct.error) as error:
            raise ValueError(name + '_NOT_FLOAT32') from error
    return digest.hexdigest()


def _trajectory(trace):
    actions, queries = trace['actions'], trace['query_steps']
    _check(bool(actions) and len(actions) <= 500, 'ORIGINAL_DECISION_BUDGET')
    _check(actions[-1] == 0 and 0 not in actions[:-1], 'STOP_NOT_SINGLE_TERMINAL')
    _check(all(action in range(4) for action in actions), 'ACTION_VOCABULARY')
    _check(queries and queries[0] == 0 and queries == sorted(set(queries)), 'QUERY_ORDER')
    _check(queries[-1] < len(actions), 'QUERY_OUTSIDE_EXECUTION')
    _check(len(trace['rgb_sha256']) == len(actions), 'STOP_EXTRA_OR_MISSING_OBSERVATION')
    _check(trace['partition'] in ('FIT', 'DEV'), 'NONTRAIN_SPLIT')
    _check(trace['kind'] in ('RECOVERY', 'PRESERVATION'), 'UNREGISTERED_KIND')


def validate_chunk(trace, plan, generated_ids, actor_rows, native_rows, *, header_ids,
                   action_token_ids, eos_token_id, context_query_start,
                   memory_replay_exclusive_end, actor_width=3584, pad_token_id=None):
    """Validate one requested query; matrices contain action rows only, before forcing.

    ``generated_ids`` begins at the assistant header, matching the actual old
    StreamVLN wrapper. Only harmless EOS/pad suffixes may be trimmed. A token
    for an unexecuted action is rejected, never admitted as a training row.
    """
    _trajectory(trace)
    _check(len(action_token_ids) == 4 and len(set(action_token_ids)) == 4, 'ACTION_TOKEN_MAPPING')
    _check(eos_token_id not in action_token_ids and bool(header_ids), 'HEADER_EOS_CONTRACT')
    query_index = plan['query_index']
    _check(0 <= query_index < len(trace['query_steps']), 'QUERY_INDEX')
    start = trace['query_steps'][query_index]
    end = trace['query_steps'][query_index + 1] if query_index + 1 < len(trace['query_steps']) else len(trace['actions'])
    executed = trace['actions'][start:end]
    _check(1 <= len(executed) <= 4, 'CHUNK_ACTION_BUDGET')
    _check(plan['trajectory_id'] == trace['id'] and plan['split'] == trace['partition']
           and plan['kind'] == trace['kind'], 'PLAN_TRAJECTORY_IDENTITY')
    _check(plan['request_id'] == f'{trace["id"]}:q{query_index}', 'REQUEST_IDENTITY')
    _check(plan['query_start_step'] == start and plan['original_actor_context_query_start'] == start,
           'PLAN_QUERY_CONTEXT')
    _check(context_query_start == start and memory_replay_exclusive_end == start + 1,
           'LATER_PHYSICAL_OBSERVATION_OR_MEMORY_LEAK')
    _check(plan['allowed_memory_feature_index'] == start and plan['memory_replay_exclusive_end'] == start + 1,
           'PLAN_MEMORY_BOUNDARY')
    _check(plan['chunk_length'] == len(executed) == len(plan['actions']), 'PLAN_CHUNK_LENGTH')
    expected = list(header_ids) + [action_token_ids[action] for action in executed] + [eos_token_id]
    generated_ids = list(generated_ids)
    _check(generated_ids[:len(expected)] == expected, 'GENERATED_HEADER_ACTION_EOS_MISMATCH')
    allowed_tail = {eos_token_id}
    if pad_token_id is not None:
        _check(pad_token_id not in action_token_ids, 'PAD_IS_ACTION_TOKEN')
        allowed_tail.add(pad_token_id)
    _check(all(token in allowed_tail for token in generated_ids[len(expected):]), 'UNEXECUTED_TOKEN_SUFFIX')
    token_rows = []
    for offset, action in enumerate(executed):
        original = plan['actions'][offset]
        step = start + offset
        supervised = trace['kind'] == 'PRESERVATION' or step >= trace['cutoff']
        old_known = offset == 0 and supervised
        _check(original['environment_step'] == step and original['action_token_offset'] == offset
               and original['executed_action'] == action, 'EXECUTED_ACTION_ALIGNMENT')
        _check(original['original_actor_context_query_start'] == start
               and original['allowed_memory_feature_index'] == start, 'TOKEN_CONTEXT_LEAK')
        _check(original['physical_feature_index_audit_only'] == step, 'PHYSICAL_AUDIT_ALIGNMENT')
        _check(original['autoregressive_prefix_action_ids'] == executed[:offset], 'TOKEN_PREFIX_ALIGNMENT')
        _check(original['recapture_required'] == (offset > 0), 'RECAPTURE_MASK_CHANGED')
        _check(original['existing_actor_feature_index'] == (query_index if offset == 0 else None),
               'OLD_ACTOR_REUSED_FOR_LATER_TOKEN')
        _check(original['original_existing_actor_supervision_known'] == old_known
               and original['original_action_in_registered_supervised_region'] == supervised,
               'ORIGINAL_KNOWN_MASK_CHANGED')
        _check(original['label_admission'] == 'UNCHANGED_NO_NEW_TRAINING_ADMISSION', 'LABEL_ADMISSION_CHANGED')
        expected_next = None if action == 0 else step + 1
        expected_status = 'TERMINAL_STOP_NO_NEW_OBSERVATION' if action == 0 else 'ACTUALLY_OBSERVED'
        _check(original['terminal_stop'] == (action == 0)
               and original['physical_next_feature_index_audit_only'] == expected_next
               and original['next_observation_status'] == expected_status, 'STOP_OBSERVATION_CONTRACT')
        token_rows.append(dict(environment_step=step, action_token_offset=offset,
            generated_token_index=len(header_ids) + offset, executed_action=action,
            captured_actor_row=offset, captured_native_row=offset,
            original_existing_actor_supervision_known=old_known,
            original_action_in_registered_supervised_region=supervised,
            recapture_required=offset > 0, allowed_memory_feature_index=start,
            terminal_stop=action == 0, physical_next_feature_index_audit_only=expected_next))
    _check(sum(row['recapture_required'] for row in token_rows) == plan['missing_actor_positions'],
           'MISSING_POSITION_COUNT')
    return dict(status='CHUNK_CONTRACT_VALID', request_id=plan['request_id'],
        trajectory_id=trace['id'], split=trace['partition'], kind=trace['kind'], query_index=query_index,
        query_start_step=start, memory_replay_exclusive_end=start + 1,
        accepted_generated_ids=expected, ignored_eos_pad_suffix=len(generated_ids) - len(expected),
        captured_actor_rows=len(executed), actor_width=actor_width,
        actor_float32_sha256=_matrix_hash(actor_rows, len(executed), actor_width, 'ACTOR'),
        native_float32_sha256=_matrix_hash(native_rows, len(executed), 4, 'NATIVE'),
        native_scope='Caller supplies actual pre-forcing logits; CPU contract checks alignment/finiteness, not model execution.',
        token_rows=token_rows)


def admit_trajectory(trace, plans, chunks, *, actual_actions, actual_rgb_sha256):
    """Admit only all requested chunks plus a complete exact physical replay."""
    _trajectory(trace)
    _check(list(actual_actions) == trace['actions'], 'ACTUAL_REPLAY_ACTIONS_MISMATCH')
    _check(list(actual_rgb_sha256) == trace['rgb_sha256'], 'ACTUAL_REPLAY_RGB_MISMATCH')
    _check(plans and all(plan['trajectory_id'] == trace['id'] for plan in plans), 'PLAN_GROUP_IDENTITY')
    expected = {plan['request_id']: plan for plan in plans}
    received = {chunk['request_id']: chunk for chunk in chunks}
    _check(len(expected) == len(plans) and len(received) == len(chunks), 'DUPLICATE_QUERY')
    _check(set(received) == set(expected), 'INCOMPLETE_TRAJECTORY_RECAPTURE_GROUP')
    counts = Counter()
    for key, plan in expected.items():
        chunk = received[key]
        _check(chunk['status'] == 'CHUNK_CONTRACT_VALID' and chunk['trajectory_id'] == trace['id'], 'CHUNK_NOT_VALID')
        _check(chunk['query_start_step'] == plan['query_start_step']
               and chunk['memory_replay_exclusive_end'] == plan['memory_replay_exclusive_end'], 'CHUNK_CONTEXT_CHANGED')
        _check(len(chunk['token_rows']) == len(plan['actions']), 'CHUNK_RECEIPT_ROW_COUNT')
        for row, original in zip(chunk['token_rows'], plan['actions']):
            for name in ('environment_step', 'action_token_offset', 'executed_action',
                         'original_existing_actor_supervision_known', 'original_action_in_registered_supervised_region',
                         'recapture_required', 'allowed_memory_feature_index', 'terminal_stop'):
                _check(row[name] == original[name], 'CHUNK_RECEIPT_PLAN_MISMATCH')
            counts['captured_actor_rows'] += 1
            counts['new_actor_positions'] += row['recapture_required']
            counts['new_terminal_stop_positions'] += row['recapture_required'] and row['terminal_stop']
            counts['new_positions_in_registered_supervised_region'] += row['recapture_required'] and row['original_action_in_registered_supervised_region']
            counts['new_positions_in_unknown_prefix'] += row['recapture_required'] and not row['original_action_in_registered_supervised_region']
    return dict(status='TRAJECTORY_RECAPTURE_ADMITTED', trajectory_id=trace['id'], split=trace['partition'],
        kind=trace['kind'], physical_actions=len(actual_actions), active_stop=True,
        physical_replay_exact=True, request_ids=sorted(received), captured_queries=len(chunks),
        **dict(counts), no_new_training_admission=True)


def summarize_coverage(expected_plans, admitted_groups):
    """Report fixed planned coverage, retaining missing trajectories in denominators."""
    expected = defaultdict(Counter)
    by_id = defaultdict(list)
    for plan in expected_plans:
        by_id[plan['trajectory_id']].append(plan)
        group = expected[plan['split'] + ':' + plan['kind']]
        group['planned_queries'] += 1
        group['planned_missing_actor_positions'] += plan['missing_actor_positions']
        group['planned_missing_stop_positions'] += sum(a['recapture_required'] and a['terminal_stop'] for a in plan['actions'])
    for plans in by_id.values():
        expected[plans[0]['split'] + ':' + plans[0]['kind']]['planned_trajectories'] += 1
    seen = set()
    for result in admitted_groups:
        identity = result['trajectory_id']
        _check(identity in by_id and identity not in seen, 'UNKNOWN_OR_DUPLICATE_ADMITTED_GROUP')
        seen.add(identity)
        plans = by_id[identity]
        _check(result['status'] == 'TRAJECTORY_RECAPTURE_ADMITTED' and result['physical_replay_exact'], 'GROUP_NOT_ADMITTED')
        _check(set(result['request_ids']) == {plan['request_id'] for plan in plans}, 'GROUP_COVERAGE_CHANGED')
        _check(result['split'] == plans[0]['split'] and result['kind'] == plans[0]['kind'], 'GROUP_SPLIT_CHANGED')
        _check(result['new_actor_positions'] == sum(plan['missing_actor_positions'] for plan in plans), 'GROUP_ACTOR_COVERAGE_CHANGED')
        _check(result['new_terminal_stop_positions'] == sum(a['recapture_required'] and a['terminal_stop'] for plan in plans for a in plan['actions']),
               'GROUP_STOP_COVERAGE_CHANGED')
        group = expected[result['split'] + ':' + result['kind']]
        group['admitted_trajectories'] += 1
        group['captured_queries'] += result['captured_queries']
        group['captured_missing_actor_positions'] += result['new_actor_positions']
        group['captured_missing_stop_positions'] += result['new_terminal_stop_positions']
    total = Counter()
    for group in expected.values():
        for name in ('admitted_trajectories', 'captured_queries', 'captured_missing_actor_positions', 'captured_missing_stop_positions'):
            group.setdefault(name, 0)
        group['remaining_missing_actor_positions'] = group['planned_missing_actor_positions'] - group['captured_missing_actor_positions']
        group['remaining_missing_stop_positions'] = group['planned_missing_stop_positions'] - group['captured_missing_stop_positions']
        total.update(group)
    return dict(status='COMPLETE' if seen == set(by_id) else 'INCOMPLETE',
        totals=dict(total), by_split_kind={key: dict(value) for key, value in sorted(expected.items())},
        missing_trajectory_ids=sorted(set(by_id) - seen), no_new_training_admission=True)
