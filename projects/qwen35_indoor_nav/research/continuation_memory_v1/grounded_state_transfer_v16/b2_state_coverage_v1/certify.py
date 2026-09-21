"""Certify terminal-absent counterparts; the SEE2 checker is unchanged."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continuation_service import exact_pose
from evaluator_v16 import legacy, evaluate, state_sequence, query_context

HISTORIES = ('seen', 'seen_sham', 'missing', 'missing_sham')
QUERIES = ('direct_stop', 'acquire_anchor', 'return_terminal')


def certify(compiler, histories, suffixes, traces):
    if set(histories) != set(HISTORIES) or set(suffixes) != set(QUERIES):
        raise ValueError('INCOMPLETE_FAMILY')
    if len({tuple(v) for v in histories.values()}) != 4:
        raise ValueError('DUPLICATE_HISTORIES')
    if len({len(v) for v in histories.values()}) != 1 or len({tuple(sorted(Counter(v).items())) for v in histories.values()}) != 1:
        raise ValueError('LENGTH_OR_ACTION_COUNT_SHORTCUT')
    if suffixes['direct_stop'] != ['S'] or any(suffixes[q][0] == 'S' for q in QUERIES[1:]):
        raise ValueError('MISSING_REAL_CONTINUATION')
    if set(traces) != {h+'__'+q for h in HISTORIES for q in QUERIES}:
        raise ValueError('MISSING_PHYSICAL_CROSS_CELL')
    cutoff = len(histories['seen'])
    first = traces['seen__direct_stop']
    labels = {}; ages = {}
    for h in HISTORIES:
        reference = traces[h+'__direct_stop']
        if reference['observations'][0] != first['observations'][0]:
            raise ValueError('INITIAL_INPUT_STATE_MISMATCH')
        if histories[h][-8:] != histories['seen'][-8:]:
            raise ValueError('RECENT_ACTION_MISMATCH')
        for t in (cutoff-1, cutoff):
            a, b = reference['observations'][t], first['observations'][t]
            if a['rgb_hash'] != b['rgb_hash'] or a['semantic_hash'] != b['semantic_hash'] or not exact_pose(a['pose'], b['pose']):
                raise ValueError('CURRENT_RAW_OR_PHYSICAL_STATE_MISMATCH')
        events = compiler.atoms(reference['observations'])
        indices = [t for t,e in enumerate(events) if e['anchor']]
        seen = h.startswith('seen')
        if bool(indices) != seen or any(e['anchor'] for e in events[-8:]):
            raise ValueError('HISTORY_EVENT_NOT_ISOLATED_FROM_RECENT_WINDOW')
        if events[-1]['terminal']:
            raise ValueError('TERMINAL_MUST_BE_ABSENT_AT_TAKEOVER')
        ages[h] = cutoff-max(indices) if indices else None
        for q in QUERIES:
            trace = traces[h+'__'+q]
            if trace.get('interior_state_assignments') != 0 or not legacy.complete(trace):
                raise ValueError('ILLEGAL_OR_INCOMPLETE_TRACE')
            if trace['actions'] != histories[h]+suffixes[q] or len(trace['actions']) > 500:
                raise ValueError('ACTION_OR_BUDGET_MISMATCH')
            if trace['observations'][:cutoff+1] != reference['observations']:
                raise ValueError('SAME_HISTORY_REPLAY_CHANGED')
            if trace['observations'][cutoff:] != traces['seen__'+q]['observations'][cutoff:]:
                raise ValueError('SHARED_CONTINUATION_NOT_IDENTICAL')
            for task in ('task_A', 'task_T'):
                result = evaluate(compiler, trace, task, cutoff)
                expected = q != 'direct_stop' and (task == 'task_T' or seen or q == 'acquire_anchor')
                if result['safe_v16_label'] != ('PASS' if expected else 'FAIL'):
                    raise ValueError('OBSERVED_CROSS_LABEL_PATTERN_MISSING')
                labels[h+'__'+q+'__'+task] = dict(label=result['safe_v16_label'],
                    state=state_sequence(compiler,trace['observations'],task)[cutoff],
                    query=query_context(compiler,trace,task,cutoff))
    return dict(status='CERTIFIED_TERMINAL_ABSENT_COUNTERPART', labels=labels, anchor_age=ages,
                physical_executions=12, distinct_histories=4, crossed_labels=24,
                controlled_history_action_pairs=2, same_state_sham_pairs=2, task_T_control_pairs=2,
                task_type='observable_see2_then_stop.v4', terminal_present_at_takeover=False,
                natural_navigation_generalization=False)
