"""Dependency-injected one-bundle orchestration. No file I/O or simulator imports.

The caller supplies frozen configurations and shared durable batch ledgers. All
physical evidence is emitted; only certification traces are retained in memory.
"""
import copy
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parent.parent
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
from core_bridge import FamilyFactory, Reject, BudgetExceeded


def run_bundle(config, backend, compiler, budget, freeze_ledger, emit):
    """Return a JSON-compatible bundle result; never auto-retry or reset budgets.

    config fields: bundle_id, configurations, context={house_id,asset_config},
    task_a, task_b. Coordinates/order are copied verbatim, never snapped/reordered.
    Integrity/persistence/programming errors propagate; only explicit Reject and
    BudgetExceeded become the declared physical/resource result categories.
    """
    required = {'bundle_id', 'configurations', 'context', 'task_a', 'task_b'}
    if not isinstance(config, dict) or not required <= set(config):
        raise ValueError('BUNDLE_CONFIGURATION_FIELDS')
    bundle = config['bundle_id']
    if not isinstance(bundle, str) or not bundle or not callable(emit):
        raise ValueError('BUNDLE_ID_OR_EMITTER')
    if (not isinstance(config['configurations'], list) or not config['configurations']
            or len(config['configurations']) > 64):
        raise ValueError('FROZEN_CONFIGURATION_COUNT')
    if budget.snapshot()['active'] is not None:
        raise ValueError('SHARED_BUDGET_ALREADY_ACTIVE')
    if bundle in freeze_ledger.records:
        raise ValueError('BUNDLE_ALREADY_ATTEMPTED_NO_AUTOMATIC_RETRY')
    if set(compiler.eligible) != {'anchor_A', 'anchor_B', 'terminal', 'irrelevant'}:
        raise ValueError('ROLE_ELIGIBILITY_KEYS')

    phase, phase_started = 'discovery', False
    candidate, certificate = None, None
    retained = []
    counts = {'confirmed_actions': 0, 'trace_returns': 0, 'complete_traces': 0,
              'discovery_trace_returns': 0, 'certification_trace_returns': 0,
              'complete_certification_traces': 0, 'collisions_in_returned_traces': 0,
              'executed_stops': 0}
    certification_indices = {}
    grid_keys = []
    factory = None

    def forward(kind, value):
        # Persist first. Failure stops the caller rather than claiming accepted
        # evidence that was not durably recorded. Raw trace objects are not edited.
        emit(kind, value)
        if kind == 'action_completed':
            counts['confirmed_actions'] += 1
        elif kind == 'trace':
            counts['trace_returns'] += 1
            counts[phase+'_trace_returns'] += 1
            counts['complete_traces'] += int(value.get('complete') is True)
            counts['collisions_in_returned_traces'] += value.get('collisions', 0)
            if phase == 'certification':
                seed = value['seed']
                index = certification_indices.get(seed, 0)
                if seed not in (1109, 2209, 3309) or index >= len(grid_keys):
                    raise ValueError('UNEXPECTED_CERTIFICATION_TRACE')
                history, continuation = grid_keys[index]
                expected = candidate['histories'][history] + candidate['continuations'][continuation]
                if value['actions'] != expected:
                    raise ValueError('CERTIFICATION_TRACE_ORDER_OR_ACTION_MISMATCH')
                certification_indices[seed] = index+1
                retained.append({'seed': seed, 'history': history,
                                 'continuation': continuation, 'trace': copy.deepcopy(value)})
                counts['complete_certification_traces'] += int(value.get('complete') is True)

    def finish_phase():
        nonlocal phase_started
        if phase_started:
            # BudgetLedger.finish_phase explicitly permits closing exhausted
            # phases. It preserves the original totals; no check_time reset.
            budget.finish_phase()
            phase_started = False

    def finish(status, reason=None, details=None):
        finish_phase()
        if factory is not None:
            counts['executed_stops'] = factory.runner.counts['executed_stops']
        result = {'bundle_id': bundle, 'status': status, 'reason': reason,
                  'details': details or {}, 'counters': dict(counts),
                  'candidate': copy.deepcopy(candidate), 'certificate': certificate,
                  'certification_traces': retained, 'budget_snapshot': budget.snapshot(),
                  'scientific_pass': False, 'training_admission': False,
                  'coordinate_transform': 'none_frozen_source_coordinates',
                  'automatic_replacement': False}
        emit('bundle_closed', {k: v for k, v in result.items()
                               if k not in ('candidate', 'certificate', 'certification_traces')})
        return result

    try:
        budget.start_bundle(bundle, 'discovery')
        phase_started = True
        empty = [name for name, ids in compiler.eligible.items() if not ids]
        if empty:
            freeze_ledger.start(bundle)
            freeze_ledger.reject_discovery(bundle, 'METADATA_INELIGIBLE:'+','.join(empty))
            return finish('metadata_ineligible', 'ELIGIBLE_SET_EMPTY', {'empty_roles': empty})
        factory = FamilyFactory(backend, compiler, budget, forward,
                                config['task_a'], config['task_b'], context=copy.deepcopy(config['context']))
        candidate = factory.discover(copy.deepcopy(config['configurations']), freeze_ledger, bundle)
        finish_phase()
        phase = 'certification'
        grid_keys = [(h, c) for h in candidate['histories'] for c in candidate['continuations']]
        budget.start_bundle(bundle, 'certification')
        phase_started = True
        certificate = factory.replay_seeds(candidate)
        if (certification_indices != {1109: 9, 2209: 9, 3309: 9}
                or counts['complete_certification_traces'] != 27
                or certificate['replays'] != 27 or certificate['evaluations'] != 54):
            raise ValueError('CERTIFICATION_EVIDENCE_COUNT_MISMATCH')
        budget.check_time()
        freeze_ledger.certify(bundle, True)
        return finish('certified')
    except BudgetExceeded as error:
        record = freeze_ledger.records.get(bundle)
        if record is not None and record['status'] == 'discovering':
            freeze_ledger.reject_discovery(bundle, str(error), resource_censored=True)
        elif record is not None and record['status'] == 'frozen':
            freeze_ledger.certify(bundle, False, 'RESOURCE_CENSORED:'+str(error))
        return finish('resource_censored', str(error), {'phase': phase})
    except Reject as error:
        if phase == 'discovery':
            record = freeze_ledger.records.get(bundle)
            if record is None:
                # Constructor Reject before discover() starts its ledger.
                freeze_ledger.start(bundle)
            freeze_ledger.reject_discovery(bundle, error.code)
            return finish('discovery_rejected', error.code, error.details)
        freeze_ledger.certify(bundle, False, error.code)
        return finish('certification_rejected', error.code, error.details)
    except BaseException:
        # No continued search after unknown/integrity/persistence errors. Preserve
        # failure evidence and close only our active phase if persistence permits.
        finish_phase()
        raise
