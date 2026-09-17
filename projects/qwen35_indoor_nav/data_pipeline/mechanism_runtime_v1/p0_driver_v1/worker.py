"""Fixed five-bundle construction diagnostic, never a training launcher."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
sys.path[:0] = [str(HERE), str(RUNTIME)]
from prepare import sha
from core_bridge import Compiler, BudgetLedger, FreezeLedger, BudgetExceeded
from runtime_journal import Journal
from family_job import run_bundle


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def execute(out):
    cfg = json.loads((out/'EXECUTION_CONFIG.json').read_text())
    auth = json.loads((out/'EXECUTION_AUTH.json').read_text())
    assert auth['approved'] and auth['P0_authorized'] and cfg['runtime_allowed']
    assert cfg['phase'] == 'five_fixed_bundle_constructability_P0'
    assert sha(out/'EXECUTION_CONFIG.json') == auth['config_sha256']
    for name, value in auth['code_sha256'].items():
        assert sha(RUNTIME/name) == value, name
    smoke = RUNTIME/'smoke_v1/result.json'
    assert sha(smoke) == cfg['smoke_result_sha256'] and json.loads(smoke.read_text())['runtime_pass']
    from habitat_backend import HabitatBackend
    from guard import ContentStore
    from exporter import export_family
    from loader import FamilyLoader
    started = time.monotonic()
    results, global_error, budget, ledger, current_bundle = [], None, None, None, None
    (out/'bundles').mkdir()
    backend = None
    try:
        # Leave 4 GiB for trace/journal/export metadata and supervisory closure.
        with ContentStore(out/'content', 12*1024**3) as store, Journal(out/'journal', cfg) as journal:
            budget = BudgetLedger(cfg['budget'], persist=lambda s: journal.append('budget', s))
            ledger = FreezeLedger(persist=lambda s: journal.append('freeze_ledger', s))
            for row in cfg['candidates']:
                budget.check_time()
                bundle_started = time.monotonic()
                current_bundle = row['candidate_id']
                folder = out/'bundles'/row['candidate_id']
                folder.mkdir()
                (folder/'traces').mkdir()
                journal.append('bundle_started', {'candidate_id': row['candidate_id'], 'house_id': row['house_id']})
                for name, value in row['assets'].items():
                    assert sha(Path(name)) == value, 'ASSET_CHANGED:'+name
                permit = {**cfg, 'scene_glb': row['scene_glb']}
                backend = HabitatBackend(row['scene_glb'], cfg['gpu_device'], cfg['roles'], store, permit)
                save(folder/'SEMANTIC_INVENTORY.json', {'objects': backend.objects, 'eligible': backend.eligible})
                compiler = Compiler(backend.compiler_roles, cfg['tasks'], backend.eligible)
                index = 0
                def emit(kind, value):
                    nonlocal index
                    if kind == 'trace':
                        path = folder/'traces'/('%06d.json' % index)
                        save(path, value)
                        journal.append('trace_saved', {'bundle': row['candidate_id'], 'index': index,
                                                      'sha256': sha(path), 'complete': value['complete']})
                        index += 1
                    else:
                        journal.append(kind, {'bundle': row['candidate_id'], 'value': value})
                job = {'bundle_id': row['candidate_id'], 'configurations': row['configurations'],
                       'context': {'house_id': row['house_id'], 'asset_config': row['assets']},
                       'task_a': 'task_A', 'task_b': 'task_B'}
                result = run_bundle(job, backend, compiler, budget, ledger, emit)
                certification_traces = result.pop('certification_traces', [])
                result.update(candidate_id=row['candidate_id'], house_id=row['house_id'],
                              saved_trace_count=index, export_loader_pass=None,
                              split='candidate_fit_pool', training_admission=False, scientific_pass=False)
                save(folder/'BUNDLE_PHYSICAL_RESULT.json', result)
                results.append(result)
                if result['status'] == 'certified':
                    grids = {seed: {} for seed in (1109, 2209, 3309)}
                    for item in certification_traces:
                        grids[item['seed']][item['history'], item['continuation']] = item['trace']
                    assert all(len(g) == 9 for g in grids.values())
                    export_family(folder/'export_v4', compiler, result['candidate'], grids[1109], out/'content',
                                  family_id=row['candidate_id']+'_P0_v4', split='candidate_fit_pool',
                                  provenance={'source_route_sha256': row['source_physical_route_sha256'],
                                              'kind': 'real_three_seed_candidate_pending_diversity_and_shortcut_admission'},
                                  repeated_traces=[grids[2209], grids[3309]])
                    loader = FamilyLoader(folder/'export_v4', compiler)
                    save(folder/'SUPERVISION_READBACK.json', loader.validate_supervision_contract())
                    payloads = 0
                    for prefix in loader.prefix_index:
                        for record in loader.prefix_records(prefix):
                            loader.policy_payload(record)
                            payloads += 1
                    for cell in loader.cells:
                        for record in loader.action_stream(cell['cell_id']):
                            loader.policy_payload(record)
                            payloads += 1
                    result.update(export_loader_pass=True, policy_payloads_verified=payloads)
                result['wall_seconds'] = time.monotonic()-bundle_started
                save(folder/'result.json', result)
                journal.append('bundle_closed', {'candidate_id': row['candidate_id'], 'status': result['status']})
                backend.close()
                backend = None
                current_bundle = None
    except BaseException as error:
        global_error = {'error': repr(error), 'traceback': traceback.format_exc(),
                        'resource_censored': isinstance(error, BudgetExceeded) or getattr(error, 'resource_censored', False)}
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            if budget is not None:
                save(out/'BUDGET_FINAL.json', budget.snapshot())
            if ledger is not None:
                save(out/'FREEZE_LEDGER.json', ledger.records)
    result = {'decision': 'P0_CONSTRUCTION_DIAGNOSTIC_CLOSED' if global_error is None else 'P0_STOPPED_WITH_ERROR',
              'planned': len(cfg['candidates']), 'closed_bundles': len(results),
              'bundles': results, 'error': global_error,
              'interrupted_bundle': current_bundle,
              'physical_certified_candidates': sum(r['status'] == 'certified' for r in results),
              'new_geometric_families': None, 'diversity_audit_required': True,
              'total_reserved_actions': budget.snapshot()['total_reserved_actions'] if budget else None,
              'wall_seconds': time.monotonic()-started, 'training_admission': False,
              'scientific_pass': False, 'P1_authorized': False}
    save(out/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'bundles'}, ensure_ascii=False), flush=True)
    return int(global_error is not None)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    out = p.parse_args().output.resolve()
    assert out.is_relative_to(RUNTIME) and out != RUNTIME
    raise SystemExit(execute(out))
