"""One explicitly authorized existing-family backend integration smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core_bridge import Compiler, FamilyFactory, BudgetLedger, digest
from runtime_journal import Journal
from prepare import DATA, sha


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def execute(out):
    cfg = json.loads((out/'EXECUTION_CONFIG.json').read_text())
    assert cfg['runtime_allowed'] is True and cfg['phase'] == 'existing_family_backend_smoke'
    auth = json.loads((out/'EXECUTION_AUTH.json').read_text())
    assert auth['approved'] and auth['config_sha256'] == sha(out/'EXECUTION_CONFIG.json')
    for relative, expected in auth['code_sha256'].items():
        assert sha(HERE/relative) == expected, relative
    for path, expected in cfg['assets'].items():
        assert sha(Path(path)) == expected, path
    from habitat_backend import HabitatBackend
    from guard import ContentStore
    from exporter import export_family
    from loader import FamilyLoader
    store = ContentStore(out/'content', cfg['disk_cap_bytes'])
    backend, result = None, None
    started = time.monotonic()
    traces = []
    trace_dir = out/'traces'
    trace_dir.mkdir()
    try:
        with Journal(out/'journal', cfg) as journal:
            budget = BudgetLedger(cfg['budget'], persist=lambda state: journal.append('budget', state))
            budget.start_bundle('existing_family_smoke', 'certification')
            backend = HabitatBackend(cfg['scene_glb'], cfg['gpu_device'], cfg['roles'], store, cfg)
            assert backend.eligible == cfg['expected_eligible'], 'ELIGIBLE_MAPPING_DIFFERENCE'
            save(out/'SEMANTIC_INVENTORY.json', {'eligible': backend.eligible, 'objects': backend.objects})
            compiler = Compiler(backend.compiler_roles, cfg['tasks'], backend.eligible)
            def emit(kind, value):
                if kind == 'trace':
                    index = len(traces)
                    traces.append(value)
                    save(trace_dir/('%03d.json' % index), value)
                    journal.append('trace_saved', {'index': index, 'sha256': sha(trace_dir/('%03d.json' % index))})
                else:
                    journal.append(kind, value)
            context = {'house_id': cfg['house_id'], 'asset_config': cfg['assets']}
            factory = FamilyFactory(backend, compiler, budget, emit, 'task_A', 'task_B', context=context)
            candidate = cfg['candidate']
            candidate.update(context=context, context_key=factory.context_key)
            candidate['candidate_hash'] = digest(candidate)
            matrix = factory.validate_matrix(candidate, cfg['seed'])
            save(out/'MATRIX.json', matrix)
            assert len(traces) == 9
            hmap = {'H_A': 'H_T', 'H_B': 'H_K', 'H_A_I': 'H_T_I'}
            cmap = {'C0': 'C0', 'C_A': 'C_T', 'C_B': 'C_K'}
            keys = [(h, c) for h in candidate['histories'] for c in candidate['continuations']]
            differences = []
            for key, trace in zip(keys, traces):
                h, c = key
                path = DATA/'physical_traces'/f"1109_{hmap[h]}_{cmap[c]}.json"
                old = json.loads(path.read_text())
                assert old['actions'] == trace['actions'] and len(old['observations']) == len(trace['observations'])
                fields = ('rgb_hash', 'semantic_hash', 'pose')
                diff = sum(any(old_o[k] != new_o[k] for k in fields)
                           for old_o, new_o in zip(old['observations'], trace['observations']))
                differences.append({'history': h, 'continuation': c, 'observation_differences': diff,
                                    'reference_trace_sha256': sha(path)})
            save(out/'LEGACY_REPLAY_DIFFERENTIAL.json', differences)
            assert not any(r['observation_differences'] for r in differences), 'PIXEL_OR_POSE_DIFFERENCE'
            export_family(out/'export_v4', compiler, candidate, dict(zip(keys, traces)), out/'content',
                          family_id='F17_TK_B_backend_smoke_reexport_v4', split='interface_only',
                          provenance={'kind': 'existing_family_new_backend_replay_not_new_family',
                                      'runtime_source': str(out), 'original_manifest_sha256': cfg['source_manifest_sha256']})
            dataset = FamilyLoader(out/'export_v4', compiler)
            save(out/'SUPERVISION_READBACK.json', dataset.validate_supervision_contract())
            prefix_count, action_count, ce_count = 0, 0, 0
            for prefix_id in dataset.prefix_index:
                for record in dataset.prefix_records(prefix_id):
                    dataset.policy_payload(record)
                    prefix_count += 1
            for cell in dataset.cells:
                supervision = dataset.supervision(cell['cell_id'])
                ce_count += sum(supervision['action_loss_mask'])
                for record in dataset.action_stream(cell['cell_id']):
                    dataset.policy_payload(record)
                    action_count += 1
            assert prefix_count == 1242 and ce_count == len(dataset.owners['owners'])
            save(out/'LOADER_READBACK.json', {'prefix_decisions': prefix_count, 'action_stream_decisions': action_count,
                 'ce_owners': ce_count, 'cells': len(dataset.cells), 'all_policy_RGB_verified': True})
            budget.finish_phase()
            save(out/'BUDGET_FINAL.json', budget.snapshot())
            result = {'decision': 'REAL_BACKEND_V4_INTEGRATION_SMOKE_PASS', 'runtime_pass': True,
                      'real_replays': len(traces), 'task_evaluations': len(matrix['rows']),
                      'pass_cells': sum(r['outcome'] == 'pass' for r in matrix['rows']),
                      'confirmed_action_returns': factory.runner.counts['confirmed_action_returns'],
                      'stops': factory.runner.counts['executed_stops'], 'export_loader_pass': True,
                      'legacy_pixel_pose_exact': True, 'new_physical_families': 0, 'scientific_pass': False}
    except BaseException as error:
        result = {'decision': 'INTEGRATION_SMOKE_FAIL', 'runtime_pass': False,
                  'error': repr(error), 'traceback': traceback.format_exc(),
                  'completed_traces': len(traces), 'new_physical_families': 0, 'scientific_pass': False}
    finally:
        if backend is not None:
            backend.close()
        store.close()
        result['wall_seconds'] = time.monotonic()-started
        save(out/'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['runtime_pass'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(HERE) and out != HERE
    raise SystemExit(execute(out))
