"""Post-run diagnostic accounting; does not convert failed probes to training data."""
import collections
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent
sys.path.insert(0, str(RUNTIME))
from prepare import sha, sealed, LINE
from core_bridge import digest
from runtime_journal import Journal


def save(name, value):
    with (HERE/name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    out = HERE/'run'
    supervisor = json.loads((out/'SUPERVISOR_RESULT.json').read_text())
    cfg = json.loads((out/'EXECUTION_CONFIG.json').read_text())
    auth = json.loads((out/'EXECUTION_AUTH.json').read_text())
    assert supervisor['cleanup_complete'], 'CLEANUP_MUST_BE_RESOLVED_FIRST'
    assert sha(out/'EXECUTION_CONFIG.json') == auth['config_sha256']
    assert all(sha(RUNTIME/k) == v for k, v in auth['code_sha256'].items())
    with Journal(out/'journal', cfg, resume=True):
        pass
    expected = json.loads((HERE/'PROTECTED_BEFORE.json').read_text())
    for relative, expected_sha in expected.items():
        root = LINE/relative
        assert sha(root/'SHA256SUMS') == expected_sha
        sealed(root)
    rows = {r['candidate_id']: {'candidate_id': r['candidate_id'], 'house_id': r['house_id'],
             'planned_configurations': len(r['configurations']), 'started': False,
             'status': 'NOT_STARTED', 'physical_actions_confirmed': 0, 'collision_returns': 0,
             'saved_traces': 0, 'complete_probe_traces': 0, 'unknown_evidence_observations': 0,
             'completed_configuration_rejections': 0, 'configuration_reasons': collections.Counter(),
             'route_reasons': collections.Counter(), 'reject_details': [],
             'observed_role_frame_counts': collections.Counter(),
             'preflight_frozen': False, 'certified': False} for r in cfg['candidates']}
    last_budget = None
    trace_refs = []
    with (out/'journal/events.jsonl').open() as f:
        for line in f:
            event = json.loads(line)
            kind, payload = event['kind'], event['payload']
            if kind == 'budget':
                last_budget = payload
            elif kind == 'bundle_started':
                rows[payload['candidate_id']]['started'] = True
                rows[payload['candidate_id']]['status'] = 'INTERRUPTED_WITHOUT_FINAL_RESULT'
            elif kind == 'trace_saved':
                trace_refs.append(payload)
            elif kind == 'action_completed':
                row = rows[payload['bundle']]
                row['physical_actions_confirmed'] += 1
                row['collision_returns'] += int(payload['value']['collision'])
            elif kind == 'discovery_rejected':
                row, value = rows[payload['bundle']], payload['value']
                row['completed_configuration_rejections'] += 1
                row['configuration_reasons'][value['reason']] += 1
                row['reject_details'].append(value)
            elif kind == 'route_rejected':
                rows[payload['bundle']]['route_reasons'][payload['value']['reason']] += 1
            elif kind == 'freeze_ledger':
                for bundle, record in payload.items():
                    rows[bundle]['preflight_frozen'] = 'candidate' in record
                    rows[bundle]['certified'] = record['status'] == 'certified'
    rgb, semantic = set(), set()
    for record in trace_refs:
        row = rows[record['bundle']]
        path = out/'bundles'/record['bundle']/'traces'/('%06d.json' % record['index'])
        assert sha(path) == record['sha256']
        trace = json.loads(path.read_text())
        assert trace['trace_hash'] == digest({k: v for k, v in trace.items() if k != 'trace_hash'})
        row['saved_traces'] += 1
        row['complete_probe_traces'] += int(trace['complete'])
        for observation in trace['observations']:
            row['unknown_evidence_observations'] += int(not observation['evidence_complete'])
            rgb.add(observation['rgb_hash'])
            semantic.add(observation['semantic_hash'])
            for role, ids in observation['see2'].items():
                if ids:
                    row['observed_role_frame_counts'][role] += 1
    for key, row in rows.items():
        folder = out/'bundles'/key
        result_path = folder/'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            row.update(status=result['status'], reason=result['reason'], details=result['details'],
                       wall_seconds=result['wall_seconds'], export_loader_pass=result['export_loader_pass'])
            assert row['physical_actions_confirmed'] == result['counters']['confirmed_actions']
            assert row['saved_traces'] == result['counters']['trace_returns']
        inventory_path = folder/'SEMANTIC_INVENTORY.json'
        if inventory_path.exists():
            inv = json.loads(inventory_path.read_text())
            row['eligible_instance_counts'] = {k: len(v) for k, v in inv['eligible'].items()}
        row['unexhausted_or_not_started_configurations'] = row['planned_configurations']-row['completed_configuration_rejections']
    counts = collections.Counter(r['status'] for r in rows.values())
    certified = sum(r['certified'] for r in rows.values())
    started = sum(r['started'] for r in rows.values())
    # A nonzero family set requires a dedicated physical geometry audit; never
    # assume each different candidate ID means a new independent family.
    final = {'node': 'Q35N_MECHANISM_FIVE_BUNDLE_P0_V1',
             'decision': 'P0_YIELD_DIAGNOSTIC_CLOSED' if started == 5 and supervisor['returncode'] == 0 else 'P0_INTERRUPTED_DIAGNOSTIC',
             'planned_bundles': 5, 'started_bundles': started, 'status_counts': dict(counts),
             'preflight_frozen': sum(r['preflight_frozen'] for r in rows.values()),
             'certified_candidates': certified, 'new_geometric_families': 0 if certified == 0 else None,
             'exact_duplicates': 0 if certified == 0 else None, 'geometry_clusters': 0 if certified == 0 else None,
             'certified_per_planned': certified/5, 'certified_per_started': certified/started if started else None,
             'confirmed_physical_actions': sum(r['physical_actions_confirmed'] for r in rows.values()),
             'reserved_physical_actions': last_budget['total_reserved_actions'] if last_budget else None,
             'collision_returns': sum(r['collision_returns'] for r in rows.values()),
             'saved_probe_traces': sum(r['saved_traces'] for r in rows.values()),
             'unique_RGB_referenced_by_saved_traces': len(rgb),
             'unique_semantic_referenced_by_saved_traces': len(semantic),
             'trace_file_and_internal_hash_verification': True, 'journal_chain_pass': True,
             'protected_hashes_pass': True, 'gpu_cleanup_complete': supervisor['cleanup_complete'],
             'supervisor': supervisor, 'rows': list(rows.values()),
             'training_admission': False, 'scientific_pass': False, 'P1_authorized': False,
             'SFT_results_read': False, 'SFT_modified': False,
             'source_scope': 'five_existing_FIT_houses_only_not_unseen_generalization',
             'completed_probe_is_not_certified_family': True}
    save('result.json', final)
    save('SCENE_EXPOSURE.json', {'house_ids': [r['house_id'] for r in cfg['candidates']],
         'asset_hashes_reverified': True, 'all_existing_FIT': True,
         'reserved_56_house_assets_read': False, 'new_rendered_house_ids': [r['house_id'] for r in rows.values() if r['started']],
         'independent_test_generalization_claim': False})
    print(json.dumps({k: v for k, v in final.items() if k not in ('rows', 'supervisor')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
