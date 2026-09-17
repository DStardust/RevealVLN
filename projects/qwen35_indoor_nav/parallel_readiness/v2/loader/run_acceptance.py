"""Writes CPU-only acceptance artifacts strictly beside this script."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import loader as L
import test_loader


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    start = time.monotonic()
    before = L.verify_protected()
    write('PROTECTED_BEFORE.json', before)
    write('CODE_LOCK.json', {p.name: L.sha(p.read_bytes()) for p in OUT.iterdir() if p.suffix == '.py' or p.name == 'SPEC_ZH.md'})
    result, log = test_loader.run_tests()
    (OUT / 'TEST_LOG.txt').write_text(log)
    stats = {}
    if result.wasSuccessful():
        data = test_loader.RealLoaderTests.data
        lengths = {t + '/' + h: len(data.prefixes[(t, h)]) for t, h in data.prefix_groups()}
        queries = {s['query_hash']: data.query_input(s['sample_id']) for s in data.supervision}
        for path in sorted((L.DATA / 'content').glob('*.rgb.npy')):
            L.rgb_from_npy(path.read_bytes(), 'sha256:' + path.name.split('.')[0])
        stats = {'families': 1, 'houses': 1, 'split': 'interface_only', 'independent_confirmation_samples': 0,
                 'prefix_groups': 6, 'prefix_decision_counts': lengths, 'prefix_total_decisions': sum(lengths.values()),
                 'rgb_shape': [224, 224, 3], 'rgb_dtype': 'uint8', 'all_rgb_arrays_verified': len(list((L.DATA / 'content').glob('*.rgb.npy'))),
                 'recent_rgb_max': 2, 'recent_action_max': 8, 'cross_cells': len(data.supervision),
                 'outcomes': dict(collections.Counter(s['outcome'] for s in data.supervision)),
                 'distinct_semantic_queries': len(queries), 'query_sequence_lengths': sorted(len(q) for q in queries.values()),
                 'query_flat_scalar_lengths': sorted(sum(len(row) for row in q) for q in queries.values()),
                 'unique_action_ce_owners': len(data.dedup['owners']),
                 'all_action_targets_including_masked': sum(len(s['action_targets']) for s in data.supervision),
                 'action_stream_decisions_including_warmup': sum(len(data.trace(s)['observations']) for s in data.supervision),
                 'query_encoding': 'ordered_typed_integer_tuples_no_neural_encoder', 'category_vocab': L.CATEGORY, 'room_vocab': L.ROOM}
        write('REAL_DATA_SHAPES.json', stats)
    after = L.verify_protected()
    write('PROTECTED_AFTER.json', after)
    outcome = {'status': 'CPU_LOADER_ACCEPTANCE_PASS' if result.wasSuccessful() and before == after else 'FAIL',
               'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'protected_unchanged': before == after, 'elapsed_seconds': time.monotonic()-start,
               'scientific_pass': False, 'gpu_operations': 0, 'model_calls': 0, 'simulator_replays': 0,
               'training_updates': 0, 'mechanism_training_authorized': False,
               'limitations': ['single_exposed_interface_family', 'numerically_normalized_real_history_not_raw_exact_join',
                               'no_neural_query_encoder', 'no_Qwen_full_history_gradient_acceptance',
                               'no_actual_model_cache_isolation_test', 'no_navigation_or_generalization_benefit']}
    write('result.json', outcome)
    print(json.dumps(outcome, ensure_ascii=False))
    print(log if not result.wasSuccessful() else json.dumps(stats, ensure_ascii=False))
    return 0 if outcome['status'] == 'CPU_LOADER_ACCEPTANCE_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
