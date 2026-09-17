import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('planner', HERE / 'dry_run.py')
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
SPLIT = json.loads((planner.LINE / PROTOCOL['source']['split']).read_text())
ROWS = [json.loads(line) for line in (planner.LINE / PROTOCOL['source']['manifest']).read_text().splitlines()]


class PlannerTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(planner.build(ROWS, SPLIT, PROTOCOL), planner.build(list(reversed(ROWS)), SPLIT, PROTOCOL))

    def test_counts_and_round_robin(self):
        inv, candidates = planner.build(ROWS, SPLIT, PROTOCOL)
        self.assertEqual((len(inv), len(candidates)), (61, 20))
        self.assertEqual([x['house_id'] for x in candidates[:5]], SPLIT['fit_pilot'])
        self.assertEqual(sum(x['source_physical_routes'] for x in inv), 3603)
        self.assertEqual(sum(x['source_records'] for x in inv), 10819)

    def test_no_reserved_candidates(self):
        _, candidates = planner.build(ROWS, SPLIT, PROTOCOL)
        self.assertFalse({x['house_id'] for x in candidates} & set(SPLIT['reserved_unassigned']))

    def test_no_instructions_or_coordinates(self):
        _, candidates = planner.build(ROWS, SPLIT, PROTOCOL)
        self.assertNotIn('"instruction":', json.dumps(candidates))
        self.assertTrue(all(x['runtime_u_positions'] is None for x in candidates))

    def test_executable_rejected(self):
        p = copy.deepcopy(PROTOCOL); p['executable'] = True
        with self.assertRaises(ValueError): planner.build(ROWS, SPLIT, p)

    def test_split_overlap_rejected(self):
        s = copy.deepcopy(SPLIT); s['reserved_unassigned'][0] = s['fit_pilot'][0]
        with self.assertRaises(ValueError): planner.build(ROWS, s, PROTOCOL)

    def test_eval_source_rejected(self):
        rows = [dict(ROWS[0], source_split='val_unseen')]
        with self.assertRaises(ValueError): planner.build(rows, SPLIT, PROTOCOL)

    def test_wrong_hash_rejected(self):
        rows = [dict(ROWS[0], source_sha256='0' * 64)]
        with self.assertRaises(ValueError): planner.build(rows, SPLIT, PROTOCOL)

    def test_no_claim_of_generated_geometry(self):
        _, candidates = planner.build(ROWS, SPLIT, PROTOCOL)
        self.assertTrue(all(x['new_geometry'] == 'unknown' and not x['certified'] for x in candidates))
        self.assertEqual(len({(x['house_id'], x['source_physical_route_sha256']) for x in candidates}), 20)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PlannerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    planner.dump(HERE / 'TEST_RESULTS.json', {'tests': result.testsRun, 'failures': len(result.failures),
                  'errors': len(result.errors), 'passed': result.wasSuccessful(),
                  'runtime_or_scientific_validation': False})
    raise SystemExit(0 if result.wasSuccessful() else 1)
