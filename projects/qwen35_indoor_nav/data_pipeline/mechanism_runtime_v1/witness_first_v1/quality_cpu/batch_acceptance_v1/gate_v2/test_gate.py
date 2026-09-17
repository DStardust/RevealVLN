import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gate_test_v2',HERE/'gate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def batch(n,count=3):
    return {'run_root':'batch'+str(n),'bindings_verified':True,'all_frozen_candidates_attempted_and_terminal':True,
        'attempts':[{'house_id':'house'+str(n),'hub_position':[i*2.,0.,0.],'quality_pass':True,
                     'source_and_phase_binding_verified':True,'control_type':'completed_subgoal_revisit_placement_not_event_free_detour'} for i in range(count)]}

class Tests(unittest.TestCase):
    def test_three_batches_allowed(self):
        self.assertTrue(m.evaluate([batch(0),batch(1),batch(2)])['preliminary_cross_house_production_pass'])
    def test_two_batches_with_three_houses_allowed(self):
        a,b=batch(0),batch(1);b['attempts'][2]['house_id']='house2'
        self.assertTrue(m.evaluate([a,b])['preliminary_cross_house_production_pass'])
    def test_single_batch_rejected(self):self.assertFalse(m.evaluate([batch(0)])['preliminary_cross_house_production_pass'])
    def test_all_included_failures_retained(self):
        rows=[batch(0),batch(1),batch(2)];rows[1]['bindings_verified']=False
        self.assertFalse(m.evaluate(rows)['preliminary_cross_house_production_pass'])
    def test_recovery_does_not_fill_six_hubs(self):
        recovered=batch(2)['attempts'];rows=[batch(0),batch(1)]
        for r in rows:r['attempts'][2]['quality_pass']=False
        self.assertFalse(m.evaluate(rows,recovered)['preliminary_cross_house_production_pass'])
    def test_same_hub_new_role_or_seed_not_new(self):
        rows=[batch(0),batch(1),batch(2)];rows[0]['attempts'][1]['hub_position']=[0.,0.,0.]
        self.assertFalse(m.evaluate(rows)['preliminary_cross_house_production_pass'])
    def test_missing_attempt_denominator_rejected(self):
        rows=[batch(0),batch(1),batch(2)];rows[0]['all_frozen_candidates_attempted_and_terminal']=False
        self.assertFalse(m.evaluate(rows)['preliminary_cross_house_production_pass'])
    def test_cohort_must_preexist_runtime_locks(self):
        with tempfile.TemporaryDirectory(prefix='cpu_',dir=HERE) as temp:
            root=Path(temp);runs=[root/'a',root/'b']
            for run in runs:run.mkdir()
            manifest=root/'COHORT.json'
            manifest.write_text(json.dumps({'schema_version':'q35n.production_cohort.v2','criteria_version':'AT_LEAST_TWO_BATCHES_V2',
                'cohort_id':'cpu_only','mechanism_version':'cpu','transport_version':'cpu','declared_run_roots':list(map(str,runs))}))
            for run in runs:
                (run/'INPUT_LOCK.json').write_text(json.dumps({str(manifest):hashlib.sha256(manifest.read_bytes()).hexdigest()}))
                (run/'PROCESS.json').write_text(json.dumps({'started_unix':time.time()+1}))
            self.assertEqual(len(m.validate_cohort(manifest,runs)[1]),2)
            (runs[1]/'INPUT_LOCK.json').write_text('{}')
            with self.assertRaises(ValueError):m.validate_cohort(manifest,runs)

if __name__=='__main__':unittest.main()
