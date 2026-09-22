"""Supplement-specific denominator, isolation and shared-device contracts."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from aggregate_counts import combine,target_met
from coexist import available,verify_prior_lease_pid

class Tests(unittest.TestCase):
    def test_201_meets_gap_not_200(self):
        self.assertFalse(target_met(599,200));self.assertTrue(target_met(599,201))
    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):combine([dict(family_id='P',house='H')],[dict(family_id='P',house='J')])
    def test_same_house_rejected(self):
        with self.assertRaisesRegex(ValueError,'HOUSE_OVERLAP'):combine([dict(family_id='P',house='H')],[dict(family_id='Q',house='H')])
    def test_disjoint_merge(self):
        self.assertEqual(len(combine([dict(family_id='P',house='H')],[dict(family_id='Q',house='J')])),2)
    def test_never_reassign_prior_active_device(self):
        ds=[dict(gpu=i) for i in range(8)]
        self.assertEqual([d['gpu'] for d in available(ds,{4,5,6,7},{0,1})],[2,3])
        self.assertEqual(len(available(ds,set(),{0,1})),6)
    def test_fit_selection(self):
        manifest=read(HERE/'HOUSE_MANIFEST.json')
        hs={x['house'] for x in manifest['houses']}
        self.assertEqual(len(hs),16);self.assertTrue(hs<=set(manifest['ordinary_fit_houses']))
        self.assertFalse(hs&set(manifest['excluded']))
        self.assertFalse(hs&set(manifest['ordinary_check_houses']))
        self.assertFalse(hs&set(manifest['prior_memory_excluded']))
    def test_frozen_collector_and_quality(self):
        for name in ('collect.py','quality.py','continuation_service.py'):
            self.assertEqual(sha(HERE/name),sha(PARENT/'event_data_scale20_v1'/name))
    def test_prior_data_seal(self):
        files=read(HERE/'HOUSE_MANIFEST.json')['prior_parent_files']
        self.assertEqual(len(files),599)
        for p,h in files.items():self.assertEqual(sha(LINE/p),h)
    def test_reject_unrecognized_lease(self):
        import os
        monitor=load('supplement_test_monitor',V16/'pipeline.py')
        with self.assertRaisesRegex(ValueError,'UNRECOGNIZED'):
            verify_prior_lease_pid(os.getpid(),dict(prior_pipeline='/not/our/task',prior_unit='not-our.service'),monitor)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(HERE/'CPU_TEST_RESULT.json',dict(tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),success=result.wasSuccessful(),
        new_physical_samples=0,new_training_updates=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)

