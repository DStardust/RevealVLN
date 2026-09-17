import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('language_ready_v2_prepare',HERE/'prepare.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)

class Tests(unittest.TestCase):
    def setUp(self):
        self.source=p.read(p.OLD/'CONFIG_DRAFT.json');self.old=p.read(p.OLD/'COHORT_V2.json')
    def test_all_candidates_exactly_unchanged(self):
        cfg,_=p.derive(self.source,self.old);self.assertEqual(cfg['candidates'],self.source['candidates'])
    def test_only_top_level_node_cohort_change(self):
        cfg,_=p.derive(self.source,self.old)
        self.assertEqual({k for k in cfg if cfg[k]!=self.source[k]},{'node','evaluation_cohort_manifest'})
    def test_exact_new_assignments(self):
        _,c=p.derive(self.source,self.old)
        self.assertEqual([m['batch_id'] for m in c['members']],['batch_03r1','batch_04r1','batch_05r1','batch_06r1'])
        self.assertEqual([i for m in c['members'] for i in m['source_indices']],list(range(12)))
        self.assertEqual([m['gpu_device'] for m in c['members']],[1,5,1,5])
    def test_no_mutation_old_source(self):
        source=copy.deepcopy(self.source);prior=copy.deepcopy(self.old);p.derive(self.source,self.old)
        self.assertEqual(source,self.source);self.assertEqual(prior,self.old)
    def test_runtime_presence_rejected(self):
        path=Mock();path.exists.return_value=True;entry=Mock();entry.name='PROCESS.json';path.iterdir.return_value=[entry]
        with self.assertRaises(AssertionError):p.preaction_check(path)
    def test_prior_cohort_only_provenance(self):
        cfg,c=p.derive(self.source,self.old)
        self.assertNotEqual(cfg['evaluation_cohort_manifest'],self.source['evaluation_cohort_manifest'])
        self.assertTrue(c['superseded_unstarted_cohort']['not_current_cohort'])
        self.assertFalse(c['runtime_allowed']);self.assertFalse(c['scientific_pass'])

if __name__=='__main__':unittest.main()
