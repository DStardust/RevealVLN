"""CPU wording and prospective cohort invariants; not a runtime cohort PASS."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('language_ready_prepare',HERE/'prepare.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)

class Tests(unittest.TestCase):
    def test_all_twelve_structural_invariants(self):
        old=p.read(p.SOURCE/'CONFIG_DRAFT.json');new,changes=p.revise(old,p.load_language())
        self.assertEqual(len(changes),12)
        for before,after in zip(old['candidates'],new['candidates']):self.assertEqual(p.invariant(before),p.invariant(after))
    def test_action_change_detected(self):
        old=p.read(p.SOURCE/'CONFIG_DRAFT.json')['candidates'][0];new=copy.deepcopy(old);new['components']['a'].append('F')
        self.assertNotEqual(p.invariant(old),p.invariant(new))
    def test_deterministic_idempotent_wording(self):
        old=p.read(p.SOURCE/'CONFIG_DRAFT.json');first,_=p.revise(old,p.load_language());second,changes=p.revise(first,p.load_language())
        self.assertTrue(all(not r['changes'] for r in changes))
        self.assertEqual([r['tasks'] for r in first['candidates']],[r['tasks'] for r in second['candidates']])
    def test_four_batch_members_exact_once(self):
        cfg,_=p.revise(p.read(p.SOURCE/'CONFIG_DRAFT.json'),p.load_language());cohort=p.cohort(cfg,'0'*64)
        self.assertEqual(cohort['schema_version'],'q35n.production_cohort.v2')
        self.assertEqual([m['gpu_device'] for m in cohort['members']],[1,5,1,5])
        self.assertEqual([i for m in cohort['members'] for i in m['source_indices']],list(range(12)))
        self.assertEqual(len(set(cohort['declared_run_roots'])),4)
    def test_same_hub_in_one_batch_is_rejected(self):
        cfg=p.read(p.SOURCE/'CONFIG_DRAFT.json');cfg['candidates'][1]['house_id']=cfg['candidates'][0]['house_id']
        cfg['candidates'][1]['configuration']['u_position']=cfg['candidates'][0]['configuration']['u_position']
        with self.assertRaises(AssertionError):p.cohort(cfg,'0'*64)

if __name__=='__main__':unittest.main()
