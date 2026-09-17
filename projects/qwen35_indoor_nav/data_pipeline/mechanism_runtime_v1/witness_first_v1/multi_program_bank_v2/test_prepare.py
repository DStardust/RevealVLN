import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('bank_v2_prepare_test',HERE/'prepare.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
class Tests(unittest.TestCase):
    def test_exact_reverse_transport(self):
        old=(p.OLD/'prepare.py').read_text();new=p.adapt(old)
        for a,b in reversed(p.patches()):new=new.replace(b,a)
        self.assertEqual(new,old)
    def test_source_mutation_rejected(self):
        with self.assertRaises(AssertionError):p.adapt((p.OLD/'prepare.py').read_text()+'\n')
    def test_no_active_source(self):
        new=p.adapt((p.OLD/'prepare.py').read_text())
        self.assertIn('SPIN_RUNS=[]',new);self.assertNotIn("SCOUTS=[c.WF/'scout_v1",new)
        self.assertIn(str(p.SCOUT),new)
    def test_original_count_query_thresholds_unchanged(self):
        old=(p.OLD/'prepare.py').read_text();new=p.adapt(old)
        for token in ('c.select_programs(viable,24)','maxcont>160','max(estimates.values())>160','c.solve(components'):
            self.assertEqual(old.count(token),new.count(token));self.assertGreater(old.count(token),0)
    def test_language_structural_preservation(self):
        cfg=p.read(p.OLD/'handoff_v1/CONFIG_DRAFT.json');new,changes=p.normalize(cfg,p.load_language())
        self.assertEqual(len(new['candidates']),12);self.assertEqual(len(changes),12)
        for a,b in zip(cfg['candidates'],new['candidates']):
            self.assertEqual(a['components'],b['components']);self.assertEqual(a['expected_eligible'],b['expected_eligible'])
    def test_language_idempotent(self):
        cfg=p.read(p.OLD/'language_ready_v1/CONFIG_DRAFT.json');_,changes=p.normalize(cfg,p.load_language())
        self.assertTrue(all(not row['changes'] for row in changes))

if __name__=='__main__':unittest.main()
