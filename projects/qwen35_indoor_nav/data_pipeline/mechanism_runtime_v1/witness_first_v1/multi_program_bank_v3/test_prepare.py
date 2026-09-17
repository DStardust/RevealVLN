import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('bank3_prepare',HERE/'prepare.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
class Tests(unittest.TestCase):
    def test_exact_reverse(self):
        old=p.PREVIOUS.read_text();new=p.adapt(old)
        for a,b in reversed(p.patches()):new=new.replace(b,a)
        self.assertEqual(new,old)
    def test_mutation_rejected(self):
        with self.assertRaises(AssertionError):p.adapt(p.PREVIOUS.read_text()+'\n')
    def test_new_source_exact(self):
        m=p.load();self.assertEqual(m.HOUSES,['8WUmhLawc2A','D7N2EKCX4Sj','E9uDoFAP3SH'])
        self.assertEqual(m.HERE,HERE);self.assertEqual(m.SCOUT,p.WF/'bulk_source_v1/next_shard_runtime_v1/run_v1')
    def test_unchanged_original_bank_algorithm(self):
        m=p.load();source=(m.OLD/'prepare.py').read_text();new=m.adapt(source)
        for a,b in reversed(m.patches()):new=new.replace(b,a)
        self.assertEqual(new,source)
    def test_runtime_input_and_parent_both_bound(self):
        source=p.adapt(p.PREVIOUS.read_text())
        self.assertIn("read(source/'INPUT_LOCK.json')==lock",source)
        self.assertIn("sha(SCOUT.parent/'SOURCE_LOCK.json')",source)
        self.assertIn("'shard':1,'gpu':2",source)
    def test_language_and_thresholds_unchanged(self):
        m=p.load();cfg=m.read(m.OLD/'handoff_v1/CONFIG_DRAFT.json');new,changes=m.normalize(cfg,m.load_language())
        self.assertEqual(len(changes),12)
        for a,b in zip(cfg['candidates'],new['candidates']):self.assertEqual(a['components'],b['components'])
        bank=m.adapt((m.OLD/'prepare.py').read_text())
        self.assertIn('c.select_programs(viable,24)',bank);self.assertIn('maxcont>160',bank)

if __name__=='__main__':unittest.main()
