import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import source
class Tests(unittest.TestCase):
    def test_only_never_attempted(self):
        rows=[{'job_id':v} for v in ('a','b','c')]
        pending,partial=source.classify(rows,[{'job_id':'a'}],{'a','b'})
        self.assertEqual(pending,[{'job_id':'c'}]);self.assertEqual(partial,['b'])
    def test_unknown_directory_rejected(self):
        with self.assertRaises(AssertionError):source.classify([{'job_id':'a'}],[],{'b'})
    def test_ledger_without_directory_rejected(self):
        with self.assertRaises(AssertionError):source.classify([{'job_id':'a'}],[{'job_id':'a'}],set())
    def test_duplicate_terminal_rejected(self):
        with self.assertRaises(AssertionError):source.classify([{'job_id':'a'}],[{'job_id':'a'}]*2,{'a'})
    def test_actual_evidence(self):
        rows,e=source.evidence();self.assertEqual(len(rows),1363)
        self.assertEqual(e['instruction_aliases'],4089)
        self.assertAlmostEqual(e['prior_wall_seconds'],184.1455736728385)
        self.assertEqual([s['terminal'] for s in e['shards']],[60,0])
        self.assertEqual([len(s['interrupted']) for s in e['shards']],[1,0])
if __name__=='__main__':unittest.main(verbosity=2)
