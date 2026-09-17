from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import source
class Tests(unittest.TestCase):
    def test_only_unattempted(self):
        jobs=[dict(job_id=x) for x in 'abcd']
        rows,partial=source.classify(jobs,[dict(job_id='a',status='FAILED'),dict(job_id='b',status='CERTIFIED')],{'a','b','c'})
        self.assertEqual(rows,[dict(job_id='d')]);self.assertEqual(partial,['c'])
    def test_missing_terminal_dir_rejected(self):
        with self.assertRaises(AssertionError):source.classify([dict(job_id='a')],[dict(job_id='a')],set())
    def test_unknown_route_rejected(self):
        with self.assertRaises(AssertionError):source.classify([dict(job_id='a')],[],{'foreign'})
    def test_duplicate_ledger_rejected(self):
        with self.assertRaises(AssertionError):source.classify([dict(job_id='a')],[dict(job_id='a')]*2,{'a'})
    def test_completely_unstarted(self):
        jobs=[dict(job_id='a')];self.assertEqual(source.classify(jobs,[],set()),(jobs,[]))
if __name__=='__main__':unittest.main(verbosity=2)
