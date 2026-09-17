from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import audit
class Tests(unittest.TestCase):
    def test_terminal_only_not_partial_or_unattempted(self):
        jobs=[dict(job_id=x) for x in 'abc']
        self.assertEqual(audit.terminal_partition(jobs,[dict(job_id='a')],{'a','b'}),([dict(job_id='a')],['b']))
    def test_unknown_directory_rejected(self):
        with self.assertRaises(AssertionError):audit.terminal_partition([],[],{'unknown'})
    def test_duplicate_ledger_rejected(self):
        with self.assertRaises(AssertionError):audit.terminal_partition([dict(job_id='a')],[dict(job_id='a')]*2,{'a'})
    def test_paths_do_not_target_new_production(self):
        self.assertEqual(audit.OLDROOT,audit.FULL/'production/shard_0002')
        self.assertNotIn('auto_generation_v1/production',str(audit.OLDROOT))
        self.assertEqual(audit.ROOT,Path('/mnt/data_nas/deeprobotics/daiyang/vla'))
if __name__=='__main__':unittest.main(verbosity=2)
