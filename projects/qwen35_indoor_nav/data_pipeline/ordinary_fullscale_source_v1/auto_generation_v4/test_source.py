import sys
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import source
class Tests(unittest.TestCase):
    def test_actual_evidence(self):
        rows,e=source.evidence();self.assertEqual(len(rows),1362)
        self.assertEqual([s['terminal'] for s in e['shards']],[0,0])
        self.assertEqual(e['shards'][0]['interrupted'],['rxr_3babd2c8afa6f2dacc9c'])
        self.assertAlmostEqual(e['prior_wall_seconds'],209.07168883481062)
        self.assertNotIn('rxr_3babd2c8afa6f2dacc9c',{j['job_id'] for j in rows})
    def test_terminal_and_partial_excluded(self):
        rows=[{'job_id':j} for j in ('a','b','c')]
        self.assertEqual(source.classify(rows,[{'job_id':'a'}],{'a','b'}),([{'job_id':'c'}],['b']))
    def test_unknown_dir_rejected(self):
        with self.assertRaises(AssertionError):source.classify([{'job_id':'a'}],[],{'b'})
if __name__=='__main__':unittest.main(verbosity=2)
