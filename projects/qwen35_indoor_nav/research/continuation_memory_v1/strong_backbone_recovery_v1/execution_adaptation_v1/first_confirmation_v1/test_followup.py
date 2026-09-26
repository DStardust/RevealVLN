import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from prepare import complement
from review import bootstrap,old


class FollowupTests(unittest.TestCase):
    def test_complement_keeps_order_and_never_reads_scores(self):
        full=[dict(id=i,score=100-i) for i in range(6)];prior=[full[4],full[1]]
        self.assertEqual([e['id'] for e in complement(full,prior)],[0,2,3,5])
        with self.assertRaisesRegex(ValueError,'DUPLICATE_OR_UNKNOWN'):complement(full,[dict(id=8)])
        with self.assertRaisesRegex(ValueError,'DUPLICATE_OR_UNKNOWN'):complement(full,prior+prior)

    def test_partial_followup_has_no_admission_sr(self):
        p=dict(heads={'CURRENT_FIRST_s42':{},'DELTA_FIRST_s42':{}},comparisons={'pair':dict(CURRENT='CURRENT_FIRST_s42',DELTA='DELTA_FIRST_s42')})
        entries=[dict(id=1,house='a'),dict(id=2,house='b')]
        row=dict(house='a',outcomes={a:dict(success=int(a.startswith('DELTA')),spl=0.,steps=10) for a in ['NATIVE',*p['heads']]})
        r=old.summarize(p,entries,{1:row});self.assertIsNone(r['arms']['DELTA_FIRST_s42']['sr'])
        self.assertEqual(r['paired']['pair']['difference_identification_bounds'],[0.,1.])

    def test_house_bootstrap_keeps_seed_pairing(self):
        rows={i:dict(house=h,outcomes={a:dict(success=s) for a,s in [('NATIVE',0),('DELTA_FIRST_s42',1),('DELTA_FIRST_s43',1)]}) for i,h in enumerate(['a','a','b'])}
        result=bootstrap(rows,'NATIVE','DELTA_FIRST_s{seed}',[42,43],repetitions=100)
        self.assertEqual(result['houses'],2);self.assertEqual(result['interval95'],[1.,1.])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(FollowupTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    with (HERE/'CPU_TEST_RESULT.json').open('x') as f:json.dump(dict(tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),passed=result.wasSuccessful(),gpu_hours=0),f,indent=2)
    raise SystemExit(not result.wasSuccessful())
