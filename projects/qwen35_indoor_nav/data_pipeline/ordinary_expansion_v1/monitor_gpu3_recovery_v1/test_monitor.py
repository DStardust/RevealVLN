import copy
import json
from pathlib import Path
import sys
import time
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import monitor as m

def fixture():
    lane=dict(gpu=3,completed=140,replay_certified=140,strict_routes=103,strict_decisions=6772,
        stage='FAILED',result={'error':'SHARD_WALL_BUDGET'},target=3253)
    other=[dict(gpu=g,completed=100,replay_certified=99,strict_routes=90,strict_decisions=5000,stage='PRODUCING',result={},target=3253 if g==4 else 3155) for g in (4,5)]
    return dict(lanes=[lane]+other,completed=340,strict_routes=283,strict_decisions=16772,note='base')

class Tests(unittest.TestCase):
    def test_counts_and_other_lanes(self):
        base=fixture();before=copy.deepcopy(base)
        p=[dict(completed=3,replay_certified_routes=3,audited_routes=3,audited_instruction_conditioned_decisions=200,shard=0,heartbeat_unix=10)]
        d=m.augment(base,p,{'strict_routes':140,'instruction_conditioned_decisions':9000},{'strict_pass':True},{},True)
        self.assertEqual(base,before);self.assertEqual(d['completed'],343)
        self.assertEqual(d['strict_routes'],323);self.assertEqual(d['strict_decisions'],19200)
        self.assertEqual(d['lanes'][1:],base['lanes'][1:]);self.assertEqual(d['lanes'][0]['stage'],'PRODUCING')
        self.assertEqual(d['lanes'][0]['prior_failure'],{'error':'SHARD_WALL_BUDGET'})

    def test_no_salvage_does_not_invent(self):
        d=m.augment(fixture(),[],{}, {},{},False)
        self.assertEqual(d['strict_routes'],283);self.assertEqual(d['lanes'][0]['stage'],'RECOVERY_PREPARED')

    def test_failure_and_closure(self):
        for result,stage in [({'error':'timeout'},'FAILED'),({'error':None},'CLOSED')]:
            d=m.augment(fixture(),[],{}, {},result,True);self.assertEqual(d['lanes'][0]['stage'],stage)

    def test_no_double_count(self):
        p=[dict(completed=3113,shard=63,heartbeat_unix=10)]
        with self.assertRaises(AssertionError):m.augment(fixture(),p,{}, {},{},True)

    def test_live_collector_readonly(self):
        d=m.load_collector()()
        self.assertEqual(len(d['lanes']),3)
        self.assertEqual(d['training_status'],'STOPPED')

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    with (HERE/'CPU_TESTS.json').open('x') as f:json.dump(dict(passed=r.wasSuccessful(),tests=r.testsRun,errors=len(r.errors),failures=len(r.failures),unix=time.time(),sha256={p.name:m.sha(p) for p in HERE.glob('*.py')}),f,indent=2)
    raise SystemExit(0 if r.wasSuccessful() else 1)
