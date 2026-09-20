"""Pure contract checks; toy records are never emitted as research data."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, LINE, RESEARCH, c, load, raw_window, pose_delta
compiler = load('v15_test_compiler', LINE/'data_pipeline/mechanism_factory_v2/compiler.py')


class Contracts(unittest.TestCase):
    def test_window_uses_only_executed_prefix(self):
        trace = dict(actions=['L']*10+['R','S'], observations=[dict(rgb_hash=str(i)) for i in range(12)])
        self.assertEqual(raw_window(trace, 10), dict(rgb=['9','10'], executed=['L']*8))
        trace['actions'][10] = 'F'
        trace['observations'][11]['rgb_hash'] = 'future'
        self.assertEqual(raw_window(trace, 10), dict(rgb=['9','10'], executed=['L']*8))

    def test_sensor_pose_difference_cannot_hide(self):
        a = dict(position=[0,0,0], rotation=[1,0,0,0], sensors={})
        b = dict(position=[0,1e-6,0], rotation=[1,0,0,0], sensors={})
        self.assertEqual(pose_delta(dict(a,sensors={'rgb':a}), dict(a,sensors={'rgb':b})), 1e-6)

    def test_see2_requires_order_and_stop(self):
        comp = compiler.Compiler({'a':['bed','bedroom'],'t':['tv_monitor','living room']},
            {'g':dict(anchor='a',terminal='t',instruction='See bed, then tv, then stop.')}, {'a':[1],'t':[2]})
        observations = [dict(step=i, pixels=p, evidence_complete=True)
            for i,p in enumerate([{'1':300},{'1':300},{'2':300},{'2':300}])]
        trace = dict(actions=['L','R','L','S'], observations=observations, complete=True, collisions=0)
        self.assertEqual(comp.evaluate(trace,'g'), 'pass')
        trace['actions'] = ['L','R','L']
        self.assertEqual(comp.evaluate(trace,'g'), 'fail')
        trace['observations'][1]['evidence_complete'] = False
        self.assertEqual(comp.evaluate(trace,'g'), 'unknown')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    c.write(RESEARCH/'RAW_CPU_TEST_RESULT.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        scope='Raw causal window, complete sensor pose, frozen SEE2 stop/unknown contracts. No simulator or learned model validation.'), True)
    raise SystemExit(not result.wasSuccessful())
