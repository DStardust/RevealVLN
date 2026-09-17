import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('witness_audit',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def obs(step,pixels):return dict(step=step,pixels=pixels,evidence_complete=True)

class WitnessTests(unittest.TestCase):
    def test_exact_threshold(self):
        events=audit.instance_events([obs(0,{'1':256}),obs(1,{'1':256})])
        self.assertIn(1,events[1]);self.assertEqual(events[0],{})
    def test_no_cross_instance_confirmation(self):
        self.assertEqual(audit.instance_events([obs(0,{'1':500}),obs(1,{'2':500})])[1],{})
    def test_255_is_not_event(self):
        self.assertEqual(audit.instance_events([obs(0,{'1':255}),obs(1,{'1':256})])[1],{})
    def test_unknown_is_not_negative(self):
        with self.assertRaises(AssertionError):audit.instance_events([dict(step=0,pixels={},evidence_complete=False)])
    def test_prefix_blocks_simultaneous_forbidden(self):
        row=dict(role_first={'a':(3,1),'b':(3,2)},closed_loop=False,actions=10,path='x',sha256='h')
        self.assertIsNone(audit.witness([row],'a',['b']))
    def test_later_forbidden_does_not_taint_prefix(self):
        row=dict(role_first={'a':(3,1),'b':(4,2)},closed_loop=False,actions=10,path='x',sha256='h')
        self.assertEqual(audit.witness([row],'a',['b'])['prefix_cutoff'],3)
        self.assertIsNone(audit.witness([row],'a',['b'],closed=True))
    def test_distinct_raw_same_query_semantics_rejected(self):
        self.assertFalse(audit.semantic_distinct(['["chair","living room","armchair"]','["chair","living room","chair"]']))

if __name__=='__main__':unittest.main()
