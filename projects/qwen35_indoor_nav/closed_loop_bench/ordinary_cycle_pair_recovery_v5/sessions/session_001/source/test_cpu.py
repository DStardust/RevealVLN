import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
from cycle_policy import CycleRecovery


class Tests(unittest.TestCase):
    def test_fixed_stratified_order(self):
        rows=c.read(HERE/'PAIR_ORDER.json')
        self.assertEqual(sorted(r['index'] for r in rows),list(range(100)))
        self.assertEqual(len({r['house'] for r in rows[:5]}),5)
        self.assertEqual(sum(r['inference_order'][0]=='A' for r in rows),50)

    def test_finite_unequal_logits_are_separate_from_action_agreement(self):
        a=dict(raw={'x':1},processed={'x':2},logits=[2.,1.,0.,-1.],native_action='move_forward',executed_action='move_forward',override=False)
        b=copy.deepcopy(a);b['logits'][0]+=.001
        r=c.prefix_compare(a,b)
        self.assertTrue(r['input_prefix_matched'] and r['action_prefix_matched'])
        self.assertFalse(r['logits_bitwise_equal'])
        b['native_action']='turn_left'
        self.assertEqual(c.prefix_compare(a,b)['argmax_flip_count'],1)
        b['logits'][0]=float('nan')
        with self.assertRaises(c.PairError): c.prefix_compare(a,b)

    def test_input_mismatch_and_inclusive_first_override(self):
        a=dict(raw={'x':1},processed={'x':2},logits=[2.,1.,0.,-1.],native_action='move_forward',executed_action='move_forward',override=False)
        b=copy.deepcopy(a);b.update(override=True,executed_action='turn_left')
        self.assertTrue(c.prefix_compare(a,b)['action_prefix_matched'])
        b['processed']['x']=3
        self.assertFalse(c.prefix_compare(a,b)['input_prefix_matched'])

    def test_cycle_reset_ties_stop_and_executed_history(self):
        guard=CycleRecovery();rgb=[bytes(224*224*3)];logits=[2.,1.,1.,0.]
        first,_=guard.choose('go',rgb,[],logits);self.assertEqual(first,'move_forward')
        second,_=guard.choose('go',rgb,[],logits);self.assertEqual(second,'turn_left')
        guard.reset();self.assertEqual(guard.choose('go',rgb,[],logits)[0],first)
        self.assertEqual(guard.choose('go',rgb,[],[0.,0.,0.,1.])[0],'STOP')
        window=c.Window();frame=base64.b64encode(rgb[0]).decode()
        window.receive(dict(done=False,instruction='go',rgb=frame))
        window.receive(dict(done=False,rgb=frame),executed=second)
        self.assertEqual(window.executed,['turn_left'])
        window.receive(dict(done=False,instruction='next',rgb=frame))
        self.assertEqual(window.executed,[])

    def test_budget_includes_stop_and_recovery(self):
        self.assertEqual(c.advance('STOP',499,False),(500,True))
        self.assertEqual(c.advance('turn_left',499,False),(500,True))
        with self.assertRaises(ValueError): c.advance('move_forward',500,True)

    def test_atomic_complete_result_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            path=Path(folder)/'PAIR.json'
            c.write(path,{'complete':True},True)
            with self.assertRaises(FileExistsError): c.write(path,{'complete':False},True)
            self.assertTrue(c.read(path)['complete'])

    def test_stop_metric_is_not_range_only(self):
        import numpy as np
        metric=c.load('v5_cpu_official_metric',c.TINY/'metrics.py')
        class Sim:
            def get_agent_state(self): return SimpleNamespace(position=np.array([0.,0.,0.]))
            def geodesic_distance(self,*args,**kwargs): return 1.
        values=metric.OfficialMetrics(Sim(),[0.,0.,1.])
        self.assertEqual(values.update(False)['success'],0.)
        self.assertEqual(values.update(True)['success'],1.)

    def test_half_pair_and_unsealed_pair_are_not_admitted(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder, patch.object(c,'HERE',Path(folder)):
            session=Path(folder)/'sessions/session_001'
            pair=session/'pairs/pair_000';pair.mkdir(parents=True)
            c.write(pair/'A.json',dict(complete=True))
            self.assertEqual(c.committed_pairs(),{})
            c.write(pair/'PAIR.json',dict(rank=0,valid_behavioral_pair=True))
            self.assertEqual(c.committed_pairs(),{})
            c.write(session/'STATE_SEAL_001.json',dict(unchanged=True,pair_ranks=[0]))
            self.assertEqual(set(c.committed_pairs()),{0})

    def test_process_cleanup_refuses_changed_owner(self):
        ownership=c.load('v5_cpu_ownership',c.V3/'launch.py')
        proc=SimpleNamespace(pid=123)
        with patch.object(ownership,'group_members',return_value=[{'pid':123}]), patch.object(ownership,'identity',return_value={'start_ticks':2}), patch.object(ownership.os,'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError,'REFUSE_SIGNAL'):
                ownership.terminate_owned(proc,{'start_ticks':1})
            kill.assert_not_called()

    def test_processed_tensor_identity_covers_values_shape_dtype(self):
        import torch
        a=torch.tensor([[1,2]],dtype=torch.int64)
        self.assertNotEqual(c.tensor_identity(a),c.tensor_identity(a.float()))
        self.assertNotEqual(c.tensor_identity(a),c.tensor_identity(a.reshape(2,1)))
        self.assertNotEqual(c.tensor_identity(a),c.tensor_identity(a+1))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    c.write(HERE/'CPU_TEST_RESULT.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),scope='V5 orchestration CPU contracts, not model/runtime acceptance'))
    raise SystemExit(not result.wasSuccessful())
