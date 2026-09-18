import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import json
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evaluate
import review


def decision(action='move_forward',executed='move_forward'):
    return dict(raw=dict(key='same'),processed=dict(ids='same'),logits=[2.,1.,0.,-1.],
                native_action=action,executed_action=executed,override=executed!=action)


class Tests(unittest.TestCase):
    def test_all_declared_runtime_controls(self):
        self.assertEqual(evaluate.ARMS,('A','B','C','D'))
        self.assertEqual(review.ARMS,evaluate.ARMS)
        self.assertIn(('D','C'),evaluate.CONTRASTS)
        self.assertEqual(len(evaluate.CONTRASTS),6)

    def test_native_stop_and_four_class_choice(self):
        self.assertEqual(evaluate.selected_action([0,0,0,1],[9,8,7,0]),('STOP','STOP'))
        self.assertEqual(evaluate.selected_action([1,0,0,0],[0,2,1,0]),('move_forward','turn_left'))
        self.assertEqual(evaluate.selected_action([1,0,0,0],[0,0,0,2]),('move_forward','STOP'))

    def test_joint_termination_while_other_arms_continue(self):
        self.assertFalse(evaluate.active_contrast({'C':decision()},'A','B'))
        self.assertTrue(evaluate.active_contrast({'A':decision(),'B':decision()},'A','B'))
        with self.assertRaises(evaluate.c.PairError):evaluate.active_contrast({'A':decision()},'A','B')

    def test_intentional_difference_and_numeric_invalidity_are_separate(self):
        a,b=decision(executed='turn_left'),decision()
        _,different=evaluate.compare_prefix(a,b)
        self.assertTrue(different)
        b['raw']['key']='changed'
        with self.assertRaises(evaluate.c.PairError):evaluate.compare_prefix(a,b)
        b=decision(action='turn_right')
        with self.assertRaises(evaluate.c.PairError):evaluate.compare_prefix(a,b)
        b=decision();b['logits'][0]=float('nan')
        with self.assertRaises(evaluate.c.PairError):evaluate.compare_prefix(a,b)

    def test_channel_memory_reset_and_budget(self):
        module=evaluate.c.load('v9_test_mem',evaluate.MEMORY.parent/'query_reader_repair_v2/model.py')
        net=module.MemoryPolicy(6,10,slots=2,width=4)
        a,b=net.reset(1,'cpu'),net.reset(1,'cpu')
        changed,_=net.update(torch.ones(1,6),a)
        self.assertTrue(torch.equal(b,torch.zeros_like(b)))
        self.assertTrue(torch.equal(a,torch.zeros_like(a)))
        self.assertFalse(torch.equal(changed,b))
        self.assertTrue(torch.equal(net.reset(1,'cpu'),b))
        self.assertEqual(evaluate.c.advance('STOP',499,False,500),(500,True))
        self.assertEqual(evaluate.c.advance('turn_left',499,False,500),(500,True))

    def test_paired_win_loss_denominator(self):
        def episode(i,s):return dict(episode_id=str(i),house='house',success=s,spl=.5*s,ndtw=.2)
        rows=[dict(episodes={'A':episode(i,a),'B':episode(i,b)}) for i,(a,b) in enumerate([(0,1),(1,1),(0,0)])]
        result=review.contrast(rows,'A','B')
        self.assertAlmostEqual(result['delta_sr'],1/3)
        self.assertEqual(result['wins'],['0'])
        self.assertEqual(result['retained_successes'],['1'])
        self.assertEqual(result['losses'],[])

    def test_repeated_seeds_are_not_independent_episodes(self):
        values={i:dict(house='h'+str(i%2),deltas=[1,1,1]) for i in range(4)}
        result=review.clustered_interval(values,100)
        self.assertEqual(result['independent_episode_units'],4)
        self.assertEqual(result['houses'],2)
        self.assertEqual(result['episode_resample_95pct'],[1,1])
        self.assertEqual(result['house_resample_95pct'],[1,1])

    def test_incomplete_seeds_cannot_produce_complete_benefit(self):
        with patch.object(review.c,'read',return_value={'method_seeds':[1209,1210,1211]}), \
             patch.object(review,'summarize',side_effect=[{'complete_quartets':100},{'complete_quartets':20},{'complete_quartets':0}]):
            result=review.aggregate()
        self.assertEqual(result['complete_quartets'],120)
        self.assertEqual(result['episode_executions'],480)
        self.assertEqual(result['status'],'PARTIAL_FIXED_SEED_REPLICATION')
        self.assertNotIn('mean_sr',result)

    def test_wrong_seed_receipt_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root=Path(folder)
            session=root/'seed_1209/sessions/session_001'
            pair=session/'pairs/pair_000';pair.mkdir(parents=True)
            (session/'STATE_SEAL_001.json').write_text(json.dumps(dict(unchanged=True,quartet_ranks=[0])))
            (pair/'QUARTET.json').write_text(json.dumps(dict(rank=0,method_seed=1210,valid_behavioral_quartet=True)))
            with patch.object(review,'HERE',root):
                with self.assertRaisesRegex(AssertionError,'SEED_BINDING_MISMATCH'):review.committed(1209)


if __name__=='__main__':unittest.main()
