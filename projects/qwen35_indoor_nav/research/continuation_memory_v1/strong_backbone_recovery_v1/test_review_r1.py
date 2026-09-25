"""Empty and partial diagnostics must not manufacture an evaluation score."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import common as u
from memory_v2 import ExecutionMemory
from review_memory_r1 import review_split


class ReviewTests(unittest.TestCase):
    def test_empty_check_has_no_model_construction_or_score(self):
        result=review_split({'groups':[]},{})
        self.assertEqual(result['status'],'NO_ADMITTED_SAMPLES')
        self.assertEqual(len(result['arms']),4)
        for row in result['arms']:
            self.assertEqual(row['total'],0)
            self.assertIsNone(row['correct'])
            self.assertIsNone(row['teacher_action_accuracy'])

    def test_real_denominator_excludes_padding_and_unknown(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        model=ExecutionMemory(8);states={a:model.state_dict() for a in ('BC','B2','OURS')}
        g=dict(features=torch.zeros(1,2,3,8),actor_features=torch.zeros(1,2,3,8),lengths=torch.tensor([[2,3]]),
            base_action_logits=torch.tensor([[[[1.,0,0,0]]*3]*2]),action_known=torch.tensor([[[False,True,True],[False,False,True]]]),
            action_targets=torch.tensor([[[0,0,3],[0,0,1]]]),family='fixture',house='fixture_house',tasks=['task'],
            histories=['h1','h2'],continuations=['c1','c2'])
        result=review_split({'groups':[g]},states)
        for row in result['arms']:
            self.assertEqual((row['correct'],row['total']),(1,2))
            self.assertEqual(row['teacher_action_accuracy'],.5)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReviewTests))
    u.write(u.HERE/'REVIEW_CPU_TEST_RESULT.json',dict(tests=result.testsRun,successful=result.wasSuccessful(),scope='Empty split, unknown target and padded-step accounting'))
    raise SystemExit(not result.wasSuccessful())
