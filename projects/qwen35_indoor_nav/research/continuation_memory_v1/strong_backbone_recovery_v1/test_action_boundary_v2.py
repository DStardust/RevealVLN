import unittest
import torch
from action_boundary_v2 import ActionBoundary

class BoundaryTests(unittest.TestCase):
    def test_header_is_not_an_action(self):
        b=ActionBoundary([7,8,9])
        for seq in ([],[7],[7,8]):self.assertIsNone(b.offset(torch.tensor([seq],dtype=torch.long)))
        self.assertEqual(b.offset(torch.tensor([[7,8,9]])),0)
        self.assertEqual(b.offset(torch.tensor([[7,8,9,2]])),1)

    def test_prompt_length_does_not_shift_boundary(self):
        b=ActionBoundary([7,8,9]);self.assertIsNone(b.offset(torch.tensor([[1,2]])))
        self.assertEqual(b.offset(torch.tensor([[1,2,7,8,9]])),0)

    def test_malformed_header_fails(self):
        b=ActionBoundary([7,8,9]);b.offset(torch.empty(1,0,dtype=torch.long))
        with self.assertRaisesRegex(ValueError,'UNEXPECTED'):b.offset(torch.tensor([[7,6]]))

    def test_each_generation_resets(self):
        for _ in range(2):
            b=ActionBoundary([7,8,9]);b.offset(torch.empty(1,0,dtype=torch.long))
            self.assertEqual(b.offset(torch.tensor([[7,8,9]])),0)

    def test_nonzero_changes_real_action_only(self):
        b=ActionBoundary([7,8,9]);actions=[1,2,3,4];generated=[]
        for i,token in enumerate([7,8,9,1]):
            scores=torch.full((1,10),-10.);scores[0,token]=10.
            offset=b.offset(torch.tensor([generated],dtype=torch.long))
            revised=scores.clone()
            if offset==0:revised[0,actions[1]]+=25
            if i<3:self.assertTrue(torch.equal(scores,revised))
            generated.append(int(revised.argmax(-1)))
        self.assertEqual(generated,[7,8,9,2])

    def test_zero_residual_preserves_stop(self):
        b=ActionBoundary([7,8,9]);b.offset(torch.empty(1,0,dtype=torch.long))
        self.assertEqual(b.offset(torch.tensor([[7,8,9]])),0)
        scores=torch.tensor([[5.,1.,2.,3.]])
        self.assertTrue(torch.equal(scores,scores+torch.zeros_like(scores)))
        self.assertEqual(int(scores.argmax()),0)

if __name__=='__main__':unittest.main()

