"""Regression checks for early-query evidence and unchanged causal policy."""
import importlib.util
from pathlib import Path
import unittest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('reader_fix_under_test', HERE/'model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1209)
        self.fixed = module.MemoryPolicy(8,8,slots=2,width=4)
        self.old = module.original.MemoryPolicy(8,8,slots=2,width=4)
        self.old.load_state_dict(self.fixed.state_dict())

    def test_long_shared_tail_does_not_erase_query_difference(self):
        memory = self.fixed.reset(1,'cpu')
        q1 = torch.tensor([[1]+[2]*220+[3]])
        q2 = torch.tensor([[4]+[2]*220+[3]])
        with torch.no_grad():
            self.assertTrue(torch.equal(self.old.reader(memory,q1),self.old.reader(memory,q2)))
            delta = (self.fixed.reader(memory,q1)-self.fixed.reader(memory,q2)).abs().max()
            self.assertGreater(float(delta),1e-7)

    def test_early_query_embedding_has_gradient(self):
        captured=[]
        def hook(module,args,out):
            out.retain_grad(); captured.append(out)
        handle=self.fixed.query_embedding.register_forward_hook(hook)
        self.fixed.reader(self.fixed.reset(1,'cpu'),torch.tensor([[1]+[2]*220+[3]])).sum().backward()
        handle.remove()
        self.assertGreater(float(captured[0].grad[:,0].norm()),0.)

    def test_same_parameters_memory_action_and_query_firewall(self):
        self.assertEqual(list(self.fixed.state_dict()),list(self.old.state_dict()))
        features=torch.randn(2,20,8)
        with torch.no_grad():
            old,_=self.old.encode(features);fixed,_=self.fixed.encode(features)
            self.assertTrue(torch.equal(old,fixed))
            base=torch.randn(2,4)
            actions=self.fixed.action_logits(fixed[:,-1],base)
            self.assertTrue(torch.equal(actions,self.old.action_logits(old[:,-1],base)))
            snapshot=fixed.clone()
            self.fixed.reader(fixed[:,-1],torch.tensor([[1,2,3],[4,2,3]]))
            self.assertTrue(torch.equal(snapshot,fixed))
            self.assertTrue(torch.equal(actions,self.fixed.action_logits(fixed[:,-1],base)))
        with self.assertRaises(TypeError):
            self.fixed.update(features[:,0],fixed[:,-1],query=torch.ones(2,3))


if __name__=='__main__':
    unittest.main()
