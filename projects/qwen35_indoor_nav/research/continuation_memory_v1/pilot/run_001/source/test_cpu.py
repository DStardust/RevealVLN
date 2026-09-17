"""Small CPU unit checks; actual Qwen-feature gradient evidence is produced by run.py."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from model import MemoryPolicy


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1209)
        self.net=MemoryPolicy(8,300,slots=2,width=4)
        self.features=torch.randn(2,249,8)

    def test_query_is_reader_only_and_action_reads_memory(self):
        with torch.no_grad():
            states,_=self.net.encode(self.features)
            memory=states[:,-1]
            before=memory.clone();base=torch.zeros(2,4)
            actions=self.net.action_logits(memory,base)
            self.net.reader(memory,torch.tensor([[1,2],[3,4]]))
            self.assertTrue(torch.equal(before,memory))
            self.assertTrue(torch.equal(actions,self.net.action_logits(memory,base)))
            self.assertFalse(torch.equal(actions,self.net.action_logits(torch.zeros_like(memory),base)))
        with self.assertRaises(TypeError):self.net.update(self.features[:,0],memory,query=torch.tensor([1]))

    def test_long_event_write_has_gradient_and_detach_cuts_it(self):
        states,writes=self.net.encode(self.features,retain_steps=[14])
        self.net.reader(states[:,-1],torch.tensor([[1,2],[3,4]])).sum().backward()
        self.assertGreater(float(writes[14].grad.norm()),0.)
        self.net.zero_grad(set_to_none=True)
        memory=self.net.reset(2,'cpu');first=None
        for t in range(249):
            memory,write=self.net.update(self.features[:,t],memory,retain_write=t==14)
            if t==14: first=write;memory=memory.detach()
        self.net.reader(memory,torch.tensor([[1,2],[3,4]])).sum().backward()
        self.assertIsNone(first.grad)

    def test_reset_and_causal_prefix_not_affected_by_later_features(self):
        with torch.no_grad():
            first,_=self.net.encode(self.features[:,:20])
            changed=self.features.clone();changed[:,20:]*=100
            full,_=self.net.encode(changed)
            self.assertTrue(torch.equal(first,full[:,:20]))
            repeated,_=self.net.encode(self.features[:,:20])
            self.assertTrue(torch.equal(first,repeated))


if __name__=='__main__':
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    with (HERE/'CPU_TEST_RESULT.json').open('x') as out:
        json.dump(dict(passed=result.wasSuccessful(),tests=result.testsRun,real_qwen_features=False,gpu_used=False),out,indent=2)
    raise SystemExit(not result.wasSuccessful())
