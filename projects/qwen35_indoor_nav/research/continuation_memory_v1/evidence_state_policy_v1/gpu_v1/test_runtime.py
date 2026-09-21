"""CPU integration contracts for the new simulator-facing state representation."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from shared import *
from evaluate_continuations import registry_value,prefix_audit

class RuntimeContracts(unittest.TestCase):
    def test_registry_and_prefix_stream(self):
        torch.set_num_threads(2)
        data=read(CPU/'DATA.json');cfg=read(HERE/'PROTOCOL.json')
        reg=registry_value(data['raw_families'],cfg)
        self.assertEqual(len(reg['conditions']),128);self.assertEqual(len(reg['slots']),1152)
        self.assertEqual(reg['conditions'],read(CONTROL/'EVALUATION_REGISTRY.json')['conditions'])
        cache=torch.load(CONTROL/'features/FEATURES.pt',map_location='cpu',weights_only=True)
        row=data['families'][0]['sequences'][0];indices=row['features'];cut=row['cutoff']
        x=cache['features'][indices].float().unsqueeze(0);y=cache['logits'][indices].float().unsqueeze(0)
        for mode in cfg['arms']:
            net=make_head(mode+'_1209')
            with torch.no_grad():
                state=prefill(net,[x[:,t] for t in range(cut)])
                logits,state,detail=net.step(x[:,cut],y[:,cut],state)
                full=net(x[:,:cut+1],y[:,:cut+1],torch.ones(1,cut+1,dtype=torch.bool))
                self.assertTrue(torch.equal(logits,full['logits'][:,-1]))
                self.assertTrue(torch.equal(detail['state'],full['state'][:,-1]))
                self.assertEqual(set(state_identity(state)),{'memory','belief','first'})
        self.assertEqual(len(reg['models']),9)

    def test_ownership_cleanup_rejects_foreign(self):
        # The production resource helper must refuse a substituted owner.
        import subprocess
        monitor=load('evidence_test_monitor',V16/'pipeline.py')
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],start_new_session=True)
        try:
            owner=monitor.process_identity(child.pid);wrong=dict(owner,uid=owner['uid']+1)
            with self.assertRaises(Exception):monitor.cleanup(child,wrong)
            self.assertIsNone(child.poll())
        finally:
            child.terminate();child.wait(timeout=10)

if __name__=='__main__':unittest.main()
