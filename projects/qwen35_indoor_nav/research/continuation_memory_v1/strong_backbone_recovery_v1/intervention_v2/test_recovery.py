"""CPU checks of the actual gate/perturbation boundaries and data isolation."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
import common as u
import torch
from intervention_gate_v1 import InterventionGate,label
from selection import choose_once,forced_action,disagreement


class Gate(torch.nn.Module):
    def __init__(self,accept):super().__init__();self.accept=accept;self.calls=0
    def forward(self,*args):
        self.calls+=1
        return torch.tensor([[0.,10.,-10.]]) if self.accept else torch.tensor([[0.,-10.,10.]])


class Tests(unittest.TestCase):
    def test_pipeline_imports_its_own_manifest_builder(self):
        spec=importlib.util.spec_from_file_location('recovery_pipeline_test',HERE/'pipeline.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(Path(sys.modules[module.prepare_unseen.__module__].__file__).resolve(),HERE/'prepare_recovery.py')
    def test_real_outcome_labels(self):
        self.assertEqual([label(a,b) for a,b in [(0,0),(1,1),(0,1),(1,0)]],[0,0,1,2])
    def test_shared_perturbation_not_a_gate_example(self):
        r=dict(native_token=1,method_token=2,forced_prefix_query=True)
        self.assertFalse(disagreement(r));r['forced_prefix_query']=False;self.assertTrue(disagreement(r))
    def test_eight_turns_are_real_bounded_chunks(self):
        for step in (0,4):
            self.assertEqual([forced_action([12]*8,step,i,99) for i in range(5)],[12,12,12,12,99])
        self.assertIsNone(forced_action([12]*8,8,0,99))
        self.assertIsNone(forced_action([12]*8,0,None,99))
        self.assertIsNone(forced_action([],0,0,99))
    def test_reject_once_stays_native(self):
        g=Gate(False);native=torch.tensor([[0.,2.,0.,0.]]);proposal=torch.tensor([[0.,0.,3.,0.]])
        result,accepted,e=choose_once(g,torch.zeros(1,3584),native,proposal,list(range(4)),None)
        self.assertTrue(torch.equal(result,native));self.assertFalse(accepted)
        g.accept=True
        result,accepted,e=choose_once(g,torch.zeros(1,3584),native,proposal,list(range(4)),accepted)
        self.assertTrue(torch.equal(result,native));self.assertEqual(g.calls,1);self.assertIsNone(e)
    def test_accept_once_keeps_remaining_method(self):
        g=Gate(True);native=torch.tensor([[0.,2.,0.,0.]]);proposal=torch.tensor([[0.,0.,3.,0.]])
        result,accepted,e=choose_once(g,torch.zeros(1,3584),native,proposal,list(range(4)),None)
        self.assertTrue(torch.equal(result,proposal));self.assertTrue(accepted)
        g.accept=False
        result,accepted,e=choose_once(g,torch.zeros(1,3584),native,proposal,list(range(4)),accepted)
        self.assertTrue(torch.equal(result,proposal));self.assertEqual(g.calls,1)
    def test_no_choice_before_real_disagreement(self):
        g=Gate(True);native=torch.tensor([[0.,2.,0.,0.]])
        result,accepted,e=choose_once(g,torch.zeros(1,3584),native,native+1,list(range(4)),None)
        self.assertIsNone(accepted);self.assertEqual(g.calls,0);self.assertTrue(torch.equal(result,native))
    def test_gate_backward_updates(self):
        torch.manual_seed(42);gate=InterventionGate(3592);opt=torch.optim.AdamW(gate.parameters(),lr=.001)
        actor=torch.randn(6,3584);native=torch.randn(6,4);method=torch.randn(6,4)
        before=gate.linear.weight.detach().clone()
        loss=torch.nn.functional.cross_entropy(gate(actor,native,method),torch.tensor([0,1,2,0,1,2]))
        loss.backward();self.assertTrue(torch.isfinite(gate.linear.weight.grad).all());opt.step()
        self.assertFalse(torch.equal(before,gate.linear.weight))
    def test_nonfinite_gate_fails(self):
        class Invalid(Gate):
            def forward(self,*args):return torch.tensor([[float('nan'),0.,0.]])
        with self.assertRaisesRegex(ValueError,'NONFINITE_GATE_LOGITS'):
            choose_once(Invalid(True),torch.zeros(1,3584),torch.tensor([[0.,2.,0.,0.]]),torch.tensor([[0.,0.,3.,0.]]),list(range(4)),None)


def asset_checks(root):
    u.verify_sources(root/'train');audit=u.read(root/'SPLIT_AUDIT.json');rows=u.read(root/'train/DATA_MANIFEST.json')['episodes']
    assert len(rows)==800 and len({r['route_family'] for r in rows})==400
    families={}
    for r in rows:families.setdefault(r['route_family'],[]).append(r)
    for group in families.values():
        assert len(group)==2 and len({r['partition'] for r in group})==1
        assert sorted(len(r['forced_prefix']) for r in group)==[0,8]
    assert not set(audit['fit_houses'])&set(audit['dev_houses']) and not audit['unseen_overlap']
    unseen=u.read(root/'UNSEEN_MANIFEST.json')['episodes'];assert len(unseen)==200 and all(not r.get('forced_prefix') for r in unseen)
    assert 'EU6Fwq7SyZv' not in {r['house'] for r in unseen}
    return dict(new_route_families=400,new_conditions=800,unseen_conditions=200,source_hashes_verified=True,
        train_dev_house_disjoint=True,train_unseen_house_disjoint=True,unseen_forced_actions=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);a=p.parse_args();torch.set_num_threads(4)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    assets=asset_checks(a.run) if a.run and r.wasSuccessful() else None
    u.write(HERE/'CPU_TEST_RESULT.json',dict(status='PASSED' if r.wasSuccessful() else 'FAILED',tests=r.testsRun,
        failures=[str(x) for x in r.failures],errors=[str(x) for x in r.errors],assets=assets,
        synthetic_test_labels_only=True,not_model_efficacy=True))
    raise SystemExit(0 if r.wasSuccessful() else 1)
