"""Synthetic CPU audit contract tests, not navigation-effect evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
CASE=HERE.parents[1]/'closed_loop_bench/ordinary_visual_stall_guard_v1'
def load(name):
    s=importlib.util.spec_from_file_location('review_'+name,CASE/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
guard=load('guard');audit=load('guard_audit')
RGB=bytes(224*224*3)
def write_rows(path,rows):
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='synthetic_guard_audit_',dir=HERE)
        self.run=Path(self.tmp.name);self.folder=self.run/'lanes/lane_00';self.folder.mkdir(parents=True)
        self.p=dict(lanes=1,episode_count=2,max_steps=500)
        self.policies=[];self.steps=[]
        for i in range(2):
            controller=guard.VisualStallGuard();controller.observe_rgb(RGB)
            for st in range(1,51):
                proposed='STOP' if st==50 else 'move_forward';receipt=controller.choose(proposed)
                logits=[1.,0.,0.,0.] if proposed!='STOP' else [0.,0.,0.,1.]
                self.policies.append(dict(index=i,step=st,logits=logits,model_proposed_action=proposed,guard=receipt,action=receipt['action']))
                self.steps.append(dict(index=i,step=st,action=receipt['action'],rgb_sha256=hashlib.sha256(RGB).hexdigest()))
                if proposed!='STOP':controller.observe_rgb(RGB,receipt['action'])
        write_rows(self.folder/'INTERFACE.jsonl',[dict(index=i,rgb_sha256=hashlib.sha256(RGB).hexdigest()) for i in range(2)])

    def tearDown(self):self.tmp.cleanup()
    def run_audit(self):
        write_rows(self.folder/'POLICY_STEPS.jsonl',self.policies)
        write_rows(self.folder/'STEPS_PRIVILEGED.jsonl',self.steps)
        return audit.audit(self.run,self.p)

    def test_independent_replay_and_cap(self):
        result=self.run_audit();self.assertTrue(result['passed'])
        self.assertEqual(result['total_actions'],100);self.assertEqual(result['total_interventions'],8)
        self.assertEqual(result['intervened_episodes'],2)

    def test_wrong_proposal(self):
        self.policies[0]['model_proposed_action']='STOP'
        with self.assertRaises(AssertionError):self.run_audit()

    def test_wrong_actual_execution(self):
        self.steps[8]['action']='move_forward'
        with self.assertRaises(AssertionError):self.run_audit()

    def test_wrong_observed_hash(self):
        self.policies[8]['guard']['rgb_sha256']='not-the-observed-frame'
        with self.assertRaises(AssertionError):self.run_audit()

    def test_wrong_guard_count(self):
        self.policies[8]['guard']['interventions_after']=2
        with self.assertRaises(AssertionError):self.run_audit()

    def test_missing_policy_step(self):
        self.policies.pop()
        with self.assertRaises(AssertionError):self.run_audit()

    def test_missing_physical_step(self):
        self.steps.pop()
        with self.assertRaises(AssertionError):self.run_audit()

    def test_nonfinite_logits(self):
        self.policies[0]['logits'][0]=float('nan')
        with self.assertRaises(AssertionError):self.run_audit()

    def test_extra_budget_not_allowed(self):
        self.p['max_steps']=49
        with self.assertRaises(AssertionError):self.run_audit()


if __name__=='__main__':unittest.main()
