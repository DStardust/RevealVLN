import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('production_queue_tested', HERE/'queue.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='test_queue_', dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.script = HERE/'queue.py'
        self.command = [str(q.ROOT/'projects/qwen35_indoor_nav/.envs/q35n_habitat_v017_g0r/bin/python3'), '-I', '-B', str(self.script)]
        self.receipt = self.base/'receipt.json'
        self.job = dict(id='batch_test', gpu=2, command=self.command, max_seconds=100, transport_upper_seconds=100,
            completion=[dict(path=str(self.receipt), equals={'error':None, 'restoration.restored':True})],
            input_hashes={str(self.script):q.digest(self.script)})
        self.plan = dict(training_allowed=False, automatic_retry=False, wall_seconds=200, jobs=[self.job])

    def test_valid(self): q.validate(self.plan)
    def test_no_gpu0(self):
        self.job['gpu']=0
        with self.assertRaisesRegex(ValueError,'GPU0'): q.validate(self.plan)
    def test_duplicate_job(self):
        self.plan['jobs'].append(copy.deepcopy(self.job))
        with self.assertRaisesRegex(ValueError,'UNIQUE'):q.validate(self.plan)
    def test_no_training(self):
        self.plan['training_allowed']=True
        with self.assertRaisesRegex(ValueError,'DATA_ONLY'):q.validate(self.plan)
    def test_no_retry(self):
        self.plan['automatic_retry']=True
        with self.assertRaisesRegex(ValueError,'DATA_ONLY'):q.validate(self.plan)
    def test_audit_included_in_reserved_time(self):
        self.job['audit_command']=self.command
        self.job['audit_seconds']=50
        with self.assertRaisesRegex(ValueError,'PLUS_AUDIT'):q.validate(self.plan)
    def test_missing_transport_bound(self):
        del self.job['transport_upper_seconds']
        with self.assertRaisesRegex(ValueError,'UPPER_BOUND'):q.validate(self.plan)
    def test_no_shell(self):
        self.job['command']=['bash','-c','echo forbidden']
        with self.assertRaises(ValueError):q.validate(self.plan)
    def test_no_other_project(self):
        with self.assertRaisesRegex(ValueError,'PATH'):q.scoped(q.ROOT.parent/'elsewhere.py')
    def test_symlink_rejected(self):
        alias=self.base/'alias';alias.symlink_to(self.script)
        with self.assertRaisesRegex(ValueError,'PATH'):q.scoped(alias)
    def test_entry_must_be_frozen(self):
        self.job['input_hashes']={str(HERE/'test_queue.py'):'unused'}
        with self.assertRaisesRegex(ValueError,'ENTRY'):q.validate(self.plan)
    def test_changed_source_rejected(self):
        self.job['input_hashes'][str(self.script)]='0'*64
        with self.assertRaisesRegex(ValueError,'CHANGED'):q.verify_job(self.job)
    def test_no_replay_completed(self):
        q.write_new(self.receipt,{'error':None,'restoration':{'restored':True}})
        with self.assertRaisesRegex(ValueError,'OLD_ATTEMPT'):q.verify_job(self.job)
    def test_receipt_exact_booleans(self):
        q.write_new(self.receipt,{'error':None,'restoration':{'restored':1}})
        with self.assertRaisesRegex(ValueError,'COMPLETION'):q.receipts(self.job['completion'])
    def test_receipt_success(self):
        q.write_new(self.receipt,{'error':None,'restoration':{'restored':True}})
        self.assertEqual(q.receipts(self.job['completion']),{str(self.receipt):q.digest(self.receipt)})
    def test_no_overwrite(self):
        q.write_new(self.receipt, {})
        with self.assertRaises(FileExistsError):q.write_new(self.receipt, {})
    def test_deadline_and_drain(self):
        self.assertTrue(q.ready_to_start(self.job,50,200,False))
        self.assertFalse(q.ready_to_start(self.job,101,200,False))
        self.assertFalse(q.ready_to_start(self.job,0,200,True))
    def test_queue_advances_two_jobs(self):
        second=copy.deepcopy(self.job);second['id']='batch_two'
        second['completion'][0]['path']=str(self.base/'receipt2.json')
        self.plan['jobs'].append(second)
        plan_path=self.base/'PLAN.json';q.write_new(plan_path,self.plan)
        q.write_new(self.base/'MAIN_AGENT_APPROVAL.json',dict(approved=True,
            plan_sha256=q.digest(plan_path),queue_source_sha256=q.digest(self.script)))
        calls=[]
        class Child:
            pid=12345
            def __init__(self,*args,**kwargs):
                job=self.plan['jobs'][len(calls)]
                calls.append(job['id'])
                q.write_new(Path(job['completion'][0]['path']),{'error':None,'restoration':{'restored':True}})
            def wait(self):return 0
        Child.plan=self.plan
        with patch.object(q.subprocess,'Popen',Child),patch.object(q.signal,'signal'):
            q.execute(plan_path,2)
        result=q.read(self.base/'lane_gpu_2/RESULT.json')
        self.assertEqual(calls,['batch_test','batch_two'])
        self.assertIsNone(result['error'])
        self.assertEqual(set(result['states'].values()),{'PRODUCED_PENDING_AUDIT'})
    def test_failure_does_not_start_next(self):
        second=copy.deepcopy(self.job);second['id']='batch_two';second['completion'][0]['path']=str(self.base/'receipt2.json')
        self.plan['jobs'].append(second)
        plan_path=self.base/'PLAN.json';q.write_new(plan_path,self.plan)
        q.write_new(self.base/'MAIN_AGENT_APPROVAL.json',dict(approved=True,
            plan_sha256=q.digest(plan_path),queue_source_sha256=q.digest(self.script)))
        class Failed:
            pid=12345
            def __init__(self,*args,**kwargs):pass
            def wait(self):return 1
        with patch.object(q.subprocess,'Popen',Failed) as popen,patch.object(q.signal,'signal'):
            with self.assertRaises(SystemExit):q.execute(plan_path,2)
        result=q.read(self.base/'lane_gpu_2/RESULT.json')
        self.assertEqual(result['states']['batch_test'],'FAILED_NO_AUTOMATIC_RETRY')
        self.assertEqual(result['states']['batch_two'],'PENDING')
        self.assertFalse((self.base/'batch_two_RESERVED.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
