"""Independent desired-safety tests. Fake children only; no GPU/process launch."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'queue.py'
SOURCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('independent_queue_safety', SOURCE)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class Safety(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=HERE, prefix='cpu_')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        # Isolate all locks/receipts/logs; only the source is read outside here.
        self.here_patch = patch.object(q, 'HERE', self.base)
        self.here_patch.start()
        self.addCleanup(self.here_patch.stop)
        q.write_new(self.base / 'queue.py', {'fixture': True})
        self.job = dict(id='cpu_job', gpu=2, max_seconds=100, transport_upper_seconds=80,
            command=[str(q.ROOT / '.tools/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3'),
                     '-I', '-B', str(SOURCE)],
            completion=[dict(path=str(self.base/'receipt.json'), equals={'closed': True})],
            input_hashes={str(SOURCE): q.digest(SOURCE)})
        self.plan = dict(training_allowed=False, automatic_retry=False,
                         wall_seconds=200, jobs=[self.job])

    def prepared(self):
        path=self.base/'PLAN.json'
        q.write_new(path,self.plan)
        q.write_new(self.base/'MAIN_AGENT_APPROVAL.json',dict(approved=True,
            plan_sha256=q.digest(path),queue_source_sha256=q.digest(self.base/'queue.py')))
        return path

    def test_reject_path_escape_job_id(self):
        self.job['id']='../sibling'
        with self.assertRaises(ValueError): q.validate(self.plan)

    def test_reject_unbounded_audit_timeout(self):
        self.job['audit_command']=self.job['command']
        self.job['audit_seconds']=None
        with self.assertRaises(ValueError): q.validate(self.plan)

    def test_invalid_audit_cannot_consume_total_reservation(self):
        self.job['audit_command']=self.job['command']
        self.job['audit_seconds']=150
        with self.assertRaises(ValueError): q.validate(self.plan)

    def test_transport_plus_audit_must_fit_reservation(self):
        self.job['audit_command']=self.job['command']
        self.job['audit_seconds']=30
        with self.assertRaises(ValueError): q.validate(self.plan)

    def test_valid_transport_plus_audit_boundary(self):
        self.job['audit_command']=self.job['command']
        self.job['audit_seconds']=20
        q.validate(self.plan)

    def test_post_launch_log_failure_still_drains_child(self):
        path=self.prepared(); waits=[]; kwargs_seen=[]
        class Child:
            pid=12345
            def __init__(self,*args,**kwargs): kwargs_seen.append(kwargs)
            def wait(self): waits.append('wait'); return 0
            def poll(self): return None if not waits else 0
        original=q.append
        def append(path,value):
            if value.get('event')=='child_started': raise OSError('injected fsync failure')
            return original(path,value)
        with patch.object(q.subprocess,'Popen',Child),patch.object(q.signal,'signal'),patch.object(q,'append',append):
            with self.assertRaises(SystemExit): q.execute(path,2)
        self.assertTrue(waits,'child must drain before mutex release after logging failure')
        self.assertTrue(kwargs_seen[0].get('start_new_session'),'child must not inherit terminal signal group')
        inherited=kwargs_seen[0].get('pass_fds',())
        self.assertEqual(len(inherited),1,'child must inherit the mutex to cover parent hard death')
        self.assertIs(type(inherited[0]),int)

    def test_drain_during_input_verification_does_not_launch(self):
        path=self.prepared(); handlers={}; launched=[]
        original=q.verify_job
        def verify(job):
            original(job)
            handlers[q.signal.SIGTERM](q.signal.SIGTERM,None)
        def handler(number,callback): handlers[number]=callback
        class Child:
            pid=12345
            def __init__(self,*args,**kwargs):
                launched.append(True)
                q.write_new(self_receipt,{'closed':True})
            def wait(self): return 0
            def poll(self): return 0
        self_receipt=self.base/'receipt.json'
        with patch.object(q.subprocess,'Popen',Child),patch.object(q.signal,'signal',handler),patch.object(q,'verify_job',verify):
            q.execute(path,2)
        self.assertEqual(launched,[],'drain arrived before physical child creation')

    def test_hash_verification_elapsed_time_rechecked_before_launch(self):
        path=self.prepared(); clock=[0]; launched=[]
        original=q.verify_job
        def verify(job):
            original(job)
            clock[0]=150
        class Child:
            pid=12345
            def __init__(self,*args,**kwargs):
                launched.append(True)
                q.write_new(self_receipt,{'closed':True})
            def wait(self): return 0
            def poll(self): return 0
        self_receipt=self.base/'receipt.json'
        with patch.object(q.subprocess,'Popen',Child),patch.object(q.signal,'signal'),patch.object(q,'verify_job',verify),patch.object(q.time,'monotonic',lambda:clock[0]):
            q.execute(path,2)
        self.assertEqual(launched,[],'hash work consumed the available transport reservation')


if __name__=='__main__':
    print('READ_ONLY_REVIEW_SOURCE_SHA256='+SOURCE_SHA,flush=True)
    unittest.main(verbosity=2)
