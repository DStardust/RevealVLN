import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import readiness_v1 as r


class Clock:
    def __init__(self):self.t=0.
    def __call__(self):return self.t
    def sleep(self,value):self.t+=value


def sample(util):return {'utilization':util,'memory_mib':753,'processes':{1:{'mib':220}},'uuid':'test'}


class ReadinessTests(unittest.TestCase):
    def helper(self,values,check=None):
        clock=Clock();events=[];queries=[];normal=[]
        values=list(values)
        def query(timeout):
            queries.append(timeout)
            return values.pop(0) if len(values)>1 else values[0]
        obj=r.FirstIdleQuery(lambda:normal.append(True) or sample(99),query,
            check or (lambda s:533),events.append,clock=clock,sleeper=clock.sleep)
        return obj,clock,events,queries,normal

    def test_busy_to_actual_idle(self):
        idle=sample(0);obj,clock,events,queries,normal=self.helper([sample(70),sample(50),idle])
        self.assertIs(obj(),idle);self.assertEqual(clock.t,2)
        self.assertEqual([e['snapshot']['utilization'] for e in events],[70,50,0])
        self.assertEqual(normal,[]);self.assertEqual(len(queries),3)

    def test_immediate_idle_no_sleep(self):
        idle=sample(0);obj,clock,events,queries,_=self.helper([idle])
        self.assertIs(obj(),idle);self.assertEqual(clock.t,0);self.assertEqual(len(events),1)

    def test_later_calls_original_without_readiness(self):
        obj,clock,events,queries,normal=self.helper([sample(0)])
        obj();later=obj()
        self.assertEqual(later['utilization'],99);self.assertEqual(len(normal),1)
        self.assertEqual(len(queries),1);self.assertEqual(len(events),1)

    def test_busy_timeout(self):
        obj,clock,events,queries,normal=self.helper([sample(40)])
        with self.assertRaises(TimeoutError):obj()
        self.assertEqual(clock.t,60);self.assertEqual(events[-1]['event'],'readiness_timeout')
        self.assertFalse(obj.completed);self.assertEqual(normal,[])

    def test_failed_wait_cannot_restart_budget(self):
        obj,clock,_,_,_=self.helper([sample(40)])
        with self.assertRaises(TimeoutError):obj()
        with self.assertRaises(RuntimeError):obj()
        self.assertEqual(clock.t,60)

    def test_external_resource_guard_rejection(self):
        def guard(snapshot):raise AssertionError('EXTERNAL_RESOURCE_LOAD')
        obj,clock,events,queries,_=self.helper([sample(0)],guard)
        with self.assertRaisesRegex(AssertionError,'EXTERNAL_RESOURCE_LOAD'):obj()
        self.assertEqual(clock.t,0);self.assertEqual(len(queries),1)
        self.assertEqual(events[0]['event'],'readiness_query_or_guard_error')

    def test_query_error_propagated(self):
        obj,_,events,_,_=self.helper([sample(0)])
        def error(timeout):raise OSError('GPU_QUERY_FAILED')
        obj.initial_query=error
        with self.assertRaises(OSError):obj()
        self.assertIsNone(events[0]['snapshot'])

    def test_remaining_time_bounds_query_timeout(self):
        clock=Clock();timeouts=[];events=[]
        def query(timeout):
            timeouts.append(timeout);clock.t+=min(14.8,timeout);return sample(50)
        obj=r.FirstIdleQuery(lambda:sample(0),query,lambda s:0,events.append,clock=clock,sleeper=clock.sleep)
        with self.assertRaises(TimeoutError):obj()
        self.assertLess(timeouts[-1],15);self.assertEqual(clock.t,60)

    def test_late_idle_is_not_accepted(self):
        obj,clock,events,_,_=self.helper([sample(0)])
        def query(timeout):clock.t=60.;return sample(0)
        obj.initial_query=query
        with self.assertRaises(TimeoutError):obj()
        self.assertFalse(obj.completed)

    def test_no_snapshot_mutation(self):
        value=sample(0);before=json.dumps(value,sort_keys=True)
        obj,_,_,_,_=self.helper([value]);obj()
        self.assertEqual(json.dumps(value,sort_keys=True),before)

    def test_source_builder_uses_original_guards_and_sources_unchanged(self):
        paths=[r.HERE/'shared.py',r.RUNTIME/'compact_loop_v2/run.py']
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        cfg={'gpu_device':2,'gpu_uuid':'GPU-be1b30d0-517b-b079-871b-de195d35a1a2',
             'supervision_wall_seconds':3900,'budget':{'total_seconds':3600}}
        module=r.build_supervisor(r.HERE/'batch_999',cfg)
        self.assertIn('GPU_NOT_IDLE',module.main.__code__.co_consts)
        self.assertIn('EXTERNAL_RESOURCE_LOAD',module.check_gpu.__code__.co_consts)
        self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def temporary_batch(self,root):
        batch=root/'batch_999r1';out=batch/'run_v1';out.mkdir(parents=True)
        (out/'EXECUTION_CONFIG.json').write_text(json.dumps({'runtime_allowed':True,'executable':True,'training_allowed':False,'gpu_device':2}))
        return batch,out

    def test_outer_preworker_failure_receipt(self):
        with tempfile.TemporaryDirectory(dir=r.HERE,prefix='readiness_cpu_') as td:
            root=Path(td);batch,out=self.temporary_batch(root)
            def build(batch,cfg):raise RuntimeError('PREWORKER_TEST_ERROR')
            with patch.object(r,'HERE',root):
                with self.assertRaisesRegex(RuntimeError,'PREWORKER_TEST_ERROR'):r.run_main(batch,module_builder=build)
            receipt=json.loads((out/'LAUNCH_RESULT.json').read_text())
            self.assertEqual(receipt['exception']['message'],'PREWORKER_TEST_ERROR')
            self.assertFalse(receipt['process_record_present']);self.assertFalse(receipt['first_real_idle_observed'])

    def test_outer_receipt_after_real_query_simulation(self):
        with tempfile.TemporaryDirectory(dir=r.HERE,prefix='readiness_cpu_') as td:
            root=Path(td);batch,out=self.temporary_batch(root);clock=Clock();normal_calls=[]
            module=types.SimpleNamespace(gpu=lambda:normal_calls.append(True) or sample(99),
                subprocess=types.SimpleNamespace(check_output=lambda *a,**kw:json.dumps(sample(0))),
                parse_gpu=json.loads,check_gpu=lambda x:533)
            def main():module.gpu();module.gpu();raise RuntimeError('AFTER_IDLE_BEFORE_WORKER')
            module.main=main
            with patch.object(r,'HERE',root):
                with self.assertRaises(RuntimeError):r.run_main(batch,module_builder=lambda b,c:module,clock=clock,sleeper=clock.sleep)
            receipt=json.loads((out/'LAUNCH_RESULT.json').read_text())
            self.assertTrue(receipt['first_real_idle_observed']);self.assertEqual(normal_calls,[True])
            self.assertEqual(receipt['external_processes_stopped_by_launcher'],0)

    def test_existing_attempt_rejected_without_receipt_overwrite(self):
        with tempfile.TemporaryDirectory(dir=r.HERE,prefix='readiness_cpu_') as td:
            root=Path(td);batch,out=self.temporary_batch(root);(out/'SUPERVISOR.lock').write_text('old')
            with patch.object(r,'HERE',root):
                with self.assertRaises(ValueError):r.run_main(batch)
            self.assertEqual((out/'SUPERVISOR.lock').read_text(),'old')
            self.assertFalse((out/'LAUNCH_RESULT.json').exists())


if __name__=='__main__':unittest.main()
