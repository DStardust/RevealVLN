import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tested_clock_ledger',HERE/'ledger.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
LIMITS=dict(total_actions=8,total_seconds=20,discovery_actions=4,discovery_seconds=5,
            certification_actions=4,certification_seconds=8)

class Clock:
    def __init__(self,value=0.):self.value=value;self.calls=0
    def __call__(self):
        self.calls+=1
        if isinstance(self.value,BaseException):raise self.value
        return self.value

def fixture(cls,limits=None):
    clock=Clock();saved=[]
    obj=cls(limits or LIMITS,clock=clock,persist=lambda s:saved.append(copy.deepcopy(s)))
    return obj,clock,saved

class Tests(unittest.TestCase):
    def pair(self):return [fixture(c) for c in (m.SingleCommitBudgetLedger,m.ClockBatchingBudgetLedger)]
    def test_success_snapshots_differential(self):
        rows=self.pair()
        operations=[(1.,'start_bundle',('f','discovery')),(1.1,'check_time',()),(1.2,'check_time',()),
            (1.3,'reserve_action',()),(1.4,'check_time',()),(1.5,'check_time',()),(1.6,'reserve_action',()),
            (1.7,'finish_phase',()),(2.,'start_bundle',('f','certification')),(2.1,'check_time',()),
            (2.2,'reserve_action',()),(2.3,'finish_phase',())]
        for tick,name,args in operations:
            values=[]
            for ledger,clock,saved in rows:clock.value=tick;values.append(getattr(ledger,name)(*args))
            self.assertEqual(values[0],values[1]);self.assertEqual(rows[0][0].snapshot(),rows[1][0].snapshot())
        self.assertEqual(rows[0][2][-1],rows[1][2][-1]);self.assertLess(len(rows[1][2]),len(rows[0][2]))
        self.assertEqual(rows[0][1].calls,rows[1][1].calls)
    def test_success_check_is_pending_not_durable(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);clock.value=2.;before=len(saved)
        self.assertEqual(ledger.check_time(),2.);self.assertEqual(len(saved),before)
        self.assertTrue(ledger.pending_clock_only);self.assertEqual(ledger.durable_snapshot()['last_clock'],0.)
        self.assertEqual(ledger.snapshot()['last_clock'],2.)
    def test_phase_start_combines_clock_and_structure(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);clock.value=1.;ledger.check_time();before=len(saved)
        ledger.start_bundle('f');self.assertEqual(len(saved)-before,1);self.assertFalse(ledger.pending_clock_only)
        self.assertEqual(saved[-1],ledger.snapshot())
    def test_same_phase_retry_flushes_without_reset(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);ledger.start_bundle('f');ledger.reserve_action()
        clock.value=1.;ledger.check_time();before=len(saved);ledger.start_bundle('f')
        self.assertEqual(len(saved)-before,1);self.assertEqual(saved[-1]['bundles']['f']['discovery']['started'],0.)
        self.assertEqual(saved[-1]['total_reserved_actions'],1)
    def test_total_time_exact_boundary_persist_before_raise(self):
        for cls in (m.SingleCommitBudgetLedger,m.ClockBatchingBudgetLedger):
            ledger,clock,saved=fixture(cls);clock.value=20.
            with self.assertRaises(m.BudgetExceeded):ledger.check_time()
            self.assertEqual(saved[-1]['last_clock'],20.)
    def test_phase_time_exact_boundary_persist_before_raise(self):
        for cls in (m.SingleCommitBudgetLedger,m.ClockBatchingBudgetLedger):
            ledger,clock,saved=fixture(cls);clock.value=1.;ledger.start_bundle('f');clock.value=6.
            with self.assertRaises(m.BudgetExceeded):ledger.check_time()
            self.assertEqual(saved[-1]['last_clock'],6.);self.assertEqual(saved[-1]['total_reserved_actions'],0)
    def test_reserve_time_denial_is_durable(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);ledger.start_bundle('f');clock.value=5.
        with self.assertRaises(m.BudgetExceeded):ledger.reserve_action()
        self.assertEqual(saved[-1]['last_clock'],5.);self.assertEqual(saved[-1]['total_reserved_actions'],0)
    def test_action_limit_no_refund(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);ledger.start_bundle('f');ledger.reserve_action(4)
        with self.assertRaises(m.BudgetExceeded):ledger.reserve_action()
        self.assertEqual(saved[-1]['total_reserved_actions'],4)
    def test_invalid_count_clock_snapshot_still_durable(self):
        for count in (0,-1,True,1.5):
            ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);ledger.start_bundle('f');clock.value=1.
            with self.assertRaises(m.ProtocolError):ledger.reserve_action(count)
            self.assertEqual(saved[-1]['last_clock'],1.);self.assertEqual(saved[-1]['total_reserved_actions'],0)
    def test_no_active_reservation_rejected(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);clock.value=1.
        with self.assertRaises(m.ProtocolError):ledger.reserve_action()
        self.assertEqual(saved[-1]['last_clock'],1.)
    def test_phase_conflict_flushes_pending_error_clock(self):
        rows=self.pair()
        for ledger,clock,saved in rows:
            ledger.start_bundle('f');clock.value=1.
            with self.assertRaises(m.ProtocolError):ledger.start_bundle('g')
        self.assertEqual(rows[0][2][-1],rows[1][2][-1])
    def test_closing_exhausted_phase_still_permitted_no_restart(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);ledger.start_bundle('f');clock.value=6.
        ledger.finish_phase();self.assertEqual(saved[-1]['bundles']['f']['discovery']['finished'],6.)
        with self.assertRaises(m.ProtocolError):ledger.start_bundle('f')
    def test_live_rollback_against_pending_watermark(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);clock.value=3.;ledger.check_time();clock.value=2.
        with self.assertRaises(m.ProtocolError):ledger.check_time()
        self.assertEqual(ledger.snapshot()['last_clock'],3.)
    def test_invalid_clock_and_clock_baseexception_do_not_commit(self):
        for value in (True,float('nan'),KeyboardInterrupt()):
            ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);before=len(saved);clock.value=value
            with self.assertRaises(BaseException):ledger.check_time()
            self.assertEqual(len(saved),before);self.assertEqual(ledger.snapshot()['last_clock'],0.)
    def test_persist_baseexceptions_poison_no_backend(self):
        for exc in (OSError('disk'),KeyboardInterrupt(),SystemExit(3)):
            ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);ledger.start_bundle('f');calls=[]
            def fail(s):raise exc
            ledger.persist=fail
            with self.assertRaises(BaseException):ledger.step_after_durable_reservation(lambda:calls.append(1))
            self.assertEqual(calls,[]);self.assertTrue(ledger.poisoned)
            self.assertEqual(ledger.snapshot()['total_reserved_actions'],1)
            ledger.persist=lambda s:None
            for op in (ledger.reserve_action,ledger.check_time,ledger.finish_phase):
                with self.assertRaises(m.ProtocolError):op()
    def test_time_limit_persist_error_takes_precedence(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);clock.value=20.
        def fail(s):raise OSError('no durable time-limit record')
        ledger.persist=fail
        with self.assertRaises(OSError):ledger.check_time()
        self.assertTrue(ledger.poisoned)
    def test_backend_error_preserves_reservation(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);ledger.start_bundle('f')
        def fail():raise RuntimeError('backend')
        with self.assertRaises(RuntimeError):ledger.step_after_durable_reservation(fail)
        self.assertEqual(saved[-1]['total_reserved_actions'],1);self.assertFalse(ledger.poisoned)
    def test_resumed_durable_start_and_downtime_not_reset(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);clock.value=1.;ledger.start_bundle('f');ledger.reserve_action()
        durable=ledger.durable_snapshot();clock.value=4.;ledger.check_time()
        resumed=m.ClockBatchingBudgetLedger(LIMITS,state=durable,clock=lambda:6.,persist=lambda s:None)
        self.assertEqual(resumed.snapshot()['created'],0.);self.assertEqual(resumed.snapshot()['bundles']['f']['discovery']['started'],1.)
        with self.assertRaises(m.BudgetExceeded):resumed.start_bundle('f')
        self.assertEqual(resumed.snapshot()['total_reserved_actions'],1)
    def test_rollback_before_durable_watermark_resume_denied(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);clock.value=2.;ledger.start_bundle('f')
        with self.assertRaises(m.ProtocolError):m.ClockBatchingBudgetLedger(LIMITS,state=ledger.durable_snapshot(),clock=lambda:1.,persist=lambda s:None)
    def test_lost_watermark_detection_difference_is_explicit(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);clock.value=1.;ledger.start_bundle('f');durable=ledger.durable_snapshot()
        clock.value=3.;ledger.check_time()
        # This counterexample is documented, not hidden: reboot/rollback into
        # an unpersisted watermark gap cannot be detected from old state alone.
        resumed=m.ClockBatchingBudgetLedger(LIMITS,state=durable,clock=lambda:2.,persist=lambda s:None)
        self.assertEqual(resumed.snapshot()['bundles']['f']['discovery']['started'],1.)
    def test_no_callback_disallowed(self):
        with self.assertRaises(m.ProtocolError):m.ClockBatchingBudgetLedger(LIMITS)

    def test_removed_callback_fails_even_on_time_only_check(self):
        ledger,clock,saved=fixture(m.ClockBatchingBudgetLedger);ledger.persist=None
        with self.assertRaises(m.ProtocolError):ledger.check_time()
        self.assertTrue(ledger.poisoned)

    def test_real_journal_reviewed_reload_keeps_original_start(self):
        p=m.RUNTIME/'runtime_journal.py';s=importlib.util.spec_from_file_location('clock_reload_journal',p)
        j=importlib.util.module_from_spec(s);s.loader.exec_module(j)
        with tempfile.TemporaryDirectory(prefix='cpu_reload_',dir=HERE) as temp:
            root=Path(temp)/'journal';clock=Clock()
            with j.Journal(root,{'cpu_fixture':True}) as journal:
                ledger=m.ClockBatchingBudgetLedger(LIMITS,clock=clock,persist=lambda value:journal.append('budget',value))
                clock.value=1.;ledger.start_bundle('f');clock.value=2.;ledger.reserve_action()
                clock.value=3.;ledger.check_time();self.assertTrue(ledger.pending_clock_only)
            # Simulate loss of Python memory, not physical power failure. No phase
            # close/checkpoint is inserted to conceal the pending clock watermark.
            with j.Journal(root,{'cpu_fixture':True},resume=True) as journal:
                durable=[r['payload'] for r in journal._records if r['kind']=='budget'][-1]
                self.assertEqual(durable['last_clock'],2.)
                clock.value=4.;ledger=m.ClockBatchingBudgetLedger(LIMITS,state=durable,clock=clock,persist=lambda value:journal.append('budget',value))
                ledger.start_bundle('f');state=ledger.snapshot()
                self.assertEqual(state['created'],0.);self.assertEqual(state['bundles']['f']['discovery']['started'],1.)
                self.assertEqual(state['total_reserved_actions'],1)
                clock.value=6.
                with self.assertRaises(m.BudgetExceeded):ledger.check_time()
                self.assertEqual(journal._records[-1]['payload']['last_clock'],6.)

    def test_real_journal_fsync_faults_poison_before_mock_action(self):
        p=m.RUNTIME/'runtime_journal.py';s=importlib.util.spec_from_file_location('clock_fault_journal',p)
        j=importlib.util.module_from_spec(s);s.loader.exec_module(j)
        for failing_fsync in (1,2,3):
            with tempfile.TemporaryDirectory(prefix='cpu_fault_',dir=HERE) as temp:
                with j.Journal(Path(temp)/'journal',{'cpu_fixture':True}) as journal:
                    ledger=m.ClockBatchingBudgetLedger(LIMITS,clock=lambda:0.,persist=lambda value:journal.append('budget',value))
                    ledger.start_bundle('f');real=j.os.fsync;counter=[];actions=[]
                    def fault(fd):
                        counter.append(fd)
                        if len(counter)==failing_fsync:raise OSError('injected actual journal fsync failure')
                        return real(fd)
                    with patch.object(j.os,'fsync',side_effect=fault):
                        with self.assertRaises(OSError):ledger.step_after_durable_reservation(lambda:actions.append(1))
                    self.assertTrue(ledger.poisoned);self.assertEqual(actions,[])
                    self.assertEqual(ledger.snapshot()['total_reserved_actions'],1)

    def test_real_journal_three_fsync_per_reserve_before_backend(self):
        p=m.RUNTIME/'runtime_journal.py';s=importlib.util.spec_from_file_location('clock_test_journal',p)
        journal_module=importlib.util.module_from_spec(s);s.loader.exec_module(journal_module)
        with tempfile.TemporaryDirectory(prefix='cpu_real_',dir=HERE) as temp:
            with journal_module.Journal(Path(temp)/'journal',{'cpu_fixture':True}) as journal:
                clock=Clock();ledger=m.ClockBatchingBudgetLedger(LIMITS,clock=clock,persist=lambda value:journal.append('budget',value))
                ledger.start_bundle('f');ledger.check_time();ledger.check_time()
                actual=journal_module.os.fsync;calls=[]
                def synced(fd):calls.append(fd);return actual(fd)
                with patch.object(journal_module.os,'fsync',side_effect=synced):
                    def backend():
                        self.assertEqual(len(calls),3)
                        head=json.loads((journal.root/'HEAD.json').read_text())
                        self.assertEqual(head['last_hash'],journal._records[-1]['hash'])
                        self.assertEqual(journal._records[-1]['payload'],ledger.snapshot())
                    ledger.step_after_durable_reservation(backend)
                ledger.finish_phase()

if __name__=='__main__':unittest.main()
