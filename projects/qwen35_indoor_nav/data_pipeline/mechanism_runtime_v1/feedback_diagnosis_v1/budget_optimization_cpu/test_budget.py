"""Memory-only CPU tests: no real journal, fsync benchmark, simulator, or GPU."""
import copy
import importlib.util
from pathlib import Path
import unittest

if __package__:
    from . import budget
else:
    spec=importlib.util.spec_from_file_location("single_commit_budget_cpu",Path(__file__).resolve().with_name("budget.py"))
    budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)


class Clock:
    def __init__(self,value=100.0):self.value=value
    def __call__(self):return self.value


def limits(**changes):
    value={"total_actions":8,"total_seconds":50,"discovery_actions":3,"discovery_seconds":10,
           "certification_actions":4,"certification_seconds":20}
    value.update(changes)
    return value


class SingleCommitTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.old_records=[];self.new_records=[]
        self.old=budget.OriginalBudgetLedger(limits(),clock=self.clock,persist=self.old_records.append)
        self.new=budget.SingleCommitBudgetLedger(limits(),clock=self.clock,persist=self.new_records.append)

    def pair(self,method,*args):
        def invoke(instance):
            try:return ("return",getattr(instance,method)(*args))
            except Exception as exc:return ("raise",type(exc),str(exc))
        old=invoke(self.old);new=invoke(self.new)
        self.assertEqual(old,new)
        self.assertEqual(self.old.snapshot(),self.new.snapshot())
        self.assertEqual(self.old_records[-1],self.new_records[-1])
        return new

    def test_normal_calls_and_one_commit_saved_each(self):
        self.pair("start_bundle","a")
        for count in (1,2):
            self.clock.value+=.1
            a,b=len(self.old_records),len(self.new_records)
            self.pair("reserve_action",count)
            self.assertEqual(len(self.old_records)-a,2)
            self.assertEqual(len(self.new_records)-b,1)

    def test_action_limit_reject_retains_clock_only_commit(self):
        self.pair("start_bundle","a");self.pair("reserve_action",3)
        self.clock.value+=1
        a,b=len(self.old_records),len(self.new_records)
        self.assertEqual(self.pair("reserve_action")[1],budget.BudgetExceeded)
        self.assertEqual(len(self.old_records)-a,1);self.assertEqual(len(self.new_records)-b,1)

    def test_total_action_limit_across_phases(self):
        self.pair("start_bundle","a");self.pair("reserve_action",3);self.pair("finish_phase")
        self.pair("start_bundle","a","certification");self.pair("reserve_action",4);self.pair("finish_phase")
        self.pair("start_bundle","b");self.pair("reserve_action")
        self.assertEqual(self.pair("reserve_action")[1],budget.BudgetExceeded)

    def test_discovery_exact_time_boundary(self):
        self.pair("start_bundle","a");self.clock.value+=10
        self.assertEqual(self.pair("reserve_action")[1],budget.BudgetExceeded)

    def test_total_exact_time_boundary(self):
        self.clock.value+=50
        self.assertEqual(self.pair("reserve_action")[1],budget.BudgetExceeded)

    def test_certification_time_boundary(self):
        self.pair("start_bundle","a","certification");self.clock.value+=20
        self.assertEqual(self.pair("reserve_action")[1],budget.BudgetExceeded)

    def test_invalid_counts_equivalent_and_no_reservation(self):
        self.pair("start_bundle","a")
        for count in (0,-1,True,1.5,"2",None):
            self.clock.value+=.1
            with self.subTest(count=count):
                self.assertEqual(self.pair("reserve_action",count)[1],budget.ProtocolError)
        self.assertEqual(self.new.snapshot()["total_reserved_actions"],0)

    def test_no_active_bundle_rejection(self):
        self.clock.value+=1
        self.assertEqual(self.pair("reserve_action")[1],budget.ProtocolError)

    def test_clock_rollback_does_not_write(self):
        self.pair("start_bundle","a");self.clock.value-=1
        a,b=len(self.old_records),len(self.new_records)
        self.assertEqual(self.pair("reserve_action")[1],budget.ProtocolError)
        self.assertEqual((len(self.old_records),len(self.new_records)),(a,b))

    def test_nonfinite_clock_does_not_write(self):
        self.pair("start_bundle","a")
        for value in (float("nan"),float("inf"),True):
            self.clock.value=value
            self.assertEqual(self.pair("reserve_action")[1],budget.ProtocolError)

    def test_resume_same_start_and_counts(self):
        self.pair("start_bundle","a");self.pair("reserve_action",2);state=self.new.snapshot()
        self.clock.value+=3
        self.old=budget.OriginalBudgetLedger(limits(),state=state,clock=self.clock,persist=self.old_records.append)
        self.new=budget.SingleCommitBudgetLedger(limits(),state=state,clock=self.clock,persist=self.new_records.append)
        self.pair("start_bundle","a");self.pair("reserve_action")
        self.assertEqual(self.new.snapshot()["bundles"]["a"]["discovery"]["started"],100)
        self.assertEqual(self.pair("reserve_action")[1],budget.BudgetExceeded)

    def test_no_restarting_finished_phase(self):
        self.pair("start_bundle","a");self.pair("finish_phase")
        self.assertEqual(self.pair("start_bundle","a")[1],budget.ProtocolError)

    def test_expired_phase_may_close_without_reset(self):
        self.pair("start_bundle","a");self.clock.value+=11
        self.pair("finish_phase")
        self.assertEqual(self.new.snapshot()["total_reserved_actions"],0)

    def test_persist_ack_precedes_backend(self):
        self.new.start_bundle("a");events=[]
        def persist(state):events.append(("durable_ack",copy.deepcopy(state)))
        self.new.persist=persist
        def backend(action):
            self.assertEqual(events[-1][0],"durable_ack")
            self.assertEqual(events[-1][1]["total_reserved_actions"],1)
            self.assertEqual(events[-1][1],self.new.snapshot())
            events.append(("backend",action));return "ok"
        self.assertEqual(self.new.step_after_durable_reservation(backend,"F"),"ok")
        self.assertEqual([x[0] for x in events],["durable_ack","backend"])

    def test_persist_failure_before_or_after_possible_commit_poisoned(self):
        for failure_point in ("before_write","event_written","event_synced","head_written","head_renamed","directory_synced"):
            with self.subTest(failure_point=failure_point):
                ledger=budget.SingleCommitBudgetLedger(limits(),clock=self.clock,persist=lambda s:None)
                ledger.start_bundle("a");calls=[];potentially_durable=[]
                def fail(state):
                    if failure_point!="before_write":potentially_durable.append(copy.deepcopy(state))
                    raise OSError(failure_point)
                ledger.persist=fail
                with self.assertRaises(OSError):ledger.step_after_durable_reservation(lambda:calls.append(1))
                self.assertEqual(calls,[]);self.assertTrue(ledger.poisoned)
                self.assertEqual(ledger.snapshot()["total_reserved_actions"],1)
                ledger.persist=lambda s:None
                with self.assertRaises(budget.ProtocolError):ledger.step_after_durable_reservation(lambda:calls.append(1))
                self.assertEqual(calls,[])
                if potentially_durable:self.assertEqual(potentially_durable[-1]["total_reserved_actions"],1)

    def test_baseexception_during_persist_also_poisons(self):
        self.new.start_bundle("a")
        def interrupt(state):raise KeyboardInterrupt()
        self.new.persist=interrupt
        with self.assertRaises(KeyboardInterrupt):self.new.reserve_action()
        self.assertTrue(self.new.poisoned)

    def test_rejection_persist_failure_dominates_and_no_backend(self):
        self.new.start_bundle("a");self.new.reserve_action(3)
        def fail(state):raise OSError("reject snapshot cannot be committed")
        self.new.persist=fail;calls=[]
        with self.assertRaises(OSError):self.new.step_after_durable_reservation(lambda:calls.append(1))
        self.assertEqual(calls,[]);self.assertTrue(self.new.poisoned)
        self.assertEqual(self.new.snapshot()["total_reserved_actions"],3)

    def test_backend_exception_stays_charged_without_refund(self):
        self.new.start_bundle("a")
        def backend():raise OSError("action completion unknown")
        with self.assertRaises(OSError):self.new.step_after_durable_reservation(backend)
        self.assertEqual(self.new.snapshot()["total_reserved_actions"],1)
        self.assertEqual(self.new_records[-1]["total_reserved_actions"],1)

    def test_missing_or_removed_callback_forbidden(self):
        with self.assertRaises(budget.ProtocolError):budget.SingleCommitBudgetLedger(limits(),clock=self.clock)
        self.new.start_bundle("a");self.new.persist=None
        with self.assertRaises(budget.ProtocolError):self.new.reserve_action()
        self.assertTrue(self.new.poisoned)

    def test_callback_gets_copy_not_mutable_live_state(self):
        self.new.start_bundle("a")
        def persist(state):state["total_reserved_actions"]=-500
        self.new.persist=persist;self.new.reserve_action()
        self.assertEqual(self.new.snapshot()["total_reserved_actions"],1)

    def test_resume_changed_limits_and_corrupt_total_rejected(self):
        state=self.new.snapshot();state["total_reserved_actions"]=1
        with self.assertRaises(budget.ProtocolError):
            budget.SingleCommitBudgetLedger(limits(),state=state,clock=self.clock,persist=lambda s:None)
        with self.assertRaises(budget.ProtocolError):
            budget.SingleCommitBudgetLedger(limits(total_actions=10),state=self.new.snapshot(),clock=self.clock,persist=lambda s:None)


if __name__=="__main__":unittest.main()
