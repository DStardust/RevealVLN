"""Small real-Journal CPU integration plus fake physical backend tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import adapters as a
else:
    spec=importlib.util.spec_from_file_location("witness_adapters_test",Path(__file__).resolve().with_name("adapters.py"))
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class Clock:
    def __init__(self):self.value=100.
    def __call__(self):return self.value

def limits(actions=20):
    return {"total_actions":actions,"total_seconds":30,"discovery_actions":actions,"discovery_seconds":10,
            "certification_actions":actions,"certification_seconds":10}

class Compiler:
    def atoms(self,observations):return [{"fixture":[]} for _ in observations]

class Backend:
    def __init__(self,clock,step_advance=0,observe_advance=0,fail_step=False,collision=False):
        self.clock=clock;self.step_advance=step_advance;self.observe_advance=observe_advance
        self.fail_step=fail_step;self.collision=collision;self.calls=[];self.x=0
    def reset(self,position,yaw,seed):self.x=position[0];self.calls=[]
    def observe(self):
        self.clock.value+=self.observe_advance
        base={"position":[self.x,0,0],"rotation":[1,0,0,0]}
        pose=dict(base,sensors={"rgb":copy.deepcopy(base),"semantic":copy.deepcopy(base)})
        return {"pixels":{"1":10},"pose":pose,"rgb_hash":"a"*64,"semantic_hash":"b"*64,"evidence_complete":True}
    def step(self,action):
        self.calls.append(action);self.clock.value+=self.step_advance
        if self.fail_step:raise OSError("backend status unknown")
        if action=="F":self.x+=1
        return self.collision
    def reconstruct(self,pose):self.x=pose["position"][0]

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="journal_trace_cpu_",dir=a.HERE)
        self.root=Path(self.temp.name);self.clock=Clock();self.cfg={"cpu_fixture":True}
    def tearDown(self):self.temp.cleanup()
    def journal(self,name="journal"):
        return a.Journal(self.root/name,self.cfg)
    def budget(self,journal,n=20):
        ledger=a.durable_budget(journal,limits(n),clock=self.clock);ledger.start_bundle("fixture");return ledger

    def test_real_journal_reservation_resume_and_fsync_count(self):
        with self.journal() as journal:
            ledger=self.budget(journal)
            before=len(journal._records)
            import os
            with patch.object(a.journal_module.os,"fsync",wraps=os.fsync) as sync:
                ledger.reserve_action()
            self.assertEqual(len(journal._records)-before,1);self.assertEqual(sync.call_count,3)
            expected=ledger.snapshot()
        with a.Journal(self.root/"journal",self.cfg,resume=True) as journal:
            self.assertEqual(journal.latest("budget"),expected)
            resumed=a.durable_budget(journal,limits(),clock=self.clock,state=journal.latest("budget"))
            resumed.start_bundle("fixture");resumed.reserve_action()
            self.assertEqual(resumed.snapshot()["total_reserved_actions"],2)

    def test_durable_head_precedes_backend(self):
        with self.journal() as journal:
            ledger=self.budget(journal)
            def backend():
                head=json.loads((journal.root/"HEAD.json").read_text())
                record=json.loads((journal.root/"events.jsonl").read_text().splitlines()[-1])
                self.assertEqual(head["last_hash"],record["hash"])
                self.assertEqual(record["payload"]["total_reserved_actions"],1)
                return "called"
            self.assertEqual(ledger.step_after_durable_reservation(backend),"called")

    def test_real_event_to_head_failure_preserved_and_resume_rejected(self):
        journal=self.journal();ledger=self.budget(journal);calls=[]
        with patch.object(journal,"_commit_head",side_effect=OSError("after event fsync before HEAD")):
            with self.assertRaises(OSError):ledger.step_after_durable_reservation(lambda:calls.append(1))
        self.assertTrue(ledger.poisoned);self.assertEqual(calls,[])
        self.assertEqual(ledger.snapshot()["total_reserved_actions"],1)
        journal.close();raw=(self.root/"journal/events.jsonl").read_bytes()
        with self.assertRaises(a.JournalError):a.Journal(self.root/"journal",self.cfg,resume=True)
        self.assertEqual((self.root/"journal/events.jsonl").read_bytes(),raw)

    def test_real_fsync_failures_never_execute_backend(self):
        import os
        for nth in (1,2,3):
            with self.subTest(fsync=nth):
                journal=self.journal("sync"+str(nth));ledger=self.budget(journal);original=os.fsync;count=0;calls=[]
                def fsync(fd):
                    nonlocal count
                    count+=1
                    if count==nth:raise OSError("injected fsync failure")
                    return original(fd)
                with patch.object(a.journal_module.os,"fsync",side_effect=fsync):
                    with self.assertRaises(OSError):ledger.step_after_durable_reservation(lambda:calls.append(1))
                self.assertTrue(ledger.poisoned);self.assertEqual(calls,[])
                with self.assertRaises(a.budget_module.ProtocolError):ledger.reserve_action()
                journal.close()
                if nth in (1,2):
                    with self.assertRaises(a.JournalError):a.Journal(self.root/("sync"+str(nth)),self.cfg,resume=True)
                else:
                    # Rename reached the OS before directory-sync failure; the
                    # extant files may verify. That does not prove crash durability.
                    with a.Journal(self.root/("sync"+str(nth)),self.cfg,resume=True) as recovered:
                        self.assertEqual(recovered.latest("budget")["total_reserved_actions"],1)

    def test_budget_reject_is_durable_and_no_backend(self):
        with self.journal() as journal:
            ledger=self.budget(journal,1);ledger.reserve_action();calls=[]
            with self.assertRaises(a.BudgetExceeded):ledger.step_after_durable_reservation(lambda:calls.append(1))
            self.assertEqual(calls,[]);self.assertEqual(journal.latest("budget")["total_reserved_actions"],1)

    def test_normal_outputs_and_events_exactly_equal(self):
        for actions in ([],["F","L","R"],["F","S"],["L","R","S"]):
            with self.subTest(actions=actions):
                outputs=[]
                for idx,klass in enumerate((a.bridge.TraceRunner,a.PartialTraceRunner)):
                    with self.journal("normal"+str(len(actions))+"_"+str(idx)+"_"+str(actions[0] if actions else "empty")) as journal:
                        ledger=self.budget(journal);events=[]
                        runner=klass(Backend(self.clock),Compiler(),ledger,lambda k,v:events.append((k,copy.deepcopy(v))))
                        outputs.append((runner.run([0,0,0],0,actions),events,dict(runner.counts)))
                self.assertEqual(outputs[0],outputs[1])

    def test_normal_numerical_join_output_equal(self):
        outputs=[]
        for idx,klass in enumerate((a.bridge.TraceRunner,a.PartialTraceRunner)):
            backend=Backend(self.clock);backend.reset([0,0,0],0,1109);pose=backend.observe()["pose"]
            join={"step":2,"registered_histories":[["L","R"]],"position":[0,0,0],"yaw_bin":0,"target_pose":pose}
            with self.journal("join"+str(idx)) as journal:
                ledger=self.budget(journal);events=[]
                runner=klass(backend,Compiler(),ledger,lambda k,v:events.append((k,copy.deepcopy(v))))
                outputs.append((runner.run([0,0,0],0,["L","R","L","S"],join=join),events))
        self.assertEqual(outputs[0],outputs[1])

    def test_cutoff_after_step_keeps_confirmed_action_not_fake_next_frame(self):
        with self.journal() as journal:
            ledger=self.budget(journal);backend=Backend(self.clock,step_advance=10)
            runner=a.PartialTraceRunner(backend,Compiler(),ledger,lambda k,v:journal.append(k,v))
            with self.assertRaises(a.BudgetExceeded):runner.run([0,0,0],0,["F","F"])
            trace=journal.latest("trace")
            self.assertEqual(trace["actions"],["F"]);self.assertEqual(len(trace["observations"]),1)
            self.assertFalse(trace["complete"]);self.assertFalse(trace["training_admission"])
            self.assertEqual(trace["reserved_actions_this_trace"],1)
            self.assertEqual(ledger.snapshot()["total_reserved_actions"],1)
            self.assertTrue(runner.partial_emit_acknowledged)

    def test_cutoff_after_observe_preserves_unvalidated_return_separately(self):
        with self.journal() as journal:
            ledger=self.budget(journal);backend=Backend(self.clock,observe_advance=10)
            runner=a.PartialTraceRunner(backend,Compiler(),ledger,lambda k,v:journal.append(k,v))
            with self.assertRaises(a.BudgetExceeded):runner.run([0,0,0],0,["F"])
            trace=journal.latest("trace")
            self.assertEqual(trace["actions"],[]);self.assertEqual(trace["observations"],[])
            self.assertEqual(len(trace["raw_observation_returns"]),1)
            self.assertFalse(trace["raw_observation_returns"][0]["validated"])

    def test_action_budget_cutoff_retains_only_executed_prefix(self):
        with self.journal() as journal:
            ledger=self.budget(journal,1);runner=a.PartialTraceRunner(Backend(self.clock),Compiler(),ledger,lambda k,v:journal.append(k,v))
            with self.assertRaises(a.BudgetExceeded):runner.run([0,0,0],0,["F","F","F"])
            trace=journal.latest("trace")
            self.assertEqual(trace["actions"],["F"]);self.assertEqual(len(trace["observations"]),2)

    def test_expired_before_reset_has_no_invented_state(self):
        with self.journal() as journal:
            ledger=self.budget(journal);self.clock.value+=10
            runner=a.PartialTraceRunner(Backend(self.clock),Compiler(),ledger,lambda k,v:journal.append(k,v))
            with self.assertRaises(a.BudgetExceeded):runner.run([0,0,0],0,["F"])
            trace=journal.latest("trace")
            self.assertFalse(trace["initial_state_confirmed_by_reset"])
            self.assertEqual(trace["raw_observation_returns"],[])

    def test_partial_persistence_failure_propagates_and_never_continues(self):
        with self.journal() as journal:
            ledger=self.budget(journal,1);backend=Backend(self.clock)
            def emit(kind,value):
                if kind=="trace":raise OSError("partial trace sink failure")
            runner=a.PartialTraceRunner(backend,Compiler(),ledger,emit)
            with self.assertRaises(OSError) as exc:runner.run([0,0,0],0,["F","F"])
            self.assertIsInstance(exc.exception.__context__,a.BudgetExceeded)
            self.assertEqual(backend.calls,["F"]);self.assertEqual(ledger.snapshot()["total_reserved_actions"],1)
            self.assertFalse(runner.partial_emit_acknowledged)

    def test_backend_error_not_relabelled_or_refunded(self):
        with self.journal() as journal:
            ledger=self.budget(journal);runner=a.PartialTraceRunner(Backend(self.clock,fail_step=True),Compiler(),ledger,lambda k,v:journal.append(k,v))
            with self.assertRaises(OSError):runner.run([0,0,0],0,["F"])
            self.assertEqual(ledger.snapshot()["total_reserved_actions"],1)
            self.assertIsNone(journal.latest("trace"))

    def test_unconfirmed_backend_budget_exception_not_called_confirmed(self):
        with self.journal() as journal:
            ledger=self.budget(journal);backend=Backend(self.clock)
            runner=a.PartialTraceRunner(backend,Compiler(),ledger,lambda k,v:journal.append(k,v))
            with patch.object(backend,"step",side_effect=a.BudgetExceeded("unknown backend action status")):
                with self.assertRaises(a.BudgetExceeded):runner.run([0,0,0],0,["F"])
            trace=journal.latest("trace")
            self.assertEqual(trace["actions"],[]);self.assertEqual(trace["reserved_actions_this_trace"],1)
            self.assertEqual(len(trace["unknown_backend_attempts"]),1)

    def test_partial_capture_does_not_carry_over_between_traces(self):
        with self.journal() as journal:
            ledger=self.budget(journal,1)
            runner=a.PartialTraceRunner(Backend(self.clock),Compiler(),ledger,lambda k,v:journal.append(k,v))
            normal=runner.run([0,0,0],0,["F"])
            self.assertTrue(normal["complete"])
            with self.assertRaises(a.BudgetExceeded):runner.run([0,0,0],0,["F"])
            partial=journal.latest("trace")
            self.assertEqual(partial["actions"],[]);self.assertEqual(partial["reserved_actions_this_trace"],0)
            self.assertEqual(len(partial["observations"]),1)


if __name__=="__main__":unittest.main()
