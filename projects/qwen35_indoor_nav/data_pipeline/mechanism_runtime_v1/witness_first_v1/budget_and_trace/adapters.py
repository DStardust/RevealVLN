"""Opt-in CPU-tested adapters; never modify sealed classes or active workers."""
import copy
import hashlib
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]

def load_exact(name,path,manifest):
    entries={line.split(None,1)[1]:line.split(None,1)[0] for line in manifest.read_text().splitlines()}
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entries[path.name]:
        raise ImportError("sealed adapter dependency hash mismatch: "+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

bridge=load_exact("witness_readonly_core",RUNTIME/"core_bridge.py",RUNTIME/"SHA256SUMS")
journal_module=load_exact("witness_readonly_journal",RUNTIME/"runtime_journal.py",RUNTIME/"SHA256SUMS")
budget_root=RUNTIME/"feedback_diagnosis_v1/budget_optimization_cpu"
budget_module=load_exact("witness_single_commit_budget",budget_root/"budget.py",budget_root/"SHA256SUMS")
Journal=journal_module.Journal
JournalError=journal_module.JournalError
SingleCommitBudgetLedger=budget_module.SingleCommitBudgetLedger
BudgetExceeded=bridge.BudgetExceeded


def durable_budget(journal,limits,*,clock,state=None):
    """Bind to the original real Journal; no asynchronous/buffering callback."""
    if not isinstance(journal,Journal):
        raise TypeError("durable_budget requires this adapter's verified Journal")
    return SingleCommitBudgetLedger(limits,state=state,clock=clock,
                                    persist=lambda snapshot:journal.append("budget",snapshot))


class _CapturingBackend:
    """Instance-local forwarding only; no global backend or factory patches."""
    def __init__(self,backend):self.delegate=backend;self.clear()
    def clear(self):
        self.confirmed=[];self.returns=[];self.reset_completed=False;self.unknown_attempts=[]
    def __getattr__(self,name):return getattr(self.delegate,name)
    def reset(self,*args,**kwargs):
        result=self.delegate.reset(*args,**kwargs);self.reset_completed=True;return result
    def observe(self):
        result=self.delegate.observe()
        self.returns.append({"return_index":len(self.returns),"confirmed_action_count":len(self.confirmed),
                             "raw":copy.deepcopy(result),"validated":False})
        return result
    def step(self,action):
        try:result=self.delegate.step(action)
        except BaseException as error:
            self.unknown_attempts.append({"action":action,"error":repr(error)});raise
        if type(result) is bool:
            self.confirmed.append({"action":action,"collision":result})
        else:
            self.unknown_attempts.append({"action":action,"error":"invalid collision return type"})
        return result


class PartialTraceRunner(bridge.TraceRunner):
    """Original normal output exactly; diagnostic partial trace on budget cutoff.

    Callers must pass a durable emit sink for runtime use. The original
    BudgetExceeded is re-raised AFTER emit succeeds, so resource-stop control flow
    is unchanged. If emit fails, its exception propagates with BudgetExceeded as
    context; no partial record is claimed durable and execution must stop.

    Partial observations are actual validated returns only; numerical joins can
    produce repeated step indices. Raw returns interrupted before validation are
    retained separately and never promoted to training evidence. Partial traces
    always have complete=false/training_admission=false and must not be compiled
    as ordinary complete trajectories.
    """
    def __init__(self,backend,compiler,budget,emit):
        self.capture=_CapturingBackend(backend)
        super().__init__(self.capture,compiler,budget,emit)
        self.last_partial=None
        self.partial_emit_acknowledged=False
        self._validated=[]

    def observe(self,step):
        record=super().observe(step)
        self.capture.returns[-1]["validated"]=True
        self._validated.append(copy.deepcopy(record))
        return record

    def run(self,position,yaw,actions,seed=1109,join=None):
        self.capture.clear();self._validated=[];self.last_partial=None;self.partial_emit_acknowledged=False
        before=self.budget.snapshot()["total_reserved_actions"]
        try:
            return super().run(position,yaw,actions,seed=seed,join=join)
        except BudgetExceeded as error:
            confirmed=[record["action"] for record in self.capture.confirmed]
            snapshot=self.budget.snapshot()
            partial={"actions":confirmed,"observations":copy.deepcopy(self._validated),
                     "complete":False,"training_admission":False,"diagnostic_only":True,
                     "failure_reason":str(error),"failure_type":"BudgetExceeded",
                     "collisions":sum(record["collision"] for record in self.capture.confirmed),
                     "seed":seed,"initial_position":list(position),"initial_yaw_bin":yaw,
                     "initial_state_confirmed_by_reset":self.capture.reset_completed,
                     "normalization_events":[],"normalization_events_complete":False,
                     "observation_semantics":"actual validated chronological returns; repeated steps possible at joins",
                     "raw_observation_returns":copy.deepcopy(self.capture.returns),
                     "unknown_backend_attempts":copy.deepcopy(self.capture.unknown_attempts),
                     "confirmed_action_count":len(confirmed),
                     "reserved_actions_this_trace":snapshot["total_reserved_actions"]-before,
                     "budget_snapshot":snapshot}
            partial["trace_hash"]=bridge.digest(partial)
            self.last_partial=copy.deepcopy(partial)
            self.emit("trace",partial)
            self.partial_emit_acknowledged=True
            raise
