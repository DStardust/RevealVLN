"""Opt-in CPU proposal: defer only successful clock-only durable snapshots.

Every call reads/checks the clock. Each action reservation remains synchronously
durable before return. No asynchronous writes, background flushing, state schema
change, action refund, phase reset, or automatic runtime integration.
"""
import hashlib
import importlib.util
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parents[1]
SOURCE=RUNTIME/'feedback_diagnosis_v1/budget_optimization_cpu/budget.py'
SOURCE_SHA256='fd85570ea3a0cf98adbd887005bf9e1d9f6ee3a60214adb16ce44dc8b29169e7'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA256:
    raise ImportError('sealed SingleCommitBudgetLedger changed')
spec=importlib.util.spec_from_file_location('clock_batching_single_commit',SOURCE)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
SingleCommitBudgetLedger=parent.SingleCommitBudgetLedger
BudgetExceeded,ProtocolError=parent.BudgetExceeded,parent.ProtocolError


class ClockBatchingBudgetLedger(SingleCommitBudgetLedger):
    """Same in-memory snapshots, deliberately fewer durable clock watermarks.

After a successful time-only check, snapshot() is NOT a durable snapshot. A crash
may lose only those recent clock watermark updates; created, phase started and
reserved actions stay at the prior durable values. A reviewed same-boot resume
must use the real committed Journal state, never the pending in-memory snapshot.

This does not preserve the old ability to reject every hypothetical rollback
into the lost-watermark interval after a crash. Same-process rollback is checked
against every observed clock value. Cross-boot/unreviewed resume is not authorized.
"""
    def __init__(self,limits=None,state=None,clock=time.monotonic,persist=None):
        self._clock_dirty=False
        self.clock_checks_since_commit=0
        self._last_durable_state=None
        super().__init__(limits=limits,state=state,clock=clock,persist=persist)

    def _commit(self):
        super()._commit()  # BaseException -> poisoned, no successful return.
        self._last_durable_state=self.snapshot()
        self._clock_dirty=False
        self.clock_checks_since_commit=0

    def durable_snapshot(self):
        import copy
        return copy.deepcopy(self._last_durable_state)

    @property
    def pending_clock_only(self):return self._clock_dirty

    def check_time(self):
        if self.poisoned:raise ProtocolError('persistence failed')
        if not callable(self.persist):
            self.poisoned=True
            raise ProtocolError('durable callback removed; ledger poisoned')
        now=self._raw_now()
        if now<self.state['last_clock']:raise ProtocolError('monotonic clock rollback')
        self.state['last_clock']=now
        error=None
        if now-self.state['created']>=self.limits['total_seconds']:
            error=BudgetExceeded('total wall-time budget: resource_censored')
        else:
            active=self.state['active']
            if active is not None:
                bundle,name=active
                if now-self.state['bundles'][bundle][name]['started']>=self.limits[name+'_seconds']:
                    error=BudgetExceeded(name+' wall-time budget: resource_censored')
        if error is not None:
            self._commit()  # Error state durably acknowledged BEFORE rejection.
            raise error
        self._clock_dirty=True
        self.clock_checks_since_commit+=1
        return now

    def start_bundle(self,bundle_id,phase='discovery'):
        # The original operation uses check_time then writes any new phase. The
        # new phase commit naturally includes the pending clock. Same-phase retry
        # has no structural commit, so explicitly commit at that boundary too.
        try:result=super().start_bundle(bundle_id,phase)
        except (ProtocolError,BudgetExceeded):
            if not self.poisoned and self._clock_dirty:self._commit()
            raise
        if self._clock_dirty:self._commit()
        return result

    def finish_phase(self):
        if self.poisoned:raise ProtocolError('persistence failed')
        # Original behavior permits closing an exhausted phase for diagnostics.
        # It never grants new time/action budgets or resets the original start.
        return super().finish_phase()
