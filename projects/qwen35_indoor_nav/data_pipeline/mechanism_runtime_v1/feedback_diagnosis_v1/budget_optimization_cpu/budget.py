"""CPU-only, opt-in single-commit reservation; NOT wired to an active runtime.

The sealed BudgetLedger is loaded from its exact, SHA-verified source. Successful
reserve_action combines its intermediate clock-only and final reserved snapshot
into one durable callback. Other operations retain the original implementation.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parents[1]
CORE = RUNTIME.parent / "mechanism_factory_v2"
SOURCE = CORE / "planning.py"
expected = {line.split(None, 1)[1]: line.split(None, 1)[0]
            for line in (CORE / "SHA256SUMS").read_text().splitlines()}
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != expected[SOURCE.name]:
    raise ImportError("sealed planning.py hash mismatch")
planning = sys.modules.get("planning")
if planning is not None:
    if Path(getattr(planning, "__file__", "")).resolve() != SOURCE.resolve():
        raise ImportError("unrelated planning module already loaded")
else:
    spec = importlib.util.spec_from_file_location("planning", SOURCE)
    planning = importlib.util.module_from_spec(spec)
    sys.modules["planning"] = planning
    spec.loader.exec_module(planning)

OriginalBudgetLedger = planning.BudgetLedger
ProtocolError, BudgetExceeded = planning.ProtocolError, planning.BudgetExceeded


class SingleCommitBudgetLedger(OriginalBudgetLedger):
    """Same stable state/schema, fewer successful reservation log records.

    ``persist(snapshot)`` must synchronously complete durable acknowledgement
    (e.g. original journal's full event + HEAD + directory fsync) before returning,
    and raise on any uncertainty. No asynchronous or memory-only production sink.
    CPU tests use explicit fake sinks, never a production ``persist=None`` path.

    Stable success/rejection snapshots match the original; byte-identical journal
    traces are intentionally not claimed. On persistence failure the reserved
    count may be conservatively greater than the old *first-commit* failure state.
    It is never refunded, the ledger is poisoned and no backend action is called.
    Reload/recovery requires separately reviewed durable state, never this object.
    """

    def __init__(self, limits=None, state=None, clock=time.monotonic, persist=None):
        if not callable(persist):
            raise ProtocolError("single-commit ledger requires an explicit synchronous persist callback")
        super().__init__(limits=limits, state=state, clock=clock, persist=persist)

    def _commit(self):
        if self.poisoned:
            raise ProtocolError("durable persistence previously failed; reload reviewed state")
        if not callable(self.persist):
            self.poisoned = True
            raise ProtocolError("durable callback removed; ledger poisoned")
        try:
            self.persist(self.snapshot())
        except BaseException:
            # Also fail closed for an interrupted durable call, not just OSError.
            self.poisoned = True
            raise

    def reserve_action(self, count=1):
        if self.poisoned:
            raise ProtocolError("persistence failed")
        now = self._raw_now()
        if now < self.state["last_clock"]:
            raise ProtocolError("monotonic clock rollback")
        self.state["last_clock"] = now
        try:
            if now - self.state["created"] >= self.limits["total_seconds"]:
                raise BudgetExceeded("total wall-time budget: resource_censored")
            active = self.state["active"]
            if active is not None:
                bundle, name = active
                if now - self.state["bundles"][bundle][name]["started"] >= self.limits[name + "_seconds"]:
                    raise BudgetExceeded(name + " wall-time budget: resource_censored")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ProtocolError("positive integer action reservation required")
            if active is None:
                raise ProtocolError("no active bundle")
            bundle, name = active
            phase = self.state["bundles"][bundle][name]
            if phase["actions"] + count > self.limits[name + "_actions"]:
                raise BudgetExceeded(name + " action budget: resource_censored")
            if self.state["total_reserved_actions"] + count > self.limits["total_actions"]:
                raise BudgetExceeded("total action budget: resource_censored")
        except (ProtocolError, BudgetExceeded):
            # Old check_time persisted this clock-only snapshot before rejecting.
            # Retain that durable state/error precedence on every expected reject.
            self._commit()
            raise
        phase["actions"] += count
        self.state["total_reserved_actions"] += count
        self._commit()  # Must acknowledge durable reservation BEFORE returning.
        return self.state["total_reserved_actions"]

    def step_after_durable_reservation(self, backend_step, *args, **kwargs):
        """Optional integration helper; exactly one charged primitive invocation.

        A backend exception leaves its reservation charged, as in the old ledger.
        This helper does not implement retry, refund, or new task/action policy.
        """
        if not callable(backend_step):
            raise ProtocolError("backend_step must be callable")
        self.reserve_action()
        return backend_step(*args, **kwargs)
