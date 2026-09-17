"""CPU-only deterministic enumeration, fail-closed budgets and family accounting.

No simulator, filesystem writer or model is imported. A runtime caller must provide
a durable ``persist`` callback before backend calls; the callback must atomically
commit and fsync the complete snapshot, and must never silently discard errors.
"""

import copy
import hashlib
import json
import math
import time


class ProtocolError(ValueError):
    pass


class BudgetExceeded(RuntimeError):
    """Resource censoring, never a physical/scientific negative label."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _position(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ProtocolError("position must have exactly three coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ProtocolError("position must be finite numeric xyz")
    return tuple(float(v) for v in value)


def select_candidates(records, stage="P0"):
    """Validate the registered route-rank/house order; return 5 or all 20 seeds.

    This does not authorize P1 execution or assert candidate constructability.
    The caller verifies the sealed input manifest hash before supplying records.
    """
    if stage not in ("P0", "ALL") or len(records) != 20:
        raise ProtocolError("expected the frozen 20-seed table and P0 or ALL")
    houses = [r["house_id"] for r in records[:5]]
    if len(set(houses)) != 5:
        raise ProtocolError("first batch must cover five distinct houses")
    routes = {h: [] for h in houses}
    for i, row in enumerate(records):
        if row["house_id"] != houses[i % 5] or row["route_rank"] != i // 5:
            raise ProtocolError("candidate reordering or house/rank substitution")
        if row["candidate_id"] != "MP5_%02d" % i or row["frozen_house_group"] != "FIT_PILOT":
            raise ProtocolError("unregistered candidate identity or split")
        expected_stage = "P0" if i < 5 else "P1_PROPOSED"
        if row["stage"] != expected_stage:
            raise ProtocolError("stage mismatch")
        sha = row["source_physical_route_sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ProtocolError("invalid route SHA256")
        routes[row["house_id"]].append(sha)
    if any(v != sorted(set(v)) for v in routes.values()):
        raise ProtocolError("routes must be distinct and ascending within house")
    return copy.deepcopy(records[:5] if stage == "P0" else records)


def enumerate_configurations(start_position, reference_path):
    """Exact source-coordinate uniqueness (no unregistered quantization)."""
    positions = []
    for value in [start_position] + list(reference_path):
        pos = _position(value)
        if pos not in positions:
            positions.append(pos)
        if len(positions) == 8:
            break
    out = []
    for tail in ("LRLRLRLR", "FFFFFFFF"):
        for yaw in (0, 12, 6, 18):
            for index, pos in enumerate(positions):
                out.append({"index": len(out), "u_index": index, "u_position": list(pos),
                            "yaw_bin": yaw, "public_tail": tail})
    assert len(out) <= 64
    return out


def cache_key(house_id, asset_config, roles, extra=None):
    """Role identities remain meaningful here: aliases cannot cross-pollute cache."""
    if not isinstance(house_id, str) or not house_id or not asset_config or not roles:
        raise ProtocolError("house, asset/config provenance and roles required")
    return digest({"version": 2, "house": house_id, "asset_config": asset_config,
                   "roles": roles, "extra": extra})


DEFAULT_LIMITS = {"total_actions": 200000, "total_seconds": 6000,
                  "discovery_actions": 20000, "discovery_seconds": 720,
                  "certification_actions": 20000, "certification_seconds": 480}


class BudgetLedger:
    """Conservative pre-call reservations; failed/unknown actions stay charged.

    Monotonic timestamps include downtime on same-host restart. A clock rollback
    (including reboot) fails closed: explicit reviewed recovery is then required.
    Only one active bundle/phase is permitted; do not share without external lock.
    ``persist=None`` is ONLY for CPU fixtures, not durable runtime safety.
    """

    def __init__(self, limits=None, state=None, clock=time.monotonic, persist=None):
        self.clock, self.persist, self.poisoned = clock, persist, False
        self.limits = dict(DEFAULT_LIMITS if limits is None else limits)
        if set(self.limits) != set(DEFAULT_LIMITS):
            raise ProtocolError("budget fields mismatch")
        for key, value in self.limits.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ProtocolError("positive finite budgets required")
            if key.endswith("actions") and not isinstance(value, int):
                raise ProtocolError("action budgets must be integers")
        now = self._raw_now()
        if state is None:
            self.state = {"version": 2, "limits": copy.deepcopy(self.limits), "created": now,
                          "last_clock": now, "total_reserved_actions": 0,
                          "active": None, "bundles": {}}
        else:
            self.state = copy.deepcopy(state)
            if self.state.get("version") != 2 or self.state.get("limits") != self.limits:
                raise ProtocolError("cannot reset or alter resumed budget")
            if now < self.state["last_clock"]:
                raise ProtocolError("monotonic clock rollback; resume denied")
            self._validate_state()
        self._commit()

    def _raw_now(self):
        now = self.clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ProtocolError("invalid monotonic clock")
        return now

    def _validate_state(self):
        created, last = self.state["created"], self.state["last_clock"]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (created, last)) or created > last:
            raise ProtocolError("invalid resumed clock bounds")
        total = 0
        active_phases = []
        for bundle, phases in self.state["bundles"].items():
            for name, phase in phases.items():
                if name not in ("discovery", "certification") or type(phase["actions"]) is not int or phase["actions"] < 0:
                    raise ProtocolError("invalid resumed phase ledger")
                if phase["actions"] > self.limits[name + "_actions"]:
                    raise ProtocolError("resumed phase already exceeds budget")
                started, finished = phase["started"], phase["finished"]
                if not isinstance(started, (int, float)) or not math.isfinite(started) or not created <= started <= last:
                    raise ProtocolError("invalid phase start")
                if finished is None:
                    active_phases.append([bundle, name])
                elif not isinstance(finished, (int, float)) or not math.isfinite(finished) or not started <= finished <= last:
                    raise ProtocolError("invalid phase finish")
                total += phase["actions"]
        if type(self.state["total_reserved_actions"]) is not int or total != self.state["total_reserved_actions"] or total > self.limits["total_actions"]:
            raise ProtocolError("corrupt action total")
        expected_active = [] if self.state["active"] is None else [self.state["active"]]
        if active_phases != expected_active:
            raise ProtocolError("corrupt active phase")

    def _commit(self):
        if self.poisoned:
            raise ProtocolError("durable persistence previously failed; reload reviewed state")
        if self.persist is not None:
            try:
                self.persist(self.snapshot())
            except Exception:
                self.poisoned = True
                raise

    def snapshot(self):
        return copy.deepcopy(self.state)

    def check_time(self):
        if self.poisoned:
            raise ProtocolError("persistence failed")
        now = self._raw_now()
        if now < self.state["last_clock"]:
            raise ProtocolError("monotonic clock rollback")
        self.state["last_clock"] = now
        self._commit()
        if now - self.state["created"] >= self.limits["total_seconds"]:
            raise BudgetExceeded("total wall-time budget: resource_censored")
        active = self.state["active"]
        if active is not None:
            bundle, name = active
            if now - self.state["bundles"][bundle][name]["started"] >= self.limits[name + "_seconds"]:
                raise BudgetExceeded(name + " wall-time budget: resource_censored")
        return now

    def start_bundle(self, bundle_id, phase="discovery"):
        if not isinstance(bundle_id, str) or not bundle_id or phase not in ("discovery", "certification"):
            raise ProtocolError("invalid bundle/phase")
        now = self.check_time()
        wanted = [bundle_id, phase]
        if self.state["active"] == wanted:
            return  # Crash retry: same original start/counters, never a reset.
        if self.state["active"] is not None:
            raise ProtocolError("finish active phase before starting another")
        phases = self.state["bundles"].setdefault(bundle_id, {})
        if phase in phases:
            raise ProtocolError("finished phase cannot restart/reset")
        phases[phase] = {"started": now, "finished": None, "actions": 0}
        self.state["active"] = wanted
        self._commit()

    def reserve_action(self, count=1):
        self.check_time()
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ProtocolError("positive integer action reservation required")
        if self.state["active"] is None:
            raise ProtocolError("no active bundle")
        bundle, name = self.state["active"]
        phase = self.state["bundles"][bundle][name]
        if phase["actions"] + count > self.limits[name + "_actions"]:
            raise BudgetExceeded(name + " action budget: resource_censored")
        if self.state["total_reserved_actions"] + count > self.limits["total_actions"]:
            raise BudgetExceeded("total action budget: resource_censored")
        phase["actions"] += count
        self.state["total_reserved_actions"] += count
        self._commit()  # Must complete BEFORE a physical backend call.
        return self.state["total_reserved_actions"]

    def finish_phase(self):
        # Closing an exhausted phase is permitted; it does not grant a reset.
        now = self._raw_now()
        if now < self.state["last_clock"]:
            raise ProtocolError("monotonic clock rollback")
        self.state["last_clock"] = now
        if self.state["active"] is None:
            raise ProtocolError("no active phase")
        bundle, name = self.state["active"]
        self.state["bundles"][bundle][name]["finished"] = now
        self.state["active"] = None
        self._commit()


class FreezeLedger:
    """First accepted preflight is immutable; certification rejection is final."""

    def __init__(self, records=None, persist=None):
        self.records = copy.deepcopy(records or {})
        self.persist = persist
        self.poisoned = False

    def _commit(self):
        if self.poisoned:
            raise ProtocolError("freeze persistence previously failed")
        if self.persist:
            try:
                self.persist(copy.deepcopy(self.records))
            except Exception:
                self.poisoned = True
                raise

    def start(self, bundle_id):
        if self.poisoned:
            raise ProtocolError("freeze persistence failed")
        if bundle_id in self.records:
            raise ProtocolError("bundle already started; use linked resume")
        self.records[bundle_id] = {"status": "discovering", "attempt": 1, "events": []}
        self._commit()

    def resume(self, bundle_id, reason):
        rec = self.records[bundle_id]
        if reason != "crash" or rec["status"] not in ("discovering", "frozen"):
            raise ProtocolError("only linked crash recovery of unfinished bundle allowed")
        rec["attempt"] += 1
        rec["events"].append({"event": "crash_retry", "attempt": rec["attempt"]})
        self._commit()

    def freeze(self, bundle_id, candidate):
        rec = self.records[bundle_id]
        if rec["status"] != "discovering":
            raise ProtocolError("cannot replace first frozen candidate")
        frozen = copy.deepcopy(candidate)
        value = digest(frozen)
        rec.update(status="frozen", candidate=frozen, candidate_sha256=value)
        rec["events"].append({"event": "preflight_frozen", "candidate_sha256": value})
        self._commit()
        return value

    def certify(self, bundle_id, passed, reason=""):
        rec = self.records[bundle_id]
        if type(passed) is not bool or rec["status"] != "frozen":
            raise ProtocolError("certification requires frozen candidate and Boolean decision")
        rec["status"] = "certified" if passed else "certification_rejected"
        rec["events"].append({"event": rec["status"], "reason": reason})
        self._commit()

    def reject_discovery(self, bundle_id, reason, resource_censored=False):
        rec = self.records[bundle_id]
        if rec["status"] != "discovering":
            raise ProtocolError("cannot reopen frozen or closed bundle")
        rec["status"] = "resource_censored" if resource_censored else "discovery_rejected"
        rec["events"].append({"event": rec["status"], "reason": reason})
        self._commit()


def exact_duplicate_key(record):
    """Only canonical physical content enters this key, not arbitrary labels.

    record fields: house_id, asset_config, merge_state, witnessed_instances,
    histories (mapping alias -> trajectory), continuations (same).
    Trajectories must already be canonical physical state/action sequences.
    """
    return digest({"house": record["house_id"], "asset_config": record["asset_config"],
                   "merge_state": record["merge_state"],
                   "witnessed_instances": sorted(set(map(str, record["witnessed_instances"]))),
                   "histories": sorted(set(canonical(v) for v in record["histories"].values())),
                   "continuations": sorted(set(canonical(v) for v in record["continuations"].values()))})


def geometry_clusters(records):
    """Transitive connected components, never cross houses (Habitat Y is up).

    Requires merge_position xyz, witnessed_instances and occupancy_positions xyz.
    Aliases/seed IDs do not enter geometry. Empty occupancy is not evidence of
    overlap; it cannot trigger the witnessed-set+Jaccard branch.
    """
    positions = [_position(r["merge_position"]) for r in records]
    cells = [{tuple(math.floor(v / .25) for v in _position(p))
              for p in r["occupancy_positions"]} for r in records]
    witnesses = [set(map(str, r["witnessed_instances"])) for r in records]
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(records):
        for j in range(i):
            if a["house_id"] != records[j]["house_id"]:
                continue
            x, y, z = positions[i]
            xx, yy, zz = positions[j]
            nearby = math.hypot(x - xx, z - zz) < 1 and abs(y - yy) < .5
            union = cells[i] | cells[j]
            overlap = bool(witnesses[i]) and witnesses[i] == witnesses[j] and bool(union) and len(cells[i] & cells[j]) / len(union) >= .8
            if nearby or overlap:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(len(records)):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda group: group[0])
