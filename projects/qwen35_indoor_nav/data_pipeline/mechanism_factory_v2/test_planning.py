"""Synthetic CPU control-flow fixtures only; no Habitat or model execution."""
import copy
import importlib.util
from pathlib import Path
import unittest

_spec = importlib.util.spec_from_file_location("q35n_planning_test", Path(__file__).with_name("planning.py"))
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


def limits(**kw):
    out = dict(total_actions=8, total_seconds=50, discovery_actions=3,
               discovery_seconds=10, certification_actions=4, certification_seconds=20)
    out.update(kw)
    return out


def candidates():
    return [{"house_id": "house%d" % (i % 5), "route_rank": i // 5,
             "candidate_id": "MP5_%02d" % i, "frozen_house_group": "FIT_PILOT",
             "stage": "P0" if i < 5 else "P1_PROPOSED",
             "source_physical_route_sha256": "%064x" % (i // 5 + 1)} for i in range(20)]


class EnumerationTests(unittest.TestCase):
    def test_first_batch_and_all(self):
        self.assertEqual(len(p.select_candidates(candidates())), 5)
        self.assertEqual(len(p.select_candidates(candidates(), "ALL")), 20)

    def test_candidate_order_not_replaced(self):
        rows = candidates()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(p.ProtocolError):
            p.select_candidates(rows)

    def test_duplicate_route_rejected(self):
        rows = candidates()
        rows[5]["source_physical_route_sha256"] = rows[0]["source_physical_route_sha256"]
        with self.assertRaises(p.ProtocolError):
            p.select_candidates(rows)

    def test_no_reserved_split(self):
        rows = candidates()
        rows[7]["frozen_house_group"] = "RESERVED"
        with self.assertRaises(p.ProtocolError):
            p.select_candidates(rows)

    def test_exact_order_cap64(self):
        configs = p.enumerate_configurations([0, 0, 0], [[i, 0, 0] for i in range(20)])
        self.assertEqual(len(configs), 64)
        self.assertEqual([r["u_index"] for r in configs[:8]], list(range(8)))
        self.assertEqual(configs[8]["yaw_bin"], 12)
        self.assertEqual(configs[32]["public_tail"], "FFFFFFFF")
        self.assertEqual(configs[-1]["u_position"], [7.0, 0.0, 0.0])

    def test_fewer_positions_never_padded(self):
        configs = p.enumerate_configurations([0, 0, 0], [[0, 0, 0], [1, 0, 0]])
        self.assertEqual(len(configs), 16)

    def test_nonfinite_position_rejected(self):
        for value in ([float("nan"), 0, 0], [False, 0, 0], [0, 0]):
            with self.assertRaises(p.ProtocolError):
                p.enumerate_configurations(value, [])

    def test_cache_isolation_and_canonical_order(self):
        self.assertEqual(p.cache_key("h", {"a": 1, "b": 2}, {"r": [1]}),
                         p.cache_key("h", {"b": 2, "a": 1}, {"r": [1]}))
        base = p.cache_key("h", {"a": 1}, {"r": [1]})
        for key in (p.cache_key("g", {"a": 1}, {"r": [1]}),
                    p.cache_key("h", {"a": 2}, {"r": [1]}),
                    p.cache_key("h", {"a": 1}, {"r": [2]})):
            self.assertNotEqual(base, key)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.ledger = p.BudgetLedger(limits(), clock=self.clock)
        self.ledger.start_bundle("one")

    def test_action_exact_limit_and_no_refund(self):
        for _ in range(3):
            self.ledger.reserve_action()
        with self.assertRaises(p.BudgetExceeded):
            self.ledger.reserve_action()
        self.assertEqual(self.ledger.snapshot()["total_reserved_actions"], 3)

    def test_backend_failure_stays_charged(self):
        try:
            self.ledger.reserve_action()
            raise OSError("unknown whether simulated backend acted")
        except OSError:
            pass
        self.assertEqual(self.ledger.snapshot()["total_reserved_actions"], 1)

    def test_crash_resume_does_not_reset(self):
        self.ledger.reserve_action(2)
        state = self.ledger.snapshot()
        self.clock.value += 3
        resumed = p.BudgetLedger(limits(), state=state, clock=self.clock)
        resumed.start_bundle("one")
        self.assertEqual(resumed.snapshot()["bundles"]["one"]["discovery"]["started"], 100)
        self.assertEqual(resumed.snapshot()["total_reserved_actions"], 2)
        resumed.reserve_action()
        with self.assertRaises(p.BudgetExceeded):
            resumed.reserve_action()

    def test_time_exact_boundary(self):
        self.clock.value += 10
        with self.assertRaises(p.BudgetExceeded):
            self.ledger.reserve_action()
        self.assertEqual(self.ledger.snapshot()["total_reserved_actions"], 0)

    def test_finished_phase_cannot_restart(self):
        self.ledger.finish_phase()
        with self.assertRaises(p.ProtocolError):
            self.ledger.start_bundle("one")

    def test_total_limit_across_bundles(self):
        self.ledger.reserve_action(3)
        self.ledger.finish_phase()
        self.ledger.start_bundle("one", "certification")
        self.ledger.reserve_action(4)
        self.ledger.finish_phase()
        self.ledger.start_bundle("two")
        self.ledger.reserve_action()
        with self.assertRaises(p.BudgetExceeded):
            self.ledger.reserve_action()

    def test_total_walltime_includes_downtime(self):
        self.ledger.finish_phase()
        state = self.ledger.snapshot()
        self.clock.value += 51
        resumed = p.BudgetLedger(limits(), state=state, clock=self.clock)
        with self.assertRaises(p.BudgetExceeded):
            resumed.start_bundle("two")

    def test_clock_rollback_rejected(self):
        self.clock.value -= 1
        with self.assertRaises(p.ProtocolError):
            self.ledger.check_time()
        with self.assertRaises(p.ProtocolError):
            p.BudgetLedger(limits(), state=self.ledger.snapshot(), clock=self.clock)

    def test_mutated_resume_limits_and_corrupt_total(self):
        with self.assertRaises(p.ProtocolError):
            p.BudgetLedger(limits(total_actions=100), state=self.ledger.snapshot(), clock=self.clock)
        state = self.ledger.snapshot()
        state["total_reserved_actions"] = 2
        with self.assertRaises(p.ProtocolError):
            p.BudgetLedger(limits(), state=state, clock=self.clock)

    def test_invalid_action_count(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(p.ProtocolError):
                self.ledger.reserve_action(value)

    def test_durable_reservation_precedes_backend(self):
        written = []
        ledger = p.BudgetLedger(limits(), clock=self.clock, persist=lambda s: written.append(s))
        ledger.start_bundle("one")
        ledger.reserve_action()
        self.assertEqual(written[-1]["total_reserved_actions"], 1)
        self.assertEqual(ledger.snapshot(), written[-1])

    def test_persist_failure_poisoned(self):
        def fail(state):
            raise OSError("disk full")
        self.ledger.persist = fail
        with self.assertRaises(OSError):
            self.ledger.reserve_action()
        self.ledger.persist = None
        with self.assertRaises(p.ProtocolError):
            self.ledger.reserve_action()

    def test_close_expired_phase_without_reset(self):
        self.clock.value += 12
        self.ledger.finish_phase()
        with self.assertRaises(p.ProtocolError):
            self.ledger.start_bundle("one")
        self.ledger.start_bundle("two")


class FreezeTests(unittest.TestCase):
    def test_first_preflight_frozen_immutable(self):
        ledger = p.FreezeLedger()
        ledger.start("a")
        candidate = {"history": [1, 2]}
        ledger.freeze("a", candidate)
        candidate["history"].append(3)
        self.assertEqual(ledger.records["a"]["candidate"]["history"], [1, 2])
        with self.assertRaises(p.ProtocolError):
            ledger.freeze("a", {"history": [9]})

    def test_certification_fail_never_replaced(self):
        ledger = p.FreezeLedger()
        ledger.start("a")
        ledger.freeze("a", {"x": 1})
        ledger.certify("a", False, "actual replay mismatch")
        for operation in (lambda: ledger.freeze("a", {"x": 2}),
                          lambda: ledger.start("a"), lambda: ledger.resume("a", "crash")):
            with self.assertRaises(p.ProtocolError):
                operation()

    def test_linked_crash_preserves_frozen_candidate(self):
        ledger = p.FreezeLedger()
        ledger.start("a")
        key = ledger.freeze("a", {"x": 1})
        ledger.resume("a", "crash")
        self.assertEqual(ledger.records["a"]["candidate_sha256"], key)
        self.assertEqual(ledger.records["a"]["attempt"], 2)
        ledger.certify("a", True)

    def test_censor_is_not_negative(self):
        ledger = p.FreezeLedger()
        ledger.start("a")
        ledger.reject_discovery("a", "action cap", resource_censored=True)
        self.assertEqual(ledger.records["a"]["status"], "resource_censored")


def geometry(house="h", x=0, y=0, z=0, witnesses=(1,), occupancy=()):
    return {"house_id": house, "merge_position": [x, y, z],
            "witnessed_instances": list(witnesses), "occupancy_positions": list(occupancy)}


class GeometryTests(unittest.TestCase):
    def test_transitive_components(self):
        self.assertEqual(p.geometry_clusters([geometry(x=x) for x in (0, .9, 1.8, 4)]), [[0, 1, 2], [3]])

    def test_house_and_floor_separation(self):
        self.assertEqual(p.geometry_clusters([geometry(), geometry(house="g"), geometry(y=.5)]), [[0], [1], [2]])

    def test_exact_horizontal_boundary(self):
        self.assertEqual(p.geometry_clusters([geometry(), geometry(x=1)]), [[0], [1]])

    def test_overlap_connects_distant_merge(self):
        points = [[i, 0, 0] for i in range(5)]
        a = geometry(occupancy=points)
        b = geometry(x=20, occupancy=points[:4])
        self.assertEqual(p.geometry_clusters([a, b]), [[0, 1]])
        b["witnessed_instances"] = [2]
        self.assertEqual(p.geometry_clusters([a, b]), [[0], [1]])

    def test_empty_overlap_not_evidence(self):
        self.assertEqual(p.geometry_clusters([geometry(), geometry(x=4)]), [[0], [1]])

    def test_duplicates_ignore_alias_seed_wording(self):
        a = {"house_id": "h", "asset_config": {"mesh_sha": "a"}, "merge_state": [1, 2, 3],
             "witnessed_instances": [2, 1], "histories": {"H_A": [[0, "L"], [1, "R"]]},
             "continuations": {"C": [[0, "F"]]}, "seed": 1, "instruction": "first"}
        b = copy.deepcopy(a)
        b.update(seed=2, instruction="different", family_id="renamed")
        b["histories"] = {"swapped_alias": a["histories"]["H_A"]}
        b["witnessed_instances"] = [1, 2]
        self.assertEqual(p.exact_duplicate_key(a), p.exact_duplicate_key(b))
        b["house_id"] = "another"
        self.assertNotEqual(p.exact_duplicate_key(a), p.exact_duplicate_key(b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
