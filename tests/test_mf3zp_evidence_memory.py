import unittest

from revealnav_mf3.evidence_memory import EvidenceItem, EvidenceMemory, EvidenceStatus, reject_forbidden_evidence_mapping


class EvidenceMemoryTest(unittest.TestCase):
    def item(self):
        return EvidenceItem("e1", "c1", 1, (0, 0, 10, 10), "B1", .8, 1, 1, EvidenceStatus.RESOLVED, "1"*64, "2"*64)

    def test_memory_and_staleness(self):
        memory = EvidenceMemory([self.item()])
        self.assertEqual(memory.resolved_constraints(), ("c1",))
        memory.mark_stale(2)
        self.assertEqual(memory.items()[0].status, EvidenceStatus.STALE)

    def test_conflict_and_outcome_rejected(self):
        memory = EvidenceMemory([self.item()])
        with self.assertRaises(ValueError):
            memory.add(EvidenceItem("e1", "c1", 1, None, None, .1, 1, 1, "OBSERVED", "1"*64, "2"*64))
        for key in ("reward", "delta_utility", "future_pose", "oracle_state"):
            with self.assertRaises(ValueError):
                reject_forbidden_evidence_mapping({key: 1})


if __name__ == "__main__":
    unittest.main()
