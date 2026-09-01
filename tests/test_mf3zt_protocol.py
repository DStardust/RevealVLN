import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/training/mf3zt_evidence_memory_decision_probe_v1"
PUBLIC_CLOSED = {
    "val_seen": False,
    "val_unseen": False,
    "test": False,
    "test_challenge": False,
}


class Mf3ztProtocolTest(unittest.TestCase):
    def test_protocol_freezes_the_requested_design(self):
        protocol = json.loads(
            (OUT / "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_PROTOCOL.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(protocol["status"], "SEALED_BEFORE_MF3ZT_RESULTS")
        self.assertEqual(
            protocol["source_commit"],
            "e24c4f6a62b6e86cd143e911d0dd9ae103daa209",
        )
        self.assertEqual(
            protocol["evidence"]["ontology"],
            [
                "LANDMARK_SEEN",
                "LANDMARK_PASSED",
                "RELATION_SATISFIED",
                "ORDINAL_COUNT",
                "DIRECTIONAL_CONTEXT",
            ],
        )
        self.assertEqual(protocol["evidence"]["K_MEM"], 8)
        self.assertEqual(protocol["evaluation"]["folds"], 5)
        self.assertEqual(protocol["evaluation"]["bootstrap"]["replicates"], 10_000)
        self.assertEqual(protocol["evaluation"]["bootstrap"]["seed"], 20_260_901)
        self.assertEqual(
            protocol["conditional_model"]["arms"],
            [
                "ETP_CURRENT",
                "ETP_PLUS_EVIDENCE_MEMORY",
                "ETP_PLUS_SHUFFLED_MEMORY",
            ],
        )
        self.assertTrue(protocol["frozen_ETP"]["ETP_frozen"])
        self.assertTrue(
            protocol["memory_required_definition"][
                "classified_before_target_evaluation"
            ]
        )
        self.assertIn(
            "candidate_target",
            protocol["memory_required_definition"]["forbidden_inputs"],
        )
        self.assertEqual(protocol["public_split_access"], PUBLIC_CLOSED)

    def test_protocol_is_pre_result_and_does_not_claim_a_population(self):
        protocol = json.loads(
            (OUT / "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_PROTOCOL.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            protocol["population"]["status"],
            "NOT_MATERIALIZED_PENDING_TARGET_SUPPORT_GATE",
        )
        self.assertIsNone(protocol["population"]["sha256"])
        self.assertFalse(protocol["execution"]["training_started"])
        self.assertFalse(protocol["execution"]["full_navigation_run"])
        self.assertFalse(protocol["execution"]["checkpoint_generated"])

    def test_historical_revisions_remain_byte_identical(self):
        expected = {
            "METHOD_REVISION_3ZQ_ORACLE_REVEALSKILL_HEADROOM.md": "356e4afdb81c2e92b258dfaea20197db8c84ec1aa1475fb859c7d425bf7ebbf1",
            "METHOD_REVISION_3ZR_OPTION_BOUND_SUPPORT.md": "85c876b5feeb6f60dc88c0ba4dc647530117ac78e00ac8ff22743ff0a8c689b1",
            "artifacts/training/mf3zq_oracle_revealskill_headroom_v1/MF3ZQ_ORACLE_HEADROOM_PROTOCOL.json": "4a317eea21878ac257da48c7999ed02a8104f8d71cfdc82baaa298c2bd7fd13d",
            "artifacts/training/mf3zq_oracle_revealskill_headroom_v1/MF3ZQ_ORACLE_HEADROOM_RESULT.json": "bba5137f919fa38a88fc085773f842b2e7667a0749963a1b8831934357a68346",
            "artifacts/training/mf3zr_option_bound_support_v1/MF3ZR_OPTION_BOUND_SUPPORT_PROTOCOL.json": "68db7d58b21e78ecbfa18ac24065f15789ec8c23ec65d3070b242620c1e184f5",
            "artifacts/training/mf3zr_option_bound_support_v1/MF3ZR_OPTION_BOUND_SUPPORT_RESULT.json": "c513d6a9ed527276cc9ff7df074d18ca6e837dc484cdb8c9750c082752328ce6",
        }
        for name, digest in expected.items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
