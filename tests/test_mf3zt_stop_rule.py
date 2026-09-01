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


class Mf3ztStopRuleTest(unittest.TestCase):
    def test_target_audit_fails_only_the_missing_domain(self):
        audit = json.loads(
            (OUT / "MF3ZT_DECISION_TARGET_SUPPORT_AUDIT.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(audit["status"], "MF3ZT_DECISION_TARGET_SUPPORT_FAIL")
        self.assertEqual(
            audit["domain_support"]["R2R"]["legal_rankable_target_rows"], 0
        )
        self.assertGreater(
            audit["domain_support"]["RxR"]["legal_rankable_target_rows"], 0
        )
        frozen_r2r = audit["frozen_observation_corpus_audit"]["R2R"]
        self.assertEqual(frozen_r2r["events"], 40)
        self.assertEqual(frozen_r2r["prefixes"], 163)
        self.assertEqual(frozen_r2r["candidate_instances"], 523)
        self.assertEqual(frozen_r2r["target_fields_in_causal_rows"], [])
        self.assertEqual(frozen_r2r["target_fields_in_arrays"], [])
        self.assertEqual(
            audit["rxr_exact_target_audit"]["rankable_target_rows"], 1428
        )
        self.assertFalse(audit["decision_population_authorized"])
        self.assertFalse(audit["reranker_or_training_authorized"])
        self.assertFalse(audit["outcome_or_utility_payload_parsed"])
        self.assertEqual(audit["public_split_access"], PUBLIC_CLOSED)

    def test_result_marks_unrun_quantities_as_unrun_not_zero(self):
        result = json.loads(
            (OUT / "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_RESULT.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(result["status"], "MF3ZT_DECISION_TARGET_SUPPORT_FAIL")
        self.assertEqual(
            result["final_pass_fail"],
            "MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_FAIL",
        )
        self.assertEqual(result["scientific_evidence_about_memory"], "NOT_OBSERVED")
        self.assertEqual(result["metrics_per_domain"], "NOT_RUN")
        self.assertEqual(result["scene_bootstrap_CI"], "NOT_RUN")
        self.assertIsNone(result["population"]["total_decisions"])
        self.assertFalse(result["execution"]["training_started"])
        self.assertFalse(result["execution"]["full_navigation_run"])
        self.assertFalse(result["execution"]["checkpoint_generated"])
        self.assertEqual(result["public_split_access"], PUBLIC_CLOSED)

    def test_target_failure_created_no_downstream_artifact(self):
        forbidden = [
            "MF3ZT_DECISION_POPULATION.jsonl",
            "MF3ZT_EVIDENCE_MEMORY.jsonl",
            "MF3ZT_FOLD_ASSIGNMENTS.json",
            "MF3ZT_OOF_PREDICTIONS.jsonl",
            "MF3ZT_BOOTSTRAP.json",
            "MF3ZT_RERANKER.pt",
            "MF3ZT_RERANKER.pth",
            "MF3ZT_RERANKER.ckpt",
        ]
        self.assertEqual([name for name in forbidden if (OUT / name).exists()], [])
        for name in (
            "evidence_memory_builder.py",
            "evidence_memory_retrieval.py",
            "evidence_memory_reranker.py",
            "evidence_memory_metrics.py",
        ):
            self.assertFalse((ROOT / "revealnav_mf3" / name).exists())


if __name__ == "__main__":
    unittest.main()
