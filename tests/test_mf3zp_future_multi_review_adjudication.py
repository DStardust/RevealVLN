from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mf3zp_formal_v12", ROOT / "scripts/audit_mf3zp_labels_v1_2.py"
)
assert SPEC is not None and SPEC.loader is not None
formal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formal)


def _row(reviewer: str) -> dict[str, object]:
    factors = [
        {"step": step, "instantiated": True, "distinguishable": True, "resolved": True}
        for step in range(3)
    ]
    return {
        "schema_version": formal.SCHEMA,
        "reviewer_id": reviewer,
        "reviewer_blinded_to_outcomes": True,
        "reviewer_blinded_to_qwen_factors": True,
        "event_id": "event-1",
        "constraint_graph_sha256": "graph-hash",
        "constraint_reviews": {
            "c1": {"dec_role": "DEC_REQUIRED", "factor_by_step": factors},
        },
        "review_complete": True,
    }


class FutureMultiReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "artifacts/training")
        self.root = Path(self.temp.name)
        self.expected = {
            "event-1": {
                "constraint_graph_sha256": "graph-hash",
                "constraint_ids": ["c1"],
                "steps": [0, 1, 2],
            }
        }
        self.paths = []
        for index in range(3):
            path = self.root / f"review-{index}.jsonl"
            path.write_text(json.dumps(_row(f"reviewer-{index}")) + "\n", encoding="utf-8")
            self.paths.append(path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_kappa_pass_without_adjudicator_is_pending_not_pass(self) -> None:
        result = formal.audit_formal_reviews(
            self.paths, expected_population=self.expected
        )
        self.assertEqual(result["status"], "MF3ZP_LABEL_VALIDITY_PENDING_ADJUDICATION")
        self.assertFalse(result["oracle_headroom_authorized"])

    def test_distinct_adjudicator_and_complete_artifact_are_required_for_gold(self) -> None:
        adjudication = self.root / "adjudication.json"
        adjudication.write_text(json.dumps({
            "schema_version": formal.ADJUDICATION_SCHEMA,
            "adjudicator_id": "adjudicator-1",
            "adjudicator_blinded_to_outcomes": True,
            "items": {},
            "adjudication_complete": True,
        }), encoding="utf-8")
        gold = self.root / "gold.jsonl"
        result = formal.audit_formal_reviews(
            self.paths, expected_population=self.expected,
            adjudication_path=adjudication, gold_path=gold,
        )
        self.assertEqual(result["status"], "MF3ZP_LABEL_VALIDITY_PASS")
        self.assertTrue(result["oracle_headroom_authorized"])
        self.assertTrue(gold.is_file())
        self.assertIsNotNone(result["gold"]["sha256"])
        labels = {
            row["item_id"]: row["label"]
            for row in map(json.loads, gold.read_text(encoding="utf-8").splitlines())
        }
        self.assertEqual(labels["event-1::c1::UAD"], "D")

    def test_reviewer_cannot_also_adjudicate(self) -> None:
        adjudication = self.root / "adjudication.json"
        adjudication.write_text(json.dumps({
            "schema_version": formal.ADJUDICATION_SCHEMA,
            "adjudicator_id": "reviewer-0",
            "adjudicator_blinded_to_outcomes": True,
            "items": {},
            "adjudication_complete": True,
        }), encoding="utf-8")
        with self.assertRaises(formal.FormalAuditError):
            formal.audit_formal_reviews(
                self.paths, expected_population=self.expected,
                adjudication_path=adjudication, gold_path=self.root / "gold.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
