import sys
import unittest
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
sys.path.insert(0, str(ROOT / "scripts"))

import run_cr5_causal_prefix_language as language  # noqa: E402


class FullSetLanguageValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_all = language.USE_ALL_BRANCHES
        self.old_schema = language.RESPONSE_SCHEMA_VERSION
        language.USE_ALL_BRANCHES = True
        language.RESPONSE_SCHEMA_VERSION = (
            "revealnav-fullset-causal-prefix-language-v2"
        )
        self.event = {
            "event_id": "event",
            "target_branch_id": "BR01",
            "candidate_branch_ids": ["BR01", "BR02", "BR03"],
            "branch_current_runs": {
                "BR01": [[5, 8]], "BR02": [[1, 3]], "BR03": [[2, 4]],
            },
        }
        self.input_event = {
            "deterministic_segments": [{"segment_id": "S1", "text": "x"}]
        }
        self.prefix_record = {
            "branch_current": {"BR01": True, "BR02": False, "BR03": False}
        }

    def tearDown(self) -> None:
        language.USE_ALL_BRANCHES = self.old_all
        language.RESPONSE_SCHEMA_VERSION = self.old_schema

    def response(self):
        return {
            "schema_version": "revealnav-fullset-causal-prefix-language-v2",
            "event_id": "event",
            "prefix_index": 6,
            "evidence_status": "CLOSED",
            "recognizable_branch_ids": ["BR01", "BR02", "BR03"],
            "branches_visually_distinguishable": True,
            "instruction_uniquely_selects_one": True,
            "selected_branch_id": "BR01",
            "decisive_clause_ids": ["S1"],
            "competing_branch_supported_by_causal_history": True,
            "future_evidence_required": False,
            "confidence": 0.9,
            "rationale": "All exits are distinguishable and the clause is decisive.",
        }

    def test_closed_requires_every_declared_branch(self) -> None:
        value = self.response()
        self.assertEqual(language.validate_response(
            value, self.event, self.input_event, 6, self.prefix_record
        ), [])
        value["recognizable_branch_ids"].remove("BR03")
        self.assertIn("closed_semantic_invariants", language.validate_response(
            value, self.event, self.input_event, 6, self.prefix_record
        ))

    def test_closed_requires_selected_branch_to_be_current(self) -> None:
        value = self.response()
        value["selected_branch_id"] = "BR02"
        self.assertIn("full_set_causal_availability", language.validate_response(
            value, self.event, self.input_event, 6, self.prefix_record
        ))


if __name__ == "__main__":
    unittest.main()
