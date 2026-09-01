import json
import unittest

from revealnav_mf3.qwen_evidence_annotation import QWEN_MODEL, instruction_request, reject_forbidden_annotation_payload


class QwenIsolationTest(unittest.TestCase):
    def test_fixed_model_and_instruction_only_request(self):
        payload = instruction_request("Walk past the sofa.")
        self.assertEqual(payload["model"], "qwen3.8-max")
        serialized = json.dumps(payload).lower()
        for forbidden in ("delta_utility", "ndtw", "sdtw", "catastrophe", "which action was better"):
            self.assertNotIn(forbidden, serialized)

    def test_forbidden_fields_fail_closed(self):
        for value in ({"reward": 1}, {"nested": {"future_frame": "x"}}, {"oracle_state": True}):
            with self.assertRaises(ValueError):
                reject_forbidden_annotation_payload(value)


if __name__ == "__main__":
    unittest.main()
