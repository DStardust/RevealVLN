import copy
import json
import unittest

from revealnav_mf3.codex_visual_review import (
    REVIEW_TEMPLATE,
    SOURCE,
    TRAINED_ROLES,
    VisualReviewError,
    read_json,
    read_jsonl,
    validate_manual_event,
)


class CodexIndependentVisualReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read_json(SOURCE)
        cls.templates = read_jsonl(REVIEW_TEMPLATE)

    def test_complete_fixed_population_validates(self):
        self.assertTrue(self.source["labels_complete"])
        self.assertEqual(len(self.source["events"]), 80)
        self.assertEqual(len(self.templates), 80)
        for index, (manual, template) in enumerate(
            zip(self.source["events"], self.templates, strict=True)
        ):
            self.assertEqual(manual["index"], index)
            value = validate_manual_event(manual, template)
            self.assertTrue(any(
                item["dec_role"] in TRAINED_ROLES
                for item in value["constraints"].values()
            ))

    def test_review_source_declares_no_qwen_factor_use(self):
        policy = self.source["review_policy"]
        self.assertFalse(policy["qwen_factor_labels_read"])
        self.assertFalse(policy["qwen_uad_labels_read"])
        self.assertFalse(policy["qwen_rationales_read"])
        self.assertFalse(policy["old_training_results_used"])

    def test_factor_or_role_omission_fails_closed(self):
        manual = copy.deepcopy(self.source["events"][0])
        template = self.templates[0]
        del manual["factors"]["c2"]["0"]
        with self.assertRaises(VisualReviewError):
            validate_manual_event(manual, template)

    def test_future_prefix_fails_closed(self):
        manual = copy.deepcopy(self.source["events"][0])
        template = copy.deepcopy(self.templates[0])
        template["prefixes"][0]["step"] = template["decision_step"] + 1
        with self.assertRaises(VisualReviewError):
            validate_manual_event(manual, template)

    def test_source_contains_no_outcome_payload(self):
        serialized = json.dumps(self.source, sort_keys=True).casefold()
        for key in (
            '"delta_utility"', '"reward"', '"outcome"', '"success"',
            '"spl"', '"ndtw"', '"sdtw"', '"catastrophe"',
        ):
            self.assertNotIn(key, serialized)


if __name__ == "__main__":
    unittest.main()
