import unittest

from revealnav_mf3.evidence_memory_probe import (
    ProbeAuditError,
    TargetSourceSummary,
    evaluate_target_support,
)


def exact_source(dataset: str, source_id: str) -> TargetSourceSummary:
    return TargetSourceSummary(
        source_id=source_id,
        dataset=dataset,
        provenance="exact_train_native_action_or_candidate_supervision",
        preexisting=True,
        train_development_only=True,
        public_split_accessed=False,
        exact_same_episode_prefix=True,
        exact_candidate_set_alignment=True,
        target_rows=60,
        rankable_target_rows=55,
        raw_scene_count=12,
        unique_episode_count=50,
        accepted=True,
    )


class Mf3ztDecisionTargetSupportTest(unittest.TestCase):
    def test_both_domains_are_mandatory(self):
        value = evaluate_target_support([exact_source("RxR", "rxr")])
        self.assertFalse(value["passed"])
        self.assertEqual(value["status"], "MF3ZT_DECISION_TARGET_SUPPORT_FAIL")
        self.assertEqual(value["domain_support"]["R2R"]["legal_rankable_target_rows"], 0)

    def test_exact_sources_in_both_domains_pass_preflight(self):
        value = evaluate_target_support(
            [exact_source("R2R", "r2r"), exact_source("RxR", "rxr")]
        )
        self.assertTrue(value["passed"])
        self.assertEqual(value["status"], "MF3ZT_DECISION_TARGET_SUPPORT_PASS")

    def test_native_policy_action_is_rejected_even_if_candidate_aligned(self):
        native = TargetSourceSummary(
            source_id="native-self-label",
            dataset="R2R",
            provenance="frozen_native_action_self_label",
            preexisting=True,
            train_development_only=True,
            public_split_accessed=False,
            exact_same_episode_prefix=True,
            exact_candidate_set_alignment=True,
            target_rows=100,
            rankable_target_rows=100,
            raw_scene_count=20,
            unique_episode_count=80,
            accepted=False,
            rejection_reasons=("NATIVE_IS_PREDICTION_NOT_SUPERVISION",),
        )
        value = evaluate_target_support([native, exact_source("RxR", "rxr")])
        self.assertFalse(value["passed"])
        self.assertEqual(value["domain_support"]["R2R"]["accepted_source_ids"], [])

    def test_public_or_route_derived_source_cannot_be_marked_accepted(self):
        illegal = TargetSourceSummary(
            source_id="route-derived",
            dataset="R2R",
            provenance="route_truth_reconstruction",
            preexisting=True,
            train_development_only=True,
            public_split_accessed=False,
            exact_same_episode_prefix=True,
            exact_candidate_set_alignment=True,
            target_rows=60,
            rankable_target_rows=60,
            raw_scene_count=12,
            unique_episode_count=50,
            accepted=True,
        )
        with self.assertRaises(ProbeAuditError):
            evaluate_target_support([illegal])

    def test_unknown_provenance_is_not_an_implicit_allow(self):
        unknown = TargetSourceSummary(
            source_id="unknown",
            dataset="R2R",
            provenance="some_new_target_kind",
            preexisting=True,
            train_development_only=True,
            public_split_accessed=False,
            exact_same_episode_prefix=True,
            exact_candidate_set_alignment=True,
            target_rows=60,
            rankable_target_rows=60,
            raw_scene_count=12,
            unique_episode_count=50,
            accepted=True,
        )
        with self.assertRaises(ProbeAuditError):
            evaluate_target_support([unknown])


if __name__ == "__main__":
    unittest.main()
