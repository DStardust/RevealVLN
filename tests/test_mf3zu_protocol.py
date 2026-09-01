import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from revealnav_mf3.mf3zu_protocol import (
    ARMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CANDIDATE_BINDING_DIM,
    EVIDENCE_FEATURE_DIM,
    EVIDENCE_ONTOLOGY,
    EVIDENCE_RECORD_DIM,
    EXPECTED_POPULATION_ROWS,
    FOLD_SALT,
    IMPLEMENTATION_FILES,
    K_MEM,
    PUBLIC_CLOSED,
    QWEN_EXTRACTOR,
    REVISION,
    TRAINING_SEED,
    ProtocolError,
    build_protocol,
    verify_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_READY = all((ROOT / relative).is_file() for relative in IMPLEMENTATION_FILES)


@unittest.skipUnless(IMPLEMENTATION_READY, "parallel MF3ZU implementation is not complete")
class Mf3zuProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = build_protocol()

    def assert_protocol_mutation_rejected(self, *path_and_value):
        value = copy.deepcopy(self.protocol)
        *keys, replacement = path_and_value
        target = value
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = replacement
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ProtocolError):
                verify_protocol(path)

    def test_revision_is_explicitly_rxr_only_and_preserves_mf3zt(self):
        value = self.protocol
        self.assertEqual(REVISION, "mf3zu_rxr_evidence_memory_feasibility_v1")
        self.assertEqual(value["revision"], REVISION)
        self.assertEqual(
            value["status"],
            "SEALED_BEFORE_MF3ZU_FEASIBILITY_POPULATION_AND_RESULTS",
        )
        self.assertEqual(value["scope"]["dataset"], "RxR")
        self.assertTrue(value["scope"]["candidate_ranking_feasibility_only"])
        self.assertFalse(value["revision_relationship"]["R2R_in_scope"])
        self.assertFalse(value["revision_relationship"]["MF3ZT_modified"])
        self.assertTrue(value["revision_relationship"]["MF3ZT_failure_preserved"])
        self.assertFalse(
            value["revision_relationship"]["may_be_reported_as_MF3ZT_two_domain_pass"]
        )

    def test_population_and_target_order_are_frozen(self):
        value = self.protocol
        self.assertEqual(
            value["population"]["selection_rule"],
            "candidate_mask_count>=2_and_exact_target_feature_slot_active",
        )
        self.assertFalse(value["population"]["sampling"])
        self.assertEqual(value["population"]["expected_rows"], EXPECTED_POPULATION_ROWS)
        self.assertTrue(value["population"]["exact_target_accessed_for_support_eligibility"])
        self.assertFalse(value["population"]["target_value_in_sanitized_population"])
        self.assertFalse(value["population"]["baseline_score_or_correctness_selection"])
        self.assertTrue(
            value["exact_target_boundary"][
                "sanitized_population_and_exact_targets_are_separate_artifacts"
            ]
        )
        self.assertFalse(value["exact_target_boundary"]["annotation_may_open_exact_targets"])
        self.assertFalse(
            value["exact_target_boundary"][
                "trainer_may_open_exact_targets_before_evidence_manifest_frozen"
            ]
        )
        self.assertFalse(
            value["feature_to_physical_step_mapping"][
                "feature_row_equals_physical_step_assumed"
            ]
        )
        self.assertEqual(value["scene_split"]["salt"], FOLD_SALT)

    def test_extractor_evidence_model_and_training_are_fixed(self):
        value = self.protocol
        self.assertEqual(value["evidence"]["ontology"], list(EVIDENCE_ONTOLOGY))
        self.assertEqual(value["evidence"]["K_MEM"], K_MEM)
        self.assertEqual(value["evidence"]["extractor"], QWEN_EXTRACTOR)
        self.assertEqual(
            value["evidence"]["human_review"]["status"],
            "SKIPPED_BY_USER_FOR_THIS_ATTEMPT",
        )
        self.assertFalse(value["evidence"]["human_review"]["human_verified"])
        self.assertFalse(value["evidence"]["human_review"]["gold_labels"])
        self.assertEqual(
            value["evidence"]["fixed_record_feature"]["record_dimensions"],
            EVIDENCE_RECORD_DIM,
        )
        self.assertEqual(
            value["evidence"]["fixed_record_feature"]["candidate_binding"],
            CANDIDATE_BINDING_DIM,
        )
        self.assertEqual(
            value["evidence"]["fixed_record_feature"]["candidate_feature_dimensions"],
            EVIDENCE_FEATURE_DIM,
        )
        model = value["model_and_training"]
        self.assertEqual(model["arms"], list(ARMS))
        self.assertEqual(model["A"], "original frozen ETP masked native_scores; no training")
        self.assertEqual(model["epochs"], 40)
        self.assertEqual(model["batch_size"], 64)
        self.assertEqual(model["learning_rate"], 0.001)
        self.assertEqual(model["weight_decay"], 0.0001)
        self.assertEqual(model["optimizer"], "AdamW")
        self.assertEqual(model["loss"], "candidate_set_cross_entropy")
        self.assertEqual(model["seed"], TRAINING_SEED)
        self.assertFalse(model["early_stopping"])
        self.assertFalse(model["best_checkpoint_selection"])
        self.assertFalse(model["architecture_sweep"])
        self.assertFalse(model["threshold_search"])
        self.assertFalse(model["hyperparameter_grid"])
        self.assertFalse(model["multi_seed_rescue"])
        self.assertTrue(model["B_C_common_initialization"])
        self.assertTrue(model["B_C_common_batch_order"])

    def test_pass_rule_and_external_boundaries_are_frozen(self):
        value = self.protocol
        self.assertEqual(value["evaluation"]["bootstrap"]["replicates"], BOOTSTRAP_REPLICATES)
        self.assertEqual(value["evaluation"]["bootstrap"]["seed"], BOOTSTRAP_SEED)
        self.assertEqual(value["pass_fail"]["domain"], "RxR")
        self.assertFalse(value["pass_fail"]["R2R_required"])
        self.assertEqual(value["public_split_access"], PUBLIC_CLOSED)
        self.assertFalse(value["execution"]["full_navigation_run"])
        self.assertFalse(value["execution"]["checkpoint_generated"])
        self.assertFalse(value["execution"]["exact_target_values_opened_for_training"])
        self.assertEqual(
            value["pass_fail"]["status_on_pass"],
            "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_PASS",
        )
        self.assertEqual(
            value["pass_fail"]["status_on_fail"],
            "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_FAIL",
        )

    def test_verify_protocol_rejects_every_sealed_boundary_drift(self):
        mutations = (
            (("population", "expected_rows"), 1_427),
            (("population", "expected_episodes"), 153),
            (("population", "expected_raw_scenes"), 58),
            (("population", "sampling"), True),
            (("population", "baseline_score_or_correctness_selection"), True),
            (("exact_target_boundary", "annotation_may_open_exact_targets"), True),
            (("exact_target_boundary", "memory_required_may_inventory_exact_targets"), True),
            (("evidence", "human_review", "human_verified"), True),
            (("evidence", "human_review", "gold_labels"), True),
            (("evidence", "K_MEM"), 9),
            (("evidence", "fixed_record_feature", "record_dimensions"), 78),
            (("model_and_training", "learning_rate"), 0.002),
            (("model_and_training", "weight_decay"), 0.0),
            (("model_and_training", "batch_size"), 32),
            (("model_and_training", "epochs"), 41),
            (("model_and_training", "seed"), 1),
            (("model_and_training", "architecture_sweep"), True),
            (("evaluation", "bootstrap", "replicates"), 9_999),
            (("evaluation", "bootstrap", "seed"), 1),
            (("pass_fail", "memory_required_B_minus_A_Acc_positive"), False),
            (("pass_fail", "memory_required_B_minus_C_Acc_lower95_positive"), False),
            (("pass_fail", "memory_not_required_B_minus_A_Acc_min"), -0.02),
            (("public_split_access", "val_unseen"), True),
            (("scope", "full_navigation"), True),
            (("execution", "full_navigation_run"), True),
            (("execution", "checkpoint_generated"), True),
            (("execution", "checkpoint_for_deployment"), True),
        )
        for keys, replacement in mutations:
            with self.subTest(path=".".join(keys)):
                self.assert_protocol_mutation_rejected(*keys, replacement)

    def test_historical_mf3zt_files_remain_byte_identical(self):
        expected = {
            "METHOD_REVISION_3ZT_EVIDENCE_MEMORY_DECISION_PROBE.md": "3a962ee53567d79d58dfca049c24374e3d5b35a3891c02af42ff168cb11c12c2",
            "artifacts/training/mf3zt_evidence_memory_decision_probe_v1/MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_PROTOCOL.json": "71ba83a89b58eb7797a953cbdae8b03d51dd4fdae7b6618c451a511d7fd01af2",
            "artifacts/training/mf3zt_evidence_memory_decision_probe_v1/MF3ZT_DECISION_TARGET_SUPPORT_AUDIT.json": "31b10d600e8bce5a1c82c82481b146ed75489de287ac2d049c43508b0d6a958b",
            "artifacts/training/mf3zt_evidence_memory_decision_probe_v1/MF3ZT_EVIDENCE_MEMORY_DECISION_PROBE_RESULT.json": "0910c61b9cf5d45e845447bf60a1df7901d05aa62f3dfd37b3090429712b608e",
        }
        for relative, digest in expected.items():
            with self.subTest(path=relative):
                self.assertEqual(
                    hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                    digest,
                )


if __name__ == "__main__":
    unittest.main()
