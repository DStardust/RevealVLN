"""Pre-collection identifiability gate tests for MF3ZN-TUAD v1."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.tuad_identifiability import (
    canonical_audit_event_id,
    causal_observability_audit,
    decision_time_uad_truth,
    deterministic_review_pilot_indices,
    identifiability_gate,
    label_validity_audit,
    oracle_relevance_audit,
)
from revealnav_mf3.temporal_uad_features import causal_sequence_features
from revealnav_mf3.temporal_uad_schema import (
    CausalTemporalStep,
    TemporalSequence,
)
from revealnav_mf3.tuad_protocol import TUADProtocolError


AUDIT_SPEC = importlib.util.spec_from_file_location(
    "mf3zn_identifiability_entrypoint_for_test",
    ROOT / "scripts/audit_mf3zn_uad_identifiability.py",
)
if AUDIT_SPEC is None or AUDIT_SPEC.loader is None:
    raise RuntimeError("cannot import MF3ZN identifiability entrypoint")
AUDIT_ENTRY = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT_ENTRY)


class IdentifiabilityAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(31)

    def test_oracle_relevance_is_per_domain_scene_oof(self):
        scenes = np.repeat([f"scene-{index}" for index in range(20)], 8)
        datasets = np.tile(["R2R", "RxR"], len(scenes) // 2)
        oracle = self.rng.normal(size=(len(scenes), 2))
        current = self.rng.normal(size=(len(scenes), 2))
        utility = 2.0 * oracle[:, 0] - oracle[:, 1]
        result = oracle_relevance_audit(
            current,
            oracle,
            utility,
            scenes,
            datasets,
            bootstrap_replicates=100,
        )
        self.assertEqual(result["status"], "ORACLE_RELEVANCE_PASS")
        self.assertEqual(set(result["domains"]), {"R2R", "RxR"})
        self.assertTrue(all(
            item["delta_huber"]["lower_95"] > 0.0
            for item in result["domains"].values()
        ))

    @staticmethod
    def temporal_sequence(history_shift: float = 0.0) -> TemporalSequence:
        steps = []
        for index in range(3):
            policy = np.asarray([
                0.2,
                0.1,
                0.3 + index + (history_shift if index == 0 else 0.0),
                0.2,
                0.4,
            ], dtype=np.float64)
            steps.append(CausalTemporalStep(
                step=index,
                native_action_id="native",
                candidate_action_ids=("native", "runner"),
                policy_features=policy,
                instruction_embedding=np.asarray([0.1, 0.2]),
                checkpoint_embedding=np.asarray([index + 0.1, 0.3]),
                action_embeddings=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            ))
        return TemporalSequence.create(
            dataset="RxR", scene_id="scene", episode_id="episode",
            decision_step=2, steps=steps,
        )

    def test_entrypoint_rebuilds_full_current_rows_from_strict_sequences(self):
        original = self.temporal_sequence()
        mutated_history = self.temporal_sequence(history_shift=5.0)
        snapshot, temporal, mask = AUDIT_ENTRY._causal_probe_tensors((original,))
        changed_snapshot, changed_temporal, _ = AUDIT_ENTRY._causal_probe_tensors(
            (mutated_history,)
        )
        isolated = TemporalSequence.create(
            dataset=original.dataset,
            scene_id=original.scene_id,
            episode_id=original.episode_id,
            decision_step=original.decision_step,
            steps=(original.steps[-1],),
        )
        expected = causal_sequence_features(isolated)[-1]
        self.assertTrue(np.array_equal(snapshot[0, -1], expected))
        self.assertTrue(np.array_equal(snapshot[0, -1], changed_snapshot[0, -1]))
        self.assertFalse(np.array_equal(temporal[0, -1], changed_temporal[0, -1]))
        self.assertTrue(mask.all())

    def test_result_verifier_exact_compares_deterministic_recomputation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            sources = {}
            for name in ("protocol", "causal", "oracle", "reviews"):
                path = root / f"{name}.json"
                path.write_text("{}", encoding="utf-8")
                sources[name] = path
            provenance = {
                "protocol": {"path": str(sources["protocol"].relative_to(ROOT))},
                "causal_probe": {"path": str(sources["causal"].relative_to(ROOT))},
                "oracle_labels": {"path": str(sources["oracle"].relative_to(ROOT))},
                "label_reviews": {"path": str(sources["reviews"].relative_to(ROOT))},
            }
            expected = {
                "schema_version": "revealnav-mf3zn-identifiability-result/1",
                "status": "MF3ZN_IDENTIFIABILITY_PASS",
                "collection_authorized": True,
                "provenance": provenance,
            }
            result_path = root / "result.json"
            result_path.write_text(json.dumps(expected), encoding="utf-8")
            with mock.patch.object(
                AUDIT_ENTRY, "run_audit", return_value=expected,
            ) as recompute:
                self.assertEqual(
                    AUDIT_ENTRY.verify_identifiability_result(result_path),
                    expected,
                )
                recompute.assert_called_once()
            forged = dict(expected)
            forged["collection_authorized"] = False
            result_path.write_text(json.dumps(forged), encoding="utf-8")
            with mock.patch.object(AUDIT_ENTRY, "run_audit", return_value=expected):
                with self.assertRaisesRegex(
                    TUADProtocolError, "differs from deterministic recomputation",
                ):
                    AUDIT_ENTRY.verify_identifiability_result(result_path)

    def test_canonical_event_identity_is_outcome_free_and_content_bound(self):
        first = canonical_audit_event_id("RxR", "scene", "episode", 4)
        repeated = canonical_audit_event_id("RxR", "scene", "episode", 4)
        changed = canonical_audit_event_id("RxR", "scene", "episode", 5)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 64)

    def observability_fixture(self):
        scenes = np.repeat([f"scene-{index}" for index in range(20)], 12)
        datasets = np.tile(["R2R", "RxR"], len(scenes) // 2)
        state = np.tile(np.asarray([0, 1, 2, 0, 1, 2]), len(scenes) // 6)
        rows, prefixes = len(scenes), 3
        factor_mask = np.ones((rows, prefixes), dtype=np.bool_)
        target_in_set = np.repeat((state != 0)[:, None], prefixes, axis=1)
        separated = np.repeat((state == 2)[:, None], prefixes, axis=1)
        evidence = separated.copy()
        reveal_truth = state != 0
        expiry_truth = state == 2
        reveal = np.zeros((rows, prefixes), dtype=np.bool_)
        reveal[:, -1] = reveal_truth
        expiry = np.zeros((rows, prefixes), dtype=np.bool_)
        expiry[:, -1] = expiry_truth
        reveal_risk = np.ones((rows, prefixes), dtype=np.bool_)
        expiry_risk = np.ones((rows, prefixes), dtype=np.bool_)
        snapshot = self.rng.normal(
            scale=0.001, size=(rows, prefixes, 2)
        )
        temporal = np.zeros((rows, prefixes, 5), dtype=np.float64)
        temporal[:, :, 0] = target_in_set
        temporal[:, :, 1] = separated
        temporal[:, :, 2] = evidence
        temporal[:, :, 3] = reveal_truth[:, None]
        temporal[:, :, 4] = expiry_truth[:, None]
        return (
            snapshot, temporal, target_in_set, separated, evidence,
            factor_mask, reveal, reveal_risk, expiry, expiry_risk,
            scenes, datasets,
        )

    def test_causal_history_must_improve_all_three_metrics_in_each_domain(self):
        result = causal_observability_audit(
            *self.observability_fixture(), bootstrap_replicates=100
        )
        self.assertEqual(result["status"], "CAUSAL_OBSERVABILITY_PASS")
        for item in result["domains"].values():
            self.assertGreater(item["delta_uad_macro_f1"]["lower_95"], 0.0)
            self.assertGreater(item["delta_reveal_nll"]["lower_95"], 0.0)
            self.assertGreater(item["delta_expiry_nll"]["lower_95"], 0.0)

    def test_label_validity_enforces_kappa_and_scene_balance(self):
        truth = list("UAD") * 4
        evidence = [False, True] * 6
        result = label_validity_audit(
            truth,
            truth,
            evidence,
            evidence,
            np.repeat(["s1", "s2", "s3"], 4),
            minimum_events=12,
        )
        self.assertEqual(result["status"], "LABEL_VALIDITY_PASS")
        with self.assertRaises(ValueError):
            label_validity_audit(
                truth,
                truth,
                evidence,
                evidence,
                ["s1"] * 10 + ["s2"] * 2,
                minimum_events=12,
            )

    def test_uad_truth_is_frozen_k3_and_mask_strict(self):
        target = np.asarray([
            [False, False, False, False],
            [True, True, True, False],
            [True, True, True, False],
        ], dtype=np.bool_)
        separated = np.asarray([
            [False, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
        ], dtype=np.bool_)
        evidence = separated.copy()
        mask = np.asarray([
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, False],
        ], dtype=np.bool_)
        self.assertEqual(
            decision_time_uad_truth(target, separated, evidence, mask).tolist(),
            ["U", "A", "D"],
        )
        with self.assertRaises(ValueError):
            decision_time_uad_truth(
                target.astype(np.float64), separated, evidence, mask,
            )

    def test_observability_rejects_implicit_or_invalid_censor_masks(self):
        values = list(self.observability_fixture())
        values[7] = values[7].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "Boolean matrix"):
            causal_observability_audit(
                *values, bootstrap_replicates=10,
            )
        values = list(self.observability_fixture())
        values[7][1, -1] = False
        with self.assertRaisesRegex(ValueError, "while at risk"):
            causal_observability_audit(
                *values, bootstrap_replicates=10,
            )

    def test_review_pilot_is_deterministic_balanced_and_covers_all_scenes(self):
        scenes = np.repeat([f"scene-{index:02d}" for index in range(39)], 10)
        event_ids = np.asarray([f"event-{index:03d}" for index in range(390)])
        first = deterministic_review_pilot_indices(event_ids, scenes)
        second = deterministic_review_pilot_indices(event_ids, scenes)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 300)
        counts = np.unique(scenes[first], return_counts=True)[1]
        self.assertEqual(len(counts), 39)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        capacities = [1, 1, 3, 4, 9, *([10] * 34)]
        sparse_scenes = np.concatenate([
            np.repeat(f"sparse-{index:02d}", count)
            for index, count in enumerate(capacities)
        ])
        sparse_ids = np.asarray([
            f"sparse-event-{index:03d}" for index in range(len(sparse_scenes))
        ])
        sparse = deterministic_review_pilot_indices(sparse_ids, sparse_scenes)
        selected_counts = dict(zip(
            *np.unique(sparse_scenes[sparse], return_counts=True), strict=True,
        ))
        maximum = max(selected_counts.values())
        self.assertEqual(len(selected_counts), 39)
        self.assertTrue(all(
            selected_counts[f"sparse-{index:02d}"] == capacity
            or maximum - selected_counts[f"sparse-{index:02d}"] <= 1
            for index, capacity in enumerate(capacities)
        ))
        with self.assertRaisesRegex(ValueError, "exactly 39"):
            deterministic_review_pilot_indices(event_ids[:300], scenes[:300])

    def test_failed_subaudit_blocks_collection(self):
        domains = {domain: {"pass": True} for domain in ("R2R", "RxR")}
        passed_a = {
            "audit": "oracle_relevance", "status": "ORACLE_RELEVANCE_PASS",
            "failures": [], "domains": domains,
        }
        passed_b = {
            "audit": "causal_observability",
            "status": "CAUSAL_OBSERVABILITY_PASS", "failures": [],
            "domains": domains,
        }
        passed_c = {
            "audit": "label_validity", "status": "LABEL_VALIDITY_PASS",
            "scene_balanced": True, "uad_kappa": 0.8,
            "evidence_closure_kappa": 0.8,
        }
        passed = identifiability_gate(passed_a, passed_b, passed_c)
        self.assertTrue(passed["collection_authorized"])
        failed = identifiability_gate(
            {
                "audit": "oracle_relevance",
                "status": "TEMPORAL_ORACLE_RELEVANCE_FAIL",
                "failures": ["RxR:failed"], "domains": domains,
            },
            passed_b,
            passed_c,
        )
        self.assertEqual(failed["status"], "MF3ZN_IDENTIFIABILITY_FAIL")
        self.assertFalse(failed["collection_authorized"])
        self.assertFalse(failed["public_authorization"])

        forged = identifiability_gate(
            {
                "audit": "oracle_relevance", "status": "ORACLE_RELEVANCE_PASS",
                "failures": ["RxR:still_failed"], "domains": domains,
            },
            passed_b,
            passed_c,
        )
        self.assertFalse(forged["collection_authorized"])

    def test_missing_domain_fails_closed(self):
        with self.assertRaises(ValueError):
            oracle_relevance_audit(
                np.zeros((10, 1)),
                np.ones((10, 1)),
                np.ones(10),
                [f"s{index}" for index in range(10)],
                ["RxR"] * 10,
                bootstrap_replicates=10,
            )


if __name__ == "__main__":
    unittest.main()
