"""Fail-closed tests for the sealed MF3ZN-TUAD protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEALER_PATH = ROOT / "scripts/seal_mf3zn_tuad_protocol.py"
SPEC = importlib.util.spec_from_file_location("mf3zn_protocol_sealer", SEALER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load MF3ZN protocol sealer")
SEALER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEALER)

AUDIT_PATH = ROOT / "scripts/audit_mf3zn_uad_identifiability.py"
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "mf3zn_identifiability_entrypoint", AUDIT_PATH
)
if AUDIT_SPEC is None or AUDIT_SPEC.loader is None:
    raise RuntimeError("cannot load MF3ZN identifiability entrypoint")
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)

from revealnav_mf3.tuad_protocol import (  # noqa: E402
    ACTION_VALUE_HIDDEN_SIZE,
    ALLOWED_COMMANDS,
    CONTROLS,
    ENSEMBLE_REDUCTION,
    FIXED_SEEDS,
    FIXED_REPORTING_SEEDS,
    FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED,
    GRU_HIDDEN_SIZE,
    IDENTIFIABILITY_AUDITS,
    IDENTIFIABILITY_EXPECTED_DOMAIN_COUNTS,
    IDENTIFIABILITY_EXPECTED_ROWS,
    IDENTIFIABILITY_EXPECTED_SCENES,
    LATTICE_ID,
    METHOD_ID,
    OUTER_FOLDS,
    RAW_SCENE_OOF_FOLDS,
    TEAL_REVISION,
    TUAD_REVISION,
    TUADProtocolError,
    build_protocol,
    canonical_json_bytes,
    protocol_payload,
    validate_protocol,
    verify_protocol,
)
from revealnav_mf3.tuad_selection import REQUIRED_POLICIES  # noqa: E402


class MF3ZNTUADProtocolTest(unittest.TestCase):
    def test_family_tombstone_and_identifiers_are_machine_fixed(self):
        self.assertIs(FROZEN_SINGLE_DECISION_GATE_FAMILY_STOPPED, True)
        self.assertEqual(TUAD_REVISION, "mf3zn_tuad_v1")
        self.assertEqual(TEAL_REVISION, "mf3zn_teal_v1")
        self.assertEqual(METHOD_ID, TUAD_REVISION)
        self.assertEqual(LATTICE_ID, TEAL_REVISION)
        value = build_protocol(ROOT)
        self.assertIs(value["family_tombstone"]["value"], True)

    def test_model_and_development_have_no_selection_surface(self):
        value = build_protocol(ROOT)
        self.assertEqual(GRU_HIDDEN_SIZE, 64)
        self.assertEqual(value["model"]["stage_1"]["hidden_size"], 64)
        self.assertIs(value["model"]["joint_end_to_end_training"], False)
        self.assertEqual(value["model"]["stage_2"]["native_value"], 0.0)
        self.assertEqual(ACTION_VALUE_HIDDEN_SIZE, 64)
        self.assertEqual(value["model"]["stage_2"]["hidden_size"], 64)
        self.assertIs(value["model"]["stage_2"]["native_bypasses_network"], True)
        development = value["development"]
        self.assertEqual(development["architecture_grid"], [])
        self.assertIs(development["architecture_selection"], False)
        self.assertEqual(development["weight_decay_grid"], [])
        self.assertIs(development["weight_decay_selection"], False)
        self.assertEqual(development["threshold_grid"], [])
        self.assertIs(development["threshold_selection"], False)
        self.assertEqual(tuple(development["fixed_reporting_seeds"]), FIXED_REPORTING_SEEDS)
        self.assertEqual(FIXED_SEEDS, (20260831, 20260832, 20260833))
        self.assertEqual(len(FIXED_REPORTING_SEEDS), 3)
        self.assertIs(development["seed_selection"], False)
        self.assertEqual(development["ensemble_reduction"], ENSEMBLE_REDUCTION)
        self.assertEqual(ENSEMBLE_REDUCTION, "elementwise_median")
        self.assertEqual(RAW_SCENE_OOF_FOLDS, 5)
        self.assertEqual(OUTER_FOLDS, RAW_SCENE_OOF_FOLDS)
        self.assertEqual(development["outer_folds"], RAW_SCENE_OOF_FOLDS)
        self.assertEqual(development["fold_unit"], "raw_mp3d_scene")
        self.assertIs(development["inner_model_selection"], False)

    def test_identifiability_a_b_c_gate_collection_and_define_both_stops(self):
        value = build_protocol(ROOT)
        gate = value["identifiability_gate"]
        self.assertEqual(tuple(gate["all_required"]), IDENTIFIABILITY_AUDITS)
        population = value["source_population"]
        self.assertEqual(population["rows"], IDENTIFIABILITY_EXPECTED_ROWS)
        self.assertEqual(population["raw_mp3d_scenes"], IDENTIFIABILITY_EXPECTED_SCENES)
        self.assertEqual(population["domain_counts"], IDENTIFIABILITY_EXPECTED_DOMAIN_COUNTS)
        self.assertIs(population["cohort_substitution_allowed"], False)
        self.assertEqual(
            set(gate["audits"]),
            {"oracle_relevance", "causal_observability", "label_validity"},
        )
        self.assertIs(gate["required_before_collection"], True)
        self.assertIs(gate["collection_authorized_at_seal"], False)
        self.assertEqual(set(value["stop_rules"]), {"stop_a", "stop_b"})
        self.assertIn("before new treatment collection", value["stop_rules"]["stop_a"]["action"])
        self.assertIn("consumed development universe", value["stop_rules"]["stop_b"]["action"])

    def test_controls_and_one_shot_scientific_gates_are_explicit(self):
        value = build_protocol(ROOT)
        self.assertEqual(tuple(value["controls"]["arms"]), CONTROLS)
        self.assertEqual(CONTROLS, REQUIRED_POLICIES)
        self.assertIn("current-only", CONTROLS)
        self.assertIn("temporal-no-UAD-supervision", CONTROLS)
        self.assertIn("oracle-UAD", CONTROLS)
        self.assertIn("runner-only-support", CONTROLS)
        gates = value["scientific_gates"]
        self.assertIn("complete_five_fold_oof", gates["correctness"])
        self.assertIn("every_fold_domain_total_utility >= 0", gates["per_domain_utility"])
        self.assertIn("bootstrap_95pct_lower_bound > 0", gates["temporal_contribution_per_domain"][1])
        self.assertIs(gates["hide_domain_results"], False)

    def test_no_confirmation_or_public_entrypoint_or_authorization(self):
        value = build_protocol(ROOT)
        self.assertEqual(ALLOWED_COMMANDS, ("seal", "verify"))
        self.assertEqual(SEALER.parse_args([]).command, "seal")
        self.assertEqual(value["entrypoints"]["allowed_scope"], "protocol_sealer_only")
        self.assertEqual(tuple(value["entrypoints"]["allowed"]), ALLOWED_COMMANDS)
        self.assertEqual(len(value["entrypoints"]["source_sealed_scientific"]), 3)
        self.assertIsNone(value["entrypoints"]["confirmation"])
        self.assertIsNone(value["entrypoints"]["public_evaluation"])
        authorization = value["authorization"]
        self.assertIs(authorization["confirmation"], False)
        self.assertIs(authorization["public_unseen"], False)
        self.assertTrue(all(
            access is False
            for access in authorization["public_split_access"].values()
        ))
        self.assertFalse(hasattr(SEALER, "RESULT"))

    def test_protocol_build_is_deterministic_and_hashes_all_sources(self):
        first = build_protocol(ROOT)
        second = build_protocol(ROOT)
        self.assertEqual(protocol_payload(ROOT), first)
        self.assertEqual(first, second)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(
            set(first["source_files"]),
            {
                "car_source_protocol",
                "method_revision",
                "temporal_schema",
                "temporal_labels",
                "temporal_features",
                "temporal_model",
                "temporal_action_value",
                "temporal_exact_lattice",
                "tuad_identifiability",
                "tuad_selection",
                "protocol_implementation",
                "protocol_sealer",
                "identifiability_audit_entrypoint",
                "lattice_collection_entrypoint",
                "tuad_training_entrypoint",
            },
        )
        for item in first["source_files"].values():
            self.assertEqual(len(item["sha256"]), 64)
            self.assertGreater(item["bytes"], 0)

    def test_seal_is_atomic_canonical_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested/MF3ZN_TUAD_PROTOCOL.json"
            value = SEALER.seal_protocol(output, project_root=ROOT)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), value)
            self.assertEqual(output.read_bytes(), canonical_json_bytes(value))
            self.assertEqual(verify_protocol(output, root=ROOT), value)
            self.assertFalse(output.with_name(output.name + ".part").exists())
            self.assertEqual([path.name for path in output.parent.iterdir()], [output.name])
            with self.assertRaises(TUADProtocolError):
                SEALER.seal_protocol(output, project_root=ROOT)

    def test_verify_refuses_protocol_or_source_inventory_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "MF3ZN_TUAD_PROTOCOL.json"
            value = SEALER.seal_protocol(output, project_root=ROOT)
            value["development"]["architecture_selection"] = True
            output.write_bytes(canonical_json_bytes(value))
            with self.assertRaises(TUADProtocolError):
                SEALER.verify_protocol(output, project_root=ROOT)

            current = build_protocol(ROOT)
            current["source_files"]["method_revision"]["sha256"] = "0" * 64
            with self.assertRaises(TUADProtocolError):
                validate_protocol(current, ROOT)

    def test_identifiability_entrypoint_rejects_substituted_small_cohort(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
            root = Path(directory)
            protocol_path = root / "MF3ZN_TUAD_PROTOCOL.json"
            SEALER.seal_protocol(protocol_path, project_root=ROOT)
            causal_path = root / "causal.npz"
            oracle_path = root / "oracle.npz"
            review_path = root / "reviews.json"
            np.savez(
                causal_path,
                event_id=np.asarray(["event-1"]),
                scene_id=np.asarray(["scene-1"]),
                dataset=np.asarray(["RxR"]),
                source_canonical_identity_sha256=np.asarray(
                    build_protocol(ROOT)["source_population"][
                        "canonical_identity_sha256"
                    ]
                ),
                snapshot_features=np.zeros((1, 1), dtype=np.float64),
                temporal_summary=np.zeros((1, 1), dtype=np.float64),
            )
            np.savez(
                oracle_path,
                event_id=np.asarray(["event-1"]),
                delta_utility=np.zeros(1),
                target_in_set=np.zeros(1),
                candidate_separated=np.zeros(1),
                evidence_closed=np.zeros(1),
                uad_state=np.asarray(["U"]),
                reveal_offset=np.zeros(1),
                expiry_offset=np.zeros(1),
                reveal_event=np.zeros(1),
                expiry_event=np.zeros(1),
            )
            review_path.write_text(json.dumps({
                "event_id": ["review-1"],
                "scene_id": ["scene-1"],
                "uad_rater_a": ["U"],
                "uad_rater_b": ["U"],
                "evidence_rater_a": [False],
                "evidence_rater_b": [False],
            }), encoding="utf-8")
            with self.assertRaisesRegex(
                TUADProtocolError, "source universe drift"
            ):
                AUDIT.run_audit(
                    protocol_path, causal_path, oracle_path, review_path
                )


if __name__ == "__main__":
    unittest.main()
