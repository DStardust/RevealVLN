from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from revealnav_mf3.mf3zu_evidence_memory import (
    ConfidenceClass,
    MF3ZUContractError,
    memory_required,
    validate_evidence_response,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_mf3zu_evidence_contract_projection_v1r3.py"
SPEC = importlib.util.spec_from_file_location(
    "mf3zu_evidence_contract_projection_v1r3_test_module",
    RUNNER_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import MF3ZU v1r3 projection runner")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _project(record):
    request = record["request"]
    return RUNNER.project_unsupported_history(
        record["response"]["invalid_parsed_response"],
        expected_atom_ids=RUNNER.v1r2._expected_atom_ids(request),
        decision_step=int(request["decision_step"]),
        allowed_candidate_ids=list(request["candidate_alias_to_action_id"]),
        graph=record["graph"],
    )


def _forbidden_value_path(path: Path) -> bool:
    name = path.name.casefold()
    parts = {part.casefold() for part in path.parts}
    return (
        "exact_target" in name
        or name.endswith("_result.json")
        or bool(
            parts
            & {
                "val_seen",
                "val_unseen",
                "test_challenge",
                "test",
            }
        )
    )


class MF3ZUEvidenceContractProjectionV1R3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The real retained v1r2 partition is itself part of this regression.
        # Guard every pathlib-backed read so this audit cannot silently widen
        # into the separate target, a result, or a public split.
        original_open = Path.open
        forbidden_attempts: list[str] = []

        def guarded_open(path: Path, *args, **kwargs):
            if _forbidden_value_path(path):
                forbidden_attempts.append(str(path))
                raise AssertionError(f"forbidden value artifact opened: {path}")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            cls.bundle = RUNNER._parent_bundle()
        if forbidden_attempts:
            raise AssertionError(forbidden_attempts)

    @classmethod
    def failed_records(cls):
        return [
            value
            for value in cls.bundle["records"]
            if value["mode"]
            == "CONSERVATIVE_UNSUPPORTED_HISTORY_PROJECTION"
        ]

    def test_actual_parent_partition_is_1416_pass_12_source_only_fail(self):
        records = self.bundle["records"]
        passed = [
            value
            for value in records
            if value["mode"] == "PARENT_PASS_BYTE_REUSE"
        ]
        failed = self.failed_records()
        self.assertEqual(len(records), 1428)
        self.assertEqual(len(passed), RUNNER.EXPECTED_PARENT_PASS)
        self.assertEqual(len(failed), RUNNER.EXPECTED_PARENT_FAIL)
        self.assertEqual((len(passed), len(failed)), (1416, 12))
        self.assertEqual(self.bundle["manifest"]["pass"], 1416)
        self.assertEqual(self.bundle["manifest"]["fail"], 12)

        projected_atoms = 0
        source_at_decision = 0
        source_missing = 0
        for record in failed:
            response = record["response"]
            request = record["request"]
            self.assertEqual(response["status"], "FAIL")
            self.assertEqual(
                response["error"],
                "MF3ZUContractError: historical source must be strictly "
                "before the decision",
            )
            _, structural_issues = RUNNER.v1r2._structural_rows(
                response["invalid_parsed_response"],
                expected_atom_ids=RUNNER.v1r2._expected_atom_ids(request),
            )
            self.assertEqual(structural_issues, [])
            invalid = []
            for atom in response["invalid_parsed_response"]["atoms"]:
                source = atom["source_step"]
                if atom["historical_status"] == "OBSERVED" and (
                    isinstance(source, bool)
                    or not isinstance(source, int)
                    or not 0 <= source < int(request["decision_step"])
                ):
                    invalid.append(atom)
                    source_at_decision += int(
                        source == int(request["decision_step"])
                    )
                    source_missing += int(source is None)
            self.assertTrue(invalid)
            self.assertEqual(len(record["operations"]), len(invalid))
            projected_atoms += len(invalid)

        self.assertEqual(projected_atoms, RUNNER.EXPECTED_PROJECTED_ATOMS)
        self.assertEqual(projected_atoms, 20)
        self.assertEqual(source_at_decision, 12)
        self.assertEqual(source_missing, 8)
        ledger = RUNNER._input_ledger_rows(self.bundle)
        self.assertEqual(len(ledger), 12)
        self.assertEqual(
            sum(len(row["projected_atom_details"]) for row in ledger), 20
        )
        self.assertTrue(
            all(
                row["candidate_target_accessed"] is False
                and row["performance_accessed"] is False
                and row["public_split_access"] is False
                for row in ledger
            )
        )

    def test_projection_has_exact_32_field_diffs_and_is_idempotent(self):
        total_diffs: list[tuple[str, str, str]] = []
        valid_observed_sources = 0
        for record in self.failed_records():
            request = record["request"]
            original = record["response"]["invalid_parsed_response"]
            before = copy.deepcopy(original)
            projected, operations = _project(record)
            self.assertEqual(original, before, "projection mutated its input")

            judgements = validate_evidence_response(
                projected,
                graph=record["graph"],
                decision_step=int(request["decision_step"]),
                allowed_candidate_ids=list(
                    request["candidate_alias_to_action_id"]
                ),
            )
            self.assertEqual(
                len(judgements),
                len(RUNNER.v1r2._expected_atom_ids(request)),
            )
            old_by_id = {
                str(row["instruction_atom_id"]): row
                for row in before["atoms"]
            }
            new_by_id = {
                str(row["instruction_atom_id"]): row
                for row in projected["atoms"]
            }
            self.assertEqual(set(old_by_id), set(new_by_id))
            expected_operation_ids: set[str] = set()
            for atom_id, old in old_by_id.items():
                new = new_by_id[atom_id]
                self.assertEqual(set(new), set(old))
                source = old["source_step"]
                invalid = old["historical_status"] == "OBSERVED" and (
                    isinstance(source, bool)
                    or not isinstance(source, int)
                    or not 0 <= source < int(request["decision_step"])
                )
                changed_fields = {
                    key for key in old if old[key] != new[key]
                }
                if invalid:
                    expected_operation_ids.add(atom_id)
                    expected_changes = {"historical_status"}
                    if source is not None:
                        expected_changes.add("source_step")
                    self.assertEqual(changed_fields, expected_changes)
                    self.assertEqual(new["historical_status"], "AMBIGUOUS")
                    self.assertIsNone(new["source_step"])
                else:
                    self.assertEqual(new, old)
                    if (
                        old["historical_status"] == "OBSERVED"
                        and isinstance(source, int)
                        and not isinstance(source, bool)
                        and 0 <= source < int(request["decision_step"])
                    ):
                        valid_observed_sources += 1
                total_diffs.extend(
                    (str(request["request_id"]), atom_id, key)
                    for key in changed_fields
                )
            self.assertEqual(
                set(operations),
                {
                    f"PROJECT_UNSUPPORTED_HISTORY:{atom_id}"
                    for atom_id in expected_operation_ids
                },
            )

            second, second_operations = RUNNER.project_unsupported_history(
                projected,
                expected_atom_ids=RUNNER.v1r2._expected_atom_ids(request),
                decision_step=int(request["decision_step"]),
                allowed_candidate_ids=list(
                    request["candidate_alias_to_action_id"]
                ),
                graph=record["graph"],
            )
            self.assertEqual(second, projected)
            self.assertEqual(second_operations, ())

        self.assertGreater(valid_observed_sources, 0)
        self.assertEqual(len(total_diffs), 32)
        self.assertEqual(
            sum(field == "historical_status" for _, _, field in total_diffs),
            20,
        )
        self.assertEqual(
            sum(field == "source_step" for _, _, field in total_diffs), 12
        )

    def test_projection_refuses_schema_or_nonobserved_source_repair(self):
        record = self.failed_records()[0]
        request = record["request"]
        projected, _ = _project(record)

        extra_field = copy.deepcopy(projected)
        extra_field["atoms"][0]["unexpected_projection_input"] = True
        with self.assertRaises((RUNNER.V1R3Error, RUNNER.v1r2.V1R2Error)):
            RUNNER.project_unsupported_history(
                extra_field,
                expected_atom_ids=RUNNER.v1r2._expected_atom_ids(request),
                decision_step=int(request["decision_step"]),
                allowed_candidate_ids=list(
                    request["candidate_alias_to_action_id"]
                ),
                graph=record["graph"],
            )

        nonobserved_source = copy.deepcopy(projected)
        row = next(
            value
            for value in nonobserved_source["atoms"]
            if value["historical_status"] != "OBSERVED"
        )
        row["source_step"] = 0
        with self.assertRaises(MF3ZUContractError):
            RUNNER.project_unsupported_history(
                nonobserved_source,
                expected_atom_ids=RUNNER.v1r2._expected_atom_ids(request),
                decision_step=int(request["decision_step"]),
                allowed_candidate_ids=list(
                    request["candidate_alias_to_action_id"]
                ),
                graph=record["graph"],
            )

    def test_parent_pass_is_accepted_only_as_a_byte_copy(self):
        record = next(
            value
            for value in self.bundle["records"]
            if value["mode"] == "PARENT_PASS_BYTE_REUSE"
        )
        request_id = str(record["request"]["request_id"])
        source = (
            RUNNER.PARENT_OUTPUT
            / "responses/evidence"
            / f"{request_id}.json"
        )
        source_bytes = source.read_bytes()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            destination = output / "responses/evidence" / source.name
            destination.parent.mkdir(parents=True)
            destination.write_bytes(source_bytes)
            value = RUNNER._validate_output_response(
                record=record,
                output=output,
                protocol_sha256="0" * 64,
            )
            self.assertEqual(destination.read_bytes(), source_bytes)
            self.assertEqual(value, json.loads(source_bytes))

            # Semantically equivalent trailing whitespace is still a byte
            # mutation and must not be accepted as parent PASS reuse.
            destination.write_bytes(source_bytes + b" ")
            with self.assertRaisesRegex(
                RUNNER.V1R3Error, "not byte-preserved"
            ):
                RUNNER._validate_output_response(
                    record=record,
                    output=output,
                    protocol_sha256="0" * 64,
                )

    def test_projected_bundle_has_21_by_16_support_fail(self):
        required_events: set[str] = set()
        required_scenes: set[str] = set()
        failed_events = {
            str(record["request"]["event_id"])
            for record in self.failed_records()
        }
        for record in self.bundle["records"]:
            request = record["request"]
            if record["mode"] == "PARENT_PASS_BYTE_REUSE":
                response = record["response"]["response"]
            else:
                response, _ = _project(record)
            judgements = validate_evidence_response(
                response,
                graph=record["graph"],
                decision_step=int(request["decision_step"]),
                allowed_candidate_ids=list(
                    request["candidate_alias_to_action_id"]
                ),
            )
            if memory_required(judgements):
                required_events.add(str(request["event_id"]))
                required_scenes.add(str(request["scene_id"]))

        self.assertEqual(len(required_events), 21)
        self.assertEqual(len(required_scenes), 16)
        self.assertTrue(required_events.isdisjoint(failed_events))
        self.assertLess(len(required_events), 50)

        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("train_mf3zu_rxr_feasibility", source)
        self.assertNotIn("load_frozen_probe_inputs", source)
        self.assertNotIn("MF3ZU_RXR_EXACT_TARGETS", source)
        self.assertNotIn("EXACT_TARGETS_PATH", source)
        self.assertNotIn("performance_result", source)

    def test_support_fail_freeze_never_authorizes_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            (output / RUNNER.PROTOCOL_NAME).write_text(
                '{"status":"fixture-sealed"}\n', encoding="utf-8"
            )
            (output / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json").write_text(
                '{"status":"PASS"}\n', encoding="utf-8"
            )

            def fake_memory_build(destination: Path):
                memory_path = destination / "MF3ZU_RXR_EVIDENCE_MEMORY.jsonl"
                memory_path.write_text("{}\n", encoding="utf-8")
                manifest = {
                    "revision": RUNNER.SCIENTIFIC_REVISION,
                    "status": "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL",
                    "rows": 1428,
                    "episodes": 154,
                    "raw_scenes": 59,
                    "memory_required_support": {
                        "minimum_decisions": 50,
                        "minimum_raw_scenes": 10,
                        "observed_decisions": 21,
                        "observed_raw_scenes": 16,
                        "pass": False,
                    },
                }
                RUNNER.atomic_json(
                    destination
                    / "MF3ZU_RXR_EVIDENCE_MEMORY_MANIFEST.json",
                    manifest,
                )
                return manifest

            bundle_audit = {
                "responses": 1428,
                "candidate_target_accessed": False,
                "performance_accessed": False,
                "public_split_access": False,
            }
            with (
                mock.patch.object(
                    RUNNER, "audit_complete_bundle", return_value=bundle_audit
                ),
                mock.patch.object(RUNNER, "_materialize_parent_views"),
                mock.patch.object(
                    RUNNER.memory_builder,
                    "build",
                    side_effect=fake_memory_build,
                ),
            ):
                provenance = RUNNER.freeze(output)

            support = RUNNER.strict_json(output / RUNNER.SUPPORT_AUDIT_NAME)
            self.assertEqual(
                provenance["status"],
                "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL",
            )
            self.assertEqual(
                support["status"],
                "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL",
            )
            self.assertEqual(support["observed_memory_required_decisions"], 21)
            self.assertEqual(support["observed_memory_required_raw_scenes"], 16)
            self.assertFalse(support["support_pass"])
            self.assertFalse(support["training_started"])
            self.assertFalse(support["training_authorized_by_this_revision"])
            self.assertFalse(provenance["training_run"])
            self.assertFalse(provenance["training_authorized_by_this_revision"])
            self.assertFalse(provenance["candidate_target_accessed"])
            self.assertFalse(provenance["performance_accessed"])
            self.assertFalse(provenance["public_split_access"])

            with (
                mock.patch.object(RUNNER, "DEFAULT_OUTPUT", output),
                mock.patch.object(
                    RUNNER, "freeze", return_value=provenance
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        str(RUNNER_PATH),
                        "freeze",
                        "--output-root",
                        str(output),
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(RUNNER.main(), 3)


if __name__ == "__main__":
    unittest.main()
