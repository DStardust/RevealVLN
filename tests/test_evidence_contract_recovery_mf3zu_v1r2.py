import copy
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from revealnav_mf3.mf3zu_evidence_memory import (
    MF3ZUContractError,
    parse_instruction_response,
    validate_evidence_response,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


recovery = _load_script(
    "mf3zu_evidence_contract_recovery_v1r2_test_module",
    "scripts/run_mf3zu_evidence_contract_recovery_v1r2.py",
)


def _graph():
    return parse_instruction_response(
        {
            "instruction_atoms": [
                {
                    "instruction_atom_id": "a01",
                    "text": "pass the painting",
                    "semantic_kind": "PASSING",
                    "depends_on": [],
                },
                {
                    "instruction_atom_id": "a02",
                    "text": "turn left",
                    "semantic_kind": "DIRECTION",
                    "depends_on": ["a01"],
                },
            ]
        },
        instruction="Pass the painting, then turn left.",
    )


def _atom(
    atom_id: str,
    *,
    active: bool = False,
    relevant: bool = False,
    historical: str = "ABSENT",
    current: str = "ABSENT",
    source_step=None,
    candidate_ids=None,
    semantic: str = "evidence is absent",
):
    return {
        "instruction_atom_id": atom_id,
        "active_for_current_ranking": active,
        "relevant_to_current_ranking": relevant,
        "historical_status": historical,
        "current_status": current,
        "source_step": source_step,
        "candidate_ids": list(candidate_ids or []),
        "semantic_value": semantic,
    }


def _valid_response(*, semantic: str = "painting was passed"):
    return {
        "atoms": [
            _atom(
                "a01",
                active=True,
                relevant=True,
                historical="OBSERVED",
                current="ABSENT",
                source_step=1,
                candidate_ids=["C00"],
                semantic=semantic,
            ),
            _atom("a02", semantic="turn evidence is absent"),
        ]
    }


def _write_json(path: Path, value: object, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        payload = json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False
        ) + "\n"
    else:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n"
    path.write_text(payload, encoding="utf-8")


class Mf3zuEvidenceContractRecoveryV1r2Test(unittest.TestCase):
    expected_ids = ("a01", "a02")
    allowed_candidates = ("C00", "C01")
    decision_step = 3

    def _classify(self, value):
        return recovery.classify_parent_failure(
            value,
            expected_atom_ids=self.expected_ids,
            decision_step=self.decision_step,
            allowed_candidate_ids=self.allowed_candidates,
        )

    def _canonicalize(self, value):
        return recovery.canonicalize_mechanical_response(
            value,
            expected_atom_ids=self.expected_ids,
            decision_step=self.decision_step,
            allowed_candidate_ids=self.allowed_candidates,
            graph=_graph(),
        )

    def test_mechanical_repairs_are_whitelisted_idempotent_and_validator_valid(self):
        # These are the only semantics-preserving repairs admitted by v1r2:
        # logical closure, clearing an inapplicable source, one uniquely
        # recoverable atom ID, the three observed provider extras, and trim.
        first = _atom(
            "a01",
            active=False,
            relevant=True,
            historical="OBSERVED",
            current="OBSERVED",
            source_step=1,
            candidate_ids=["C00"],
            semantic="  painting is visible now  ",
        )
        del first["instruction_atom_id"]
        first["relative_heading_rad"] = 0.25
        second = _atom(
            "a02",
            historical="AMBIGUOUS",
            current="ABSENT",
            source_step=0,
            semantic="turn evidence remains ambiguous",
        )
        second["informative_value"] = "ignored provider extra"
        second["standard_deviation"] = 0.0
        invalid = {"atoms": [first, second]}
        before = copy.deepcopy(invalid)

        analysis = self._classify(invalid)
        self.assertEqual(analysis.mode, recovery.MECHANICAL)
        expected_operations = {
            "CLOSE_RELEVANCE_IMPLIES_ACTIVE",
            "CLEAR_NONOBSERVED_SOURCE",
            "FILL_UNIQUE_MISSING_ATOM_ID",
            "DROP_relative_heading_rad",
            "DROP_informative_value",
            "DROP_standard_deviation",
            "STRIP_SEMANTIC_VALUE",
        }
        self.assertEqual(set(analysis.predicted_operations), expected_operations)

        normalized, operations = self._canonicalize(invalid)
        self.assertEqual(invalid, before, "raw parent response was mutated")
        self.assertEqual(set(operations), expected_operations)
        self.assertEqual(
            [row["instruction_atom_id"] for row in normalized["atoms"]],
            ["a01", "a02"],
        )
        self.assertIs(normalized["atoms"][0]["active_for_current_ranking"], True)
        self.assertIsNone(normalized["atoms"][1]["source_step"])
        self.assertEqual(
            normalized["atoms"][0]["semantic_value"],
            "painting is visible now",
        )
        for row in normalized["atoms"]:
            self.assertEqual(set(row), recovery.REQUIRED_ATOM_KEYS)

        judgements = validate_evidence_response(
            normalized,
            graph=_graph(),
            decision_step=self.decision_step,
            allowed_candidate_ids=self.allowed_candidates,
        )
        self.assertEqual(len(judgements), 2)

        normalized_again, second_operations = self._canonicalize(normalized)
        self.assertEqual(normalized_again, normalized)
        self.assertEqual(second_operations, ())

    def test_bad_observed_source_and_long_semantic_require_reannotation(self):
        cases = {
            "source_at_current_step": _valid_response(),
            "source_none": _valid_response(),
            "source_bool": _valid_response(),
            "source_negative": _valid_response(),
            "source_after_current": _valid_response(),
            "semantic_over_parent_limit": _valid_response(
                semantic="x" * 501
            ),
        }
        cases["source_at_current_step"]["atoms"][0]["source_step"] = 3
        cases["source_none"]["atoms"][0]["source_step"] = None
        cases["source_bool"]["atoms"][0]["source_step"] = True
        cases["source_negative"]["atoms"][0]["source_step"] = -1
        cases["source_after_current"]["atoms"][0]["source_step"] = 4

        for name, response in cases.items():
            with self.subTest(name=name):
                analysis = self._classify(response)
                self.assertEqual(analysis.mode, recovery.REANNOTATE)
                self.assertFalse(
                    any(
                        "DEMOT" in operation or "TRUNCAT" in operation
                        for operation in analysis.predicted_operations
                    )
                )
                with self.assertRaisesRegex(
                    recovery.V1R2Error,
                    "cannot be mechanically repaired",
                ):
                    self._canonicalize(response)

    def test_unknown_schema_identity_and_sensitive_fields_fail_closed(self):
        unknown_extra = _valid_response()
        unknown_extra["atoms"][0]["target_index"] = 0
        missing_required = _valid_response()
        del missing_required["atoms"][0]["semantic_value"]
        two_missing_ids = _valid_response()
        del two_missing_ids["atoms"][0]["instruction_atom_id"]
        del two_missing_ids["atoms"][1]["instruction_atom_id"]
        duplicate_ids = _valid_response()
        duplicate_ids["atoms"][1]["instruction_atom_id"] = "a01"
        unknown_candidate = _valid_response()
        unknown_candidate["atoms"][0]["candidate_ids"] = ["C99"]
        unknown_top_level = {
            "atoms": _valid_response()["atoms"],
            "performance": {"accuracy": 1.0},
        }

        for name, response in {
            "unknown_target_extra": unknown_extra,
            "missing_required": missing_required,
            "ambiguous_missing_ids": two_missing_ids,
            "duplicate_ids": duplicate_ids,
            "unknown_candidate": unknown_candidate,
            "performance_top_level": unknown_top_level,
        }.items():
            with self.subTest(name=name), self.assertRaises(recovery.V1R2Error):
                self._classify(response)

    def test_sealed_parent_failure_partition_is_exactly_163_and_89(self):
        parent = recovery.PARENT_OUTPUT
        manifest_path = parent / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json"
        requests_path = parent / "MF3ZU_EVIDENCE_REQUESTS.jsonl"
        if not manifest_path.is_file() or not requests_path.is_file():
            self.skipTest("sealed v1r1 response bundle is not installed")

        manifest = recovery.strict_json(manifest_path)
        failures = manifest.get("failures")
        self.assertIsInstance(failures, list)
        self.assertEqual(len(failures), recovery.EXPECTED_PARENT_FAIL)
        requests = {
            str(row["request_id"]): row
            for row in recovery.jsonl(requests_path)
        }
        modes = Counter()
        observed_bad_source = 0
        semantic_too_long = 0
        for failure in failures:
            request_id = str(failure["request_id"])
            request = requests[request_id]
            response = recovery.strict_json(
                parent / "responses/evidence" / f"{request_id}.json"
            )
            analysis = recovery.classify_parent_failure(
                response["invalid_parsed_response"],
                expected_atom_ids=recovery._expected_atom_ids(request),
                decision_step=int(request["decision_step"]),
                allowed_candidate_ids=list(
                    request["candidate_alias_to_action_id"]
                ),
            )
            modes[analysis.mode] += 1
            bad_source = "OBSERVED_HISTORY_HAS_NO_CAUSAL_SOURCE" in (
                analysis.issue_codes
            )
            long_semantic = "SEMANTIC_VALUE_EXCEEDS_500" in (
                analysis.issue_codes
            )
            observed_bad_source += int(bad_source)
            semantic_too_long += int(long_semantic)
            self.assertEqual(
                analysis.mode,
                recovery.REANNOTATE
                if bad_source or long_semantic
                else recovery.MECHANICAL,
            )

        self.assertEqual(
            modes,
            Counter(
                {
                    recovery.MECHANICAL: 163,
                    recovery.REANNOTATE: 89,
                }
            ),
        )
        self.assertEqual(observed_bad_source, 76)
        self.assertEqual(semantic_too_long, 14)
        source = (
            ROOT / "scripts/run_mf3zu_evidence_contract_recovery_v1r2.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("MF3ZU_RXR_EXACT_TARGETS", source)
        self.assertNotIn("EXACT_TARGETS_PATH", source)

    def test_parent_pass_is_reused_byte_for_byte(self):
        graph = _graph()
        request = {
            "request_id": "parent-pass-request",
            "event_id": "event-1",
            "decision_step": self.decision_step,
            "candidate_alias_to_action_id": {"C00": "g0", "C01": "g1"},
        }
        response = {"status": "PASS", "response": _valid_response()}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            parent = base / "parent"
            output = base / "output"
            source = parent / "responses/evidence/parent-pass-request.json"
            destination = output / "responses/evidence/parent-pass-request.json"
            _write_json(source, response, pretty=True)
            before = source.read_bytes()
            recovery.atomic_copy(source, destination)
            record = {
                "mode": recovery.PARENT_PASS,
                "request": request,
                "graph": graph,
            }
            with mock.patch.object(recovery, "PARENT_OUTPUT", parent):
                value = recovery._validate_recovered_response(
                    record=record,
                    output=output,
                    protocol_sha256="a" * 64,
                )
                self.assertEqual(value, response)
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(destination.read_bytes(), before)

                # Reformatting the same JSON is still forbidden: byte reuse is
                # the invariant, not merely parsed-value equality.
                _write_json(destination, response, pretty=False)
                with self.assertRaisesRegex(
                    recovery.V1R2Error, "not byte-preserved"
                ):
                    recovery._validate_recovered_response(
                        record=record,
                        output=output,
                        protocol_sha256="a" * 64,
                    )

    def test_fixed_reannotation_uses_original_validator_without_normalization(self):
        # The clarified prompt and unchanged parent validator both permit up
        # to 500 characters.  A 241..500 response therefore remains valid and
        # is passed through without any post-response edit.
        response = _valid_response(semantic="x" * 300)
        validate_evidence_response(
            response,
            graph=_graph(),
            decision_step=self.decision_step,
            allowed_candidate_ids=self.allowed_candidates,
        )
        recovery.validate_fixed_reannotation(
            response,
            graph=_graph(),
            decision_step=self.decision_step,
            allowed_candidate_ids=self.allowed_candidates,
        )
        with self.assertRaises(MF3ZUContractError):
            recovery.validate_fixed_reannotation(
                _valid_response(semantic="x" * 501),
                graph=_graph(),
                decision_step=self.decision_step,
                allowed_candidate_ids=self.allowed_candidates,
            )

    def test_invalid_reannotation_is_retained_once_and_never_canonicalized(self):
        graph = _graph()
        request = {
            "request_id": "fixed-reannotation-request",
            "event_id": "event-2",
            "decision_step": self.decision_step,
            "candidate_alias_to_action_id": {"C00": "g0", "C01": "g1"},
        }
        invalid = _valid_response()
        invalid["atoms"][0]["source_step"] = self.decision_step
        raw = json.dumps(invalid, sort_keys=True)
        provider_value = {
            "status": "PARSED_JSON",
            "provider_model": recovery.QWEN_MODEL,
            "response": invalid,
            "raw_content": raw,
            "usage": {"total_tokens": 1},
            "transport_attempts": [
                {"attempt": 1, "status": "PARSED_JSON"}
            ],
        }
        record = {
            "mode": recovery.REANNOTATE,
            "request": request,
            "graph": graph,
            "response_inventory": {
                "path": "parent/response.json",
                "bytes": 1,
                "sha256": "b" * 64,
            },
        }
        payload = {
            "model": recovery.QWEN_MODEL,
            "messages": [{"role": "system", "content": "fixed"}],
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            with (
                mock.patch.object(
                    recovery, "_reannotation_payload", return_value=payload
                ),
                mock.patch.object(
                    recovery.v1r1,
                    "_provider_request_preserving",
                    return_value=provider_value,
                ) as provider_call,
                mock.patch.object(
                    recovery,
                    "canonicalize_mechanical_response",
                    side_effect=AssertionError(
                        "reannotation must not be post-canonicalized"
                    ),
                ),
            ):
                first = recovery._reannotation_response_value(
                    api_key="unused",
                    record=record,
                    output=output,
                    protocol_sha256="c" * 64,
                )
                destination = (
                    output
                    / "responses/evidence/fixed-reannotation-request.json"
                )
                before = destination.read_bytes()
                second = recovery._reannotation_response_value(
                    api_key="unused",
                    record=record,
                    output=output,
                    protocol_sha256="c" * 64,
                )

            self.assertEqual(provider_call.call_count, 1)
            self.assertEqual(first["status"], "FAIL")
            self.assertEqual(first["raw_provider_content"], raw)
            self.assertEqual(first["invalid_parsed_response"], invalid)
            self.assertEqual(second, first)
            self.assertEqual(destination.read_bytes(), before)

    def test_protocol_keeps_outcomes_out_and_support_failure_stops_training(self):
        fake_inventory = {
            "path": "sealed/input.json",
            "bytes": 1,
            "sha256": "d" * 64,
        }
        bundle = {
            "input_manifest": {"requests": fake_inventory},
            "response_bundle_sha256": "e" * 64,
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            with (
                mock.patch.object(
                    recovery, "inventory", return_value=fake_inventory
                ),
                mock.patch.object(
                    recovery,
                    "_implementation_inventory",
                    return_value={"runner": fake_inventory},
                ),
            ):
                protocol = recovery._protocol_value(
                    bundle=bundle,
                    source_commit="f" * 40,
                    output=Path(directory),
                )

        self.assertEqual(
            protocol["fixed_partition"],
            {
                "parent_pass_byte_reuse": 1176,
                "mechanical_repair": 163,
                "fixed_reannotation": 89,
                "selection_uses_contract_errors_only": True,
                "selection_uses_support_or_performance": False,
            },
        )
        self.assertIs(
            protocol["fixed_reannotation"]["post_response_canonicalization"],
            False,
        )
        self.assertIs(protocol["fixed_reannotation"]["semantic_retry"], False)
        self.assertEqual(
            protocol["inherited_support_gate"]["minimum_decisions"], 50
        )
        self.assertEqual(
            protocol["inherited_support_gate"]["minimum_raw_scenes"], 10
        )
        self.assertIs(
            protocol["inherited_support_gate"][
                "support_failure_stops_before_training"
            ],
            True,
        )
        boundary = protocol["boundary"]
        self.assertIs(boundary["candidate_target_accessed"], False)
        self.assertIs(boundary["performance_accessed"], False)
        self.assertIs(boundary["outcome_or_utility_accessed"], False)
        self.assertIs(boundary["training_authorized_by_this_revision"], False)
        self.assertIs(boundary["training_run"], False)


if __name__ == "__main__":
    unittest.main()
