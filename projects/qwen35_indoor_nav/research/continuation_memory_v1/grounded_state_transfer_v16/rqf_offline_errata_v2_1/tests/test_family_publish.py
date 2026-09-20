from __future__ import annotations

import copy
import json
import multiprocessing
import unittest

from candidate_core import finalize_candidate_result
from common import digest_value
from family_integrity import (
    FamilyValidationError,
    MappingReferenceResolver,
    inspect_existing_publication,
    validate_existing_family_reference,
    validate_new_family_for_commit,
)
from immutable_publish import PublicationDurabilityUnconfirmed, publish_bytes_no_replace
from helpers import full_family_fixture, temporary_directory


def publish_worker(destination, payload, start, queue):
    start.wait()
    try:
        publish_bytes_no_replace(destination, payload)
    except BaseException as exc:
        queue.put(type(exc).__name__)
    else:
        queue.put("SUCCESS")


class FamilyPublishTests(unittest.TestCase):
    def fixture(self):
        return full_family_fixture()

    def test_F01_weak_shell_rejected(self):
        family, resolver, protocol, provenance = self.fixture()
        family["histories"] = {"H_A": []}
        with self.assertRaisesRegex(FamilyValidationError, "HISTORY_KEY_SET"):
            validate_new_family_for_commit(family, resolver, protocol, provenance)

    def test_F02_missing_history_rejected(self):
        family, resolver, protocol, provenance = self.fixture()
        family["histories"].pop("H_B_R")
        with self.assertRaisesRegex(FamilyValidationError, "HISTORY_KEY_SET"):
            validate_new_family_for_commit(family, resolver, protocol, provenance)

    def test_F03_missing_trace_rejected(self):
        family, resolver, protocol, provenance = self.fixture()
        family["traces"].pop("H_A__C_A")
        with self.assertRaisesRegex(FamilyValidationError, "TRACE_KEY_SET"):
            validate_new_family_for_commit(family, resolver, protocol, provenance)

    def test_F04_missing_or_bad_reference_rejected(self):
        family, resolver, protocol, provenance = self.fixture()
        family["traces"]["H_A__C0"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(FamilyValidationError, "REFERENCE_HASH"):
            validate_new_family_for_commit(family, resolver, protocol, provenance)
        family, _, protocol, provenance = self.fixture()
        with self.assertRaises(FileNotFoundError):
            validate_new_family_for_commit(family, MappingReferenceResolver({}), protocol, provenance)

    def test_F04b_existing_certification_survives_unavailable_raw_reference(self):
        family = {
            "family_id": "old",
            "traces": {"H_A__C0": {"path": "unavailable/trace.json", "sha256": "0" * 64}},
        }
        result = validate_existing_family_reference(
            family,
            MappingReferenceResolver({}),
            {"canonical_sha256": digest_value(family)},
        )
        self.assertTrue(result["value_equal"])
        self.assertEqual(result["references_unavailable_or_invalid"], 1)

    def test_F05_cutoff_and_prefix_length_are_exact(self):
        family, resolver, protocol, provenance = self.fixture()
        path = family["traces"]["H_A__C0"]["path"]
        resolver.values[path]["cutoff"] = 0
        family["traces"]["H_A__C0"]["sha256"] = digest_value(resolver.values[path])
        with self.assertRaisesRegex(FamilyValidationError, "TRACE_CUTOFF"):
            validate_new_family_for_commit(family, resolver, protocol, provenance)

    def test_F06_normal_cross_task_fail_is_valid(self):
        family, resolver, protocol, provenance = self.fixture()
        result = validate_new_family_for_commit(family, resolver, protocol, provenance)
        self.assertTrue(result["label_matrix_verified"])
        self.assertEqual(family["traces"]["H_A__C0"]["labels"]["task_B"]["safe_v16_label"], "FAIL")

    def test_F07_unknown_or_unsatisfied_labels_rejected(self):
        family, resolver, protocol, provenance = self.fixture()
        family["traces"]["H_A__C0"]["labels"]["task_A"]["safe_v16_label"] = "UNKNOWN"
        with self.assertRaisesRegex(FamilyValidationError, "LABEL_UNKNOWN"):
            validate_new_family_for_commit(family, resolver, protocol, provenance)

    def test_F08_real_finalize_path_publishes_once(self):
        family, resolver, protocol, provenance = self.fixture()
        with temporary_directory("q35n_f08_") as root:
            receipt = finalize_candidate_result({"family": family}, resolver, root, provenance, protocol)
            self.assertEqual(receipt["publication_calls"], 1)
            published = json.loads((root / family["family_id"] / "FAMILY.json").read_text())
            self.assertEqual(published["content_root"], "fixture/content")
            self.assertEqual(len(published["traces"]), 12)

    def test_F09_second_publish_does_not_change_old_content(self):
        family, resolver, protocol, provenance = self.fixture()
        with temporary_directory("q35n_f09_") as root:
            finalize_candidate_result({"family": family}, resolver, root, provenance, protocol)
            path = root / family["family_id"] / "FAMILY.json"
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                finalize_candidate_result({"family": family}, resolver, root, provenance, protocol)
            self.assertEqual(path.read_bytes(), before)

    def test_F10_competing_publishers_yield_one_complete_payload(self):
        with temporary_directory("q35n_f10_") as root:
            destination = root / "FAMILY.json"
            context = multiprocessing.get_context("fork")
            start = context.Event()
            queue = context.Queue()
            payloads = [b'{"writer":1}\n', b'{"writer":2}\n']
            processes = [context.Process(target=publish_worker, args=(destination, payload, start, queue)) for payload in payloads]
            for process in processes:
                process.start()
            start.set()
            results = [queue.get(timeout=10) for _ in processes]
            for process in processes:
                process.join(10)
            self.assertEqual(results.count("SUCCESS"), 1)
            self.assertIn(destination.read_bytes(), payloads)

    def test_F11_serialization_failure_has_no_final_name(self):
        family, resolver, protocol, provenance = self.fixture()
        family["non_finite"] = float("nan")
        with temporary_directory("q35n_f11_") as root:
            with self.assertRaises(ValueError):
                finalize_candidate_result({"family": family}, resolver, root, provenance, protocol)
            self.assertFalse((root / family["family_id"] / "FAMILY.json").exists())

    def test_F12_temp_fsync_failure_has_no_final_name(self):
        with temporary_directory("q35n_f12_") as root:
            destination = root / "FAMILY.json"

            def fail_first(fd):
                raise OSError("injected file fsync")

            with self.assertRaises(OSError):
                publish_bytes_no_replace(destination, b"{}\n", fsync=fail_first)
            self.assertFalse(destination.exists())

    def test_F13_directory_fsync_failure_preserves_final_and_reports_uncertain(self):
        with temporary_directory("q35n_f13_") as root:
            destination = root / "FAMILY.json"
            calls = 0

            def fail_second(fd):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected directory fsync")

            with self.assertRaises(PublicationDurabilityUnconfirmed):
                publish_bytes_no_replace(destination, b"{}\n", fsync=fail_second)
            self.assertEqual(destination.read_bytes(), b"{}\n")

    def test_F14_recovery_verified_partial_and_conflict(self):
        family, resolver, protocol, provenance = self.fixture()
        with temporary_directory("q35n_f14_") as root:
            same = root / "same.json"
            same.write_text(json.dumps(family))
            self.assertEqual(inspect_existing_publication(same, family, resolver, protocol, provenance)["status"], "RESUMED_VERIFIED")
            partial = root / "partial.json"
            partial.write_text("{")
            self.assertEqual(inspect_existing_publication(partial, family, resolver, protocol, provenance)["status"], "BLOCKED_PARTIAL_PUBLICATION")
            conflict = root / "conflict.json"
            conflict.write_text(json.dumps({"family_id": "other"}))
            self.assertEqual(inspect_existing_publication(conflict, family, resolver, protocol, provenance)["status"], "BLOCKED_PUBLICATION_CONFLICT")


if __name__ == "__main__":
    unittest.main()
