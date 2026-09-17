"""CPU correctness tests and optional real-fsync local-file-system benchmark.

Run as a package with project-local stdlib Python. All temporary artifacts stay
under feedback_generation_v1; cleanup removes only TemporaryDirectory-owned data.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ..guard import BudgetError, ContentStore, StoreError, _npy
from .store import FEEDBACK_ROOT, TrackedContentStore


class TrackedStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cpu_store_test_", dir=FEEDBACK_ROOT)
        self.base = Path(self.tmp.name)
        self.store = TrackedContentStore(self.base / "tracked", 2**20)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def put(self, raw=bytes(range(12)), **kwargs):
        return self.store.put_raw(raw, "rgb", kwargs.get("shape", (2, 2, 3)), "uint8")

    def test_dedup_and_exact_npy_payload(self):
        a = self.put()
        inode = Path(a["path"]).stat().st_ino
        self.assertEqual(a, self.put())
        self.assertEqual(Path(a["path"]).stat().st_ino, inode)
        self.assertEqual(Path(a["path"]).read_bytes(), _npy(bytes(range(12)), (2, 2, 3), "|u1"))
        self.assertEqual(self.store._disk_bytes(), a["file_bytes"])
        self.assertTrue(self.store.full_audit()["audit_pass"])

    def test_sequence_matches_original_and_fast_path_never_lists(self):
        rng = random.Random(351)
        values = [rng.randbytes(12) for _ in range(32)]
        values += values[::3]
        with ContentStore(self.base / "original", 2**20) as old:
            for raw in values:
                expected = old.put_raw(raw, "rgb", (2, 2, 3), "uint8")
                with patch("os.listdir", side_effect=AssertionError("fast path scanned directory")):
                    actual = self.put(raw)
                self.assertEqual({k: v for k, v in actual.items() if k != "path"},
                                 {k: v for k, v in expected.items() if k != "path"})
                self.assertEqual(Path(actual["path"]).read_bytes(), Path(expected["path"]).read_bytes())
                self.assertEqual(self.store._disk_bytes(), old._disk_bytes())
        self.assertTrue(self.store.full_audit()["audit_pass"])

    def test_capacity_boundary_then_fail_closed(self):
        limit = ContentStore.FAILURE_RESERVE + len(_npy(bytes(range(12)), (2, 2, 3), "|u1"))
        with TrackedContentStore(self.base / "tiny", limit) as small:
            small.put_raw(bytes(range(12)), "rgb", (2, 2, 3), "uint8")
            with self.assertRaises(BudgetError):
                small.put_raw(b"x" * 12, "rgb", (2, 2, 3), "uint8")
            self.assertTrue(small.poisoned)
            before = small._disk_bytes()
            with self.assertRaises(StoreError):
                small.put_raw(bytes(range(12)), "rgb", (2, 2, 3), "uint8")
            self.assertEqual(before, small._disk_bytes())
            self.assertLessEqual(before, limit)
            self.assertFalse(small.full_audit()["audit_pass"])

    def test_failed_commit_keeps_partial_and_journal(self):
        with patch("os.link", side_effect=OSError("injected link failure")):
            with self.assertRaises(OSError):
                self.put()
        self.assertTrue(self.store.poisoned)
        self.assertEqual(len(list(self.store.root.glob("*.partial"))), 1)
        event = json.loads((self.store.root / "FAILURES.jsonl").read_text())
        self.assertEqual(event["event"], "STORE_WRITE_REJECTED")
        actual = ContentStore._disk_bytes(self.store)
        self.assertEqual(actual, self.store.failure_disk_bytes)
        self.assertEqual(actual, self.store._disk_bytes())
        self.assertFalse(self.store.full_audit()["accounting_pass"])

    def test_fsync_and_failed_journal_bytes_are_charged(self):
        with patch("os.fsync", side_effect=OSError("injected fsync failure")):
            with self.assertRaises(OSError):
                self.put()
        self.assertEqual(self.store._journal_bytes, 0)
        self.assertGreater((self.store.root / "FAILURES.jsonl").stat().st_size, 0)
        self.assertEqual(self.store._disk_bytes(), ContentStore._disk_bytes(self.store))
        self.assertFalse(self.store.full_audit()["audit_pass"])

    def test_post_commit_unlink_failure_counts_both_names(self):
        with patch("os.unlink", side_effect=OSError("injected unlink failure")):
            with self.assertRaises(OSError):
                self.put()
        names = list(self.store.root.iterdir())
        self.assertEqual(len(names), 3)
        self.assertEqual(self.store._disk_bytes(), sum(p.lstat().st_size for p in names))
        self.assertFalse(self.store.full_audit()["audit_pass"])

    def test_partial_payload_write_is_counted(self):
        original_fdopen = os.fdopen
        class BrokenWriter:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def write(self, payload):
                self.stream.write(payload[:11])
                self.stream.flush()
                raise OSError("injected partial payload write")
        def fdopen(fd, mode, **kwargs):
            stream = original_fdopen(fd, mode, **kwargs)
            return BrokenWriter(stream) if mode == "wb" else stream
        with patch("os.fdopen", side_effect=fdopen):
            with self.assertRaises(OSError):
                self.put()
        partial = list(self.store.root.glob("*.partial"))
        self.assertEqual(len(partial), 1)
        self.assertEqual(partial[0].stat().st_size, 11)
        self.assertEqual(self.store.failure_disk_bytes, ContentStore._disk_bytes(self.store))
        self.assertFalse(self.store.full_audit()["audit_pass"])

    def test_semantic_and_array_interfaces(self):
        class Array:
            dtype, shape = "uint32", (1, 2)
            def tobytes(self, order):
                assert order == "C"
                return b"\x01\x00\x00\x00\x02\x00\x00\x00"
        item = self.store.put_array(Array(), "semantic")
        self.assertEqual(item["dtype"], "<u4")
        self.assertEqual(Path(item["path"]).read_bytes(), _npy(Array().tobytes("C"), (1, 2), "<u4"))
        self.assertTrue(self.store.full_audit()["audit_pass"])

    def test_invalid_input_keeps_original_journal_and_poisons(self):
        with self.assertRaises(ValueError):
            self.put(b"bad")
        self.assertTrue(self.store.poisoned)
        self.assertTrue((self.store.root / "FAILURES.jsonl").exists())
        self.assertEqual(self.store._disk_bytes(), ContentStore._disk_bytes(self.store))

    def test_corruption_detected_by_inherited_put(self):
        a = self.put()
        Path(a["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(StoreError, "corrupt"):
            self.put()
        self.assertEqual(Path(a["path"]).read_bytes(), b"corrupt")
        self.assertTrue(self.store.poisoned)

    def test_conflicting_shape_rejected(self):
        self.put()
        with self.assertRaises(StoreError):
            self.put(shape=(1, 4, 3))
        self.assertTrue(self.store.poisoned)

    def test_audit_hash_detects_same_size_inplace_tamper(self):
        a = self.put()
        path = Path(a["path"])
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        report = self.store.full_audit()
        self.assertTrue(report["accounting_pass"])
        self.assertFalse(report["audit_pass"])

    def test_symlink_scope_and_foreign_entries(self):
        alias = self.base / "alias"
        alias.symlink_to(self.store.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            TrackedContentStore(alias / "store", 2**20)
        with self.assertRaises(ValueError):
            TrackedContentStore(FEEDBACK_ROOT.parent / "forbidden_store", 2**20)
        protected = self.base / "protected"
        protected.write_bytes(b"protected")
        (self.store.root / "foreign").symlink_to(protected)
        with self.assertRaises(StoreError):
            self.put()
        self.assertFalse(self.store.full_audit()["audit_pass"])
        self.assertEqual(protected.read_bytes(), b"protected")

    def test_root_identity_and_close(self):
        self.store.close()
        self.assertTrue(self.store.final_audit["audit_pass"])
        with self.assertRaises(StoreError):
            self.put()
        self.store.close()

    def test_replaced_root_fails_without_following_new_directory(self):
        self.store.root.rename(self.base / "old_root")
        self.store.root.mkdir()
        with self.assertRaisesRegex(StoreError, "identity changed"):
            self.put()
        self.assertEqual(list(self.store.root.iterdir()), [])
        self.store.close()
        self.assertFalse(self.store.final_audit["audit_pass"])

    def test_exception_interruption_also_poisoned(self):
        with patch("os.link", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.put()
        self.assertTrue(self.store.poisoned)
        self.assertGreater(self.store.failure_disk_bytes, 0)

    def test_single_writer_thread_rejected(self):
        errors = []
        def other_writer():
            try:
                self.put()
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=other_writer)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], StoreError)
        self.assertTrue(self.store.full_audit()["audit_pass"])

    def test_exact_source_direct_script_import(self):
        spec = importlib.util.spec_from_file_location("tracked_store_direct_test", FEEDBACK_ROOT / "store.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with module.TrackedContentStore(self.base / "direct", 2**20) as direct:
            direct.put_raw(bytes(range(12)), "rgb", (2, 2, 3), "uint8")
            self.assertTrue(direct.full_audit()["audit_pass"])


def benchmark(count=500, seed=351):
    """Same deterministic RGB payload sequence; includes unmodified real fsync.

    CPU/local storage only, not a simulator/GPU or end-to-end speedup claim.
    Alternating run order avoids reporting one lucky cache/order measurement.
    """
    rng = random.Random(seed)
    values = [rng.randbytes(16 * 16 * 3) for _ in range(count)]
    digest = hashlib.sha256(b"".join(values)).hexdigest()
    rounds = []
    with tempfile.TemporaryDirectory(prefix="cpu_store_bench_", dir=FEEDBACK_ROOT) as tmp:
        for round_id, order in enumerate(((ContentStore, TrackedContentStore),
                                         (TrackedContentStore, ContentStore))):
            measurement = {}
            for klass in order:
                store = klass(Path(tmp) / f"{round_id}_{klass.__name__}", 2**24)
                started = time.perf_counter()
                for raw in values:
                    store.put_raw(raw, "rgb", (16, 16, 3), "uint8")
                elapsed = time.perf_counter() - started
                measurement[klass.__name__] = elapsed
                if isinstance(store, TrackedContentStore):
                    assert store.full_audit()["audit_pass"]
                store.close()
            rounds.append(measurement)
    return {"kind": "CPU_REAL_FSYNC_STORE_ONLY", "records_per_run": count,
            "payload_sequence_sha256": digest, "rounds_seconds": rounds,
            "median_speedup": sum(r["ContentStore"] for r in rounds) /
                               sum(r["TrackedContentStore"] for r in rounds)}


if __name__ == "__main__":
    import sys
    if "--benchmark" in sys.argv:
        print(json.dumps(benchmark(), indent=2))
    else:
        unittest.main()
