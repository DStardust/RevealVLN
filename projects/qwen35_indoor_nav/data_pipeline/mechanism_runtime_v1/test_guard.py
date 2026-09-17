"""Standard-library tests, no simulator, GPU, signal or external writes."""
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

try:
    from .guard import ContentStore, ResourceGuard, BudgetError, StoreError, RUNTIME_ROOT
except ImportError:
    from guard import ContentStore, ResourceGuard, BudgetError, StoreError, RUNTIME_ROOT


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="guard_test_", dir=RUNTIME_ROOT)
        self.base = Path(self.temp.name)
        self.store = ContentStore(self.base / "store", 1024 * 1024)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def rgb(self):
        return self.store.put_raw(bytes(range(12)), "rgb", (2, 2, 3), "uint8")

    def test_npy_roundtrip_without_numpy(self):
        item = self.rgb()
        data = Path(item["path"]).read_bytes()
        self.assertEqual(data[:8], b"\x93NUMPY\x01\x00")
        size = struct.unpack("<H", data[8:10])[0]
        header = ast.literal_eval(data[10:10+size].decode("ascii").strip())
        self.assertEqual(header, {"descr": "|u1", "fortran_order": False, "shape": (2, 2, 3)})
        self.assertEqual((10+size) % 64, 0)
        self.assertEqual(data[10+size:], bytes(range(12)))
        self.assertEqual(item["pixel_sha256"], hashlib.sha256(bytes(range(12))).hexdigest())
        self.assertEqual(item["sha256"], hashlib.sha256(data).hexdigest())

    def test_semantic_uint32(self):
        item = self.store.put_raw(struct.pack("<4I", 0, 1, 256, 65536), "semantic", (2, 2), "uint32")
        self.assertEqual(item["nbytes"], 16)
        self.assertEqual(item["dtype"], "<u4")

    def test_numpy_like_api(self):
        class Array:
            shape, dtype = (2, 2, 3), "uint8"
            def tobytes(self, order):
                assert order == "C"
                return bytes(range(12))
        self.assertEqual(self.store.put_array(Array(), "rgb"), self.rgb())

    def test_dedup_no_new_write(self):
        first = self.rgb()
        inode = Path(first["path"]).stat().st_ino
        self.assertEqual(first, self.rgb())
        self.assertEqual(Path(first["path"]).stat().st_ino, inode)
        self.assertEqual(len(list(self.store.root.iterdir())), 1)

    def test_existing_root_rejected(self):
        with self.assertRaises(FileExistsError):
            ContentStore(self.store.root, 1024 * 1024)

    def test_outside_root_rejected_before_create(self):
        with self.assertRaises(ValueError):
            ContentStore(RUNTIME_ROOT.parent / "not_authorized_guard_test", 1024 * 1024)

    def test_root_itself_rejected(self):
        with self.assertRaises(ValueError):
            ContentStore(RUNTIME_ROOT, 1024 * 1024)

    def test_symlink_parent_rejected(self):
        alias = self.base / "alias"
        alias.symlink_to(self.store.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            ContentStore(alias / "child", 1024 * 1024)

    def test_corrupt_existing_never_overwritten(self):
        item = self.rgb()
        target = Path(item["path"])
        target.write_bytes(b"broken")
        with self.assertRaises(StoreError):
            self.rgb()
        self.assertEqual(target.read_bytes(), b"broken")
        event = json.loads((self.store.root / "FAILURES.jsonl").read_text())
        self.assertEqual(event["event"], "STORE_WRITE_REJECTED")

    def test_same_raw_conflicting_shape_rejected(self):
        self.rgb()
        with self.assertRaises(StoreError):
            self.store.put_raw(bytes(range(12)), "rgb", (1, 4, 3), "uint8")

    def test_symlink_content_rejected(self):
        item = self.rgb()
        target = Path(item["path"])
        local = self.base / "protected"
        local.write_bytes(b"protected")
        target.unlink()
        target.symlink_to(local)
        with self.assertRaises(OSError):
            self.rgb()
        self.assertEqual(local.read_bytes(), b"protected")

    def test_budget_before_content_write(self):
        with ContentStore(self.base / "small", ContentStore.FAILURE_RESERVE + 1) as small:
            with self.assertRaises(BudgetError) as context:
                small.put_raw(b"\0\0\0", "rgb", (1, 1, 3), "uint8")
            self.assertTrue(context.exception.resource_censored)
            self.assertEqual(list(small.root.glob("*.npy")), [])
            self.assertLessEqual(sum(p.stat().st_size for p in small.root.iterdir()), small.max_bytes)

    def test_failed_commit_leaves_partial_and_journal(self):
        with patch("os.link", side_effect=OSError("injected atomic commit failure")):
            with self.assertRaises(OSError):
                self.rgb()
        self.assertEqual(list(self.store.root.glob("*.npy")), [])
        self.assertEqual(len(list(self.store.root.glob("*.partial"))), 1)
        self.assertTrue((self.store.root / "FAILURES.jsonl").exists())

    def test_shape_dtype_and_size_validation(self):
        cases = [(b"abc", "rgb", (1, 1, 4), "uint8"),
                 (b"abc", "rgb", (1, 1, 3), "float32"),
                 (b"abc", "rgb", (True, 1, 3), "uint8"),
                 (b"abc", "semantic", (1, 1), "uint32"),
                 (b"abcd", "semantic", (1, 1), ">u4"),
                 (b"abc", "../bad", (1, 1, 3), "uint8")]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.store.put_raw(*args)
        self.assertEqual(list(self.store.root.glob("*.npy")), [])

    def test_closed_store_rejected(self):
        self.store.close()
        with self.assertRaises(StoreError):
            self.rgb()

    def test_resource_boundary(self):
        guard = ResourceGuard(10, 20, 30)
        self.assertTrue(guard.check({"wall_seconds": 10, "ram_bytes": 20, "disk_bytes": 30})["resource_pass"])

    def test_each_resource_excess(self):
        for key in ("wall_seconds", "ram_bytes", "disk_bytes"):
            values = {"wall_seconds": 1, "ram_bytes": 1, "disk_bytes": 1}
            values[key] = 2
            with self.subTest(key=key), self.assertRaises(BudgetError) as context:
                ResourceGuard(1, 1, 1).check(values)
            self.assertEqual(context.exception.as_dict()["reason"], "RESOURCE_CENSORED")

    def test_bad_samples_and_limits(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ResourceGuard(value, 1, 1)
            with self.subTest(value=value), self.assertRaises(ValueError):
                ResourceGuard(1, 1, 1).check({"wall_seconds": value, "ram_bytes": 0, "disk_bytes": 0})
        with self.assertRaises(KeyError):
            ResourceGuard(1, 1, 1).check({})


if __name__ == "__main__":
    unittest.main()
