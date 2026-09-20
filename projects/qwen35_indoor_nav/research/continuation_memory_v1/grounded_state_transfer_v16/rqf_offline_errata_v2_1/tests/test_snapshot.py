from __future__ import annotations

import os
import unittest

from common import digest_bytes, require_within
from snapshot_io import EvidenceInvalid, EvidenceMissing, SnapshotReader, seal_snapshot
from helpers import temporary_directory


def seal_one(root, relative="source.json", payload=b'{"value":1}\n'):
    source = root / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    expected = digest_bytes(payload)
    manifest = seal_snapshot(
        {"file_records": [{"path": relative, "sha256": expected, "bytes": len(payload)}]},
        [root],
        root / "objects",
        [],
    )
    return source, expected, manifest, SnapshotReader(manifest, root / "objects")


class SnapshotTests(unittest.TestCase):
    def test_S01_source_mutation_does_not_change_snapshot(self):
        with temporary_directory("q35n_s01_") as root:
            source, expected, manifest, reader = seal_one(root)
            before = reader.read_bytes(expected)
            source.write_bytes(b'{"value":2}\n')
            (root / "new.json").write_text("{}")
            after = SnapshotReader(manifest, root / "objects").read_bytes(expected)
            self.assertEqual(before, after)

    def test_S02_object_hash_mismatch_stops(self):
        with temporary_directory("q35n_s02_") as root:
            _, expected, manifest, _ = seal_one(root)
            (root / "objects" / expected).write_bytes(b"corrupt")
            with self.assertRaisesRegex(EvidenceInvalid, "OBJECT_HASH_MISMATCH"):
                SnapshotReader(manifest, root / "objects")

    def test_S03_missing_object_never_falls_back(self):
        with temporary_directory("q35n_s03_") as root:
            source, expected, manifest, _ = seal_one(root)
            (root / "objects" / expected).unlink()
            self.assertTrue(source.exists())
            with self.assertRaisesRegex(EvidenceMissing, "OBJECT_MISSING"):
                SnapshotReader(manifest, root / "objects")

    def test_S04_log_prefix_and_incomplete_tail_are_frozen(self):
        with temporary_directory("q35n_s04_") as root:
            payload = b'{"row":1}\n{"partial":'
            source, expected, manifest, reader = seal_one(root, "events.jsonl", payload)
            record = manifest["records"][0]
            self.assertEqual(record["log_prefix"]["last_complete_line"], 1)
            self.assertGreater(record["log_prefix"]["incomplete_tail_bytes"], 0)
            source.write_bytes(payload + b"2}\n")
            self.assertEqual(reader.read_bytes(expected), payload)

    def test_S05_traversal_and_symlink_escape_are_rejected(self):
        with temporary_directory("q35n_s05_") as root:
            outside = root.parent / f"{root.name}_outside"
            outside.write_text("outside")
            try:
                with self.assertRaisesRegex(ValueError, "PATH_OUTSIDE"):
                    require_within(root / ".." / outside.name, root)
                link = root / "link"
                link.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "SYMLINK"):
                    require_within(link, root)
            finally:
                outside.unlink()

    def test_S06_revision_context_is_not_original_bytes(self):
        with temporary_directory("q35n_s06_") as root:
            context = root / "context.json"
            context.write_text("{}\n")
            manifest = seal_snapshot(
                {"file_records": []},
                [root],
                root / "objects",
                [
                    {
                        "path": context,
                        "allowed_root": root,
                        "source_path": "context.json",
                        "kind": "revision_context",
                        "equivalence": "REVISION_BOUND_CONTEXT",
                    }
                ],
            )
            self.assertEqual(manifest["records"][0]["equivalence"], "REVISION_BOUND_CONTEXT")
            self.assertIsNone(manifest["records"][0]["expected_sha256"])


if __name__ == "__main__":
    unittest.main()
