"""Single-writer incremental accounting over the sealed raw NPY store.

Only successful commits enter the fast counter. Any exception poisons this new,
non-resumable store: partial commits/journal writes are inventoried, never ignored
or retried. Payload creation, dedup verification, fsync and atomic hard-link commit
remain the original implementation. No simulator, model, or process control.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import threading

if __package__:
    from ..guard import ContentStore, StoreError
else:
    # The worker also has a direct-script entry. Resolve the sealed dependency
    # by its exact source path, never by ambient sys.path or a caught ImportError.
    import importlib.util
    import sys
    _guard_path = Path(__file__).resolve().parent.parent / "guard.py"
    _guard_module = sys.modules.get("guard")
    if _guard_module is not None:
        if Path(getattr(_guard_module, "__file__", "")).resolve() != _guard_path:
            raise ImportError("unrelated module already occupies sealed guard name")
    else:
        _guard_spec = importlib.util.spec_from_file_location("guard", _guard_path)
        _guard_module = importlib.util.module_from_spec(_guard_spec)
        sys.modules["guard"] = _guard_module
        _guard_spec.loader.exec_module(_guard_module)
    ContentStore, StoreError = _guard_module.ContentStore, _guard_module.StoreError

FEEDBACK_ROOT = Path(__file__).resolve().parent


class TrackedContentStore(ContentStore):
    """New child of feedback_generation_v1; one owning process and thread only.

    Call ``full_audit()`` before accepting a dataset and require ``audit_pass``.
    ``close()`` also stores that audit in ``final_audit`` and always closes the FD;
    it deliberately does not mask an in-flight exception with an audit error.
    External writers are prohibited. Directory-entry changes are detected before
    puts; in-place corruption is checked by inherited put_raw and final hashes.
    """

    def __init__(self, root, max_bytes):
        supplied = Path(root).absolute()
        if supplied != supplied.resolve():
            raise ValueError("symlink or noncanonical store path")
        if not supplied.is_relative_to(FEEDBACK_ROOT) or supplied == FEEDBACK_ROOT:
            raise ValueError("tracked store must be a new feedback_generation_v1 child")
        super().__init__(supplied, max_bytes)
        self._owner = (os.getpid(), threading.get_ident())
        self._committed = {}
        self._committed_bytes = 0
        self.poisoned = False
        self.poison_reason = None
        self.failure_disk_bytes = None
        self.failure_scan_error = None
        self.final_audit = None
        self._directory_stamp = self._stamp()

    def _stamp(self):
        info = os.fstat(self._fd)
        return info.st_mtime_ns, info.st_ctime_ns

    def _check_owner(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            raise StoreError("single-writer store accessed outside owning process/thread")

    def _check_root(self):
        self._check_owner()
        super()._check_root()
        if self.poisoned:
            raise StoreError("store is poisoned; preserve evidence and create a new store")
        if self._stamp() != self._directory_stamp:
            raise StoreError("unexpected directory-entry modification by another writer")

    def _disk_bytes(self):
        self._check_owner()
        # A failed fsync/journal append may leave more bytes than the journal's
        # successful-write counter; no fast path is permitted after any failure.
        if self.poisoned:
            ContentStore._check_root(self)
            return ContentStore._disk_bytes(self)
        self._check_root()
        return self._committed_bytes + self._journal_bytes

    def put_raw(self, raw, kind, shape, dtype):
        self._check_owner()
        if self.poisoned:
            raise StoreError("store is poisoned; no further writes are permitted")
        try:
            item = super().put_raw(raw, kind, shape, dtype)
            name = item["relative_path"]
            if name not in self._committed:
                self._committed[name] = (item["file_bytes"], item["sha256"])
                self._committed_bytes += item["file_bytes"]
            self._directory_stamp = self._stamp()
            return item
        except BaseException as error:
            self.poisoned = True
            self.poison_reason = f"{type(error).__name__}: {error}"
            try:
                ContentStore._check_root(self)
                self.failure_disk_bytes = ContentStore._disk_bytes(self)
            except Exception as scan_error:
                self.failure_scan_error = f"{type(scan_error).__name__}: {scan_error}"
            raise

    def full_audit(self):
        """Read-only full scan/hash audit; rejects partial, foreign and bad files.

        actual_disk_bytes counts all directory entries (including failed journal
        and partial bytes), matching the old store's apparent-byte cap convention.
        No symlink is followed. This does not repair or clear a poisoned store.
        """
        self._check_owner()
        ContentStore._check_root(self)
        names = set(os.listdir(self._fd))
        before = self._stamp()
        actual = 0
        journal = 0
        problems = []
        if before != self._directory_stamp:
            problems.append({"file": None, "reason": "directory_differs_from_last_successful_commit"})
        for name in sorted(names):
            info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
            actual += info.st_size
            if not stat.S_ISREG(info.st_mode):
                problems.append({"file": name, "reason": "non_regular_entry"})
                continue
            if name == "FAILURES.jsonl":
                journal += info.st_size
                continue
            if name not in self._committed:
                problems.append({"file": name, "reason": "uncommitted_or_foreign_entry"})
                continue
            size, expected_hash = self._committed[name]
            digest = hashlib.sha256()
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._fd)
            with os.fdopen(fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    problems.append({"file": name, "reason": "entry_changed_during_audit"})
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if info.st_size != size or digest.hexdigest() != expected_hash:
                problems.append({"file": name, "reason": "content_mismatch"})
        for name in sorted(set(self._committed) - names):
            problems.append({"file": name, "reason": "missing_committed_file"})
        if before != self._stamp():
            problems.append({"file": None, "reason": "directory_changed_during_audit"})
        expected = self._committed_bytes + self._journal_bytes
        accounting_pass = actual == expected and journal == self._journal_bytes
        audit_pass = not self.poisoned and not problems and accounting_pass and actual <= self.max_bytes
        return {"audit_pass": audit_pass, "poisoned": self.poisoned,
                "poison_reason": self.poison_reason, "accounting_pass": accounting_pass,
                "actual_disk_bytes": actual, "tracked_disk_bytes": expected,
                "committed_content_bytes": self._committed_bytes,
                "actual_journal_bytes": journal, "tracked_journal_bytes": self._journal_bytes,
                "committed_files": len(self._committed), "problems": problems,
                "max_bytes": self.max_bytes}

    def close(self):
        if self._closed:
            return
        try:
            self.final_audit = self.full_audit()
        except Exception as error:
            self.final_audit = {"audit_pass": False, "poisoned": self.poisoned,
                                "error": f"{type(error).__name__}: {error}"}
        finally:
            super().close()
