"""Local single-writer, fail-closed CPU journal; no simulator/runtime admission.

Events are append-only, canonical JSONL with a SHA256 chain. A separately fsynced
atomic HEAD replacement detects partial *and complete-record* tail truncation.
Append is acknowledged only after both files and directory metadata are fsynced.
This is NOT a multi-file atomic transaction: interruption between event and HEAD
commit makes resume fail closed, preserving all bytes for reviewed recovery.

fcntl locking is advisory and assumes a filesystem honoring flock/fsync/rename;
it cannot protect against malicious writers, storage rollback of all files,
broken network-filesystem semantics, or physical storage failure. No corruption
is repaired, truncated, ignored or auto-replayed. Each root is journal-only.
"""

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat


class JournalError(RuntimeError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise JournalError("duplicate JSON key")
        out[key] = value
    return out


def _load(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(JournalError("nonfinite JSON")))
        if _canonical(value) != raw:
            raise JournalError("noncanonical or changed JSON bytes")
        return value
    except (UnicodeError, ValueError, TypeError) as exc:
        raise JournalError("invalid journal JSON") from exc


def _write_all(fd, data):
    offset = 0
    while offset < len(data):
        count = os.write(fd, data[offset:])
        if count <= 0:
            raise JournalError("short journal write")
        offset += count


class Journal:
    def __init__(self, root, config, resume=False):
        self._lock_fd = self._log_fd = self._dir_fd = None
        self._closed = False
        self._poisoned = False
        self._records = []
        self._head = None
        allowed = Path(__file__).resolve().parent
        supplied = Path(root).absolute()
        resolved = supplied.resolve()
        if resolved == allowed or allowed not in resolved.parents:
            raise JournalError("journal must be a child of mechanism_factory_v2")
        if supplied != resolved:
            raise JournalError("symlink or noncanonical journal path forbidden")
        self.root = resolved
        self.config = json.loads(_canonical(config))
        self.config_hash = _hash(self.config)
        try:
            if resume:
                if not self.root.is_dir():
                    raise JournalError("explicit resume requires existing journal")
            else:
                self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
                parent_fd = os.open(self.root.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            self._dir_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            lock_flags = os.O_RDWR | os.O_NOFOLLOW
            if not resume:
                lock_flags |= os.O_CREAT | os.O_EXCL
            self._lock_fd = os.open(self.root / "LOCK", lock_flags, 0o600)
            self._regular(self._lock_fd)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise JournalError("journal already locked by another writer") from exc
            if resume:
                self._verify()
                self._log_fd = os.open(self.root / "events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
                self._regular(self._log_fd)
            else:
                self._log_fd = os.open(self.root / "events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                self._append("__config__", self.config)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _regular(fd):
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise JournalError("journal file must be regular and not hard-linked")

    def _read(self, name):
        fd = os.open(self.root / name, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            self._regular(fd)
            chunks = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        finally:
            os.close(fd)

    def _verify(self):
        if {p.name for p in self.root.iterdir()} != {"LOCK", "events.jsonl", "HEAD.json"}:
            raise JournalError("missing, partial-commit or unexpected journal files")
        head = _load(self._read("HEAD.json"))
        raw = self._read("events.jsonl")
        if set(head) != {"version", "count", "byte_length", "last_hash", "config_hash"} or head["version"] != 1:
            raise JournalError("invalid HEAD schema")
        if type(head["count"]) is not int or head["count"] < 1 or type(head["byte_length"]) is not int:
            raise JournalError("invalid HEAD counts")
        if len(raw) != head["byte_length"] or not raw.endswith(b"\n"):
            raise JournalError("journal truncation or uncommitted append detected")
        if head["config_hash"] != self.config_hash:
            raise JournalError("resumed config differs from frozen config")
        prev = "0" * 64
        records = []
        for seq, line in enumerate(raw.splitlines()):
            record = _load(line)
            if not isinstance(record, dict) or set(record) != {"seq", "prev", "kind", "payload", "hash"}:
                raise JournalError("invalid event schema")
            if type(record["seq"]) is not int or record["seq"] != seq or record["prev"] != prev:
                raise JournalError("event order or previous hash mismatch")
            body = {key: value for key, value in record.items() if key != "hash"}
            if record["hash"] != _hash(body):
                raise JournalError("event hash mismatch")
            if not isinstance(record["kind"], str) or not record["kind"]:
                raise JournalError("invalid event kind")
            if seq == 0:
                if record["kind"] != "__config__" or record["payload"] != self.config:
                    raise JournalError("config genesis mismatch")
            elif record["kind"].startswith("__"):
                raise JournalError("reserved event kind")
            prev = record["hash"]
            records.append(record)
        if len(records) != head["count"] or prev != head["last_hash"]:
            raise JournalError("complete-record tail truncation or HEAD mismatch")
        self._head, self._records = head, records

    def _commit_head(self, head):
        temporary = self.root / ".HEAD.next"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            _write_all(fd, _canonical(head))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, self.root / "HEAD.json")
        os.fsync(self._dir_fd)

    def _append(self, kind, payload):
        if self._closed or self._poisoned:
            raise JournalError("journal closed or poisoned by failed append")
        body = {"seq": len(self._records), "prev": self._head["last_hash"] if self._head else "0" * 64,
                "kind": kind, "payload": copy.deepcopy(payload)}
        record = dict(body, hash=_hash(body))
        encoded = _canonical(record) + b"\n"
        expected_size = self._head["byte_length"] if self._head else 0
        try:
            if os.fstat(self._log_fd).st_size != expected_size:
                raise JournalError("journal changed outside lock discipline")
            _write_all(self._log_fd, encoded)
            os.fsync(self._log_fd)
            head = {"version": 1, "count": len(self._records) + 1,
                    "byte_length": expected_size + len(encoded), "last_hash": record["hash"],
                    "config_hash": self.config_hash}
            self._commit_head(head)
        except Exception:
            self._poisoned = True
            raise
        self._head = head
        self._records.append(record)
        return copy.deepcopy(record)

    def append(self, kind, payload):
        if not isinstance(kind, str) or not kind or kind.startswith("__"):
            raise JournalError("nonempty, nonreserved event kind required")
        return self._append(kind, payload)

    def latest(self, kind):
        if self._closed or self._poisoned:
            raise JournalError("journal closed or poisoned")
        for record in reversed(self._records):
            if record["kind"] == kind:
                return copy.deepcopy(record["payload"])
        return None

    def close(self):
        if self._closed:
            return
        self._closed = True
        for name in ("_log_fd", "_lock_fd", "_dir_fd"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
