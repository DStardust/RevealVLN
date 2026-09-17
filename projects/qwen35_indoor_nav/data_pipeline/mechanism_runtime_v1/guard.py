"""CPU-safe resource checks and append-only, raw-pixel-addressed NPY store.

No GPU or process control is performed by this module. Runtime supervisor owns
its child lifecycle; resource samples must include that child's process tree.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import uuid

RUNTIME_ROOT = Path(__file__).resolve().parent


class BudgetError(RuntimeError):
    resource_censored = True

    def __init__(self, resource, observed, limit):
        self.resource, self.observed, self.limit = resource, observed, limit
        super().__init__(f"resource_censored: {resource} {observed} > {limit}")

    def as_dict(self):
        return {"reason": "RESOURCE_CENSORED", "resource": self.resource,
                "observed": self.observed, "limit": self.limit}


class ResourceGuard:
    """Limits are seconds, bytes, bytes; caller supplies measured usage.

    Pure check, not an autonomous watchdog. At-limit samples pass; before an
    operation callers must project its resource cost (as ContentStore does).
    """
    def __init__(self, wall, ram, disk):
        self.limits = dict(zip(("wall_seconds", "ram_bytes", "disk_bytes"),
                               (wall, ram, disk)))
        for value in self.limits.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("resource limits must be finite positive numbers")

    def check(self, samples):
        for key, limit in self.limits.items():
            value = samples[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid measured sample: {key}")
            if value > limit:
                raise BudgetError(key, value, limit)
        return {"resource_pass": True, "samples": dict(samples), "limits": dict(self.limits)}


class StoreError(RuntimeError):
    pass


def _npy(raw, shape, dtype):
    header = repr({"descr": dtype, "fortran_order": False, "shape": tuple(shape)}).encode("ascii")
    header += b" " * ((-10 - len(header) - 1) % 64) + b"\n"
    if len(header) > 65535:
        raise ValueError("NPY v1 header too long")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + raw


class ContentStore:
    """Single-writer new store, no resume, no unsafe existing-file replacement.

    put_array / put_raw return metadata including pixel_sha256. Hash is over
    C-contiguous raw pixels, not NPY serialization. 64 KiB of the supplied total
    max_bytes is reserved for an append-only failure journal. Once journal is
    exhausted, errors are still raised but no further disk writes are attempted.
    Interrupted .partial files are retained as evidence and charged to the cap.
    """
    FAILURE_RESERVE = 65536

    def __init__(self, root, max_bytes):
        supplied = Path(root).absolute()
        if supplied != supplied.resolve():
            raise ValueError("symlink or noncanonical store path")
        if not supplied.is_relative_to(RUNTIME_ROOT) or supplied == RUNTIME_ROOT:
            raise ValueError("store must be a new child of mechanism_runtime_v1")
        if type(max_bytes) is not int or max_bytes <= self.FAILURE_RESERVE:
            raise ValueError("max_bytes must exceed the 64 KiB failure reserve")
        # Parent creation is deliberately the supervisor's responsibility.
        supplied.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.root = supplied
        self.max_bytes = max_bytes
        self.content_limit = max_bytes - self.FAILURE_RESERVE
        self._fd = os.open(supplied, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._stat = os.fstat(self._fd)
        self._closed = False
        self._journal_bytes = 0

    def close(self):
        if not self._closed:
            os.close(self._fd)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _check_root(self):
        if self._closed:
            raise StoreError("store is closed")
        if self.root != self.root.resolve():
            raise StoreError("store path changed or became a symlink")
        current = self.root.stat()
        if (current.st_dev, current.st_ino) != (self._stat.st_dev, self._stat.st_ino):
            raise StoreError("store root identity changed")

    def _disk_bytes(self):
        return sum(os.stat(name, dir_fd=self._fd, follow_symlinks=False).st_size
                   for name in os.listdir(self._fd))

    def _failure(self, error, name):
        event = json.dumps({"event": "STORE_WRITE_REJECTED", "file": name,
                            "error": type(error).__name__, "detail": str(error)[:512]},
                           sort_keys=True).encode("utf8") + b"\n"
        if self._journal_bytes + len(event) > self.FAILURE_RESERVE:
            return
        try:
            # Never follow a substituted journal link; no overwrite/repair.
            fd = os.open("FAILURES.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                         0o600, dir_fd=self._fd)
            try:
                with os.fdopen(fd, "ab", closefd=False) as stream:
                    stream.write(event)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(fd)
            self._journal_bytes += len(event)
            os.fsync(self._fd)
        except OSError:
            # Original failure remains authoritative if the filesystem is full.
            pass

    def put_array(self, arr, kind):
        dtype = str(arr.dtype)
        return self.put_raw(arr.tobytes(order="C"), kind, tuple(arr.shape), dtype)

    def put_raw(self, raw, kind, shape, dtype):
        self._check_root()
        name = None
        try:
            if kind not in ("rgb", "semantic"):
                raise ValueError("kind must be rgb or semantic")
            if not isinstance(raw, bytes):
                raise ValueError("raw must be immutable bytes")
            shape = tuple(shape)
            if not shape or any(type(x) is not int or x <= 0 for x in shape):
                raise ValueError("shape must contain positive integers")
            if kind == "rgb":
                if dtype not in ("uint8", "|u1", "u1") or len(shape) != 3 or shape[-1] != 3:
                    raise ValueError("RGB must be uint8 HWC with exactly 3 channels")
                descr, width = "|u1", 1
            else:
                if dtype not in ("uint32", "<u4", "u4") or len(shape) != 2:
                    raise ValueError("semantic must be little-endian uint32 HW")
                descr, width = "<u4", 4
            if len(raw) != math.prod(shape) * width:
                raise ValueError("raw byte length does not match shape/dtype")
            digest = hashlib.sha256(raw).hexdigest()
            name = f"{digest}.{kind}.npy"
            payload = _npy(raw, shape, descr)
            file_digest = hashlib.sha256(payload).hexdigest()
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._fd)
            except FileNotFoundError:
                fd = None
            if fd is not None:
                with os.fdopen(fd, "rb") as stream:
                    stat = os.fstat(stream.fileno())
                    if stat.st_size != len(payload) or stream.read() != payload:
                        raise StoreError("existing content is corrupt or has conflicting metadata")
            else:
                observed = self._disk_bytes() - self._journal_bytes + len(payload)
                if observed > self.content_limit:
                    raise BudgetError("content_bytes", observed, self.content_limit)
                temporary = "." + name + "." + uuid.uuid4().hex + ".partial"
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=self._fd)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                # Hard-link commit is atomic and refuses to replace an existing
                # destination. A raced collision fails closed and keeps partial.
                os.link(temporary, name, src_dir_fd=self._fd, dst_dir_fd=self._fd,
                        follow_symlinks=False)
                os.fsync(self._fd)
                os.unlink(temporary, dir_fd=self._fd)
                os.fsync(self._fd)
            return {"pixel_sha256": digest, "sha256": file_digest,
                    "path": str(self.root / name), "relative_path": name,
                    "shape": list(shape), "dtype": descr,
                    "nbytes": len(raw), "file_bytes": len(payload)}
        except (ValueError, OSError, StoreError, BudgetError) as error:
            self._failure(error, name)
            raise
