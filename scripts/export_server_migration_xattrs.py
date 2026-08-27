#!/usr/bin/env python3
"""Export non-empty source xattrs, including raw POSIX ACL blobs.

The destination NFS rejects both user xattrs and POSIX ACLs.  This sidecar is
not a claim that those attributes are active on the destination; it is a
lossless recovery record for later restoration on a supporting filesystem.
Paths and attribute names/values are encoded as Base64 bytes so undecodable
filenames remain round-trippable.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import stat
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
INVENTORY = ROOT / "artifacts/migration/SERVER_MIGRATION_SOURCE_INVENTORY.json"
OUT_DIR = ROOT / "artifacts/migration/xattr_sidecars"


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_gzip_open(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    return temporary, gzip.open(temporary, "wt", encoding="utf-8",
                                compresslevel=6)


def walk_bytes(root: bytes):
    root_info = os.lstat(root)
    device = root_info.st_dev
    stack = [root]
    while stack:
        path = stack.pop()
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            yield path, None, "DISAPPEARED_BEFORE_LSTAT"
            continue
        yield path, info, None
        if not stat.S_ISDIR(info.st_mode) or info.st_dev != device:
            continue
        try:
            with os.scandir(path) as iterator:
                children = [entry.name for entry in iterator]
        except (FileNotFoundError, PermissionError) as error:
            yield path, None, type(error).__name__ + "_DURING_SCANDIR"
            continue
        for name in sorted(children, reverse=True):
            stack.append(os.path.join(path, name))


def source_id(path: str) -> str:
    value = path.strip("/").replace("/", "__").replace(".", "_dot_")
    return value or "root"


def export(source: str) -> dict:
    source_bytes = os.fsencode(source)
    name = source_id(source)
    output = OUT_DIR / (name + ".jsonl.gz")
    temporary, handle = atomic_gzip_open(output)
    scanned = records = attributes = errors = 0
    try:
        with handle:
            for path, info, walk_error in walk_bytes(source_bytes):
                if walk_error is not None:
                    errors += 1
                    handle.write(json.dumps({
                        "record_type": "SCAN_ERROR",
                        "path_b64": b64(path),
                        "error": walk_error,
                    }, sort_keys=True) + "\n")
                    continue
                scanned += 1
                try:
                    names = os.listxattr(path, follow_symlinks=False)
                    values = []
                    for attr_name in sorted(names):
                        attr_name_bytes = os.fsencode(attr_name)
                        attr_value = os.getxattr(
                            path, attr_name, follow_symlinks=False)
                        values.append({
                            "name_b64": b64(attr_name_bytes),
                            "value_b64": b64(attr_value),
                            "value_bytes": len(attr_value),
                        })
                    if not values:
                        continue
                    attributes += len(values)
                    records += 1
                    handle.write(json.dumps({
                        "record_type": "XATTRS",
                        "path_b64": b64(path),
                        "lstat_mode": info.st_mode,
                        "attributes": values,
                    }, sort_keys=True) + "\n")
                except (FileNotFoundError, PermissionError, OSError) as error:
                    errors += 1
                    handle.write(json.dumps({
                        "record_type": "XATTR_ERROR",
                        "path_b64": b64(path),
                        "error_type": type(error).__name__,
                        "errno": getattr(error, "errno", None),
                    }, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "source": source,
        "output": str(output.relative_to(ROOT)),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "scanned_paths": scanned,
        "paths_with_xattrs": records,
        "xattr_count": attributes,
        "error_count": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--source-count", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.source_index < args.source_count:
        raise SystemExit("invalid source shard")
    document = json.loads(INVENTORY.read_text())
    sources = [row["source"] for row in
               document["mnt_sources"] + document["root_sources"]]
    selected = [source for index, source in enumerate(sources)
                if index % args.source_count == args.source_index]
    results = [export(source) for source in selected]
    summary = {
        "manifest": "Daiyang migration raw xattr recovery sidecars",
        "revision": "server-migration-xattrs/1",
        "status": "PASS" if not any(row["error_count"] for row in results)
                  else "PASS_WITH_RECORDED_LIVE_SOURCE_ERRORS",
        "source_inventory_sha256": sha256_file(INVENTORY),
        "source_index": args.source_index,
        "source_count": args.source_count,
        "sources": results,
        "scanned_paths": sum(row["scanned_paths"] for row in results),
        "paths_with_xattrs": sum(row["paths_with_xattrs"] for row in results),
        "xattr_count": sum(row["xattr_count"] for row in results),
        "error_count": sum(row["error_count"] for row in results),
        "destination_attributes_applied": False,
        "recovery_sidecar_only": True,
    }
    path = OUT_DIR / ("shard_%02d.json" % args.source_index)
    atomic_json(path, summary)
    print(json.dumps({
        "status": summary["status"],
        "shard": args.source_index,
        "scanned_paths": summary["scanned_paths"],
        "paths_with_xattrs": summary["paths_with_xattrs"],
        "xattr_count": summary["xattr_count"],
        "errors": summary["error_count"],
        "output": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
