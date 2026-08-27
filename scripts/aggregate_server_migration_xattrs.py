#!/usr/bin/env python3
"""Seal the eight raw xattr/ACL recovery-sidecar scans."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/migration/xattr_sidecars"
INVENTORY = ROOT / "artifacts/migration/SERVER_MIGRATION_SOURCE_INVENTORY.json"
OUT = BASE / "XATTR_SIDECAR_CLOSURE.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    inventory_sha = sha256_file(INVENTORY)
    shards = []
    sources = []
    for index in range(8):
        path = BASE / ("shard_%02d.json" % index)
        if not path.is_file() or path.is_symlink():
            raise SystemExit("missing xattr shard")
        value = json.loads(path.read_text())
        if (value["status"] not in {
                    "PASS", "PASS_WITH_RECORDED_LIVE_SOURCE_ERRORS"}
                or value["source_index"] != index
                or value["source_count"] != 8
                or value["source_inventory_sha256"] != inventory_sha):
            raise SystemExit("xattr shard contract failure")
        shards.append({"path": str(path.relative_to(ROOT)),
                       "sha256": sha256_file(path)})
        sources.extend(value["sources"])
    expected = json.loads(INVENTORY.read_text())["source_count"]
    if len(sources) != expected or len({row["source"] for row in sources}) \
            != expected:
        raise SystemExit("xattr source exact closure failure")
    output = {
        "manifest": "Daiyang migration xattr and raw ACL sidecar closure",
        "revision": "server-migration-xattr-closure/1",
        "status": "PASS_NO_SOURCE_XATTRS_PRESENT"
            if not sum(row["xattr_count"] for row in sources)
            and not sum(row["error_count"] for row in sources)
            else "RECOVERY_SIDECARS_CREATED_WITH_RECORDED_FINDINGS",
        "source_inventory_sha256": inventory_sha,
        "shards": shards,
        "source_count": len(sources),
        "scanned_paths": sum(row["scanned_paths"] for row in sources),
        "paths_with_xattrs": sum(row["paths_with_xattrs"] for row in sources),
        "xattr_count": sum(row["xattr_count"] for row in sources),
        "error_count": sum(row["error_count"] for row in sources),
        "destination_nfs_xattrs_applied": False,
        "destination_nfs_posix_acls_applied": False,
        "sources": sorted(sources, key=lambda row: row["source"]),
    }
    temporary = OUT.with_name(OUT.name + ".part")
    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({
        "status": output["status"],
        "scanned_paths": output["scanned_paths"],
        "xattr_count": output["xattr_count"],
        "errors": output["error_count"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
