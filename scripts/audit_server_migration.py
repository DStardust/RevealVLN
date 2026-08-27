#!/usr/bin/env python3
"""Create the fail-closed source inventory for the server migration."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
MNT = Path("/mnt")
REMOTE = "8.130.54.48"
DESTINATION = Path("/mnt/data_nas/deeprobotics/daiyang")
OUT = ROOT / "artifacts/migration/SERVER_MIGRATION_SOURCE_INVENTORY.json"
EXCLUDED_MNT = {"zrh", "wyh", "lost+found"}
ROOT_SOURCES = [
    Path("/root/.codex"),
    Path("/root/.claude"),
    Path("/root/.claude.json"),
    Path("/root/.cache/claude"),
    Path("/root/.cache/claude-cli-nodejs"),
    Path("/root/.local/share/claude"),
    Path("/root/.local/state/claude"),
    Path("/root/.local/bin/claude"),
    Path("/root/.vscode-server/extensions/anthropic.claude-code-2.1.243-linux-x64"),
    Path("/root/.vscode-server/extensions/anthropic.claude-code-2.1.245-linux-x64"),
    Path("/root/.vscode-server/data/CachedExtensionVSIXs/anthropic.claude-code-2.1.243-linux-x64"),
    Path("/root/.vscode-server/data/CachedExtensionVSIXs/anthropic.claude-code-2.1.245-linux-x64"),
    Path("/root/.antigravity-ide-server/extensions/anthropic.claude-code-2.1.219-linux-x64"),
    Path("/root/.antigravity-ide-server/extensions/anthropic.claude-code-2.1.220-linux-x64"),
]
MIGRATION_KEY = ROOT / ".secret/migration_ed25519_8_130_54_48"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def disk_usage(path: Path) -> int:
    result = subprocess.run(
        ["du", "-sx", "--bytes", "--", str(path)],
        check=True, capture_output=True, text=True)
    return int(result.stdout.split(None, 1)[0])


def kind(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISLNK(mode):
        return "symbolic_link"
    return "other"


def destination_for_mnt(path: Path) -> Path:
    if path == Path("/mnt/daiyang"):
        return DESTINATION
    return DESTINATION / "_source_mnt_top_level" / path.name


def record(path: Path, destination: Path) -> dict:
    info = path.lstat()
    value = {
        "source": str(path),
        "destination": str(destination),
        "kind": kind(info.st_mode),
        "mode_octal": format(stat.S_IMODE(info.st_mode), "04o"),
        "lstat_bytes": info.st_size,
        "disk_usage_bytes": disk_usage(path),
        "one_file_system": True,
        "follow_symbolic_links": False,
    }
    if stat.S_ISREG(info.st_mode):
        value["sha256"] = sha256_file(path)
    if stat.S_ISLNK(info.st_mode):
        value["link_target"] = os.readlink(path)
    return value


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> int:
    names = sorted(entry.name for entry in os.scandir(MNT))
    missing_exclusions = sorted(EXCLUDED_MNT - set(names))
    if missing_exclusions:
        raise SystemExit("required exclusion path missing: "
                         + repr(missing_exclusions))
    mnt_paths = [MNT / name for name in names
                 if name not in EXCLUDED_MNT]
    if Path("/mnt/daiyang") not in mnt_paths:
        raise SystemExit("/mnt/daiyang is not in the migration set")
    mnt_records = [record(path, destination_for_mnt(path))
                   for path in mnt_paths]
    root_records = []
    for path in ROOT_SOURCES:
        if not path.exists() and not path.is_symlink():
            continue
        destination = DESTINATION / "_source_root_state" / str(path)[1:]
        root_records.append(record(path, destination))
    source_total = sum(row["disk_usage_bytes"]
                       for row in mnt_records + root_records)
    phase_output = ROOT / (
        "artifacts/phase1/rxr_train_expansion/hindsight_factory/"
        "RXR_HINDSIGHT_EVENT_CANDIDATES.json")
    if not phase_output.is_file() or phase_output.is_symlink():
        raise SystemExit("phase output missing")
    document = {
        "manifest": "Daiyang source-server migration inventory",
        "revision": "server-migration-source-inventory/1",
        "status": "SOURCE_AUDIT_PASS_REMOTE_AUTH_REQUIRED",
        "remote_host": REMOTE,
        "remote_user": "root",
        "destination_root": str(DESTINATION),
        "scope": {
            "mnt_policy": "all immediate /mnt entries except exact exclusions",
            "excluded_mnt_paths": [
                "/mnt/lost+found", "/mnt/wyh", "/mnt/zrh"],
            "excluded_paths_were_not_recursively_scanned": True,
            "root_policy": "Codex and Claude Code state/caches only",
            "source_deletion_authorized": False,
            "destination_deletion_authorized": False,
        },
        "mapping_policy": {
            "/mnt/daiyang/": str(DESTINATION) + "/",
            "/mnt/<other>/": str(
                DESTINATION / "_source_mnt_top_level/<other>/"),
            "/root/<selected>/": str(
                DESTINATION / "_source_root_state/root/<selected>/"),
        },
        "migration_credential": {
            "private_key_path": str(MIGRATION_KEY),
            "private_key_is_transfer_excluded": True,
            "public_key_path": str(MIGRATION_KEY) + ".pub",
            "fingerprint": subprocess.run(
                ["ssh-keygen", "-lf", str(MIGRATION_KEY) + ".pub"],
                check=True, capture_output=True, text=True
            ).stdout.strip(),
        },
        "completed_phase_checkpoint": {
            "path": str(phase_output.relative_to(ROOT)),
            "sha256": sha256_file(phase_output),
            "trajectory_count": 2303,
            "merged_candidate_count": 6106,
            "trajectory_with_primary_count": 2208,
        },
        "mnt_sources": mnt_records,
        "root_sources": root_records,
        "source_count": len(mnt_records) + len(root_records),
        "source_disk_usage_total_bytes": source_total,
        "minimum_destination_free_bytes": source_total + 100_000_000_000,
        "transfer": {
            "transport": "rsync over pinned SSH host key",
            "resumable": True,
            "delete_option_forbidden": True,
            "hardlinks_symlinks_modes_times_and_logical_bytes_preserved": True,
            "destination_nfs_preserves_sparse_physical_layout": False,
            "destination_nfs_supports_user_xattrs": False,
            "destination_nfs_supports_posix_acls": False,
            "source_xattrs_and_acl_blobs_exported_to_recovery_sidecar": True,
            "final_delta_required_after_quiescing_writers": True,
            "source_removed_after_success": False,
        },
    }
    atomic_json(OUT, document)
    print(json.dumps({
        "status": document["status"],
        "source_count": document["source_count"],
        "source_disk_usage_total_bytes": source_total,
        "minimum_destination_free_bytes": document[
            "minimum_destination_free_bytes"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
