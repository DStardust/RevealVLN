#!/usr/bin/env python3
"""Continuously report the RxR multi-branch pipeline up to training readiness."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
V2 = ROOT / "artifacts/phase1/rxr_train_expansion/multibranch_v2"
SNAPSHOT = V2 / "RXR_TRAINING_READINESS_MONITOR.json"
LOG = V2 / "RXR_TRAINING_READINESS_MONITOR.log"
LOCK = V2 / ".rxr_training_readiness_monitor.lock"
PID_FILE = V2 / "RXR_TRAINING_READINESS_MONITOR.pid"

LANGUAGE_LOG = V2 / "RXR_MULTIBRANCH_LANGUAGE_V2.log"
LANGUAGE_DONE = V2 / "RXR_MULTIBRANCH_LANGUAGE_V2_DONE"
LANGUAGE_GATE = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_LANGUAGE_GATE_V2.json"
MEDIA = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_MEDIA_MANIFEST_V2.json"
INDEX_DONE = V2 / "RXR_MULTIBRANCH_TRAINING_INDEX_V2_DONE"
INDEX = V2 / "RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
TX_DONE = V2 / "RXR_MULTIBRANCH_TX_V2_DONE"
TX_GATE = V2 / "RXR_MULTIBRANCH_TX_V2_GATE.json"
TX_RUNS = V2 / "tx_runs"
FEATURE_DONE = V2 / "RXR_MULTIBRANCH_FEATURE_V2_DONE"
FEATURE_GATE = V2 / "RXR_MULTIBRANCH_FEATURE_GATE_V2.json"
FEATURE_MANIFEST = V2 / "RXR_MULTIBRANCH_FEATURE_MANIFEST_V2.json"
FEATURES = V2 / "frozen_features"
AUTHORIZATION = V2 / "RXR_MULTIBRANCH_TRAINING_AUTHORIZATION_V2.json"

PROCESS_TOKENS = {
    "language": (
        "run_rxr_multibranch_language_v2_background.sh",
        "run_rxr_multibranch_causal_prefix_language_v2.py",
    ),
    "index": (
        "finalize_rxr_multibranch_training_index_v2.sh",
        "build_rxr_multibranch_training_index_v2.py",
    ),
    "tx": (
        "run_rxr_multibranch_tx_v2_after_index.sh",
        "run_rxr_multibranch_tx_v2.py",
        "rxr_multibranch_tx_v2_worker.py",
    ),
    "features": (
        "run_rxr_multibranch_feature_v2_after_tx.sh",
        "run_rxr_multibranch_feature_v2.py",
        "rxr_multibranch_feature_v2_lane.py",
    ),
    "training": ("train_revealnav_mf2_heads.py",),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def pass_sentinel(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() \
            and path.read_text().strip() == "PASS"
    except OSError:
        return False


def atomic_json(path: Path, value) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_processes() -> dict[str, list[dict]]:
    found = {key: [] for key in PROCESS_TOKENS}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cwd = (entry / "cwd").resolve()
            if cwd != ROOT and ROOT not in cwd.parents:
                continue
            command = (entry / "cmdline").read_bytes().replace(
                b"\0", b" "
            ).decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if not command:
            continue
        for stage, tokens in PROCESS_TOKENS.items():
            if any(token in command for token in tokens):
                if stage == "training" and "--smoke-test" in command:
                    continue
                found[stage].append({"pid": int(entry.name), "command": command})
    return found


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=True
        )
        rows = []
        for line in completed.stdout.splitlines():
            index, used, free, utilization = [
                value.strip() for value in line.split(",")
            ]
            rows.append({
                "index": int(index),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_percent": int(utilization),
            })
        return rows
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def language_progress() -> tuple[int, int | None]:
    media = read_json(MEDIA)
    expected = len(media.get("event_ranges", {})) if isinstance(media, dict) \
        else None
    try:
        text = LANGUAGE_LOG.read_text()
    except OSError:
        return 0, expected
    start = text.rfind("MULTIBRANCH_LANGUAGE_START")
    if start >= 0:
        text = text[start:]
    completed = set(re.findall(
        r"^(x\d{4}_ep\d+_hv\d+) CAUSAL_LANGUAGE_", text, re.MULTILINE
    ))
    return len(completed), expected


def count_json_files(path: Path) -> int:
    try:
        return sum(
            item.is_file() and not item.is_symlink()
            for item in path.glob("*.json")
        )
    except OSError:
        return 0


def progress(completed: int, expected: int | None) -> dict:
    return {
        "completed": completed,
        "expected": expected,
        "percent": round(100 * completed / expected, 2)
        if expected else None,
    }


def stage_state(*, passed: bool, active: bool, upstream: bool) -> str:
    if passed:
        return "PASS"
    if not upstream:
        return "WAITING_UPSTREAM"
    if active:
        return "RUNNING"
    return "FAILED_OR_STOPPED"


def authorized_manifest() -> tuple[Path, dict | None, dict | None, list[str]]:
    authorization = read_json(AUTHORIZATION)
    manifest_path = FEATURE_MANIFEST
    failures = []
    if isinstance(authorization, dict):
        reference = authorization.get("training_manifest", {})
        relative = reference.get("path")
        if isinstance(relative, str):
            candidate = (ROOT / relative).resolve()
            if candidate == ROOT or ROOT not in candidate.parents:
                failures.append("authorization manifest path escapes project")
            else:
                manifest_path = candidate
    manifest = read_json(manifest_path)
    if not isinstance(authorization, dict):
        failures.append("fresh full-set audit authorization is absent")
    else:
        if authorization.get("status") != "TRAINING_AUTHORIZATION_PASS":
            failures.append("training authorization status is not PASS")
        if authorization.get("training_authorized") is not True:
            failures.append("training authorization boolean is not true")
    if not isinstance(manifest, dict):
        failures.append("authorized training manifest is absent")
        return manifest_path, authorization, manifest, failures
    metadata = manifest.get("metadata", {})
    required = {
        "training_authorized": True,
        "causal_prefix_verified": True,
        "future_frames_used": 0,
        "full_candidate_sets": True,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            failures.append(f"training manifest metadata {key} mismatch")
    if isinstance(authorization, dict):
        reference = authorization.get("training_manifest", {})
        expected_sha = reference.get("sha256")
        if not isinstance(expected_sha, str) or not manifest_path.is_file() \
                or sha256_file(manifest_path) != expected_sha:
            failures.append("authorized training manifest SHA-256 mismatch")
    return manifest_path, authorization, manifest, failures


def build_snapshot() -> dict:
    processes = project_processes()
    language = read_json(LANGUAGE_GATE)
    index = read_json(INDEX)
    tx = read_json(TX_GATE)
    feature = read_json(FEATURE_GATE)

    language_pass = pass_sentinel(LANGUAGE_DONE) and isinstance(language, dict) \
        and language.get("status") == "COMPLETE_CAUSAL_CONTROLS_REQUIRED"
    index_pass = pass_sentinel(INDEX_DONE) and isinstance(index, dict) \
        and index.get("status") == "FEATURE_AND_TX_GENERATION_REQUIRED"
    tx_pass = pass_sentinel(TX_DONE) and isinstance(tx, dict) \
        and tx.get("status") == "MULTIBRANCH_TX_PASS"
    feature_pass = pass_sentinel(FEATURE_DONE) and isinstance(feature, dict) \
        and feature.get("status") == "FEATURE_GATE_PASS_AUDIT_REQUIRED"

    language_done, language_expected = language_progress()
    index_expected = len(index.get("records", [])) if isinstance(index, dict) \
        else None
    round1 = count_json_files(TX_RUNS / "round1")
    round2 = count_json_files(TX_RUNS / "round2")
    feature_files = sum(
        item.is_file() and not item.is_symlink()
        for item in FEATURES.glob("*.npz")
    ) if FEATURES.is_dir() else 0

    manifest_path, authorization, manifest, authorization_failures = \
        authorized_manifest()
    ready = feature_pass and not authorization_failures
    training_started = bool(processes["training"])
    stages = {
        "language_gate": {
            "state": stage_state(
                passed=language_pass, active=bool(processes["language"]),
                upstream=True,
            ),
            "progress": progress(language_done, language_expected),
        },
        "training_index": {
            "state": stage_state(
                passed=index_pass, active=bool(processes["index"]),
                upstream=language_pass,
            ),
            "records": index_expected,
        },
        "resource_labels_tx": {
            "state": stage_state(
                passed=tx_pass, active=bool(processes["tx"]),
                upstream=index_pass,
            ),
            "round1": progress(round1, index_expected),
            "round2": progress(round2, index_expected),
        },
        "frozen_features": {
            "state": stage_state(
                passed=feature_pass, active=bool(processes["features"]),
                upstream=tx_pass,
            ),
            "progress": progress(feature_files, index_expected),
        },
        "fresh_fullset_audit_and_authorization": {
            "state": "PASS" if ready else (
                "WAITING_UPSTREAM" if not feature_pass else "ACTION_REQUIRED"
            ),
            "authorization_path": str(AUTHORIZATION.relative_to(ROOT)),
            "training_manifest": str(manifest_path.relative_to(ROOT))
            if manifest_path == ROOT or ROOT in manifest_path.parents else None,
            "blockers": authorization_failures,
            "authorization_loaded": isinstance(authorization, dict),
            "manifest_records": len(manifest.get("records", []))
            if isinstance(manifest, dict) else None,
        },
    }
    failed = [name for name, value in stages.items()
              if value["state"] == "FAILED_OR_STOPPED"]
    if training_started:
        overall = "TRAINING_STARTED"
        next_action = "monitoring boundary reached"
    elif ready:
        overall = "READY_FOR_TRAINING"
        next_action = "main-agent review, then explicitly start training"
    elif failed:
        overall = "PIPELINE_STOPPED"
        next_action = "inspect failed stage before resuming"
    elif feature_pass:
        overall = "WAITING_FRESH_FULLSET_AUDIT"
        next_action = "build and complete the independent full-set audit"
    else:
        overall = "PREPARING_TRAINING_DATA"
        next_action = next(
            name for name, value in stages.items() if value["state"] != "PASS"
        )
    disk = shutil.disk_usage(ROOT)
    return {
        "schema_version": "revealnav-rxr-training-readiness-monitor/1",
        "updated_at": utc_now(),
        "overall_status": overall,
        "next_action": next_action,
        "stages": stages,
        "active_processes": processes,
        "resources": {
            "disk_free_bytes": disk.free,
            "disk_free_gib": round(disk.free / 1024 ** 3, 2),
            "gpus": gpu_snapshot(),
        },
        "training_started_by_monitor": False,
        "terminal": overall in ("READY_FOR_TRAINING", "TRAINING_STARTED"),
    }


def log_transition(snapshot: dict, previous: str | None) -> str:
    compact = {
        "time": snapshot["updated_at"],
        "overall": snapshot["overall_status"],
        "stages": {name: value["state"]
                   for name, value in snapshot["stages"].items()},
        "language": snapshot["stages"]["language_gate"]["progress"],
        "tx_round1": snapshot["stages"]["resource_labels_tx"]["round1"],
        "tx_round2": snapshot["stages"]["resource_labels_tx"]["round2"],
        "features": snapshot["stages"]["frozen_features"]["progress"],
        "disk_free_gib": snapshot["resources"]["disk_free_gib"],
    }
    line = json.dumps(compact, sort_keys=True)
    if line != previous:
        with LOG.open("a") as stream:
            stream.write(line + "\n")
    return line


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 5:
        raise SystemExit("interval must be at least five seconds")
    if args.once:
        print(json.dumps(build_snapshot(), indent=2, sort_keys=True))
        return 0

    V2.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("training-readiness monitor is already running")
        PID_FILE.write_text(str(os.getpid()) + "\n")
        previous = None
        try:
            while True:
                snapshot = build_snapshot()
                atomic_json(SNAPSHOT, snapshot)
                previous = log_transition(snapshot, previous)
                if snapshot["terminal"]:
                    return 0
                time.sleep(args.interval)
        finally:
            try:
                PID_FILE.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
