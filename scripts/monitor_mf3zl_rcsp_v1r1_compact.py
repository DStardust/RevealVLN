#!/usr/bin/env python3
"""One-screen, read-only monitor for the MF3ZL-RCSP v1r1 expansion."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/training/mf3zl_rcsp_v1r1"


def read_json(name: str) -> dict | None:
    path = OUT / name
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def duration(seconds) -> str:
    if seconds is None:
        return "--"
    seconds = max(0, int(float(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:d}m{seconds:02d}s"


def stage(label: str, value: dict | None) -> str:
    if value is None:
        return f"{label:<7} MISSING"
    active = value.get("active", [])
    return (
        f"{label:<7} {value.get('status', 'UNKNOWN'):<12} "
        f"{int(value.get('completed_pass', 0)):>4}/{int(value.get('total', 0)):<4} "
        f"fail={int(value.get('failed', 0)):<3} active={len(active):<2} "
        f"queued={int(value.get('queued', 0)):<4} "
        f"eta={duration(value.get('eta_s'))}"
    )


def supervisor_running() -> bool:
    needle = b"run_mf3zl_rcsp_v1r1_supervisor.sh"
    for entry in Path("/proc").glob("[0-9]*"):
        if entry.name == str(os.getpid()):
            continue
        try:
            if needle in (entry / "cmdline").read_bytes():
                return True
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return False


def main() -> int:
    native = read_json("MF3ZL_R2R_VARIANT_NATIVE_PROGRESS.json")
    target = read_json("MF3ZL_R2R_VARIANT_TARGET_PROGRESS.json")
    targets = read_json("MF3ZL_R2R_VARIANT_TARGETS.json")
    audit = read_json("MF3ZL_V1R1_DATA_SUPPORT_AUDIT.json")
    print("MF3ZL-RCSP v1r1  |  " + datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"))
    print(stage("native", native))
    print(stage("target", target))
    if targets is None:
        print("targets MISSING")
    else:
        counts = targets.get("counts", {})
        print(
            "targets SEALED     "
            f"events={counts.get('events', 0)} "
            f"R2R={counts.get('datasets', {}).get('R2R', 0)} "
            f"core={counts.get('tiers', {}).get('core', 0)} "
            f"expansion={counts.get('tiers', {}).get('expansion', 0)}"
        )
    if audit is None:
        print("audit   PENDING")
    else:
        domains = audit.get("domains", {})
        r2r = domains.get("R2R", {})
        rxr = domains.get("RxR", {})
        print(
            f"audit   {audit.get('status', 'UNKNOWN')}  "
            f"R2R={r2r.get('combined_unique_exact_events', '--')} "
            f"RxR={rxr.get('combined_unique_exact_events', '--')} "
            f"train_authorized={audit.get('rcsp_training_authorized', False)}"
        )
    print("supervisor " + ("RUNNING" if supervisor_running() else "STOPPED"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
