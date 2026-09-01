#!/usr/bin/env python3
"""On-demand, read-only MF3ZP v2 progress view."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2"


def main() -> int:
    protocol = OUT / "MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json"
    status = OUT / "MF3ZP_OBSERVATION_COLLECTION_STATUS.json"
    print("MF3ZP v2")
    if protocol.is_file():
        p = json.loads(protocol.read_text())
        print(f"protocol={p.get('status')} sha256={_sha256(protocol)}")
        pop = p.get("population", {})
        print(
            f"population events={pop.get('event_count')} episodes={pop.get('episode_count')} "
            f"domains={pop.get('domain_counts')} scenes={len(set(v.get('scene_id') for v in pop.get('events', [])))}"
        )
    else:
        print("protocol=missing")
    if status.is_file():
        s = json.loads(status.read_text())
        total = int(s.get("planned_episodes", 0) or 0)
        done = int(s.get("completed_records", 0) or 0)
        pct = 100.0 * done / total if total else 0.0
        print(
            f"collection={s.get('status')} {done}/{total} ({pct:.1f}%) "
            f"pass={s.get('pass', 0)} fail={s.get('fail', 0)} "
            f"elapsed_min={float(s.get('elapsed_seconds', 0.0) or 0.0)/60:.1f}"
        )
        failures = [v for v in s.get("results", []) if v.get("status") == "FAIL"]
        if failures:
            print("recent_failures:")
            for row in failures[-5:]:
                print(f"  {row.get('dataset')}/{row.get('episode_id')}: {row.get('error')}")
    else:
        print("collection=not_started")
    for name in (
        "MF3ZP_OBSERVATION_COLLECTION_MANIFEST.json",
        "MF3ZP_ANNOTATION_INPUT_MANIFEST.json",
        "MF3ZP_QWEN_ANNOTATION_MANIFEST.json",
        "MF3ZP_EXPLORATORY_SCOUT_RESULT.json",
    ):
        path = OUT / name
        if path.is_file():
            value = json.loads(path.read_text())
            print(f"{name} status={value.get('status')}")
    try:
        rows = subprocess.run(
            ["pgrep", "-af", "run_mf3zp_qwen_reference_v2.py"],
            check=False, capture_output=True, text=True,
        ).stdout.splitlines()
        print(f"collector_processes={len(rows)}")
    except OSError:
        pass
    return 0


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
