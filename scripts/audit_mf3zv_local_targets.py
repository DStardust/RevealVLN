#!/usr/bin/env python3
"""Validate Q3 exact native targets from outcome-free decision traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from revealnav_mf3.mf3zv_protocol import validate_protocol
from revealnav_mf3.progress_target_support import exact_target_from_trace_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_protocol(json.loads(args.protocol.read_text()))
    if json.loads(args.state_audit.read_text())["status"] != "MF3ZV_STATE_GATE_PASS":
        raise ValueError("Q3 is forbidden because Q2 did not pass")
    supported = {
        (row["dataset"], row["episode_id"])
        for row in (json.loads(line) for line in args.transitions.open() if line.strip())
    }
    manifest = json.loads(args.trace_manifest.read_text())
    rows = []
    for source in manifest["traces"]:
        key = (source["dataset"], source["episode_id"])
        if key not in supported:
            continue
        path = Path(source["path"])
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"trace hash mismatch: {path}")
        for line in data.decode("utf-8").splitlines():
            trace = json.loads(line)
            if int(trace.get("step", -1)) != int(source["decision_step"]):
                continue
            target = exact_target_from_trace_row(
                dataset=source["dataset"],
                episode_id=source["episode_id"],
                scene_id=source["scene_id"],
                row=trace,
                source_sha256=digest,
            )
            rows.append(target.to_dict())
            break
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

