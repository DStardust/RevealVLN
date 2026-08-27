#!/usr/bin/env python3
"""Run one sealed V5.13.1 R2R evaluation group episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import r2r_full_opp_worker_v5_6 as v56  # noqa: E402
from r2r_v5_6_net_advantage_controller import (  # noqa: E402
    NetAdvantageOnlyController,
    V56NetAdvantageController,
    V56NetAdvantageNoReturnController,
)


GROUPS = (
    "etp_r1",
    "v5_6",
    "net_advantage_only",
    "v5_6_net_advantage",
    "v5_6_net_advantage_no_return",
)
NET_GROUPS = frozenset(GROUPS[2:])
LOCKED_SEEDS = (20260826, 20260827, 20260828)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(value: str | None, expected_sha256: str | None) -> Path:
    if value is None or expected_sha256 is None:
        raise RuntimeError("Net-Advantage groups require checkpoint provenance")
    path = Path(value).resolve()
    if ROOT not in path.parents or path.is_symlink() or not path.is_file():
        raise RuntimeError("Net-Advantage checkpoint must be a project-local file")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("Net-Advantage checkpoint SHA-256 mismatch")
    return path


def configured_controller(controller_type, checkpoint: Path, seed: int):
    class ConfiguredController(controller_type):
        def __init__(self, controller_seed, mode, device, trace_path):
            super().__init__(
                controller_seed, mode, device, trace_path, checkpoint,
                expected_checkpoint_seed=seed,
            )

    ConfiguredController.__name__ = f"Configured{controller_type.__name__}"
    return ConfiguredController


def atomic_summary(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("val_seen", "val_unseen"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--net-advantage-checkpoint")
    parser.add_argument("--net-advantage-sha256")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if ROOT not in run_dir.parents:
        raise SystemExit("run directory must remain inside the project")
    if args.group == "etp_r1" and args.seed != 0:
        raise SystemExit("the deterministic ETP-R1 baseline uses seed 0")
    if args.group != "etp_r1" and args.seed not in LOCKED_SEEDS:
        raise SystemExit("treatment group seed is outside the locked set")
    if args.group not in NET_GROUPS and (
        args.net_advantage_checkpoint is not None
        or args.net_advantage_sha256 is not None
    ):
        raise SystemExit("non-Net-Advantage groups forbid a learned checkpoint")

    checkpoint = None
    if args.group in NET_GROUPS:
        checkpoint = checkpoint_path(
            args.net_advantage_checkpoint, args.net_advantage_sha256
        )
        controller_types = {
            "net_advantage_only": NetAdvantageOnlyController,
            "v5_6_net_advantage": V56NetAdvantageController,
            "v5_6_net_advantage_no_return": V56NetAdvantageNoReturnController,
        }
        controller_type = configured_controller(
            controller_types[args.group], checkpoint, args.seed
        )
        v56.FullOPPActionController = controller_type

    if args.group == "etp_r1":
        sys.argv = [
            "r2r_continuous_controller_worker_v5_4.py",
            "--episode-id", args.episode_id,
            "--mode", "baseline", "--split", args.split,
            "--run-dir", str(run_dir),
        ]
        v56.v54.main()
        state = None
    else:
        sys.argv = [
            "r2r_full_opp_worker_v5_6.py",
            "--episode-id", args.episode_id,
            "--mode", "revealnav", "--seed", str(args.seed),
            "--split", args.split, "--run-dir", str(run_dir),
        ]
        v56.main()
        state = v56.v55._CONTROLLER

    summary_path = run_dir / "RUN_SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    summary.update({
        "schema_version": "revealnav-r2r-v5.13.1-group-worker/1",
        "group": args.group,
        "seed": args.seed,
        "split": args.split,
        "protocol_revision": "V5.13.1",
    })
    if checkpoint is not None:
        summary["net_advantage_checkpoint"] = {
            "path": str(checkpoint.relative_to(ROOT)),
            "bytes": checkpoint.stat().st_size,
            "sha256": args.net_advantage_sha256,
            "seed": args.seed,
        }
        summary["controller"].update({
            "net_advantage_decisions": state.net_advantage_decisions,
            "net_advantage_approvals": state.net_advantage_approvals,
            "net_advantage_vetoes": state.net_advantage_vetoes,
            "net_advantage_checkpoint_seed": state.net_advantage.checkpoint_seed,
            "no_return_suppressions": getattr(
                state, "no_return_suppressions", 0
            ),
        })
    atomic_summary(summary_path, summary)
    print(json.dumps({
        "status": summary["status"], "group": args.group,
        "episode_id": args.episode_id, "seed": args.seed,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
