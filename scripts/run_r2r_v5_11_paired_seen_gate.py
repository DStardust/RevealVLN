#!/usr/bin/env python3
"""Paired val_seen metric gate for the blind V5.11 active cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_r2r_full_opp_gate_v5_6 as common  # noqa: E402
import run_r2r_v5_10_paired_seen_gate as base  # noqa: E402
import run_r2r_v5_11_fresh_activation_screen as screen  # noqa: E402
import run_r2r_v5_11_temporal_diagnostic as diagnostic  # noqa: E402


RUNNER = Path(__file__).resolve()
WORKER = ROOT / "scripts/r2r_temporal_native_control_opp_worker_v5_11.py"
OUT = ROOT / "artifacts/evaluation/mf2_r2r_v5_11_paired_seen_gate"
PROTOCOL = OUT / "R2R_V5_11_PAIRED_SEEN_PROTOCOL.json"
RESULT = OUT / "R2R_V5_11_PAIRED_SEEN_RESULT.json"
TARGET_EPISODES = base.TARGET_EPISODES
TARGET_SCENES = base.TARGET_SCENES
SEEDS = common.SEEDS


def protocol_value() -> dict:
    result = json.loads(screen.RESULT.read_text())
    if not (
        result.get("status") == "V5_11_FRESH_COHORT_READY"
        and all(result.get("gates", {}).values())
        and result.get("task_metric_payload_read") is False
        and result.get("selection_used_task_metrics") is False
    ):
        raise RuntimeError("V5.11 blind activation cohort is not ready")
    selected = result.get("selected_confirmation_cohort", [])
    scenes = {row["scene_id"] for row in selected}
    if len(selected) != TARGET_EPISODES or len(scenes) < TARGET_SCENES:
        raise RuntimeError("V5.11 active cohort size/diversity drift")
    return {
        "schema_version": "revealnav-r2r-v5.11-paired-seen-protocol/1",
        "status": "SEALED_BEFORE_V5_11_PAIRED_TASK_METRIC_GATE",
        "selection": selected,
        "selection_rule": (
            "sealed V5.11 activation screen uses controller traces and scene "
            "metadata only; task metrics are never used"
        ),
        "minimum_distinct_scenes": TARGET_SCENES,
        "distinct_scenes": len(scenes),
        "seeds": list(SEEDS),
        "treatment_runs": TARGET_EPISODES * len(SEEDS),
        "baseline": (
            "identical deterministic frozen ETP-R1 shadow trajectory; its "
            "metric file is opened only after this protocol is sealed"
        ),
        "paired_unit": "episode averaged across three locked model seeds",
        "uncertainty": "10000 deterministic episode bootstrap replicates",
        "success_gate": "mean SPL>0, nDTW>0, Success>=0",
        "screened_active_episodes": result["combined_active"],
        "sources": {
            str(RUNNER.relative_to(ROOT)): common.sha256_file(RUNNER),
            str(WORKER.relative_to(ROOT)): common.sha256_file(WORKER),
            str(screen.PROTOCOL.relative_to(ROOT)): common.sha256_file(
                screen.PROTOCOL
            ),
            str(screen.RESULT.relative_to(ROOT)): common.sha256_file(
                screen.RESULT
            ),
        },
        "paper_result": False,
        "unseen_or_test_allowed": False,
    }


def baseline_summary(episode_id: str) -> dict:
    name = f"shadow_ep_{episode_id}"
    candidates = [
        diagnostic.OUT / "runs" / name,
        screen.OUT / "runs" / name,
    ]
    matches = [path for path in candidates if (path / "RUN_SUMMARY.json").is_file()]
    if len(matches) != 1:
        raise RuntimeError("V5.11 blind baseline location is ambiguous")
    run_dir = matches[0]
    summary = json.loads((run_dir / "RUN_SUMMARY.json").read_text())
    controller = summary["controller"]
    active = (
        controller["effective_commit_interventions"]
        + controller["explore_decisions"] > 0
    )
    if not (
        summary.get("status") == "PASS"
        and summary.get("mode") == "shadow"
        and summary.get("task_metric_payload_read") is False
        and summary.get("metrics") is None
        and active
    ):
        raise RuntimeError("V5.11 baseline is not an active blind trajectory")
    paired = (
        base.executor.SCREEN / "runs" / name / "RUN_SUMMARY.json"
    )
    if summary["base_trace_sha256"] != json.loads(paired.read_text())["base_trace_sha256"]:
        raise RuntimeError("V5.11 diagnostic baseline trace drift")
    stats = list((run_dir / "etp_output").rglob(
        "stats_ep_ckpt_270_val_seen_r0_w1.json"
    ))
    if len(stats) != 1:
        raise RuntimeError("V5.11 baseline metric file is ambiguous")
    metrics = json.loads(stats[0].read_text()).get(episode_id)
    if metrics is None:
        raise RuntimeError("V5.11 baseline episode metric is absent")
    return metrics


def configure_base() -> None:
    base.WORKER = WORKER
    base.OUT = OUT
    base.PROTOCOL = PROTOCOL
    base.RESULT = RESULT
    base.SEEDS = SEEDS
    base.protocol_value = protocol_value
    base.baseline_summary = baseline_summary
    base.configure_executor()


def seal() -> None:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed V5.11 paired protocol drift")
    if not PROTOCOL.exists():
        common.atomic_json(PROTOCOL, value)
    print(json.dumps({
        "status": value["status"],
        "runs": value["treatment_runs"],
        "episodes": len(value["selection"]),
        "scenes": value["distinct_scenes"],
        "sha256": common.sha256_file(PROTOCOL),
    }))


def verify() -> None:
    base.verify()
    result = json.loads(RESULT.read_text())
    if not result.get("status", "").startswith("V5_10_PAIRED_"):
        raise RuntimeError("shared V5.11 paired verification failed")
    result["schema_version"] = "revealnav-r2r-v5.11-paired-seen-result/1"
    result["status"] = result["status"].replace("V5_10_PAIRED_", "V5_11_PAIRED_", 1)
    common.atomic_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "run", "resume", "verify"))
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("--gpus must contain unique GPU indices")
    if args.command == "seal":
        seal()
    elif args.command in ("run", "resume"):
        base.executor.execute(gpus, args.command == "resume")
    else:
        verify()


if __name__ == "__main__":
    main()
