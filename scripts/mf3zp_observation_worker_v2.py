#!/usr/bin/env python3
"""MF3ZP v2 prefix-safe frozen-native observation worker.

Some historical exact-switch runs store the *treatment* action in
``base_trace.jsonl``.  This worker never treats that row as a native action:
it reruns the frozen policy and checks only the causal action prefix strictly
before the sealed decision step.  Observations are truncated at that step,
so no post-decision branch can enter the annotation queue.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "scripts/mf3zp_observation_worker.py"
_spec = importlib.util.spec_from_file_location("mf3zp_observation_worker_v1", V1_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load sealed MF3ZP v1 worker")
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)

base = _v1.base
SCHEMA = "revealnav-mf3zp-frozen-native-observation-v2/1"


def _action_signature(row: dict) -> tuple:
    return (
        int(row["act"]),
        str(row.get("ghost_vp")),
        str(row.get("front_vp")),
        int(row.get("back_path_len", 0)),
        bool(row.get("tryout", False)),
    )


class BoundedPanoramaCapture(_v1.PanoramaCapture):
    """Do not persist frames after the causal decision prefix."""

    def __init__(self, media_dir: Path, prefix_limit: int) -> None:
        super().__init__(media_dir)
        self.prefix_limit = prefix_limit

    def capture(self, observations: dict, waypoint: dict) -> None:
        if len(self.records) > self.prefix_limit:
            return
        super().capture(observations, waypoint)


class BoundedObserver(_v1.FrozenNativeObserver):
    """Record only steps through the sealed decision step."""

    def __init__(self, arrays_dir: Path, prefix_limit: int) -> None:
        super().__init__(arrays_dir)
        self.prefix_limit = prefix_limit

    def navigation(self, kwargs: dict, result: dict) -> dict:
        if len(self.records) > self.prefix_limit:
            return result
        return super().navigation(kwargs, result)


def _source_trace(path: Path) -> tuple[Path, list[dict], str]:
    return _v1._source_trace(path)


def _atomic_summary(path: Path, value: dict) -> None:
    _v1.atomic_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("R2R", "RxR"), required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--source-native-trace", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--decision-step", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--source-trace-mode",
        choices=("native_reference", "prefix_witness"),
        default="prefix_witness",
    )
    args = parser.parse_args()
    if args.decision_step < 0 or args.gpu_id < 0:
        raise SystemExit("decision-step and gpu-id must be non-negative")

    requested_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if requested_gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    elif requested_gpu not in {str(args.gpu_id), "0"}:
        raise SystemExit("CUDA_VISIBLE_DEVICES disagrees with --gpu-id")

    source_path, source_actions, source_sha = _source_trace(args.source_native_trace)
    run_dir = _v1._new_directory(args.run_dir)
    arrays_dir = run_dir / "arrays"
    media_dir = run_dir / "media"
    arrays_dir.mkdir()
    media_dir.mkdir()
    base_trace = run_dir / "base_trace.jsonl"
    base_trace.write_text("")
    os.environ["REVEALVLN_BASE_TRACE"] = str(base_trace)

    observer = BoundedObserver(arrays_dir, args.decision_step)
    capture = BoundedPanoramaCapture(media_dir, args.decision_step)
    base._CONTROLLER = observer
    base.install_hooks()
    _v1.install_panorama_hook(capture)

    os.chdir(base.ETPR1)
    from etpr1_compat import configure_project_cache_env

    configure_project_cache_env()
    output = run_dir / "etp_output"
    argv = _v1.run_argv(args.dataset, str(args.episode_id), output)
    summary = {
        "schema_version": "revealnav-mf3zp-observation-worker-v2/1",
        "status": "RUNNING",
        "dataset": args.dataset,
        "split": "train",
        "episode_id": str(args.episode_id),
        "scene_id": args.scene_id,
        "decision_step": args.decision_step,
        "prefix_rule": "actions and observations through decision_step; source comparison strictly before decision_step",
        "source_trace_mode": args.source_trace_mode,
        "public_split_access": False,
        "target_received": False,
        "task_metric_payload_read": False,
        "action_changed": False,
        "source_native_trace": {
            "path": str(source_path.relative_to(ROOT)),
            "bytes": source_path.stat().st_size,
            "sha256": source_sha,
        },
        "argv": argv,
    }
    sys.argv = argv
    import time

    started = time.monotonic()
    try:
        import run

        run.main()
        actual_actions = [
            json.loads(line)
            for line in base_trace.read_text().splitlines()
            if line.strip()
        ]
        if len(source_actions) < args.decision_step:
            raise RuntimeError("source trace is shorter than the causal prefix")
        if len(actual_actions) <= args.decision_step:
            raise RuntimeError("native replay did not reach the sealed decision step")
        expected_prefix = [
            _action_signature(value) for value in source_actions[:args.decision_step]
        ]
        actual_prefix = [
            _action_signature(value) for value in actual_actions[:args.decision_step]
        ]
        if actual_prefix != expected_prefix:
            raise RuntimeError("causal native action prefix differs from source witness")
        if not (
            len(observer.records) == len(capture.records) == args.decision_step + 1
        ):
            raise RuntimeError("bounded causal observation cardinality drift")
        summary.update({
            "status": "PASS",
            "source_prefix_replay_exact": True,
            "source_target_action_compared": False,
            "actual_actions_through_decision": len(actual_actions[:args.decision_step + 1]),
        })
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        records_path = run_dir / "causal_prefix_records.jsonl"
        media_path = run_dir / "panorama_manifest.jsonl"
        # Bounded hooks should already have truncated these; slice again as a
        # fail-safe before the files become annotation inputs.
        observer.records = [
            value for value in observer.records
            if int(value["step"]) <= args.decision_step
        ]
        capture.records = [
            value for value in capture.records
            if int(value["step"]) <= args.decision_step
        ]
        _v1.atomic_jsonl(records_path, observer.records)
        _v1.atomic_jsonl(media_path, capture.records)
        summary.update({
            "wall_time_seconds": round(time.monotonic() - started, 3),
            "prefix_records": len(observer.records),
            "panorama_records": len(capture.records),
            "causal_prefix_records": {
                "path": str(records_path.relative_to(ROOT)),
                "bytes": records_path.stat().st_size,
                "sha256": _v1.sha256_file(records_path),
            },
            "panorama_manifest": {
                "path": str(media_path.relative_to(ROOT)),
                "bytes": media_path.stat().st_size,
                "sha256": _v1.sha256_file(media_path),
            },
            "base_trace": {
                "path": str(base_trace.relative_to(ROOT)),
                "bytes": base_trace.stat().st_size,
                "sha256": _v1.sha256_file(base_trace),
            },
            "no_outcome_or_target_input": True,
            "paper_result": False,
        })
        _atomic_summary(run_dir / "RUN_SUMMARY.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "dataset": args.dataset,
        "episode_id": str(args.episode_id),
        "decision_step": args.decision_step,
        "prefix_records": len(observer.records),
        "run_dir": str(run_dir.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
