#!/usr/bin/env python3
"""Run the sealed MF3ZU RxR feasibility stages once, fail closed, and detach.

The supervisor deliberately exposes only execution state while work is in
progress.  Scientific metrics are written only by the final trainer bundle.
Long observation replay and fixed Qwen annotation can be resumed from their
validated per-episode/per-request artifacts without changing the sealed design.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zu_protocol import (  # noqa: E402
    OUTPUT,
    POPULATION_MANIFEST_PATH,
    PROTOCOL_PATH,
    REVISION,
    verify_protocol,
)


STATUS_PATH = OUTPUT / "MF3ZU_RXR_PIPELINE_STATUS.json"
PROCESS_PATH = OUTPUT / "MF3ZU_RXR_PIPELINE_PROCESS.json"
LOG_PATH = OUTPUT / "MF3ZU_RXR_PIPELINE_SUPERVISOR.log"


class MF3ZUPipelineError(RuntimeError):
    """Raised when a sealed stage fails or its artifact does not validate."""


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    artifact: Path | None
    expected_status: str | None
    accepted_returncodes: tuple[int, ...] = (0,)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise MF3ZUPipelineError(f"stale pipeline partial: {partial}")
    partial.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MF3ZUPipelineError(f"JSON object required: {path}")
    return value


def build_stage_plan(
    *,
    python: str,
    gpu_id: int = 0,
    qwen_workers: int = 8,
) -> tuple[Stage, ...]:
    if gpu_id < 0 or qwen_workers < 1:
        raise MF3ZUPipelineError("invalid fixed-execution resources")
    output_text = str(OUTPUT)
    return (
        Stage(
            "causal_observation_replay",
            (
                python,
                "scripts/collect_mf3zu_rxr_observations.py",
                "--output-root",
                output_text,
                "--max-workers",
                "1",
                "--gpu-ids",
                str(gpu_id),
                "--max-attempts",
                "2",
            ),
            OUTPUT / "MF3ZU_OBSERVATION_COLLECTION_MANIFEST.json",
            "PASS",
        ),
        Stage(
            "fixed_qwen_annotation",
            (
                python,
                "scripts/annotate_mf3zu_rxr_evidence.py",
                "--output-root",
                output_text,
                "run",
                "--max-workers",
                str(qwen_workers),
            ),
            OUTPUT / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json",
            "PASS",
        ),
        Stage(
            "freeze_outcome_blind_evidence",
            (
                python,
                "scripts/build_mf3zu_evidence_memory.py",
                "--output-root",
                output_text,
            ),
            OUTPUT / "MF3ZU_RXR_EVIDENCE_MEMORY_MANIFEST.json",
            None,
            (0, 3),
        ),
        Stage(
            "once_only_five_fold_training",
            (
                python,
                "scripts/train_mf3zu_rxr_feasibility.py",
                "--device",
                f"cuda:{gpu_id}",
            ),
            OUTPUT / "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_RESULT.json",
            None,
        ),
        Stage(
            "immutable_result_audit",
            (python, "scripts/audit_mf3zu_rxr_feasibility_result.py"),
            OUTPUT / "MF3ZU_RXR_FEASIBILITY_RESULT_AUDIT.json",
            "MF3ZU_RXR_FEASIBILITY_RESULT_AUDIT_PASS",
        ),
        Stage(
            "full_regression",
            (python, "-m", "unittest", "discover", "-s", "tests", "-v"),
            None,
            None,
        ),
    )


def _artifact_status(stage: Stage) -> str | None:
    if stage.artifact is None:
        return None
    if not stage.artifact.is_file() or stage.artifact.is_symlink():
        raise MF3ZUPipelineError(f"stage artifact missing or unsafe: {stage.artifact}")
    return str(_read_object(stage.artifact).get("status", ""))


def _completed(stage: Stage) -> bool:
    if stage.artifact is None or not stage.artifact.is_file() or stage.artifact.is_symlink():
        return False
    status = _artifact_status(stage)
    if stage.name == "freeze_outcome_blind_evidence":
        return status in {
            "MF3ZU_RXR_EVIDENCE_MEMORY_FROZEN",
            "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL",
        }
    if stage.expected_status is None:
        return bool(status)
    return status == stage.expected_status


def _save_progress(
    *,
    state: str,
    current_stage: str | None,
    completed: Sequence[str],
    details: Mapping[str, object] | None = None,
) -> None:
    _atomic_json(
        STATUS_PATH,
        {
            "schema_version": "revealnav-mf3zu-rxr-pipeline-status/1",
            "revision": REVISION,
            "state": state,
            "current_stage": current_stage,
            "completed_stages": list(completed),
            "updated_at_utc": _now(),
            "details": dict(details or {}),
            "performance_metrics_exposed_while_running": False,
            "public_split_access": False,
            "full_navigation_run": False,
            "checkpoint_generated": False,
        },
    )


def run_pipeline(
    stages: Sequence[Stage],
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> int:
    verify_protocol(PROTOCOL_PATH)
    population = _read_object(POPULATION_MANIFEST_PATH)
    if (
        population.get("revision") != REVISION
        or population.get("status") != "MF3ZU_RXR_EXACT_SUPPORT_POPULATION_FROZEN"
        or population.get("population_rows") != 1_428
        or population.get("episodes") != 154
        or population.get("raw_scenes") != 59
    ):
        raise MF3ZUPipelineError("sealed 1428/154/59 population is missing")

    completed: list[str] = []
    _save_progress(state="RUNNING", current_stage=None, completed=completed)
    for stage in stages:
        if _completed(stage):
            completed.append(stage.name)
        else:
            _save_progress(
                state="RUNNING", current_stage=stage.name, completed=completed
            )
            result = runner(
                list(stage.command),
                cwd=ROOT,
                check=False,
                stdout=sys.stdout.buffer,
                stderr=subprocess.STDOUT,
            )
            if int(result.returncode) not in stage.accepted_returncodes:
                raise MF3ZUPipelineError(
                    f"stage {stage.name} exited {result.returncode}"
                )
            if stage.artifact is not None:
                _artifact_status(stage)
            if not _completed(stage):
                raise MF3ZUPipelineError(f"stage artifact failed validation: {stage.name}")
            completed.append(stage.name)

        if stage.name == "freeze_outcome_blind_evidence":
            support_status = _artifact_status(stage)
            if support_status == "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL":
                _save_progress(
                    state="SCIENTIFIC_STOP",
                    current_stage=None,
                    completed=completed,
                    details={"status": support_status, "training_started": False},
                )
                return 3

    result_status = _read_object(
        OUTPUT / "MF3ZU_RXR_EVIDENCE_MEMORY_FEASIBILITY_RESULT.json"
    ).get("status")
    _save_progress(
        state="COMPLETE",
        current_stage=None,
        completed=completed,
        details={"final_result_status": result_status},
    )
    return 0


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def launch_detached(*, gpu_id: int, qwen_workers: int) -> int:
    verify_protocol(PROTOCOL_PATH)
    if PROCESS_PATH.is_file() and not PROCESS_PATH.is_symlink():
        previous = _read_object(PROCESS_PATH)
        if _pid_alive(int(previous.get("pid", -1))):
            raise MF3ZUPipelineError("MF3ZU pipeline is already running")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--foreground",
        "--gpu-id",
        str(gpu_id),
        "--qwen-workers",
        str(qwen_workers),
    ]
    with LOG_PATH.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    _atomic_json(
        PROCESS_PATH,
        {
            "schema_version": "revealnav-mf3zu-rxr-pipeline-process/1",
            "revision": REVISION,
            "pid": process.pid,
            "launched_at_utc": _now(),
            "command": command,
            "status_path": str(STATUS_PATH.relative_to(ROOT)),
            "log_path": str(LOG_PATH.relative_to(ROOT)),
        },
    )
    print(json.dumps({
        "revision": REVISION,
        "pid": process.pid,
        "status_path": str(STATUS_PATH),
        "log_path": str(LOG_PATH),
        "continuous_monitoring": False,
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--foreground", action="store_true")
    mode.add_argument("--detach", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--qwen-workers", type=int, default=8)
    args = parser.parse_args()
    try:
        if args.detach:
            return launch_detached(
                gpu_id=args.gpu_id, qwen_workers=args.qwen_workers
            )
        stages = build_stage_plan(
            python=sys.executable,
            gpu_id=args.gpu_id,
            qwen_workers=args.qwen_workers,
        )
        return run_pipeline(stages)
    except BaseException as error:
        if args.foreground:
            try:
                _save_progress(
                    state="TECHNICAL_FAIL",
                    current_stage=None,
                    completed=[],
                    details={"error": f"{type(error).__name__}: {error}"},
                )
            except BaseException:
                pass
        print(
            f"MF3ZU_RXR_PIPELINE_FAIL_CLOSED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
