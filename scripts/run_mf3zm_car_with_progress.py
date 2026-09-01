#!/usr/bin/env python3
"""Run the sealed MF3ZM-CAR v1 trainer with observable work-unit progress.

This wrapper does not change model inputs, optimization, folds, seeds,
hyperparameters, candidate selection, or scientific criteria.  It wraps the
existing ensemble-fit functions to count completed nested-fit work units and
writes a small atomic JSON file that survives the launching terminal.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts/training/mf3zm_car_v1"
PROGRESS = OUT / "MF3ZM_CAR_PROGRESS.json"
OBSERVABILITY = OUT / "MF3ZM_CAR_PROGRESS_PROTOCOL.json"
SOURCE_PROTOCOL = OUT / "MF3ZM_CAR_PROTOCOL.json"
RESULT = OUT / "MF3ZM_CAR_TRAIN_DEVELOPMENT_RESULT.json"
GATE = OUT / "gates/MF3ZM_CAR_MODEL.pt"
TRAINER = ROOT / "scripts/train_mf3zm_car.py"
MONITOR = ROOT / "scripts/monitor_mf3zm_car_progress.py"

PHASE_LABELS = (
    "car_mainline",
    "car_no_scene_constraint",
    "car_soft_risk",
    "car_28d",
    "car_policy_only",
    "car_no_risk",
    "rxr_only_car",
    "r2r_only_car",
    "dsr_v1_expanded_data",
)
# Each nested phase always performs 5 outer x 3 WD x 4 inner fits.  Up to five
# outer refits and one final fit are optional, so 66 is an honest maximum.
MAX_ENSEMBLE_FITS_PER_PHASE = 66
TOTAL_WORK_UNITS = len(PHASE_LABELS) * MAX_ENSEMBLE_FITS_PER_PHASE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load project module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Progress:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.completed_phases = 0
        self.phase_label = "initializing"
        self.phase_fit_completed = 0
        self.current_fit: dict = {}
        self.status = "RUNNING"
        self.message = "loading sealed trainer"
        self.write()

    def write(self) -> None:
        completed = (
            self.completed_phases * MAX_ENSEMBLE_FITS_PER_PHASE
            + min(self.phase_fit_completed, MAX_ENSEMBLE_FITS_PER_PHASE)
        )
        percent = 100.0 * completed / TOTAL_WORK_UNITS
        atomic_json(PROGRESS, {
            "schema_version": "revealnav-mf3zm-car-progress/1",
            "status": self.status,
            "pid": os.getpid(),
            "started_unix": self.started_at,
            "updated_unix": time.time(),
            "elapsed_seconds": time.time() - self.started_at,
            "phase": self.phase_label,
            "phase_index": min(self.completed_phases + 1, len(PHASE_LABELS)),
            "phase_count": len(PHASE_LABELS),
            "phase_fit_completed": self.phase_fit_completed,
            "phase_fit_maximum": MAX_ENSEMBLE_FITS_PER_PHASE,
            "work_units_completed": completed,
            "work_units_total": TOTAL_WORK_UNITS,
            "progress_percent": min(percent, 100.0),
            "current_fit": self.current_fit,
            "message": self.message,
            "result_exists": RESULT.is_file(),
            "checkpoint_exists": GATE.is_file(),
            "progress_definition": (
                "completed ensemble fits over the pre-registered maximum; "
                "skipped optional refits complete when a phase closes"
            ),
        })

    def begin_phase(self, label: str) -> None:
        if label not in PHASE_LABELS:
            raise RuntimeError(f"unknown progress phase: {label}")
        expected = PHASE_LABELS[self.completed_phases]
        if label != expected:
            raise RuntimeError(
                f"progress phase order drift: expected {expected}, found {label}"
            )
        self.phase_label = label
        self.phase_fit_completed = 0
        self.current_fit = {}
        self.message = "phase running"
        self.write()

    def begin_fit(self, kind: str, rows: int, weight_decay: float) -> None:
        self.current_fit = {
            "kind": kind,
            "fit_number": self.phase_fit_completed + 1,
            "rows": int(rows),
            "weight_decay": float(weight_decay),
        }
        self.message = "ensemble fit running"
        self.write()

    def end_fit(self) -> None:
        self.phase_fit_completed += 1
        self.current_fit = {}
        self.message = "ensemble fit complete"
        self.write()

    def end_phase(self) -> None:
        self.completed_phases += 1
        self.phase_fit_completed = 0
        self.current_fit = {}
        self.message = "phase complete"
        self.write()

    def finish(self, exit_code: int) -> None:
        self.completed_phases = len(PHASE_LABELS)
        self.phase_label = "complete"
        self.phase_fit_completed = 0
        self.current_fit = {}
        self.status = "COMPLETE" if RESULT.is_file() else "ERROR"
        self.message = f"trainer exit code {int(exit_code)}"
        self.write()

    def error(self, exc: BaseException) -> None:
        self.status = "ERROR"
        self.message = f"{type(exc).__name__}: {exc}"
        self.current_fit = {
            **self.current_fit,
            "traceback": traceback.format_exc(limit=12),
        }
        self.write()


def _phase_for_arm(kwargs: dict) -> str:
    arm = kwargs.get("arm", "joint")
    representation = kwargs.get("representation", "semantic")
    risk_mode = kwargs.get("risk_mode", "hard")
    scene_constraint = kwargs.get("scene_constraint", True)
    if arm == "RxR":
        return "rxr_only_car"
    if arm == "R2R":
        return "r2r_only_car"
    if representation == "engineered_28d":
        return "car_28d"
    if representation == "policy_only":
        return "car_policy_only"
    if risk_mode == "soft":
        return "car_soft_risk"
    if risk_mode == "none":
        return "car_no_risk"
    if scene_constraint is False:
        return "car_no_scene_constraint"
    return "car_mainline"


def seal_observability() -> None:
    if not SOURCE_PROTOCOL.is_file():
        raise RuntimeError("sealed CAR protocol is unavailable")
    value = {
        "schema_version": "revealnav-mf3zm-car-progress-protocol/1",
        "status": "OBSERVABILITY_ONLY",
        "algorithm_revision": "mf3zm_car_v1",
        "algorithm_change": False,
        "source_protocol": {
            "path": str(SOURCE_PROTOCOL.relative_to(ROOT)),
            "bytes": SOURCE_PROTOCOL.stat().st_size,
            "sha256": sha256_file(SOURCE_PROTOCOL),
        },
        "wrapper": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "monitor": {
            "path": str(MONITOR.relative_to(ROOT)),
            "sha256": sha256_file(MONITOR),
        },
        "progress_definition": {
            "phase_labels": list(PHASE_LABELS),
            "maximum_ensemble_fits_per_phase": MAX_ENSEMBLE_FITS_PER_PHASE,
            "total_work_units": TOTAL_WORK_UNITS,
        },
        "public_split_access": False,
    }
    if OBSERVABILITY.exists():
        if json.loads(OBSERVABILITY.read_text()) != value:
            raise RuntimeError("CAR observability protocol drift")
    else:
        atomic_json(OBSERVABILITY, value)


def run() -> int:
    if RESULT.exists() or GATE.exists():
        raise RuntimeError("refusing to overwrite CAR result or checkpoint")
    seal_observability()
    progress = Progress()
    trainer = _load_module(TRAINER, "sealed_car_trainer_with_progress")

    import revealnav_mf3.car_selection as car_selection
    import revealnav_mf3.dsr_selection as dsr_selection

    original_car_fit = car_selection.fit_car_ensemble
    original_dsr_fit = dsr_selection._fit_ensemble
    original_arm = trainer._fit_arm
    original_dsr_control = trainer._fit_dsr_control

    def car_fit(*args, **kwargs):
        target = args[1] if len(args) > 1 else kwargs["target"]
        progress.begin_fit(
            "car", len(target), float(kwargs["weight_decay"])
        )
        value = original_car_fit(*args, **kwargs)
        progress.end_fit()
        return value

    def dsr_fit(*args, **kwargs):
        target = args[1] if len(args) > 1 else kwargs["target"]
        progress.begin_fit(
            "dsr", len(target), float(kwargs["weight_decay"])
        )
        value = original_dsr_fit(*args, **kwargs)
        progress.end_fit()
        return value

    def fit_arm(*args, **kwargs):
        progress.begin_phase(_phase_for_arm(kwargs))
        try:
            return original_arm(*args, **kwargs)
        finally:
            progress.end_phase()

    def fit_dsr_control(*args, **kwargs):
        progress.begin_phase("dsr_v1_expanded_data")
        try:
            return original_dsr_control(*args, **kwargs)
        finally:
            progress.end_phase()

    car_selection.fit_car_ensemble = car_fit
    dsr_selection._fit_ensemble = dsr_fit
    trainer._fit_arm = fit_arm
    trainer._fit_dsr_control = fit_dsr_control
    try:
        exit_code = int(trainer.fit())
        progress.finish(exit_code)
        return exit_code
    except BaseException as exc:
        progress.error(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(run())
