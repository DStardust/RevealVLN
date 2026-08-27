#!/usr/bin/env python3
"""Print a compact, read-only live view of scale-v2 through model training."""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2"
MULTI = BASE / "automatic/multibranch"
STATUS = BASE / "RXR_SCALE_V2_SUPERVISOR_STATUS.json"
LOG = BASE / "RXR_SCALE_V2_SUPERVISOR.log"
ANALYSIS = MULTI / "RXR_SCALE_CAUSAL_CANDIDATE_ANALYSIS.json"
LANGUAGE = MULTI / "RXR_SCALE_CAUSAL_PREFIX_LANGUAGE_GATE.json"
INDEX = MULTI / "RXR_SCALE_TRAINING_INDEX.json"
TX = MULTI / "RXR_SCALE_TX_GATE.json"
FEATURE = MULTI / "RXR_SCALE_FEATURE_GATE.json"
MODEL = BASE / "model_training/RXR_SCALE_MODEL_SUPERVISOR_STATUS.json"


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def count_suffix(path: Path, suffix: str) -> int:
    try:
        return sum(row.is_file() and row.suffix == suffix for row in path.iterdir())
    except OSError:
        return 0


def bar(done: int, total: int | None, width: int = 28) -> str:
    if not total:
        return "[" + "-" * width + "]  --"
    filled = min(width, round(width * done / total))
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} {100 * done / total:5.1f}%"


def language_progress() -> tuple[int, int, int, int]:
    gate = load(LANGUAGE)
    analysis = load(ANALYSIS) or {}
    expected = analysis.get("status_counts", {}).get(
        "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED", 0
    )
    if gate:
        counts = gate.get("counts", {})
        passed = counts.get("language_k3_pass", 0)
        failed = counts.get("language_k3_fail", 0)
        return passed + failed, expected, passed, failed
    try:
        text = LOG.read_text(errors="replace")
    except OSError:
        return 0, expected, 0, 0
    start = text.rfind("[causal_language] START")
    if start >= 0:
        text = text[start:]
    rows = re.findall(
        r"^(v2x\d+_ep\d+_hv\d+) CAUSAL_LANGUAGE_K3_(PASS_CONTROLS_REQUIRED|FAIL)$",
        text,
        re.MULTILINE,
    )
    outcomes = {event_id: outcome for event_id, outcome in rows}
    passed = sum(value.startswith("PASS") for value in outcomes.values())
    failed = sum(value == "FAIL" for value in outcomes.values())
    return len(outcomes), expected, passed, failed


def service(name: str) -> str:
    result = subprocess.run(
        ["systemctl", "is-active", name],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def active_service(names: tuple[str, ...]) -> str:
    states = [(name, service(name)) for name in names]
    for name, state in states:
        if state == "active":
            return name.removesuffix(".service") + "=active"
    return ", ".join(
        name.removesuffix(".service") + "=" + state
        for name, state in states
    )


def gpu_rows() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return ["GPU状态不可用"]
    rows = []
    for line in result.stdout.splitlines():
        index, used, free, utilization = [part.strip() for part in line.split(",")]
        rows.append(f"GPU{index}: used {int(used):5d} MiB  free {int(free):5d} MiB  util {int(utilization):3d}%")
    return rows


def main() -> int:
    supervisor = load(STATUS) or {}
    model = load(MODEL) or {}
    index = load(INDEX) or {}
    tx = load(TX) or {}
    feature = load(FEATURE) or {}
    expected = index.get("counts", {}).get("events")
    language_done, language_total, language_pass, language_fail = language_progress()
    round1 = count_suffix(MULTI / "tx_runs/round1", ".json")
    round2 = count_suffix(MULTI / "tx_runs/round2", ".json")
    features = count_suffix(MULTI / "frozen_features", ".npz")

    print("RevealNav RxR scale pipeline    " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 72)
    print(f"V2监督器 : {supervisor.get('status', 'MISSING')} / {supervisor.get('current_stage', '-')}")
    if supervisor.get("failed_stage"):
        print(f"失败阶段   : {supervisor['failed_stage']}  {supervisor.get('error', '')}")
    print(f"语言判定   : {bar(language_done, language_total)}  PASS={language_pass} FAIL={language_fail}")
    print(f"训练索引   : {index.get('status', 'WAITING')}  events={expected or 0}")
    print(f"资源标签R1 : {bar(round1, expected)}")
    print(f"资源标签R2 : {bar(round2, expected)}")
    print(f"冻结特征   : {bar(features, expected)}  gate={feature.get('status', 'WAITING')}")
    print(f"模型阶段   : {model.get('status', 'MISSING')} / {model.get('current_stage', '-')}")
    print("-" * 72)
    print("服务       : " + active_service((
        "revealnav-rxr-scale-v2-tx24.service",
        "revealnav-rxr-scale-v2-tx-accelerated.service",
        "revealnav-rxr-scale-v2-resume.service",
        "revealnav-rxr-scale-v2-durable.service",
    )) + "  MODEL=" + service(
        "revealnav-rxr-scale-model-supervisor.service"
    ))
    print("GPU:")
    for row in gpu_rows():
        print("  " + row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
