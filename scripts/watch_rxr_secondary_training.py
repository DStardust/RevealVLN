#!/usr/bin/env python3
"""Continuously show compact secondary-data and training progress."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SECONDARY = ROOT / "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
PIPELINE_STATUS = SECONDARY / "RXR_SECONDARY_AUTOMATIC_PIPELINE_STATUS.json"
PIPELINE_LOG = SECONDARY / "RXR_SECONDARY_AUTOMATIC_PIPELINE.log"
LANGUAGE_RESULTS = SECONDARY / "multibranch/prefix_language_results"
LANGUAGE_GATE = SECONDARY / (
    "multibranch/RXR_SECONDARY_CAUSAL_PREFIX_LANGUAGE_GATE.json"
)
TX_PLAN = SECONDARY / "multibranch/RXR_SECONDARY_TX_PLAN.json"
TX_GATE = SECONDARY / "multibranch/RXR_SECONDARY_TX_GATE.json"
TX_RUNS = SECONDARY / "multibranch/tx_runs"
FEATURES = SECONDARY / "multibranch/frozen_features"
FEATURE_GATE = SECONDARY / "multibranch/RXR_SECONDARY_FEATURE_GATE.json"
TRAINING_STATUS = ROOT / (
    "artifacts/evaluation/mf2_secondary_augmentation_v1/"
    "RXR_SECONDARY_TRAINING_STATUS.json"
)
COMPARISON = ROOT / (
    "artifacts/evaluation/mf2_secondary_augmentation_v1/"
    "RXR_SECONDARY_AUGMENTATION_COMPARISON_V1.json"
)
RELATIONAL_STATUS = ROOT / (
    "artifacts/evaluation/mf2_relational_augmented_v2/"
    "RXR_RELATIONAL_AUGMENTED_STATUS_V2.json"
)
RELATIONAL_COMPARISON = ROOT / (
    "artifacts/evaluation/mf2_relational_augmented_v2/"
    "RXR_RELATIONAL_AUGMENTED_COMPARISON_V2.json"
)
GOLD_RESULT = ROOT / (
    "artifacts/evaluation/mf2_relational_gold_v1/"
    "RXR_RELATIONAL_GOLD_RESULT_V1.json"
)
EXPECTED_LANGUAGE_EVENTS = 206


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def alive(pid: int | None) -> str:
    return "存活" if pid and Path(f"/proc/{pid}").exists() else "停止"


def duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}小时{minutes:02d}分" if hours else f"{minutes}分"


def language_progress() -> dict:
    terminal = Counter()
    if PIPELINE_LOG.is_file():
        text = PIPELINE_LOG.read_text(errors="replace")
        text = text.rsplit("[causal_language] START", 1)[-1]
        for line in text.splitlines():
            if " CAUSAL_LANGUAGE_K3_" in line:
                terminal[line.rsplit(" ", 1)[-1]] += 1
    files = list(LANGUAGE_RESULTS.glob("*/*.json"))
    latest = max((path.stat().st_mtime for path in files), default=0.0)
    completed = sum(terminal.values())
    passed = terminal["CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED"]
    failed = terminal["CAUSAL_LANGUAGE_K3_FAIL"]
    gate = load(LANGUAGE_GATE)
    if gate.get("status") == "COMPLETE_CAUSAL_CONTROLS_REQUIRED":
        counts = gate["counts"]
        completed = counts["frontend_causal_ready"]
        passed = counts["language_k3_pass"]
        failed = counts["language_k3_fail"]
    return {
        "completed": completed,
        "passed": passed,
        "failed": failed,
        "events_started": len({path.parent.name for path in files}),
        "responses": len(files),
        "latest_age": time.time() - latest if latest else None,
    }


def running_tx_workers() -> dict[int, str]:
    """Return the event currently assigned to each live secondary T_X worker."""
    workers = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            values = [value.decode(errors="replace") for value in args if value]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(value.endswith("rxr_secondary_tx_worker.py") for value in values):
            continue
        try:
            gpu = int(values[values.index("--gpu") + 1])
            event_id = values[values.index("--event-id") + 1]
        except (ValueError, IndexError):
            continue
        workers[gpu] = event_id
    return workers


def resource_progress(pipeline: dict) -> dict:
    plan = load(TX_PLAN)
    events = plan.get("eligible_event_ids", [])
    event_set = set(events)
    expected = len(events)
    rounds = {}
    latest = 0.0
    for round_name in ("round1", "round2"):
        completed_ids = {
            path.stem for path in (TX_RUNS / round_name).glob("*.json")
            if path.stem in event_set and path.is_file() and path.stat().st_size > 0
        }
        completed_paths = [
            TX_RUNS / round_name / f"{event_id}.json"
            for event_id in completed_ids
        ]
        latest = max(
            latest,
            max((path.stat().st_mtime for path in completed_paths), default=0.0),
        )
        per_gpu = Counter()
        for index, event_id in enumerate(events):
            if event_id in completed_ids:
                per_gpu[index % 8] += 1
        rounds[round_name] = {
            "completed": len(completed_ids),
            "per_gpu": per_gpu,
        }
    total = sum(value["completed"] for value in rounds.values())
    total_expected = expected * 2
    started = next(
        (
            row.get("started") for row in pipeline.get("stages", [])
            if row.get("name") == "resource_labels"
        ),
        None,
    )
    eta = None
    if started and 0 < total < total_expected:
        elapsed = time.time() - started
        eta = elapsed / total * (total_expected - total)
    if expected and rounds["round1"]["completed"] < expected:
        current_round = "第1轮"
    elif expected and rounds["round2"]["completed"] < expected:
        current_round = "第2轮"
    elif expected:
        current_round = "聚合/已完成"
    else:
        current_round = "尚未生成计划"
    return {
        "expected": expected,
        "rounds": rounds,
        "total": total,
        "total_expected": total_expected,
        "current_round": current_round,
        "latest_age": time.time() - latest if latest else None,
        "eta": eta,
        "workers": running_tx_workers(),
        "gate": load(TX_GATE),
    }


def running_feature_lanes() -> set[int]:
    lanes = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            values = [value.decode(errors="replace") for value in args if value]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(value.endswith("rxr_secondary_feature_lane.py")
                   for value in values):
            continue
        try:
            lanes.add(int(values[values.index("--physical-gpu") + 1]))
        except (ValueError, IndexError):
            continue
    return lanes


def feature_progress(pipeline: dict) -> dict:
    lane_events = {}
    all_events = set()
    feature_root = SECONDARY / "multibranch"
    for gpu in range(8):
        path = feature_root / f"feature_lane_gpu{gpu}_events.json"
        events = load(path) if path.is_file() else []
        if not isinstance(events, list):
            events = []
        lane_events[gpu] = events
        all_events.update(events)
    completed_paths = {
        path.stem: path for path in FEATURES.glob("*.npz")
        if path.stem in all_events and path.is_file() and path.stat().st_size > 0
    }
    per_gpu = {
        gpu: sum(event_id in completed_paths for event_id in events)
        for gpu, events in lane_events.items()
    }
    latest = max(
        (path.stat().st_mtime for path in completed_paths.values()),
        default=0.0,
    )
    completed = len(completed_paths)
    expected = len(all_events)
    started = next(
        (
            row.get("started") for row in pipeline.get("stages", [])
            if row.get("name") == "frozen_features"
        ),
        None,
    )
    eta = None
    if started and 0 < completed < expected:
        elapsed = time.time() - started
        eta = elapsed / completed * (expected - completed)
    return {
        "completed": completed,
        "expected": expected,
        "per_gpu": per_gpu,
        "lane_sizes": {gpu: len(events) for gpu, events in lane_events.items()},
        "running_lanes": running_feature_lanes(),
        "latest_age": time.time() - latest if latest else None,
        "eta": eta,
        "gate": load(FEATURE_GATE),
    }


def render() -> str:
    pipeline = load(PIPELINE_STATUS)
    training = load(TRAINING_STATUS)
    relational = load(RELATIONAL_STATUS)
    language = language_progress()
    resource = resource_progress(pipeline)
    features = feature_progress(pipeline)
    completed = language["completed"]
    elapsed = time.time() - next(
        (
            row["started"] for row in pipeline.get("stages", [])
            if row["name"] == "causal_language"
        ),
        time.time(),
    )
    eta = (
        elapsed / completed * (EXPECTED_LANGUAGE_EVENTS - completed)
        if 0 < completed < EXPECTED_LANGUAGE_EVENTS else None
    )
    completion = 100.0 * completed / EXPECTED_LANGUAGE_EVENTS
    pass_rate = 100.0 * language["passed"] / completed if completed else 0.0
    lines = [
        f"RevealNav 实时监控  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        (
            f"自动流水线: {pipeline.get('status', '--')} / "
            f"{pipeline.get('current_stage', '已结束')}  "
            f"PID {pipeline.get('pid', '--')} [{alive(pipeline.get('pid'))}]"
        ),
        (
            f"语言门: {completed}/{EXPECTED_LANGUAGE_EVENTS} ({completion:5.1f}%)  "
            f"通过 {language['passed']}  拒绝 {language['failed']}  "
            f"通过率 {pass_rate:5.1f}%"
        ),
        (
            f"响应: {language['responses']} 条 / "
            f"已启动 {language['events_started']} 个事件  "
            f"最新响应 {language['latest_age']:.1f} 秒前"
            if language["latest_age"] is not None else "响应: 尚未开始"
        ),
        f"语言门线性剩余估计: {duration(eta)}",
    ]
    if resource["expected"]:
        tx_completion = (
            100.0 * resource["total"] / resource["total_expected"]
        )
        first = resource["rounds"]["round1"]
        second = resource["rounds"]["round2"]
        lane_counts = "  ".join(
            f"GPU{gpu}:{first['per_gpu'][gpu]}/{second['per_gpu'][gpu]}"
            for gpu in range(8)
        )
        active = "  ".join(
            f"GPU{gpu}:{event_id}" for gpu, event_id
            in sorted(resource["workers"].items())
        ) or "无（轮次切换、聚合或已结束）"
        gate_status = resource["gate"].get("status")
        lines.extend([
            "-" * 70,
            (
                f"T_X 资源标签: {resource['current_round']}  "
                f"第1轮 {first['completed']}/{resource['expected']}  "
                f"第2轮 {second['completed']}/{resource['expected']}  "
                f"总进度 {tx_completion:5.1f}%"
            ),
            f"各 GPU 完成数（第1轮/第2轮）: {lane_counts}",
            f"正在执行: {active}",
            (
                f"最近标签落盘: {resource['latest_age']:.1f} 秒前  "
                f"线性剩余估计: {duration(resource['eta'])}"
                if resource["latest_age"] is not None
                else "最近标签落盘: 尚无"
            ),
        ])
        if gate_status:
            lines.append(f"T_X 最终门: {gate_status}")
    if features["expected"]:
        feature_completion = 100.0 * features["completed"] / features["expected"]
        lane_counts = "  ".join(
            f"GPU{gpu}:{features['per_gpu'][gpu]}/{features['lane_sizes'][gpu]}"
            for gpu in range(8)
        )
        active = ",".join(
            str(gpu) for gpu in sorted(features["running_lanes"])
        ) or "无（聚合、已完成或已停止）"
        lines.extend([
            "-" * 70,
            (
                f"冻结特征: {features['completed']}/{features['expected']} "
                f"({feature_completion:5.1f}%)  运行中 GPU: {active}"
            ),
            f"各 GPU lane: {lane_counts}",
            (
                f"最近特征落盘: {features['latest_age']:.1f} 秒前  "
                f"线性剩余估计: {duration(features['eta'])}"
                if features["latest_age"] is not None
                else "最近特征落盘: 尚无（模型加载中）"
            ),
        ])
        feature_gate_status = features["gate"].get("status")
        if feature_gate_status:
            lines.append(f"特征最终门: {feature_gate_status}")
    lines.extend([
        "-" * 70,
        (
            f"训练监督器: {training.get('status', '--')}  "
            f"PID {training.get('pid', '--')} [{alive(training.get('pid'))}]"
        ),
    ])
    for seed, value in sorted(training.get("runs", {}).items()):
        lines.append(
            f"  seed {seed}: {value.get('status', '--')}  "
            f"GPU {value.get('physical_gpu', '--')}  PID {value.get('pid', '--')}"
        )
    lines.append(
        f"关系模型增广: {relational.get('status', '--')}  "
        f"PID {relational.get('pid', '--')} [{alive(relational.get('pid'))}]"
    )
    for seed, value in sorted(relational.get("runs", {}).items()):
        lines.append(
            f"  relational seed {seed}: {value.get('status', '--')}  "
            f"GPU {value.get('physical_gpu', '--')}  PID {value.get('pid', '--')}"
        )
    if COMPARISON.is_file():
        comparison = load(COMPARISON)
        lines.extend([
            "-" * 70,
            f"旧冻结模型 development 对照: {comparison.get('status', '--')}",
        ])
    if RELATIONAL_COMPARISON.is_file():
        comparison = load(RELATIONAL_COMPARISON)
        lines.append(
            f"关系模型 development 对照: {comparison.get('status', '--')}"
        )
    if GOLD_RESULT.is_file():
        gold = load(GOLD_RESULT)
        effect = gold.get("selected_comparison", {}).get(
            "augmented_minus_primary_macro_f1", {}
        )
        lines.append(
            f"107-event Gold pilot: {gold.get('status', '--')}  "
            f"Macro-F1 Δ {effect.get('mean', float('nan')):+.4f}"
        )
    lines.extend(["=" * 70, "按 Ctrl+C 退出监控；不会停止后台流水线。"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    while True:
        if not args.once and os.isatty(1):
            print("\033[2J\033[H", end="")
        print(render(), flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
