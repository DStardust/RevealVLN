#!/usr/bin/env python3
"""Human-readable live monitor for the V6.x pilot/full/training campaign."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/phase1/rxr_v6"


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def progress_bar(done: int, total: int, width: int = 30) -> str:
    ratio = done / total if total else 0.0
    filled = min(width, round(width * ratio))
    return f"[{'#' * filled}{'-' * (width - filled)}] {ratio:6.1%}"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "暂不可估算"
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"约 {hours}小时{minutes:02d}分"
    if minutes:
        return f"约 {minutes}分{seconds:02d}秒"
    return f"约 {seconds}秒"


def failure_count(value) -> int:
    return len(value) if isinstance(value, list) else int(value or 0)


def average_task_time(cohort_root: Path, stage: str) -> float | None:
    values = []
    for path in cohort_root.glob(f"runs/{stage}_*/RUN_SUMMARY.json"):
        row = read_json(path)
        if row and isinstance(row.get("wall_time_s"), (int, float)):
            values.append(float(row["wall_time_s"]))
    return sum(values) / len(values) if values else None


def compact_snapshot(revision: str) -> dict:
    label = revision.upper()
    state = read_json(BASE / f"RXR_{label}_CAMPAIGN_STATE.json") or {
        "status": "NOT_STARTED"
    }
    value = dict(state)
    for cohort in (f"pilot_{revision}", f"full_{revision}"):
        root = BASE / cohort
        progress = read_json(root / "RXR_V6_PAIR_PROGRESS.json")
        manifest = read_json(root / "RXR_V6_PAIRED_DATASET_MANIFEST.json")
        if progress:
            value[cohort] = {key: progress.get(key) for key in (
                "stage", "selected", "completed", "remaining",
                "active", "failures", "rejections",
            )}
        if manifest:
            value.setdefault(cohort, {})["dataset"] = manifest["metadata"]
    return value


def status_name(status: str) -> str:
    if "PILOT_FEASIBILITY_FAIL" in status:
        return "Pilot 可行性门失败（流程已停止）"
    if "PILOT_EXECUTION_FAILED" in status:
        return "Pilot 执行失败（流程已停止）"
    if "WAITING_FOR" in status:
        return "Pilot 正在采集；完成后自动进入全量阶段"
    if "FULL_COLLECTION_RUNNING" in status:
        return "全量反事实数据正在采集"
    if "FULL_COLLECTION_FAILED" in status:
        return "全量采集失败（流程已停止）"
    if "CROSSFIT_RUNNING" in status:
        return "五折模型训练与离线验证正在运行"
    if "OFFLINE_GATE_PASS" in status:
        return "全部完成，离线科学门通过"
    if "OFFLINE_GATE_FAIL" in status:
        return "全部完成，但离线科学门未通过"
    return status


def stage_line(index: int, name: str, cohort: dict | None, state: str) -> str:
    if cohort and cohort.get("dataset"):
        data = cohort["dataset"]
        return (
            f"[完成] {index}. {name}: {data.get('pairs', '?')} 对"
            f"（正收益 {data.get('positive_pairs', '?')} 对）"
        )
    if cohort:
        done = int(cohort.get("completed") or 0)
        total = int(cohort.get("selected") or 0)
        stage = "候选采集" if cohort.get("stage") == "shadow" else "反事实回放"
        if index == 1 and "PILOT_EXECUTION_FAILED" in state:
            return f"[失败] {index}. {name}: {stage}已处理 {done}/{total}"
        if index == 2 and "FULL_COLLECTION_FAILED" in state:
            return f"[失败] {index}. {name}: {stage}已处理 {done}/{total}"
        return f"[运行] {index}. {name}: {stage} {done}/{total}"
    if index == 3 and "CROSSFIT_RUNNING" in state:
        return f"[运行] {index}. {name}"
    if index == 3 and ("OFFLINE_GATE_PASS" in state or "OFFLINE_GATE_FAIL" in state):
        return f"[完成] {index}. {name}"
    return f"[等待] {index}. {name}"


def render(revision: str) -> str:
    label = revision.upper().replace("_", ".")
    snapshot = compact_snapshot(revision)
    state = str(snapshot.get("status", "UNKNOWN"))
    pilot_name = f"pilot_{revision}"
    full_name = f"full_{revision}"
    pilot = snapshot.get(pilot_name)
    full = snapshot.get(full_name)
    lines = [
        f"RevealNav RxR V{label[1:]} 实时监控  |  {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 72,
        f"总状态：{status_name(state)}",
        "",
        stage_line(1, "Pilot 可行性采集", pilot, state),
        stage_line(2, "Full 全量反事实采集", full, state),
        stage_line(3, "五折训练 + 离线门", None, state),
    ]

    active_name, active = (
        (full_name, full) if full and not full.get("dataset") else
        (pilot_name, pilot)
    )
    if active and not active.get("dataset"):
        done = int(active.get("completed") or 0)
        total = int(active.get("selected") or 0)
        remaining = int(active.get("remaining") or max(0, total - done))
        failures = failure_count(active.get("failures"))
        rejections = failure_count(active.get("rejections"))
        workers = active.get("active") or {}
        stage = str(active.get("stage") or "unknown")
        mean = average_task_time(BASE / active_name, stage)
        eta = remaining * mean / len(workers) if mean is not None and workers else None
        lines += [
            "",
            f"当前阶段：{active_name} / {'候选采集' if stage == 'shadow' else '反事实回放'}",
            f"进度：{progress_bar(done, total)}  {done}/{total}，剩余 {remaining}",
            f"结果：已处理 {done}，有效 {done - failures - rejections}，"
            f"拒绝 {rejections}，失败 {failures}，并行任务 {len(workers)}",
            f"速度：{f'平均 {mean:.1f} 秒/任务' if mean is not None else '正在收集耗时样本'}；"
            f"预计剩余 {'已停止' if failures and not workers else duration(eta)}",
        ]
        if workers:
            jobs = []
            for slot in sorted(workers, key=lambda value: int(value)):
                row = workers[slot]
                task = str(row.get("id", "?"))
                jobs.append(f"GPU{row.get('gpu', '?')}={task.removeprefix(stage + ':')}")
            lines += ["", "正在运行：", *[f"  {job}" for job in jobs]]

    if pilot and pilot.get("dataset"):
        data = pilot["dataset"]
        pairs = int(data.get("pairs") or 0)
        positive = int(data.get("positive_pairs") or 0)
        rate = positive / pairs if pairs else 0.0
        lines += ["", f"Pilot 收益：正收益 {positive}/{pairs}（{rate:.1%}）"]

    if "OFFLINE_GATE_PASS" in state or "OFFLINE_GATE_FAIL" in state:
        lines += ["", f"最终离线门：{'PASS' if 'PASS' in state else 'FAIL'}"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", choices=("v6_1", "v6_2"), default="v6_1")
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    parser.add_argument("--json", action="store_true", help="print the legacy JSON snapshot")
    args = parser.parse_args()
    if args.json:
        print(json.dumps(compact_snapshot(args.revision), indent=2, sort_keys=True))
        return 0
    while True:
        if args.watch and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(render(args.revision), flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
