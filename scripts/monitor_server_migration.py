#!/usr/bin/env python3
"""Show an approximate live progress and ETA for the server migration."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
INVENTORY = ROOT / "artifacts/migration/SERVER_MIGRATION_SOURCE_INVENTORY.json"
LOG_DIR = ROOT / "artifacts/migration/transfer_logs"
SUPERVISOR = Path("/var/tmp/daiyang_server_migration_supervisor")
VERIFY = Path("/var/tmp/daiyang_server_migration_verify")
PRIORITY = Path("/var/tmp/daiyang_vla_priority")
REMOTE_IP = "8.130.54.48"
FILE_LINE = re.compile(r"\bsend\s+<f\S*\s+.*\s(\d+)$")
DELIVERY_RATE = re.compile(r"\bdelivery_rate\s+(\d+)bps\b")


def run(*args: str) -> str:
    result = subprocess.run(args, check=False, text=True, capture_output=True)
    return result.stdout


def human_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def human_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    if days:
        return f"{days}天 {hours}小时 {minutes}分钟"
    if hours:
        return f"{hours}小时 {minutes}分钟"
    return f"{minutes}分钟"


def completed_file_bytes(log: Path) -> int:
    total = 0
    if not log.exists():
        return total
    with log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = FILE_LINE.search(line.rstrip("\n"))
            if match:
                total += int(match.group(1))
    return total


def marker_count(prefix: str) -> int:
    return sum((LOG_DIR / f"{prefix}_part_{part}.pass").is_file() for part in range(6))


def main() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    total = int(inventory["source_disk_usage_total_bytes"])
    initial_markers = marker_count("initial")
    final_markers = marker_count("final")
    verify_markers = sum((VERIFY / f"part_{part}.pass").is_file() for part in range(6))
    closure_pass = (SUPERVISOR / "ALL_CHECKSUM_VERIFICATIONS_PASS").is_file()
    priority_ready = (PRIORITY / "VLA_WORKSET_READY").is_file()
    remote_ready = (PRIORITY / "REMOTE_VLA_READY").is_file()
    priority_active = run(
        "systemctl",
        "show",
        "daiyang-vla-priority",
        "-p",
        "ActiveState",
        "--value",
    ).strip() == "active"
    readiness_active = run(
        "systemctl",
        "show",
        "daiyang-vla-readiness",
        "-p",
        "ActiveState",
        "--value",
    ).strip() == "active"

    completed = sum(
        completed_file_bytes(LOG_DIR / f"initial_part_{part}.rsync.log")
        for part in range(6)
    )
    completed = min(completed, total)

    socket_info = run("ss", "-tin", "dst", REMOTE_IP)
    bits_per_second = sum(int(value) for value in DELIVERY_RATE.findall(socket_info))
    bytes_per_second = bits_per_second / 8
    connections = socket_info.count("ESTAB")

    active = run(
        "systemctl",
        "show",
        "daiyang-server-migration",
        "-p",
        "ActiveState",
        "--value",
    ).strip() or "not-found"

    if closure_pass:
        phase = "完成：六路校验全部通过"
    elif remote_ready:
        phase = "VLA 远端环境已通过 smoke；其余数据后台迁移"
    elif priority_ready and readiness_active:
        phase = "VLA 文件已校验；远端兼容构建与 smoke"
    elif priority_ready:
        phase = "VLA 文件已校验；等待远端启动检查"
    elif priority_active:
        phase = "VLA 工作集优先迁移"
    elif verify_markers:
        phase = "最终内容校验"
    elif final_markers or initial_markers == 6:
        phase = "最终增量收敛"
    else:
        phase = "首轮批量传输"

    print(f"更新时间: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"阶段: {phase}")
    print(f"后台服务: {active}")
    print(f"SSH 数据连接: {connections}")
    print(f"当前总传输速率: {human_bytes(bytes_per_second)}/s ({bits_per_second / 1_000_000:.1f} Mbit/s)")

    if priority_active and not priority_ready:
        source_bytes_file = PRIORITY / "source_bytes"
        if source_bytes_file.is_file():
            priority_total = int(source_bytes_file.read_text().strip())
            priority_completed = sum(
                completed_file_bytes(
                    LOG_DIR.parent / "vla_priority_logs" / f"priority_part_{part}.rsync.log"
                )
                for part in range(4)
            )
            priority_completed = min(priority_completed, priority_total)
            priority_remaining = priority_total - priority_completed
            priority_percent = priority_completed * 100 / priority_total
            print(
                "VLA 优先工作集: "
                f"{human_bytes(priority_completed)} / {human_bytes(priority_total)} "
                f"({priority_percent:.2f}%)"
            )
            if bytes_per_second > 0:
                print(
                    "VLA 可开工预计等待: "
                    f"{human_duration(priority_remaining / bytes_per_second)}"
                )

    if initial_markers == 6:
        print("首轮数据进度: 100.00%（等待或正在进行最终增量/校验）")
    else:
        percent = completed * 100 / total
        remaining = max(0, total - completed)
        print(f"日志确认已完成文件: {human_bytes(completed)} / {human_bytes(total)} ({percent:.2f}%)")
        print(f"估算剩余数据: {human_bytes(remaining)}")
        if bytes_per_second > 0:
            print(f"按当前瞬时速率估算: {human_duration(remaining / bytes_per_second)}")
        else:
            print("按当前瞬时速率估算: 暂不可用")

    print(f"首轮分片: {initial_markers}/6")
    print(f"最终同步分片: {final_markers}/6")
    print(f"校验分片: {verify_markers}/6")
    print(f"VLA 文件闭环: {'PASS' if priority_ready else 'PENDING'}")
    print(f"VLA 远端可开工: {'PASS' if remote_ready else 'PENDING'}")
    print(f"迁移闭环: {'PASS' if closure_pass else 'PENDING'}")
    print("说明: 已完成量按 rsync 完成文件日志统计，不含当前正在传输的文件；ETA 是近似值。")


if __name__ == "__main__":
    main()
