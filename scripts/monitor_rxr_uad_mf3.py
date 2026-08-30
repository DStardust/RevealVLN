#!/usr/bin/env python3
"""Print the current MF3 online-data or training status."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/phase1/mf3b_uad_online/dataset_v1"
progress = DATA / "MF3B_ONLINE_DATA_PROGRESS.json"
protocol = DATA / "MF3B_ONLINE_DATA_PROTOCOL.json"
if progress.is_file():
    value = json.loads(progress.read_text())
    elapsed = max(time.time() - protocol.stat().st_mtime, 1.0)
    completed = int(value["completed"])
    rate = completed / elapsed
    unfinished = int(value["total"]) - completed
    eta = unfinished / rate if rate else None
    print(
        f"data {value['status']}: {completed}/{value['total']} complete, "
        f"failed={value['failed']}, active={len(value['active'])}, "
        f"ETA={eta / 60:.1f} min" if eta is not None else
        f"data {value['status']}: {completed}/{value['total']} complete"
    )
    for gpu, row in sorted(value["active"].items(), key=lambda item: int(item[0])):
        print(f"  GPU {gpu}: episode {row['episode_id']} scene {row['scene_id']}")
else:
    print("data: not started")
active_training = []
uptime = float(Path("/proc/uptime").read_text().split()[0])
clock_ticks = os.sysconf("SC_CLK_TCK")
for process in Path("/proc").glob("[0-9]*"):
    try:
        command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if "scripts/train_rxr_uad_mf3.py --seed" not in command:
            continue
        stat = (process / "stat").read_text().split()
        elapsed = uptime - int(stat[21]) / clock_ticks
        active_training.append((int(process.name), command.strip(), max(elapsed, 0.0)))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
if active_training:
    print(f"training: {len(active_training)} active seeds")
    for pid, command, elapsed in sorted(active_training):
        seed = command.split("--seed", 1)[1].split()[0]
        print(f"  PID {pid}: seed {seed}, active {elapsed / 60:.1f} min")
for seed in (20260826, 20260827, 20260828):
    result = ROOT / f"artifacts/training/mf3b_uad_online_v1/seed_{seed}/RESULT.json"
    if result.is_file():
        row = json.loads(result.read_text())
        print(
            f"seed {seed}: target_acc="
            f"{row['online_calibration_target_accuracy']:.4f}, "
            f"semantic_macro_f1="
            f"{row['legacy_semantic_development']['macro_f1']:.4f}"
        )
gate = ROOT / "artifacts/evaluation/mf3b_uad_shadow_gate_v1/MF3B_UAD_SHADOW_GATE.json"
if gate.is_file():
    row = json.loads(gate.read_text())
    print(f"shadow gate: {row['status']}")
    for seed, value in row.get("shadow_members", {}).items():
        print(
            f"  seed {seed}: rescue={value['rescues']} harm={value['harms']} "
            f"interventions={value['interventions']}"
        )
metric_progress = ROOT / (
    "artifacts/evaluation/mf3b_uad_rxr_val_seen_v1/"
    "MF3B_RXR_VAL_SEEN_PROGRESS.json"
)
if metric_progress.is_file():
    row = json.loads(metric_progress.read_text())
    print(
        f"RxR val_seen {row['status']}: {row['completed']}/{row['total']}, "
        f"failed={row['failed']}, active={len(row['active'])}"
    )
metric_result = metric_progress.with_name("MF3B_RXR_VAL_SEEN_RESULT.json")
if metric_result.is_file():
    row = json.loads(metric_result.read_text())
    utility = row["aggregate_uad_seed_median_minus_baseline"]["utility"]
    print(
        f"task metric gate: {row['status']}, utility={utility['mean']:.6f}, "
        f"95% CI={utility['scene_bootstrap_95pct']}"
    )
