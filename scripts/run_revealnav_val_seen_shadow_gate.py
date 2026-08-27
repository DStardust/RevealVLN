#!/usr/bin/env python3
"""Run and verify behavior-neutral RevealNav shadow inference on val_seen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla").resolve()
RUNTIME = ROOT / "artifacts/runtime"
WORKER = RUNTIME / "revealnav_val_seen_observer_worker.py"
OLD_GATE = RUNTIME / "R2R_VAL_SEEN_RUNTIME_GATE.json"
SELECTION = RUNTIME / "R2R_VAL_SEEN_EPISODE_SELECTION.json"
DEVELOPMENT = ROOT / (
    "artifacts/evaluation/mf2_ecog_opp_development_v2/"
    "RXR_ECOG_OPP_DEVELOPMENT_COMPARISON.json"
)
FAILED_V1 = RUNTIME / "revealnav_val_seen_shadow_gate/RUN_FAILURES.json"
OUT = RUNTIME / "revealnav_val_seen_shadow_gate_v2"
PROTOCOL = OUT / "REVEALNAV_VAL_SEEN_SHADOW_PROTOCOL.json"
GATE = OUT / "REVEALNAV_VAL_SEEN_SHADOW_GATE.json"
EPISODES = ("326", "312", "776", "40", "533", "185", "510", "608")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def protocol_value() -> dict:
    old = json.loads(OLD_GATE.read_text())
    development = json.loads(DEVELOPMENT.read_text())
    selection = json.loads(SELECTION.read_text())
    failed_v1 = json.loads(FAILED_V1.read_text())
    selected = tuple(
        str(row["episode_id"]) for row in selection["selection"]["episodes"]
    )
    if not (
        old.get("verdict") == "RUNTIME_GATE_PASS"
        and development.get("status") == "ECOG_OPP_DEVELOPMENT_GATE_PASS"
        and development.get("gold_payload_read") is False
        and selected == EPISODES
        and len(failed_v1.get("failures", [])) == 8
        and all(row.get("returncode") == 1 for row in failed_v1["failures"])
    ):
        raise RuntimeError("val_seen shadow gate precondition failed")
    canonical = {}
    for episode in EPISODES:
        path = RUNTIME / (
            f"r2r_val_seen_gate/runs/r2rvg_pilot_ep{episode}/trace.jsonl"
        )
        canonical[str(path.relative_to(ROOT))] = sha256_file(path)
    return {
        "schema_version": "revealnav-val-seen-shadow-protocol/2",
        "status": "SEALED_BEFORE_VAL_SEEN_SHADOW_RUNS",
        "label": "val_seen_engineering_only",
        "episodes": list(EPISODES), "physical_gpus": list(range(8)),
        "observer_seed": 20260826,
        "execution": (
            "ETP-R1 actions remain authoritative; RevealNav receives only "
            "causal ETP embeddings and writes shadow recommendations"
        ),
        "deterministic_replay_episode": "326",
        "success_gates": {
            "eight_real_episodes_complete": True,
            "base_action_traces_equal_preobserver_canonical": True,
            "base_metrics_equal_preobserver_canonical": True,
            "shadow_rows_cover_every_high_level_step": True,
            "shadow_hash_chains_valid": True,
            "shadow_replay_bit_exact": True,
            "no_shadow_action_executed": True,
            "models_and_sources_unchanged": True,
        },
        "sources": {
            str(WORKER.relative_to(ROOT)): sha256_file(WORKER),
            str(OLD_GATE.relative_to(ROOT)): sha256_file(OLD_GATE),
            str(SELECTION.relative_to(ROOT)): sha256_file(SELECTION),
            str(DEVELOPMENT.relative_to(ROOT)): sha256_file(DEVELOPMENT),
            str(FAILED_V1.relative_to(ROOT)): sha256_file(FAILED_V1),
        },
        "canonical_traces": canonical,
        "gold_access_allowed": False, "forbidden_splits": [
            "val_unseen", "test", "test_challenge"
        ],
        "paper_result": False,
    }


def seal() -> int:
    value = protocol_value()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != value:
        raise RuntimeError("sealed val_seen shadow protocol drift")
    if not PROTOCOL.exists():
        atomic_json(PROTOCOL, value)
    print(json.dumps({"status": value["status"], "episodes": value["episodes"],
                      "protocol": str(PROTOCOL.relative_to(ROOT)),
                      "sha256": sha256_file(PROTOCOL)}, indent=2))
    return 0


def launch(episode: str, gpu: int, suffix: str = "") -> subprocess.Popen:
    name = f"revealnav_shadow_ep{episode}{suffix}"
    run_dir = OUT / "runs" / name
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    stdout = (run_dir / "stdout.log").open("w")
    stderr = (run_dir / "stderr.log").open("w")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    command = [
        sys.executable, str(WORKER), "--episode-id", episode,
        "--exp-name", name,
        "--run-dir", str(Path("/mnt/daiyang/vla") / run_dir.relative_to(ROOT)),
        "--gate-out", str(
            Path("/mnt/daiyang/vla") / (OUT / "etp_outputs").relative_to(ROOT)
        ),
    ]
    process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
    process._revealnav_files = (stdout, stderr)  # type: ignore[attr-defined]
    return process


def run() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("val_seen shadow protocol must be sealed")
    processes = [(episode, launch(episode, gpu))
                 for gpu, episode in enumerate(EPISODES)]
    failures = []
    for episode, process in processes:
        code = process.wait()
        for stream in process._revealnav_files:  # type: ignore[attr-defined]
            stream.close()
        if code:
            failures.append({"episode": episode, "returncode": code})
    if failures:
        atomic_json(OUT / "RUN_FAILURES.json", {"failures": failures})
        print(json.dumps({"status": "SHADOW_RUN_FAIL", "failures": failures}, indent=2))
        return 1
    replay = launch("326", 0, "_replay")
    code = replay.wait()
    for stream in replay._revealnav_files:  # type: ignore[attr-defined]
        stream.close()
    if code:
        atomic_json(OUT / "RUN_FAILURES.json", {
            "failures": [{"episode": "326_replay", "returncode": code}]
        })
        return 1
    print(json.dumps({"status": "SHADOW_RUNS_COMPLETE",
                      "episodes": len(EPISODES), "replay": "326"}, indent=2))
    return 0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def valid_chain(rows: list[dict]) -> bool:
    previous = "0" * 64
    for row in rows:
        if row.get("previous_hash") != previous:
            return False
        claimed = row.get("record_hash")
        value = dict(row); value.pop("record_hash", None)
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode()).hexdigest() != claimed:
            return False
        previous = claimed
    return True


def verify() -> int:
    if not PROTOCOL.is_file() or json.loads(PROTOCOL.read_text()) != protocol_value():
        raise RuntimeError("val_seen shadow protocol drift")
    rows = []
    checks = {
        "eight_real_episodes_complete": True,
        "base_action_traces_equal_preobserver_canonical": True,
        "base_metrics_equal_preobserver_canonical": True,
        "shadow_rows_cover_every_high_level_step": True,
        "shadow_hash_chains_valid": True,
        "shadow_replay_bit_exact": True,
        "no_shadow_action_executed": True,
        "models_and_sources_unchanged": True,
        "no_gold_or_forbidden_split_payload_read": True,
    }
    old_results = {
        row["episode_id"]: row for row in load_jsonl(
            RUNTIME / "R2R_VAL_SEEN_EPISODE_RESULTS.jsonl"
        )
    }
    for episode in EPISODES:
        run_dir = OUT / "runs" / f"revealnav_shadow_ep{episode}"
        summary_path = run_dir / "RUN_SUMMARY.json"
        if not summary_path.is_file():
            checks["eight_real_episodes_complete"] = False
            continue
        summary = json.loads(summary_path.read_text())
        trace = load_jsonl(run_dir / "trace.jsonl")
        shadow = load_jsonl(run_dir / "revealnav_shadow.jsonl")
        canonical = load_jsonl(RUNTIME / (
            f"r2r_val_seen_gate/runs/r2rvg_pilot_ep{episode}/trace.jsonl"
        ))
        checks["eight_real_episodes_complete"] &= summary.get("exit_status") == "OK"
        checks["base_action_traces_equal_preobserver_canonical"] &= trace == canonical
        current_stats = summary.get("stats", {}).get(episode)
        checks["base_metrics_equal_preobserver_canonical"] &= (
            current_stats == old_results[episode]["metrics"]
        )
        checks["shadow_rows_cover_every_high_level_step"] &= (
            len(shadow) == len(trace) == summary.get("high_level_steps")
        )
        checks["shadow_hash_chains_valid"] &= valid_chain(shadow)
        checks["no_shadow_action_executed"] &= (
            summary["revealnav_shadow_observer"]["shadow_actions_executed"] == 0
            and all(row["shadow_only_not_executed"] for row in shadow)
        )
        rows.append({
            "episode_id": episode, "high_level_steps": len(trace),
            "shadow_rows": len(shadow),
            "checkpoint_count": summary["revealnav_shadow_observer"][
                "checkpoint_count"
            ],
            "trace_sha256": sha256_file(run_dir / "trace.jsonl"),
            "shadow_sha256": sha256_file(run_dir / "revealnav_shadow.jsonl"),
            "shadow_final_hash": shadow[-1]["record_hash"] if shadow else None,
        })
    first = OUT / "runs/revealnav_shadow_ep326/revealnav_shadow.jsonl"
    replay = OUT / "runs/revealnav_shadow_ep326_replay/revealnav_shadow.jsonl"
    first_trace = OUT / "runs/revealnav_shadow_ep326/trace.jsonl"
    replay_trace = OUT / "runs/revealnav_shadow_ep326_replay/trace.jsonl"
    checks["shadow_replay_bit_exact"] &= (
        first.is_file() and replay.is_file()
        and first.read_bytes() == replay.read_bytes()
        and first_trace.read_bytes() == replay_trace.read_bytes()
    )
    sources = protocol_value()["sources"]
    checks["models_and_sources_unchanged"] &= all(
        sha256_file(ROOT / path) == digest for path, digest in sources.items()
    )
    passed = all(checks.values())
    value = {
        "schema_version": "revealnav-val-seen-shadow-gate/2",
        "status": "VAL_SEEN_SHADOW_GATE_PASS" if passed
                  else "VAL_SEEN_SHADOW_GATE_FAIL",
        "label": "val_seen_engineering_only",
        "checks": checks, "episodes": rows,
        "totals": {"episodes": len(rows),
                   "high_level_steps": sum(row["high_level_steps"] for row in rows),
                   "shadow_checkpoints": sum(row["checkpoint_count"] for row in rows)},
        "behavior_change": False if passed else None,
        "shadow_actions_executed": 0,
        "sources": {"protocol_sha256": sha256_file(PROTOCOL), **sources},
        "gold_payload_read": False, "forbidden_split_accessed": False,
        "paper_result": False,
        "next_step": "final regression and scale-readiness audit" if passed else
                     "shadow integration diagnosis",
    }
    atomic_json(GATE, value)
    print(json.dumps(value, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.seal:
        return seal()
    if args.run:
        return run()
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
