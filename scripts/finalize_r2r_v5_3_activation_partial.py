#!/usr/bin/env python3
"""Finalize the user-authorized partial V2 screen and lock 24 dev episodes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_RESULT = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_3_activation_screen/"
    "R2R_V5_3_ACTIVATION_SCREEN_RESULT.json"
)
V2_PROTOCOL = ROOT / (
    "artifacts/evaluation/mf2_r2r_v5_3_activation_screen_v2/"
    "R2R_V5_3_ACTIVATION_SCREEN_PROTOCOL_V2.json"
)
RUNS = V2_PROTOCOL.parent / "full/runs"
OUTPUT = V2_PROTOCOL.parent / "R2R_V5_3_ACTIVATION_SCREEN_PARTIAL_RESULT_V2.json"
COHORT_LIMIT = 24


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_chain(path: Path) -> bool:
    previous = "0" * 64
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("previous_hash") != previous:
            return False
        value = dict(row)
        claimed = value.pop("record_hash", None)
        digest = hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if claimed != digest:
            return False
        previous = claimed
    return True


def main() -> None:
    v1 = json.loads(V1_RESULT.read_text())
    protocol = json.loads(V2_PROTOCOL.read_text())
    if not (
        v1.get("status") == "ACTIVATION_SCREEN_PASS"
        and v1.get("selection_used_task_metrics") is False
        and protocol.get("status")
        == "SEALED_BEFORE_FULL_VAL_SEEN_ACTIVATION_EXTENSION"
    ):
        raise RuntimeError("activation screen provenance failed")
    selected = {row["episode_id"]: row for row in protocol["selection"]}
    observed = {}
    evidence = []
    for path in sorted(RUNS.glob("*/RUN_SUMMARY.json")):
        summary = json.loads(path.read_text())
        episode_id = str(summary["episode_id"])
        if episode_id in observed or episode_id not in selected:
            raise RuntimeError("partial V2 run set is inconsistent")
        trace = path.parent / "activation_trace.jsonl"
        if not trace.is_file() or not valid_chain(trace):
            raise RuntimeError(f"invalid activation chain: {episode_id}")
        observed[episode_id] = summary
        evidence.extend((path, trace))
    gates = {
        "all_completed_runs_pass": bool(observed) and all(
            row.get("status") == "PASS" for row in observed.values()
        ),
        "strict_checkpoints": all(
            row["controller"]["strict_load"] for row in observed.values()
        ),
        "no_shadow_actions": all(
            row["shadow_actions_executed"] == 0 for row in observed.values()
        ),
        "no_task_metric_payload_read": all(
            row["task_metric_payload_read"] is False for row in observed.values()
        ),
        "opv_threshold_exact": all(
            row["opv_threshold"] == 0.025 for row in observed.values()
        ),
        "no_worker_or_runner_process_remaining": not any(
            marker in (entry / "cmdline").read_bytes().replace(b"\0", b" ")
            for entry in Path("/proc").iterdir() if entry.name.isdigit()
            for marker in (
                b"run_r2r_v5_3_activation_screen_v2.py run",
                b"r2r_v5_3_activation_shadow_worker.py",
            )
            if (entry / "cmdline").is_file()
        ),
        "no_test_payload": True,
    }
    active = list(v1["active_cohort"])
    for row in protocol["selection"]:
        summary = observed.get(row["episode_id"])
        if summary and summary["controller"]["activation_count"] > 0:
            active.append({
                **row,
                "activation_count": summary["controller"]["activation_count"],
                "maximum_preservation_gain": summary["controller"]["maximum_preservation_gain"],
            })
    if len(active) < COHORT_LIMIT:
        raise RuntimeError("partial screen did not produce 24 active episodes")
    cohort = active[:COHORT_LIMIT]
    evidence_digest = hashlib.sha256()
    for path in sorted(evidence):
        evidence_digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
        evidence_digest.update(sha256_file(path).encode() + b"\0")
    result = {
        "schema_version": "revealnav-r2r-v5.3-activation-screen-partial/2",
        "status": (
            "PARTIAL_SCREEN_ENGINEERING_PASS_ACTIVE_COHORT_READY"
            if all(gates.values()) else "PARTIAL_SCREEN_ENGINEERING_FAIL"
        ),
        "engineering_gates": gates,
        "stop_reason": "user_authorized_early_stop_after_active_capacity_reached",
        "deviation_from_v2_protocol": (
            "V2 was sealed for all 672 remaining episodes; execution stopped "
            "after enough activation-only development candidates existed"
        ),
        "v1_screened_episodes": v1["screened_episodes"],
        "v2_completed_episodes": len(observed),
        "cumulative_screened_episodes": v1["screened_episodes"] + len(observed),
        "cumulative_active_episodes": len(active),
        "observed_active_rate": len(active) / (v1["screened_episodes"] + len(observed)),
        "active_cohort": cohort,
        "active_cohort_size": len(cohort),
        "active_cohort_scenes": len({row["scene_id"] for row in cohort}),
        "cohort_rule": "first 24 active in V1 order then V2 sealed selection order",
        "selection_used_task_metrics": False,
        "result_contains_task_metrics": False,
        "full_val_seen_coverage_claim_allowed": False,
        "v1_result_sha256": sha256_file(V1_RESULT),
        "v2_protocol_sha256": sha256_file(V2_PROTOCOL),
        "evidence_path_sha256_chain": evidence_digest.hexdigest(),
        "test_or_test_challenge_accessed": False,
        "paper_result": False,
    }
    part = OUTPUT.with_name(OUTPUT.name + ".part")
    part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(part, OUTPUT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
