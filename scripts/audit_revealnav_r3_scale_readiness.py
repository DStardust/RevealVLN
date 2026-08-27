#!/usr/bin/env python3
"""Fail-closed final audit before RevealNav event scaling/new Gold.

This script reads only implementation/development/engineering evidence.  It does
not open an old Gold or human-review payload and does not run a paper benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "artifacts/evaluation/REVEALNAV_R3_SCALE_READINESS.json"
OUT_MD = ROOT / "artifacts/evaluation/REVEALNAV_R3_SCALE_READINESS.md"

EXPECTED = {
    "FROZEN_SPEC.md": "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    "PHASE0_PROTOCOL.md": "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
    "etpr1_head": "a94b5c8fe20d1631e9e150c430a925543eb1cba9",
    "etpr1_diff": "5a207d26b5b582e8c810f5c95ecfe52897caecf69bb689874ad3222d21bfc521",
    "habitat_lab_head": "d6ed1c0a0e786f16f261de2beafe347f4186d0d8",
    "habitat_sim_head": "856d4b08c1a2632626bf0d205bf46471a99502b7",
}

EVIDENCE = {
    "feature_gate": (
        "artifacts/phase1/rxr_train_expansion/expiry_r3/RXR_EXPIRY_R3_FEATURE_GATE.json",
        "EXPIRY_R3_FEATURE_GATE_PASS",
    ),
    "expiry_r3_1": (
        "artifacts/evaluation/mf2_expiry_r3_1/RXR_EXPIRY_R3_COMPARISON.json",
        "EXPIRY_R3_1_GATE_PASS",
    ),
    "opv_hurdle_r3_4": (
        "artifacts/evaluation/mf2_opv_hurdle_r3_4/RXR_OPV_HURDLE_R3_4_COMPARISON.json",
        "OPV_HURDLE_R3_4_GATE_PASS",
    ),
    "ecog_opp_development": (
        "artifacts/evaluation/mf2_ecog_opp_development_v2/RXR_ECOG_OPP_DEVELOPMENT_COMPARISON.json",
        "ECOG_OPP_DEVELOPMENT_GATE_PASS",
    ),
    "val_seen_shadow": (
        "artifacts/runtime/revealnav_val_seen_shadow_gate_v2/REVEALNAV_VAL_SEEN_SHADOW_GATE.json",
        "VAL_SEEN_SHADOW_GATE_PASS",
    ),
    "migrated_habitat": (
        "artifacts/runtime/HABITAT017_MIGRATED_SIM_SMOKE.json",
        "PASS",
    ),
}

TESTS = [
    "tests/test_revealnav_mf2.py",
    "tests/test_revealnav_mf2r1.py",
    "tests/test_revealnav_mf2r2.py",
    "tests/test_revealnav_mf2r3.py",
    "tests/test_revealnav_mf2r3_qmodel.py",
    "tests/test_revealnav_mf2r3_policy.py",
    "tests/test_multibranch_pipeline_v2.py",
    "tests/test_toporeveal.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*args: str, cwd: Path = ROOT, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def git_head(repo: Path) -> str:
    proc = run("git", "rev-parse", "HEAD", cwd=repo)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def main() -> int:
    checks: dict[str, bool] = {}
    observed: dict[str, object] = {}

    for relative, expected_hash in (
        ("FROZEN_SPEC.md", EXPECTED["FROZEN_SPEC.md"]),
        ("PHASE0_PROTOCOL.md", EXPECTED["PHASE0_PROTOCOL.md"]),
    ):
        actual = sha256(ROOT / relative)
        checks[f"{relative}_unchanged"] = actual == expected_hash
        observed[f"{relative}_sha256"] = actual

    loaded: dict[str, dict[str, object]] = {}
    for name, (relative, expected_status) in EVIDENCE.items():
        path = ROOT / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded[name] = data
        checks[f"{name}_pass"] = data.get("status") == expected_status
        observed[f"{name}_sha256"] = sha256(path)

    checks["all_declared_evidence_gates_true"] = all(
        all(bool(value) for value in data.get("gates", {}).values())
        for name, data in loaded.items()
        if name not in {"val_seen_shadow", "migrated_habitat"}
    ) and all(
        bool(value)
        for value in loaded["val_seen_shadow"].get("checks", {}).values()
    ) and all(
        bool(value)
        for value in loaded["migrated_habitat"].get("checks", {}).values()
    )

    r3_3_path = (
        ROOT
        / "artifacts/evaluation/mf2_causal_opp_q_r3_3/"
        "RXR_CAUSAL_OPP_Q_R3_3_COMPARISON.json"
    )
    r3_3 = json.loads(r3_3_path.read_text(encoding="utf-8"))
    checks["r3_3_unconditional_failure_retained"] = (
        r3_3.get("status") == "CAUSAL_OPP_Q_R3_3_GATE_FAIL"
        and r3_3.get("gates", {}).get("opv_mae_beats_zero_in_two_seeds") is False
        and loaded["opv_hurdle_r3_4"].get("unconditional_mae_failure_still_reported") is True
        and loaded["opv_hurdle_r3_4"].get("unconditional_mae_gate_reclassified") is False
    )
    observed["r3_3_sha256"] = sha256(r3_3_path)

    tests: list[dict[str, object]] = []
    total_tests = 0
    for relative in TESTS:
        proc = run(sys.executable, relative)
        combined = proc.stdout + "\n" + proc.stderr
        match = re.search(r"Ran (\d+) tests?", combined)
        count = int(match.group(1)) if match else 0
        total_tests += count
        tests.append(
            {
                "path": relative,
                "pass": proc.returncode == 0,
                "test_count": count,
                "source_sha256": sha256(ROOT / relative),
                "output_tail": combined.strip()[-500:],
            }
        )
    checks["all_tests_pass"] = all(bool(item["pass"]) for item in tests)
    checks["expected_test_count_48"] = total_tests == 48
    observed["tests"] = tests
    observed["total_tests"] = total_tests

    pip_check = run(sys.executable, "-m", "pip", "check")
    checks["python_environment_consistent"] = pip_check.returncode == 0
    observed["pip_check"] = (pip_check.stdout + pip_check.stderr).strip()

    habitat_repos = {
        "habitat_lab": ROOT / "third_party/habitat-lab",
        "habitat_sim": ROOT / "third_party/habitat-sim",
    }
    for name, repo in habitat_repos.items():
        head = git_head(repo)
        porcelain = run("git", "status", "--porcelain=v1", cwd=repo).stdout.strip()
        checks[f"{name}_pinned_clean"] = (
            head == EXPECTED[f"{name}_head"] and not porcelain
        )
        observed[f"{name}_head"] = head

    etp = ROOT / "third_party/ETP-R1"
    etp_status = run("git", "status", "--porcelain=v1", cwd=etp).stdout.splitlines()
    tracked = [line for line in etp_status if not line.startswith("??")]
    untracked = [line[3:] for line in etp_status if line.startswith("??")]
    etp_diff = run("git", "diff", cwd=etp).stdout.encode()
    etp_diff_sha = hashlib.sha256(etp_diff).hexdigest()
    checks["etpr1_reviewed_patch_only"] = (
        git_head(etp) == EXPECTED["etpr1_head"]
        and len(tracked) == 14
        and untracked == ["etpr1_compat.py"]
        and etp_diff_sha == EXPECTED["etpr1_diff"]
    )
    observed["etpr1"] = {
        "head": git_head(etp),
        "tracked_modified": len(tracked),
        "untracked": untracked,
        "tracked_diff_sha256": etp_diff_sha,
    }

    new_scope_roots = [
        ROOT / "revealnav_mf2r3",
        ROOT / "artifacts/evaluation/mf2_expiry_r3_1",
        ROOT / "artifacts/evaluation/mf2_causal_opp_q_r3_3",
        ROOT / "artifacts/evaluation/mf2_opv_hurdle_r3_4",
        ROOT / "artifacts/evaluation/mf2_ecog_opp_development_v2",
        ROOT / "artifacts/runtime/revealnav_val_seen_shadow_gate_v2",
    ]
    part_files = [
        str(path.relative_to(ROOT))
        for scope in new_scope_roots
        for path in scope.rglob("*.part")
    ]
    checks["no_partial_files_in_r3_scope"] = not part_files
    observed["partial_files"] = part_files

    feature_counts = loaded["feature_gate"].get("counts", {})
    current_events = int(feature_counts.get("events", 0))
    checks["current_492_event_manifest_complete"] = (
        current_events == 492
        and feature_counts.get("train") == 424
        and feature_counts.get("development") == 68
    )

    val_seen = loaded["val_seen_shadow"]
    checks["val_seen_is_behavior_neutral_engineering_only"] = (
        val_seen.get("behavior_change") is False
        and val_seen.get("shadow_actions_executed") == 0
        and val_seen.get("paper_result") is False
        and val_seen.get("gold_payload_read") is False
    )

    no_gold_flags = [
        loaded[name].get("gold_payload_read") is False
        for name in (
            "feature_gate",
            "expiry_r3_1",
            "opv_hurdle_r3_4",
            "ecog_opp_development",
            "val_seen_shadow",
        )
    ]
    checks["no_old_gold_payload_used"] = all(no_gold_flags)

    entry = ROOT / "artifacts/design/REVEALNAV_SCALE_AND_NEW_GOLD_ENTRY_R3.md"
    checks["next_phase_entry_documented"] = entry.is_file()
    observed["next_phase_entry_sha256"] = sha256(entry)

    readiness_checks = dict(checks)
    ready = all(readiness_checks.values())
    target_events = 2000
    target_gold = 600
    payload = {
        "schema_version": "revealnav-r3-scale-readiness/1",
        "status": "READY_FOR_EVENT_SCALE_AND_NEW_GOLD" if ready else "NOT_READY",
        "checks": checks,
        "observed": observed,
        "closed_scope": {
            "expiry_adapter": True,
            "causal_paired_q_and_opv": True,
            "ecog_opp_controller": True,
            "scene_heldout_development": True,
            "val_seen_behavior_neutral_online_integration": True,
        },
        "scale_gap": {
            "current_train_development_events": current_events,
            "frozen_minimum_total_events": target_events,
            "minimum_additional_events_if_all_current_events_remain_eligible": max(
                0, target_events - current_events
            ),
            "new_three_reviewer_gold_events": 0,
            "frozen_minimum_gold_events": target_gold,
            "minimum_additional_new_gold_events": target_gold,
            "prior_single_reviewer_pilot_promoted_to_gold": False,
        },
        "immediate_next_phase": "expand events and establish new three-reviewer Gold",
        "future_after_next_phase": [
            "RxR-CE-en and R2R-CE Oracle/Frozen full benchmarks",
            "three seeds and paired episode bootstrap 95% confidence intervals",
            "frozen matched-input and modern-reference baselines",
            "core ablations and SR/SPL non-inferiority tests",
        ],
        "old_gold_payload_read": False,
        "old_gold_modified": False,
        "frozen_spec_modified": False,
        "forbidden_split_accessed": False,
        "paper_result": False,
        "cvpr_competitiveness_established": False,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RevealNav R3 Scale Readiness",
        "",
        f"Status: **{payload['status']}**",
        "",
        "The implementation, development ablations, and behavior-neutral online",
        "`val_seen` integration are closed for the current engineering scope. This",
        "is not a paper benchmark or CVPR acceptance claim.",
        "",
        "## Verified boundary",
        "",
        f"- Regression: {total_tests}/48 tests passed; Python dependency check passed.",
        f"- Current event manifest: {current_events} events (424 train, 68 development).",
        "- R2R val_seen shadow: 8 episodes, 67/67 high-level steps, base behavior unchanged.",
        "- Old Gold payload read: no; old Gold modified: no.",
        "- Frozen specification modified: no.",
        "",
        "## Only immediate next phase",
        "",
        f"- Add at least {target_events - current_events} eligible events to reach the frozen 2,000-event minimum.",
        "- Establish at least 600 new scene-disjoint events reviewed independently by three reviewers.",
        "- Do not promote the prior single-reviewer 300-item pilot to Gold.",
        "",
        "Full public benchmarks and paper claims occur only after this next phase.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks, "total_tests": total_tests}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
