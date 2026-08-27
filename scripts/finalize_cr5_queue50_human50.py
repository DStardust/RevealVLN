#!/usr/bin/env python3
"""Close queue50 human review after adjudicating three challenged rejects."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
REVIEW = BASE / "human_review_fast"
CAUSAL = BASE / "causal_gate"
Q36 = BASE / "causal_gate_q36"
OUT = CAUSAL / "CR5_QUEUE50_HUMAN50_ACCEPTANCE.json"
LOG = CAUSAL / "CR5_QUEUE50_HUMAN50_ACCEPTANCE.log"
SOURCES = {
    ROOT / "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    ROOT / "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
    CAUSAL / "CR5_QUEUE50_CAUSAL_GATE_ACCEPTANCE.json":
        "0f45f370eb56d7be2f3db40bb6727bb417d75e23be7b8b74d64aeb323c31e6c1",
    REVIEW / "daiyang_queue50.jsonl":
        "6b8a390f4b9e6c060aa20c834fd5567edecb63f00ce0995c6a3853ba5db3e209",
    REVIEW / "daiyang_auto_reject16.jsonl":
        "fcf17ff60bd9e07fa4e66a83741ea47136b8725bde97603cda03aed76f34f5ff",
    REVIEW / "CR5_QUEUE50_AUTO_REJECT_HUMAN_ACCEPTANCE.json":
        "4fb4bc45a2fbdd65fa80922c18c18b8ceed16a910dab67191f50e58defdfebd2",
    REVIEW / "CR5_QUEUE50_HUMAN_CHALLENGE_GEOMETRY_DIAGNOSTIC.json":
        "5cab848045e19c81bd9285b773d8eb4768ecf187b4f979c0ce9500e8f5108c8e",
    REVIEW / "CR5_QUEUE50_HUMAN_CHALLENGE_ADJUDICATION.json":
        "c8ff14d96f0de6684d409bd4899695da7b741971a8f819848cf94ab8e0c04495",
    REVIEW / "CR5_QUEUE50_Q36_CORRECTED_GEOMETRY.json":
        "80ee7482df1cfd821fa1984c1e4cbf8d88d777ce8d1b750ca118712345b9fea3",
    REVIEW / "CR5_QUEUE50_Q36_CORRECTED_CONTROLLER.json":
        "e835f838e3111c497643d598db2f434d36bb76a01790b92a3cdc8ff0c2f878df",
    Q36 / "CR5_Q36_CAUSAL_CANDIDATE_ANALYSIS.json":
        "5e0538c465322eef7f8d6016aed55e339751372a409610aee77d7e1d2614ef09",
    Q36 / "CR5_Q36_CAUSAL_PREFIX_MEDIA_MANIFEST.json":
        "05ef6acfdfa17e51456d055711b8a65ebe6cc2ed828354d93679338bb4f5cf9e",
    Q36 / "CR5_Q36_CAUSAL_PREFIX_LANGUAGE_GATE.json":
        "99dd2228aba1ba2d189d756c8eb4f86bc77bceb753ea06972e8c8a497417ff17",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def atomic_text(path: Path, value: str):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(value)
    os.replace(temporary, path)


def command(arguments, env=None):
    result = subprocess.run(
        arguments, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    return {
        "command": arguments,
        "returncode": result.returncode,
        "output": result.stdout,
        "pass": result.returncode == 0,
    }


def main() -> int:
    failures = []
    source_manifest = []
    for path, expected in SOURCES.items():
        observed = sha256_file(path) if path.is_file() else None
        passed = (path.is_file() and not path.is_symlink()
                  and observed == expected)
        source_manifest.append({
            "path": str(path.relative_to(ROOT)),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "pass": passed,
        })
        if not passed:
            failures.append("source drift: " + str(path.relative_to(ROOT)))

    original = load(CAUSAL / "CR5_QUEUE50_CAUSAL_GATE_ACCEPTANCE.json")
    auto = load(REVIEW / "CR5_QUEUE50_AUTO_REJECT_HUMAN_ACCEPTANCE.json")
    adjudication = load(
        REVIEW / "CR5_QUEUE50_HUMAN_CHALLENGE_ADJUDICATION.json")
    q36_controller = load(
        REVIEW / "CR5_QUEUE50_Q36_CORRECTED_CONTROLLER.json")
    q36_analysis = load(Q36 / "CR5_Q36_CAUSAL_CANDIDATE_ANALYSIS.json")
    q36_language = load(Q36 / "CR5_Q36_CAUSAL_PREFIX_LANGUAGE_GATE.json")
    original_review_rows = [json.loads(line) for line in
                            (REVIEW / "daiyang_queue50.jsonl").read_text(
                                ).splitlines() if line.strip()]
    original_review_ids = {row["event_id"] for row in original_review_rows}
    original_accept_ids = {
        row["event_id"] for row in original_review_rows
        if row["final_label"] == "ACCEPT"
    }
    original_reject_ids = {
        row["event_id"] for row in original_review_rows
        if row["final_label"] == "REJECT"
    }
    disposition_ids = {
        row["event_id"] for row in original["event_dispositions"]
    }
    second_rows = [json.loads(line) for line in
                   (REVIEW / "daiyang_auto_reject16.jsonl").read_text(
                       ).splitlines() if line.strip()]
    second_ids = {row["event_id"] for row in second_rows}
    if (len(original_review_rows) != 34
            or len(original_review_ids) != 34
            or len(original_accept_ids) != 28
            or len(original_reject_ids) != 6
            or any(row.get("reviewer_type") != "HUMAN"
                   for row in original_review_rows)
            or disposition_ids != original_accept_ids):
        failures.append("original human review is not 34 unique (28+6) events")
    if (len(second_rows) != 16 or len(second_ids) != 16
            or any(row.get("reviewer_type") != "HUMAN"
                   for row in second_rows)
            or original_review_ids & second_ids
            or len(original_review_ids | second_ids) != 50):
        failures.append("human review partition is not 34+16 unique events")
    if original["counts"]["human_reviewed_full_boards"] != 34:
        failures.append("original 34-board review count drift")
    if auto.get("review_count") != 16 or auto.get("unique_event_count") != 16:
        failures.append("machine-reject human review count drift")

    decisions = {row["event_id"]: row["decision"]
                 for row in adjudication["decisions"]}
    if decisions != {
            "q17_ep34158_hv05":
                "ORIGINAL_REJECT_CONFIRMED_BY_3D_COUNTERFACTUAL",
            "q24_ep28644_hv04":
                "ORIGINAL_REJECT_CONFIRMED_BY_3D_COUNTERFACTUAL",
            "q36_ep1049_hv05":
                "CORRECTED_GEOMETRY_CONTROLLER_REQUIRED",
            }:
        failures.append("challenge adjudication drift")
    if (q36_controller.get("status_counts")
            != {"CONTROLLER_PASS_CAUSAL_GATE_REQUIRED": 1}
            or q36_analysis.get("status_counts")
            != {"FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED": 1}
            or q36_language.get("counts") != {
                "frontend_causal_ready": 1,
                "language_k3_pass": 0,
                "language_k3_fail": 1,
                "topology_only_frontend_fail": 0,
                "requests_made_or_reused": 4,
            }):
        failures.append("q36 corrected causal chain drift")

    original_eligible = original["eligible_event_ids"]
    if len(original_eligible) != 16 or len(set(original_eligible)) != 16:
        failures.append("original strict T_R eligible set drift")
    resolved_rejects = {
        row["event_id"]: (
            "HUMAN_CONFIRMED_MACHINE_REJECT"
            if row["final_label"] == "CONFIRM_REJECT"
            else "ORIGINAL_REJECT_CONFIRMED_BY_3D_COUNTERFACTUAL"
            if row["event_id"] in {"q17_ep34158_hv05", "q24_ep28644_hv04"}
            else "CORRECTED_THEN_EXCLUDED_LANGUAGE_K3_FAIL"
        )
        for row in second_rows
    }
    if (sum(value == "HUMAN_CONFIRMED_MACHINE_REJECT"
            for value in resolved_rejects.values()) != 13
            or sum(value == "ORIGINAL_REJECT_CONFIRMED_BY_3D_COUNTERFACTUAL"
                   for value in resolved_rejects.values()) != 2
            or resolved_rejects.get("q36_ep1049_hv05")
            != "CORRECTED_THEN_EXCLUDED_LANGUAGE_K3_FAIL"):
        failures.append("resolved machine-reject disposition drift")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    regression = {
        "new_scripts_compile": command([
            str(ROOT / ".envs/etpr1/bin/python"), "-m", "py_compile",
            str(ROOT / "scripts/analyze_cr5_queue50_human_challenges.py"),
            str(ROOT / "scripts/adjudicate_cr5_queue50_human_challenges.py"),
            str(ROOT / "scripts/run_cr5_queue50_q36_controller_gate.py"),
            str(ROOT / "scripts/cr5_q36_causal_frontend_worker.py"),
            str(ROOT / "scripts/analyze_cr5_q36_causal_candidate.py"),
            str(ROOT / "scripts/build_cr5_q36_causal_prefix_media.py"),
            str(ROOT / "scripts/run_cr5_q36_causal_prefix_language.py"),
            str(ROOT / "scripts/finalize_cr5_queue50_human50.py"),
        ]),
        "toporeveal_24": command([
            str(ROOT / ".envs/etpr1/bin/python"),
            str(ROOT / "tests/test_toporeveal.py"), "-v",
        ], environment),
        "uv_pip_check": command([
            str(ROOT / ".tools/uv/uv"), "pip", "check", "--python",
            str(ROOT / ".envs/etpr1/bin/python"),
        ]),
    }
    if not all(row["pass"] for row in regression.values()):
        failures.append("regression command failure")
    reserve_files = sorted((ROOT / ".disk_reserve").glob("reserve_10G_*.bin"))
    reserve_pass = (len(reserve_files) == 19 and all(
        path.is_file() and not path.is_symlink()
        and path.stat().st_size == 10_737_418_240 for path in reserve_files))
    if not reserve_pass:
        failures.append("reserve contract drift")
    part_files = [str(path.relative_to(ROOT)) for path in BASE.rglob("*.part")]
    if part_files:
        failures.append("queue50 contains .part files")
    stat = os.statvfs(ROOT)
    free_bytes = stat.f_bavail * stat.f_frsize
    if free_bytes < 8 * 1024 ** 3:
        failures.append("free space below 8 GiB floor")

    output = {
        "revision": "cr5-queue50-human50-acceptance/1",
        "verdict": (
            "HUMAN50_AND_STRICT_T_R_ENGINEERING_PASS_T_X_REQUIRED"
            if not failures else "HUMAN50_ACCEPTANCE_FAIL"
        ),
        "scope": (
            "RxR-CE-en train-only queue50 human and strict-causal T_R "
            "engineering evidence; not a training or Phase-0 GO claim"
        ),
        "source_manifest": source_manifest,
        "counts": {
            "frozen_queue_trajectories": 50,
            "human_reviewed_candidates": 50,
            "original_full_board_reviews": 34,
            "machine_reject_confirmation_reviews": 16,
            "human_confirmed_machine_rejects": 13,
            "human_challenges": 3,
            "challenge_original_reject_confirmed": 2,
            "challenge_corrected_geometry": 1,
            "challenge_corrected_controller_pass": 1,
            "challenge_corrected_language_k3_fail": 1,
            "eligible_strict_causal_T_R": len(original_eligible),
            "strict_T_R_yield_over_frozen_queue": len(original_eligible) / 50,
        },
        "eligible_event_ids": original_eligible,
        "machine_reject_dispositions": resolved_rejects,
        "human_protocol": {
            "frozen_50_item_human_protocol_satisfied": True,
            "all_candidate_dispositions_resolved": True,
            "original_human_files_rewritten": False,
        },
        "correctness_finding": {
            "finding": (
                "Target-direction mismatch previously suppressed executable "
                "alternative search. Human challenge q36 found a valid "
                "counterfactual; the versioned correction passed 3-D and "
                "controller gates but was safely rejected by causal language."
            ),
            "production_requirement": (
                "Always run target-route-authoritative alternative search; "
                "route semantic direction disagreement must enter an "
                "automatic re-centering/re-grounding lane, not a final reject."
            ),
        },
        "regression": regression,
        "integrity": {
            "reserve_file_count": len(reserve_files),
            "reserve_files_untouched_contract_pass": reserve_pass,
            "part_files": part_files,
            "free_bytes": free_bytes,
            "free_space_floor_bytes": 8 * 1024 ** 3,
        },
        "phase0_go_no_go": {
            "status": "NO_GO_UNIQUE_T_X_REQUIRED",
            "human50_gate_satisfied": True,
            "strict_T_R_count": len(original_eligible),
            "unique_reproducible_T_X_count": 0,
            "reason": (
                "The human queue and strict T_R engineering gates pass, but "
                "no eligible event yet has its required SHA-bound unique "
                "controller expiry T_X."
            ),
            "next_required_stage": (
                "Generate and validate unique reproducible T_X controller "
                "witnesses for the 16 eligible strict T_R events."
            ),
        },
        "training_authorized": False,
        "forbidden_split_accessed": False,
        "future_frames_used_for_online_label": 0,
        "panoramas_used_for_online_label": 0,
        "failures": failures,
    }
    atomic_text(OUT, json.dumps(
        output, indent=2, ensure_ascii=False) + "\n")
    atomic_text(LOG, "\n".join([
        "verdict=" + output["verdict"],
        "eligible=" + ",".join(original_eligible),
        "phase0_status=" + output["phase0_go_no_go"]["status"],
        "failures=" + json.dumps(failures, ensure_ascii=False),
        "acceptance_sha256=" + sha256_file(OUT),
    ]) + "\n")
    print(json.dumps({
        "verdict": output["verdict"],
        "counts": output["counts"],
        "phase0_status": output["phase0_go_no_go"]["status"],
        "failures": failures,
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
