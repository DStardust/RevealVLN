#!/usr/bin/env python3
"""Finalize the queue50 strict-causal T_R engineering gate without claiming GO."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase0/phase0c_cr5_queue50"
HUMAN = BASE / "human_review_fast"
CAUSAL = BASE / "causal_gate"
OUT = CAUSAL / "CR5_QUEUE50_CAUSAL_GATE_ACCEPTANCE.json"
LOG = CAUSAL / "CR5_QUEUE50_CAUSAL_GATE_ACCEPTANCE.log"
SOURCES = {
    ROOT / "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    ROOT / "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
    HUMAN / "CR5_QUEUE50_FAST_REVIEW_MANIFEST.json":
        "6d75c96075d5746dc90a19c7d4a59941b17ab81e1d7a4cfa6480c607fb089017",
    HUMAN / "CR5_QUEUE50_AUTO_REJECTED.json":
        "14f549c8d0c73628335fa673b433593f7152fb6b6dd8a0abd074134b7c218403",
    HUMAN / "daiyang_queue50.jsonl":
        "6b8a390f4b9e6c060aa20c834fd5567edecb63f00ce0995c6a3853ba5db3e209",
    HUMAN / "CR5_QUEUE50_HUMAN_REVIEW_ACCEPTANCE.json":
        "27c2dab09f843f2d9d0bde9071c29395c7898551ee2555923006bbac64c1fe25",
    CAUSAL / "CR5_QUEUE50_CAUSAL_CANDIDATE_ANALYSIS.json":
        "6e85507666bd6a94746b9b9ecb4a8229fe3fb71184f8e43d49a0938044fdedae",
    CAUSAL / "CR5_QUEUE50_CAUSAL_PREFIX_MEDIA_MANIFEST.json":
        "471de5b3fb6b3ba3a2103e5b9c3e64419222de3b1654dea4e8eed6aa63f18f69",
    CAUSAL / "CR5_QUEUE50_CAUSAL_PREFIX_LANGUAGE_GATE.json":
        "eaf494b5348b0c6cdbb01a0199fe794d00cf99fb7d3b8c2059248a5a64081e23",
    CAUSAL / "CR5_QUEUE50_CAUSAL_NEGATIVE_CONTROLS_ACCEPTED.json":
        "d7b51bd7bf0f857fec907773008ae9f6048eb325d1aff2b8ce496dc8cd7409ce",
}
SCRIPT_SOURCES = {
    ROOT / "scripts/validate_cr5_queue50_human_review.py":
        "74f5440b5a82c02bdac28d795880cd939933421712b4d8b2391a5d482f5a32f2",
    ROOT / "scripts/cr5_queue50_causal_frontend_worker.py":
        "03f6d049a02aba1d6d4483f3254ab4748460dc2b96758630b66092fb9ed3feb8",
    ROOT / "scripts/analyze_cr5_causal_candidates.py":
        "e2d0533a73ef582933dd3e7f8f45f1cdf243e56f0c7c7c39265ce5ebb67fd4d6",
    ROOT / "scripts/analyze_cr5_queue50_causal_candidates.py":
        "c4002d6377c5b08f788101d1bfcb7c02f335c098ef148a72d95e812592c896d3",
    ROOT / "scripts/build_cr5_causal_prefix_media.py":
        "e903e8e1349d1c9511564dff79d3562b971f4bdea703e4cba52ca2909e6e64df",
    ROOT / "scripts/build_cr5_queue50_causal_prefix_media.py":
        "5a05bdd1650059196bce28cdf2704604e77eadf216187993d312950bebf87419",
    ROOT / "scripts/run_cr5_causal_prefix_language.py":
        "a1ee4c0c783b1521a9c69032eb1bd33ec7dae87cd04c080b866424086de7a097",
    ROOT / "scripts/run_cr5_queue50_causal_prefix_language.py":
        "35df83908ebb5f396222d2459bc6e266604cfc3daeae7224658370fd1e943086",
    ROOT / "scripts/run_cr5_causal_negative_controls.py":
        "da66e7fc92cd77647dc34f39f356cf1962734ee44d6da71fb9f06ec378061add",
    ROOT / "scripts/run_cr5_queue50_causal_negative_controls.py":
        "19194ef3d0d9c28ea22e6feee8ac0e91fcd5a4c1c6d1ca19ec27bd0843b588a8",
    ROOT / "scripts/retry_cr5_queue50_causal_controls.py":
        "cd72600936ddbdd08085ac789fb2feaee0899bd4821ecab8f98383c36b7b01e0",
    ROOT / "scripts/assemble_cr5_queue50_causal_controls.py":
        "b9136d6b2b9b2543afcb1ad1113ec2c5d69b18c8a03cd6373d793353514973fb",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def atomic_write(path: Path, data: bytes):
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


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


def validate_ref(failures, manifest, path_text, expected, kind):
    path = ROOT / path_text
    passed = (path.is_file() and not path.is_symlink()
              and ROOT.resolve() in path.resolve().parents
              and sha256_file(path) == expected)
    manifest.append({
        "kind": kind,
        "path": path_text,
        "expected_sha256": expected,
        "observed_sha256": sha256_file(path) if path.is_file() else None,
        "pass": passed,
    })
    if not passed:
        failures.append(kind + " drift: " + path_text)


def main() -> int:
    failures = []
    source_manifest = []
    for path, expected in {**SOURCES, **SCRIPT_SOURCES}.items():
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

    review_manifest = load(HUMAN / "CR5_QUEUE50_FAST_REVIEW_MANIFEST.json")
    auto_rejected = load(HUMAN / "CR5_QUEUE50_AUTO_REJECTED.json")
    human_acceptance = load(
        HUMAN / "CR5_QUEUE50_HUMAN_REVIEW_ACCEPTANCE.json")
    reviews = [json.loads(line) for line in
               (HUMAN / "daiyang_queue50.jsonl").read_text().splitlines()
               if line.strip()]
    review_by_event = {row["event_id"]: row for row in reviews}
    human_accepted = {row["event_id"] for row in reviews
                      if row["final_label"] == "ACCEPT"}
    if (review_manifest.get("screened_trajectory_count") != 50
            or review_manifest.get("full_review_board_count") != 34
            or auto_rejected.get("rejected_count") != 16
            or len(reviews) != 34 or len(review_by_event) != 34
            or len(human_accepted) != 28
            or any(row.get("reviewer_type") != "HUMAN" for row in reviews)
            or human_acceptance.get("status") != "PASS"
            or set(human_acceptance.get("accepted_event_ids", []))
            != human_accepted):
        failures.append("human-review contract drift")
    automatic_event_ids = {row["event_id"] for row in
                           auto_rejected.get("events", [])}
    if (len(automatic_event_ids) != 16
            or automatic_event_ids & set(review_by_event)):
        failures.append("automatic/human queue partition drift")

    analysis = load(CAUSAL / "CR5_QUEUE50_CAUSAL_CANDIDATE_ANALYSIS.json")
    media = load(CAUSAL / "CR5_QUEUE50_CAUSAL_PREFIX_MEDIA_MANIFEST.json")
    language = load(CAUSAL / "CR5_QUEUE50_CAUSAL_PREFIX_LANGUAGE_GATE.json")
    controls = load(
        CAUSAL / "CR5_QUEUE50_CAUSAL_NEGATIVE_CONTROLS_ACCEPTED.json")
    analysis_by_event = {row["event_id"]: row for row in analysis["events"]}
    language_by_event = {row["event_id"]: row for row in language["events"]}
    topology_by_event = {row["event_id"]: row
                         for row in language["topology_only_events"]}
    control_by_event = {row["event_id"]: row for row in controls["events"]}
    if set(analysis_by_event) != human_accepted:
        failures.append("causal analysis does not cover human ACCEPT exactly")

    dispositions = []
    for event_id in sorted(human_accepted):
        event = analysis_by_event[event_id]
        if event["status"] == "TOPOLOGY_ONLY_FRONTEND_K3_FAIL":
            disposition = "EXCLUDE_FRONTEND_K3_FAIL"
            if event_id not in topology_by_event:
                failures.append("missing topology evidence: " + event_id)
        else:
            language_event = language_by_event.get(event_id)
            if language_event is None:
                disposition = "EVIDENCE_INCOMPLETE"
                failures.append("missing language evidence: " + event_id)
            elif language_event["status"] == "CAUSAL_LANGUAGE_K3_FAIL":
                disposition = "EXCLUDE_LANGUAGE_K3_FAIL"
            else:
                control_event = control_by_event.get(event_id)
                if control_event is None:
                    disposition = "EVIDENCE_INCOMPLETE"
                    failures.append("missing controls: " + event_id)
                elif control_event["status"] == "CAUSAL_CONTROLS_PASS":
                    disposition = "ELIGIBLE_STRICT_CAUSAL_T_R"
                else:
                    disposition = "EXCLUDE_NEGATIVE_CONTROL_FAIL"
        dispositions.append({
            "event_id": event_id,
            "episode_id": event["episode_id"],
            "disposition": disposition,
            "training_label": False,
        })

    expected_counts = {
        "ELIGIBLE_STRICT_CAUSAL_T_R": 16,
        "EXCLUDE_FRONTEND_K3_FAIL": 10,
        "EXCLUDE_LANGUAGE_K3_FAIL": 1,
        "EXCLUDE_NEGATIVE_CONTROL_FAIL": 1,
    }
    disposition_counts = {key: sum(
        row["disposition"] == key for row in dispositions)
        for key in expected_counts}
    if disposition_counts != expected_counts:
        failures.append("strict-causal disposition count drift")
    eligible = [row["event_id"] for row in dispositions
                if row["disposition"] == "ELIGIBLE_STRICT_CAUSAL_T_R"]
    if len({analysis_by_event[row]["episode_id"] for row in eligible}) != 16:
        failures.append("eligible events are not episode-unique")
    if (analysis.get("future_frames_used") != 0
            or language.get("future_frames_used") != 0
            or language.get("panoramas_used") != 0
            or controls.get("future_frames_used") != 0
            or controls.get("panoramas_used") != 0
            or any(value.get("training_authorized") is not False for value in
                   (human_acceptance, analysis, media, language, controls))):
        failures.append("causal observation or training boundary drift")
    if language.get("counts") != {
            "frontend_causal_ready": 18,
            "language_k3_pass": 17,
            "language_k3_fail": 1,
            "topology_only_frontend_fail": 10,
            "requests_made_or_reused": 69,
            }:
        failures.append("language counts drift")
    if controls.get("counts") != {
            "baseline_k3_candidates": 17,
            "causal_controls_pass": 16,
            "causal_controls_fail": 1,
            "format_invalid_first_responses": 3,
            "provider_retry_requests": 4,
            "accepted_valid_retries": 3,
            "semantic_control_failures": 1,
            }:
        failures.append("negative-control counts drift")

    evidence_manifest = []
    sources = analysis["sources"]
    for name in ("geometry", "multiview_inputs", "controller", "human_review"):
        ref = sources[name]
        validate_ref(failures, evidence_manifest, ref["path"], ref["sha256"],
                     "analysis_source")
    for ref in sources["frontend_shards"]:
        validate_ref(failures, evidence_manifest, ref["path"], ref["sha256"],
                     "frontend_shard")
    for event in language["events"]:
        for ref in event["tested_prefixes"]:
            validate_ref(failures, evidence_manifest, ref["path"],
                         ref["sha256"], "language_response")
    for event in controls["events"]:
        for control in event["mllm_controls"].values():
            for ref in control["prefix_results"]:
                validate_ref(failures, evidence_manifest, ref["path"],
                             ref["sha256"], "control_response")
    for ref in media["media_manifest"]:
        validate_ref(failures, evidence_manifest, ref["path"],
                     ref["sha256"], "causal_media")
        path = ROOT / ref["path"]
        if path.is_file() and path.stat().st_size != ref["bytes"]:
            failures.append("causal media size drift: " + ref["path"])
    for ref in controls["control_media_manifest"]:
        validate_ref(failures, evidence_manifest, ref["path"],
                     ref["sha256"], "control_media")
        path = ROOT / ref["path"]
        if path.is_file() and path.stat().st_size != ref["bytes"]:
            failures.append("control media size drift: " + ref["path"])

    secret = ROOT / ".secret/qwen_api_key"
    secret_mode = secret.stat().st_mode & 0o777 if secret.is_file() else None
    leak_paths = []
    key_pattern = re.compile(rb"sk-[A-Za-z0-9]{20,}")
    scan_paths = (list(CAUSAL.rglob("*.json"))
                  + list(CAUSAL.rglob("*.md"))
                  + list((ROOT / "scripts").glob("*cr5*causal*.py")))
    for path in scan_paths:
        if key_pattern.search(path.read_bytes()):
            leak_paths.append(str(path.relative_to(ROOT)))
    if (not secret.is_file() or secret.is_symlink()
            or secret_mode != 0o600 or leak_paths):
        failures.append("secret permission or leakage failure")

    reserve_files = sorted((ROOT / ".disk_reserve").glob("reserve_10G_*.bin"))
    reserve_pass = (len(reserve_files) == 19 and all(
        path.is_file() and not path.is_symlink()
        and path.stat().st_size == 10_737_418_240 for path in reserve_files))
    if not reserve_pass:
        failures.append("reserve contract drift")
    part_files = [str(path.relative_to(ROOT)) for path in CAUSAL.rglob("*.part")]
    if part_files:
        failures.append("causal gate contains .part files")
    stat = os.statvfs(ROOT)
    free_bytes = stat.f_bavail * stat.f_frsize
    if free_bytes < 8 * 1024 ** 3:
        failures.append("free space below 8 GiB floor")

    test_environment = os.environ.copy()
    test_environment["PYTHONPATH"] = str(ROOT)
    regression = {
        "new_scripts_compile": command([
            str(ROOT / ".envs/etpr1/bin/python"), "-m", "py_compile",
            str(ROOT / "scripts/retry_cr5_queue50_causal_controls.py"),
            str(ROOT / "scripts/assemble_cr5_queue50_causal_controls.py"),
            str(ROOT / "scripts/finalize_cr5_queue50_causal_gate.py"),
        ]),
        "toporeveal_24": command([
            str(ROOT / ".envs/etpr1/bin/python"),
            str(ROOT / "tests/test_toporeveal.py"), "-v",
        ], test_environment),
        "uv_pip_check": command([
            str(ROOT / ".tools/uv/uv"), "pip", "check", "--python",
            str(ROOT / ".envs/etpr1/bin/python"),
        ]),
    }
    if not all(row["pass"] for row in regression.values()):
        failures.append("regression command failure")

    formal_human50 = len(reviews) == 50
    formal_tx = False
    output = {
        "revision": "cr5-queue50-strict-causal-tr-acceptance/1",
        "verdict": (
            "QUEUE50_T_R_ENGINEERING_PASS_FORMAL_HUMAN50_AND_T_X_REQUIRED"
            if not failures else "QUEUE50_T_R_ENGINEERING_FAIL"
        ),
        "scope": (
            "RxR-CE-en train-only strict-causal T_R engineering evidence; "
            "not a benchmark, training, performance, event-projection, or "
            "Phase-0 GO claim"
        ),
        "source_manifest": source_manifest,
        "counts": {
            "frozen_queue_trajectories": 50,
            "machine_rejected_without_human_label": 16,
            "human_reviewed_full_boards": len(reviews),
            "human_accepted_branch_events": len(human_accepted),
            **disposition_counts,
            "mllm_prefix_requests": 69,
            "negative_control_initial_requests": 102,
            "negative_control_retry_requests": 4,
        },
        "eligible_event_ids": eligible,
        "event_dispositions": dispositions,
        "evidence_manifest": evidence_manifest,
        "integrity": {
            "declared_evidence_file_count": len(evidence_manifest),
            "causal_media_count": len(media["media_manifest"]),
            "control_media_count": len(controls["control_media_manifest"]),
            "secret_file_mode_octal": (oct(secret_mode)
                                       if secret_mode is not None else None),
            "secret_leak_paths": leak_paths,
            "reserve_file_count": len(reserve_files),
            "reserve_files_untouched_contract_pass": reserve_pass,
            "part_files": part_files,
            "free_bytes": free_bytes,
            "free_space_floor_bytes": 8 * 1024 ** 3,
        },
        "regression": regression,
        "phase0_go_no_go": {
            "status": "NO_GO_FORMAL_HUMAN50_AND_UNIQUE_T_X_REQUIRED",
            "frozen_50_item_human_protocol_satisfied": formal_human50,
            "unique_reproducible_expiry_attached": formal_tx,
            "reason": [
                "The uploaded file contains 34 human-reviewed boards; the "
                "other 16 queue entries remain machine rejections, not human "
                "labels, so the frozen 50-reviewed-candidate gate is unmet.",
                "The 16 strict-causal events validate T_R only; no event yet "
                "has the required SHA-bound unique reproducible T_X witness.",
            ],
            "next_required_stage": (
                "Human-confirm the 16 machine-rejected queue entries, then "
                "derive and validate a unique controller-bound T_X for every "
                "eligible strict-causal T_R event before any training label."
            ),
        },
        "training_authorized": False,
        "forbidden_split_accessed": False,
        "future_frames_used_for_online_label": 0,
        "panoramas_used_for_online_label": 0,
        "failures": failures,
    }
    atomic_write(OUT, (json.dumps(output, indent=2, ensure_ascii=False)
                       + "\n").encode())
    atomic_write(LOG, ("\n".join([
        "verdict=" + output["verdict"],
        "eligible=" + ",".join(eligible),
        "formal_phase0_status=" + output["phase0_go_no_go"]["status"],
        "failures=" + json.dumps(failures, ensure_ascii=False),
        "acceptance_sha256=" + sha256_file(OUT),
    ]) + "\n").encode())
    print(json.dumps({
        "verdict": output["verdict"],
        "counts": output["counts"],
        "formal_phase0_status": output["phase0_go_no_go"]["status"],
        "failures": failures,
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
