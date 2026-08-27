#!/usr/bin/env python3
"""Validate and freeze the train-only CR5 strict causal pilot evidence."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
GATE = ROOT / "artifacts/phase0/phase0c_cr5_causal_gate"
OUT = GATE / "CR5_CAUSAL_PILOT_ACCEPTANCE.json"
LOG = GATE / "CR5_CAUSAL_PILOT_ACCEPTANCE.log"
SOURCES = {
    ROOT / "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    ROOT / "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
    ROOT / (
        "artifacts/phase0/phase0c_cr5_human_review_v1/reviews/daiyang.jsonl"
    ): "88eb9934cb8bc0abad3400f295e0bd1527b5d08d11189d0d4f055f61df14f1cb",
    GATE / "CR5_CAUSAL_CANDIDATE_ANALYSIS.json":
        "df4a5cd387b721b4b16a8285e376fa387458f6c1d4028505f47ded9cf9fed5c1",
    GATE / "CR5_CAUSAL_PREFIX_MEDIA_MANIFEST.json":
        "08ef7c784165d82dff3c42a6b30b3bea5dfbe05c7894b708aa559005bbdc52c6",
    GATE / "CR5_CAUSAL_PREFIX_LANGUAGE_GATE.json":
        "9fcdce2af6268e19c62b55f8a2d55639a1832a929371c3ddc32e1e3d1d4b63bc",
    GATE / "CR5_CAUSAL_NEGATIVE_CONTROLS.json":
        "82623cd4005ac65996203de4f51afab55d369088caee2c750db9d6a1a325002b",
}
SCRIPT_SOURCES = {
    ROOT / "scripts/cr5_causal_frontend_worker.py":
        "acb0cedf9170087ab3432b132b7999b76ae10d4a827546a86f58a79c51dc8d42",
    ROOT / "scripts/analyze_cr5_causal_candidates.py":
        "5e03efc52de70c916157a6b8fb009000c408af6558df8f41c010f3b63d342ebf",
    ROOT / "scripts/build_cr5_causal_prefix_media.py":
        "e59408eb9f18bc3377f6fa315064243ba82409c99924383472fb4b6646a36798",
    ROOT / "scripts/run_cr5_causal_prefix_language.py":
        "f9a0df860de1fb54b86c11d30cf936d047ab27f439f67461f515bbe3a80f2482",
    ROOT / "scripts/run_cr5_causal_negative_controls.py":
        "a4ce72f8ba724888dee3b71253a6a31069d5c1629e0337b1f2a216a126af7077",
}
REPO_CONTRACT = {
    ROOT / "third_party/habitat-sim":
        ("856d4b08c1a2632626bf0d205bf46471a99502b7", "clean"),
    ROOT / "third_party/habitat-lab":
        ("d6ed1c0a0e786f16f261de2beafe347f4186d0d8", "clean"),
    ROOT / "third_party/ETP-R1":
        ("a94b5c8fe20d1631e9e150c430a925543eb1cba9", "compat_dirty"),
}


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def command(arguments, env=None):
    result = subprocess.run(
        arguments, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {
        "command": arguments,
        "returncode": result.returncode,
        "output": result.stdout,
        "pass": result.returncode == 0,
    }


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
            failures.append("source drift: " + str(path))

    analysis = load(GATE / "CR5_CAUSAL_CANDIDATE_ANALYSIS.json")
    language = load(GATE / "CR5_CAUSAL_PREFIX_LANGUAGE_GATE.json")
    controls = load(GATE / "CR5_CAUSAL_NEGATIVE_CONTROLS.json")
    media = load(GATE / "CR5_CAUSAL_PREFIX_MEDIA_MANIFEST.json")
    reviews = [json.loads(line) for line in
               (ROOT / (
                   "artifacts/phase0/phase0c_cr5_human_review_v1/reviews/"
                   "daiyang.jsonl"
               )).read_text().splitlines() if line.strip()]
    review_by_event = {row["event_id"]: row for row in reviews}
    if len(reviews) != 10 or len(review_by_event) != 10:
        failures.append("human review is not exactly ten unique events")
    accepted = {row["event_id"] for row in reviews
                if row["final_label"] == "ACCEPT"}
    if len(accepted) != 9 or any(
            row["reviewer_type"] != "HUMAN" for row in reviews):
        failures.append("human review acceptance contract drift")

    analysis_by_event = {row["event_id"]: row
                         for row in analysis["events"]}
    language_by_event = {row["event_id"]: row
                         for row in language["events"]}
    topology_by_event = {row["event_id"]: row
                         for row in language["topology_only_events"]}
    control_by_event = {row["event_id"]: row
                        for row in controls["events"]}
    if set(analysis_by_event) != accepted:
        failures.append("causal analysis set differs from human ACCEPT set")

    dispositions = []
    for event_id in sorted(accepted):
        row = analysis_by_event[event_id]
        if row["status"] == "TOPOLOGY_ONLY_FRONTEND_K3_FAIL":
            status = "EXCLUDE_FRONTEND_K3_FAIL"
            if event_id not in topology_by_event:
                failures.append("missing topology-only evidence: " + event_id)
        else:
            language_row = language_by_event.get(event_id)
            if language_row is None:
                failures.append("missing language evidence: " + event_id)
                status = "EVIDENCE_INCOMPLETE"
            elif language_row["status"] == "CAUSAL_LANGUAGE_K3_FAIL":
                status = "EXCLUDE_LANGUAGE_K3_FAIL"
            else:
                control_row = control_by_event.get(event_id)
                if control_row is None:
                    failures.append("missing negative controls: " + event_id)
                    status = "EVIDENCE_INCOMPLETE"
                elif control_row["status"] == "CAUSAL_CONTROLS_PASS":
                    status = "ELIGIBLE_STRICT_CAUSAL_REVEAL_PILOT"
                else:
                    status = "EXCLUDE_NEGATIVE_CONTROL_FAIL"
        dispositions.append({
            "event_id": event_id,
            "episode_id": row["episode_id"],
            "disposition": status,
            "training_label": False,
        })

    expected_dispositions = {
        "ELIGIBLE_STRICT_CAUSAL_REVEAL_PILOT": 5,
        "EXCLUDE_FRONTEND_K3_FAIL": 2,
        "EXCLUDE_LANGUAGE_K3_FAIL": 1,
        "EXCLUDE_NEGATIVE_CONTROL_FAIL": 1,
    }
    observed_dispositions = {
        key: sum(row["disposition"] == key for row in dispositions)
        for key in expected_dispositions
    }
    if observed_dispositions != expected_dispositions:
        failures.append("pilot disposition counts drift")

    eligible = [row["event_id"] for row in dispositions
                if row["disposition"] ==
                "ELIGIBLE_STRICT_CAUSAL_REVEAL_PILOT"]
    if len({analysis_by_event[value]["episode_id"] for value in eligible}) != 5:
        failures.append("eligible pilot events are not episode-unique")
    if (language.get("future_frames_used") != 0
            or language.get("panoramas_used") != 0
            or controls.get("future_frames_used") != 0
            or controls.get("panoramas_used") != 0
            or language.get("training_authorized") is not False
            or controls.get("training_authorized") is not False):
        failures.append("causal/training boundary drift")
    if (language["counts"] != {
            "frontend_causal_ready": 7,
            "language_k3_pass": 6,
            "language_k3_fail": 1,
            "topology_only_frontend_fail": 2,
            "requests_made_or_reused": 29,
            }):
        failures.append("language gate counts drift")
    if (controls["counts"]["causal_controls_pass"] != 5
            or controls["counts"]["causal_controls_fail"] != 1
            or controls["counts"]["valid_mllm_responses"] != 36
            or controls["counts"]["structural_rejections"] != 12
            or len(controls["control_media_manifest"]) != 18):
        failures.append("negative control counts drift")

    evidence_paths = []
    for event in language["events"]:
        evidence_paths.extend(event["tested_prefixes"])
    for event in controls["events"]:
        for value in event["mllm_controls"].values():
            evidence_paths.extend(value["prefix_results"])
    for record in evidence_paths:
        path = ROOT / record["path"]
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != record["sha256"]
                or ROOT.resolve() not in path.resolve().parents):
            failures.append("response evidence drift: " + record["path"])
    for record in media["media_manifest"]:
        path = ROOT / record["path"]
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]):
            failures.append("causal media drift: " + record["path"])
    for record in controls["control_media_manifest"]:
        path = ROOT / record["path"]
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]):
            failures.append("control media drift: " + record["path"])

    repository_checks = []
    for repo, (expected_head, mode) in REPO_CONTRACT.items():
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False).stdout.splitlines()
        if mode == "clean":
            status_pass = not porcelain
        else:
            status_pass = (
                len(porcelain) == 15
                and sum(line.startswith(" M ") for line in porcelain) == 14
                and [line for line in porcelain if line.startswith("?? ")]
                == ["?? etpr1_compat.py"]
            )
        passed = head == expected_head and status_pass
        repository_checks.append({
            "path": str(repo.relative_to(ROOT)),
            "head": head,
            "expected_head": expected_head,
            "porcelain_line_count": len(porcelain),
            "status_contract": mode,
            "pass": passed,
        })
        if not passed:
            failures.append("repository state drift: " + str(repo))

    reserve = ROOT / ".disk_reserve"
    reserve_files = sorted(reserve.glob("reserve_10G_*.bin"))
    reserve_pass = (
        len(reserve_files) == 19
        and all(path.is_file() and not path.is_symlink()
                and path.stat().st_size == 10_737_418_240
                for path in reserve_files)
    )
    if not reserve_pass:
        failures.append("reserve contract drift")

    secret = ROOT / ".secret/qwen_api_key"
    secret_mode = secret.stat().st_mode & 0o777 if secret.is_file() else None
    secret_leak_paths = []
    scan_paths = list(GATE.rglob("*.json")) + list(GATE.rglob("*.md")) \
        + list((ROOT / "scripts").glob("*cr5*causal*.py"))
    key_pattern = re.compile(rb"sk-[A-Za-z0-9]{20,}")
    for path in scan_paths:
        if key_pattern.search(path.read_bytes()):
            secret_leak_paths.append(str(path.relative_to(ROOT)))
    secret_pass = (secret.is_file() and not secret.is_symlink()
                   and secret_mode == 0o600 and not secret_leak_paths)
    if not secret_pass:
        failures.append("secret permission or evidence leak failure")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    regression = {
        "toporeveal_24": command([
            str(ROOT / ".envs/etpr1/bin/python"),
            str(ROOT / "tests/test_toporeveal.py"), "-v"], environment),
        "uv_pip_check": command([
            str(ROOT / ".tools/uv/uv"), "pip", "check", "--python",
            str(ROOT / ".envs/etpr1/bin/python")]),
    }
    if not all(value["pass"] for value in regression.values()):
        failures.append("regression command failure")

    stat = os.statvfs(ROOT)
    free_bytes = stat.f_bavail * stat.f_frsize
    if free_bytes < 8 * 1024 ** 3:
        failures.append("free space below 8 GiB floor")
    part_files = [str(path.relative_to(ROOT)) for path in GATE.rglob("*.part")]
    if part_files:
        failures.append("gate contains .part files")

    output = {
        "revision": "cr5-strict-causal-pilot-acceptance/1",
        "verdict": ("PILOT_ENGINEERING_PASS_SCALE_REQUIRED"
                    if not failures else "PILOT_ACCEPTANCE_FAIL"),
        "scope": (
            "RxR-CE-en train-only, five episode-unique eligible events; "
            "engineering evidence only, not a benchmark or performance claim"
        ),
        "source_manifest": source_manifest,
        "counts": {
            "human_reviewed_branch_events": len(reviews),
            "human_accepted_branch_events": len(accepted),
            **observed_dispositions,
            "baseline_mllm_prefix_requests": 29,
            "negative_control_mllm_requests": 36,
            "structural_negative_rejections": 12,
        },
        "event_dispositions": dispositions,
        "eligible_event_ids": eligible,
        "repository_checks": repository_checks,
        "regression": regression,
        "integrity": {
            "original_causal_media_count": len(media["media_manifest"]),
            "control_mask_media_count": len(
                controls["control_media_manifest"]),
            "response_evidence_count": len(evidence_paths),
            "secret_file_mode_octal": (oct(secret_mode)
                                       if secret_mode is not None else None),
            "secret_leak_paths": secret_leak_paths,
            "reserve_file_count": len(reserve_files),
            "reserve_files_untouched_contract_pass": reserve_pass,
            "part_files": part_files,
            "free_bytes": free_bytes,
            "free_space_floor_bytes": 8 * 1024 ** 3,
        },
        "phase0_go_no_go": {
            "status": "NO_GO_SCALE_AND_EXPIRY_REQUIRED",
            "reason": [
                "Only 10/50 required Phase-0 candidates have this human "
                "branch review and only nine entered strict causal replay.",
                "This gate validates T_R evidence only; it does not attach "
                "the required unique reproducible T_X expiry witness.",
                "Five events are pilot examples, not enough to estimate the "
                "frozen 300-event feasibility gate reliably.",
            ],
            "next_authorized_stage": (
                "Scale the unchanged generator and controls to the frozen "
                "50-trajectory train queue, then join each eligible T_R with "
                "its SHA-bound T_X controller witness."
            ),
        },
        "training_authorized": False,
        "forbidden_split_accessed": False,
        "future_frames_used_for_online_label": 0,
        "panoramas_used_for_online_label": 0,
        "failures": failures,
    }
    atomic_write(OUT, (json.dumps(output, indent=2, ensure_ascii=False,
                                  sort_keys=False) + "\n").encode())
    log_lines = [
        "verdict=" + output["verdict"],
        "eligible=" + ",".join(eligible),
        "failures=" + json.dumps(failures, ensure_ascii=False),
        "acceptance_sha256=" + sha256_file(OUT),
    ]
    atomic_write(LOG, ("\n".join(log_lines) + "\n").encode())
    print(json.dumps({
        "verdict": output["verdict"],
        "eligible_event_ids": eligible,
        "counts": output["counts"],
        "failures": failures,
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
