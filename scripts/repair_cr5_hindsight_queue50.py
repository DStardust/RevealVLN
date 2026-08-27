#!/usr/bin/env python3
"""Repair non-semantic shape errors and retry semantic/parser failures."""

from __future__ import annotations

import concurrent.futures
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cr5_hindsight_queue50 as wrapped  # noqa: E402


runner = wrapped.runner
OUT_DIR = ROOT / "artifacts/phase0/phase0c_cr5_queue50/hindsight_locator"
RAW_RUN = OUT_DIR / "CR5_QUEUE50_HINDSIGHT_RUN.json"
EXPECTED_RAW_RUN_SHA = (
    "a3b5bf00130947a29d7579c19808d63e9ebc50a2ecbf324bbb3c05337f85c038"
)
REPAIR_DIR = OUT_DIR / "repairs"
RETRY_DIR = OUT_DIR / "retries"
RETRY_REPAIR_DIR = OUT_DIR / "retry_repairs"
OUT = OUT_DIR / "CR5_QUEUE50_HINDSIGHT_ACCEPTED_RUN.json"
SUMMARY_ERROR = re.compile(
    r"interval\[([0-9]+)\]:reference_route_choice_summary")


def load(path: Path):
    return json.loads(path.read_text())


def is_repairable(payload):
    errors = payload.get("validation_errors", [])
    return (payload.get("status") == "INVALID_MLLM_PROPOSAL"
            and bool(errors)
            and all(SUMMARY_ERROR.fullmatch(value) for value in errors))


def repair_payload(source_path: Path, destination: Path, episode, chunk):
    source = load(source_path)
    if not is_repairable(source):
        raise RuntimeError("non-whitelisted repair requested")
    normalized = copy.deepcopy(source["normalized_proposal"])
    repairs = []
    indexes = sorted({int(SUMMARY_ERROR.fullmatch(value).group(1))
                      for value in source["validation_errors"]})
    for index in indexes:
        value = normalized["candidate_intervals"][index][
            "reference_route_choice_summary"]
        if not isinstance(value, str) or len(value) <= 240:
            raise RuntimeError("repair preimage does not exceed limit")
        repaired = value[:240]
        normalized["candidate_intervals"][index][
            "reference_route_choice_summary"] = repaired
        repairs.append({
            "field": "candidate_intervals[%d]."
                     "reference_route_choice_summary" % index,
            "rule": "unicode_codepoint_prefix_truncation_to_schema_max",
            "semantic_fields_changed": False,
            "original_length": len(value),
            "repaired_length": len(repaired),
            "original_sha256": runner.sha256_bytes(value.encode()),
            "repaired_sha256": runner.sha256_bytes(repaired.encode()),
        })
    errors = runner.validate_proposal(normalized, episode, chunk)
    if errors:
        raise RuntimeError("post-repair schema failure: " + repr(errors))
    value = copy.deepcopy(source)
    value["status"] = "VALID_MLLM_PROPOSAL"
    value["normalized_proposal"] = normalized
    value["validation_errors"] = []
    value["posthoc_repairs"] = repairs
    value["repair_source"] = {
        "path": str(source_path.relative_to(ROOT)),
        "sha256": runner.sha256_file(source_path),
    }
    value["repair_scope"] = (
        "non-semantic display summary only; interval, frame, clause, kind, "
        "pattern, confidence and rationale unchanged"
    )
    runner.atomic_json(destination, value)
    return destination


def main():
    for path, expected in ((runner.INPUT, runner.EXPECTED_INPUT_SHA),
                           (runner.ACCEPTANCE,
                            runner.EXPECTED_ACCEPTANCE_SHA),
                           (RAW_RUN, EXPECTED_RAW_RUN_SHA)):
        if (not path.is_file() or path.is_symlink()
                or runner.sha256_file(path) != expected):
            raise SystemExit("pinned queue50 source drift: " + str(path))
    manifest = load(runner.INPUT)
    raw_run = load(RAW_RUN)
    episode_map = {row["episode_id"]: row
                   for row in manifest["episodes"]}
    chunk_map = {(row["episode_id"], chunk["chunk_id"]): chunk
                 for row in manifest["episodes"] for chunk in row["chunks"]}
    raw_rows = []
    retry_jobs = []
    raw_counts = Counter()
    for record in raw_run["results"]:
        path = ROOT / record["path"]
        if runner.sha256_file(path) != record["sha256"]:
            raise SystemExit("raw result SHA drift: " + record["path"])
        payload = load(path)
        raw_counts[payload["status"]] += 1
        raw_rows.append((record, path, payload))
        if payload["status"] != "VALID_MLLM_PROPOSAL" \
                and not is_repairable(payload):
            retry_jobs.append((episode_map[record["episode_id"]],
                               chunk_map[(record["episode_id"],
                                          record["chunk_id"])]))
    if (raw_counts != Counter({"VALID_MLLM_PROPOSAL": 121,
                               "INVALID_MLLM_PROPOSAL": 16,
                               "REQUEST_OR_VALIDATION_FAILURE": 1})
            or len(retry_jobs) != 3):
        raise SystemExit("unexpected raw failure distribution")

    pending_retries = []
    for episode, chunk in retry_jobs:
        retry_path = RETRY_DIR / ("ep%s_%s.json" %
                                  (episode["episode_id"], chunk["chunk_id"]))
        evidence, fingerprint = runner.request_evidence(
            manifest, episode, chunk)
        if retry_path.is_file():
            existing = load(retry_path)
            if (existing.get("request_fingerprint_sha256") == fingerprint
                    and existing.get("status") in {
                        "VALID_MLLM_PROPOSAL", "INVALID_MLLM_PROPOSAL",
                        "REQUEST_OR_VALIDATION_FAILURE"}):
                continue
        pending_retries.append((episode, chunk))
    if pending_retries:
        prompt = (ROOT / manifest["contract"]["prompt_path"]).read_text()
        key = runner.read_secret()
        original_result_dir = runner.RESULT_DIR
        runner.RESULT_DIR = RETRY_DIR
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(
                    runner.execute_one, index, len(pending_retries), manifest,
                    prompt, episode, chunk, key)
                    for index, (episode, chunk) in enumerate(
                        pending_retries, 1)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        finally:
            runner.RESULT_DIR = original_result_dir

    accepted_rows = []
    selection_counts = Counter()
    unresolved = []
    retry_usage = Counter()
    for record, source_path, payload in raw_rows:
        episode = episode_map[record["episode_id"]]
        chunk = chunk_map[(record["episode_id"], record["chunk_id"])]
        if payload["status"] == "VALID_MLLM_PROPOSAL":
            accepted_path = source_path
            selection = "RAW_FIRST_PASS"
        elif is_repairable(payload):
            accepted_path = REPAIR_DIR / source_path.name
            repair_payload(source_path, accepted_path, episode, chunk)
            selection = "RAW_NON_SEMANTIC_SHAPE_REPAIR"
        else:
            retry_path = RETRY_DIR / source_path.name
            if not retry_path.is_file():
                unresolved.append({"path": str(source_path.relative_to(ROOT)),
                                   "reason": "retry_missing"})
                continue
            retry = load(retry_path)
            for name in ("prompt_tokens", "completion_tokens",
                         "total_tokens"):
                retry_usage[name] += int(
                    retry.get("usage", {}).get(name, 0))
            if retry.get("status") == "VALID_MLLM_PROPOSAL":
                accepted_path = retry_path
                selection = "REAL_API_RETRY"
            elif is_repairable(retry):
                accepted_path = RETRY_REPAIR_DIR / retry_path.name
                repair_payload(retry_path, accepted_path, episode, chunk)
                selection = "REAL_API_RETRY_PLUS_NON_SEMANTIC_SHAPE_REPAIR"
            else:
                unresolved.append({
                    "path": str(source_path.relative_to(ROOT)),
                    "retry_path": str(retry_path.relative_to(ROOT)),
                    "retry_status": retry.get("status"),
                    "retry_validation_errors": retry.get(
                        "validation_errors"),
                    "retry_error_type": retry.get("error_type"),
                })
                continue
        accepted = load(accepted_path)
        errors = runner.validate_proposal(accepted["normalized_proposal"],
                                          episode, chunk)
        if (accepted.get("status") != "VALID_MLLM_PROPOSAL" or errors
                or accepted.get("provider_model") != runner.MODEL):
            raise RuntimeError("accepted result validation failure")
        selection_counts[selection] += 1
        accepted_rows.append({
            "episode_id": record["episode_id"],
            "chunk_id": record["chunk_id"],
            "selection": selection,
            "path": str(accepted_path.relative_to(ROOT)),
            "sha256": runner.sha256_file(accepted_path),
        })
    accepted_rows.sort(key=lambda row: (
        next(value["queue_order"] for value in manifest["episodes"]
             if value["episode_id"] == row["episode_id"]),
        row["chunk_id"]))
    usage = dict(raw_run["usage"])
    for name, value in retry_usage.items():
        usage[name] = int(usage.get(name, 0)) + value
    expected_fail_closed = (
        len(accepted_rows) == 136 and len(unresolved) == 2
        and {(row.get("path"), tuple(row.get("retry_validation_errors")
                                     or [])) for row in unresolved} == {
            ("artifacts/phase0/phase0c_cr5_queue50/hindsight_locator/"
             "proposals/ep16121_C02.json",
             ("interval[0]:reveal_clause_ids",)),
            ("artifacts/phase0/phase0c_cr5_queue50/hindsight_locator/"
             "proposals/ep28644_C00.json",
             ("interval[2]:interval_frames",
              "interval[2]:supporting_frames")),
        }
    )
    output = {
        "revision": "cr5-queue50-hindsight-accepted-run/1",
        "status": ("PASS_WITH_FAIL_CLOSED_BLOCK_REJECTIONS"
                   if expected_fail_closed else "FAIL"),
        "input_manifest_sha256": runner.EXPECTED_INPUT_SHA,
        "input_acceptance_sha256": runner.EXPECTED_ACCEPTANCE_SHA,
        "raw_first_pass": {
            "path": str(RAW_RUN.relative_to(ROOT)),
            "sha256": EXPECTED_RAW_RUN_SHA,
            "counts": dict(sorted(raw_counts.items())),
            "raw_valid_rate": raw_counts["VALID_MLLM_PROPOSAL"] / 138,
        },
        "repair_policy": {
            "allowed": "reference_route_choice_summary >240 codepoints only",
            "forbidden": (
                "interval/frame/clause/kind/pattern/confidence/rationale "
                "semantic repair"
            ),
            "original_provider_response_retained": True,
        },
        "selection_counts": dict(sorted(selection_counts.items())),
        "retry_job_count": len(retry_jobs),
        "job_count": 138,
        "valid_count": len(accepted_rows),
        "fail_closed_rejected_block_count": len(unresolved),
        "usage_including_retries": usage,
        "results": accepted_rows,
        "unresolved": unresolved,
        "future_frames_are_offline_annotation_only": True,
        "online_causal_labels_created": 0,
        "human_labels_created": 0,
        "training_authorized": False,
    }
    runner.atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "raw_counts": output["raw_first_pass"]["counts"],
        "selection_counts": output["selection_counts"],
        "valid": output["valid_count"],
        "unresolved": unresolved,
        "output": str(OUT.relative_to(ROOT)),
        "sha256": runner.sha256_file(OUT),
    }, indent=2))
    return 0 if output["status"] == \
        "PASS_WITH_FAIL_CLOSED_BLOCK_REJECTIONS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
