#!/usr/bin/env python3
"""Retry only the three format-invalid queue50 causal controls.

The first provider responses remain immutable evidence.  Retries use the exact
same causal media and instruction intervention and write to a separate tree.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cr5_queue50_causal_negative_controls as queue50  # noqa: E402


controls = queue50.controls
CAUSAL = queue50.CAUSAL
ORIGINAL_DIR = CAUSAL / "negative_control_results"
RETRY_DIR = CAUSAL / "negative_control_retry_results"
OUT = CAUSAL / "CR5_QUEUE50_CAUSAL_CONTROL_RETRY.json"
TARGETS = (
    (
        "removed_reveal_views", "q10_ep7850_hv02", 27,
        "ef64b512199eb46ec3caad4dfff16998811eb6e811236fadf22759ac3ccbb338",
    ),
    (
        "neutral_instruction", "q19_ep31649_hv02", 12,
        "3702ec4372280476b084b4e0330aff73a98e971c139f172d1b80c844ba82667f",
    ),
    (
        "neutral_instruction", "q37_ep55494_hv03", 36,
        "4a9c5e2b375612089e6e830030b6303c4554cae114e78204cc1290c07cb77057",
    ),
)


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    targets = TARGETS
    retry_dir = RETRY_DIR
    output_path = OUT
    source_dir = ORIGINAL_DIR
    source_shas = {row[0:3]: row[3] for row in TARGETS}
    source_expected_status = "INVALID_RESPONSE"
    source_expected_parsed_status = "NOT_CLOSED"
    if args.round == 2:
        targets = TARGETS[:1]
        retry_dir = CAUSAL / "negative_control_retry_results_round2"
        output_path = CAUSAL / "CR5_QUEUE50_CAUSAL_CONTROL_RETRY_ROUND2.json"
        source_dir = RETRY_DIR
        source_shas = {
            ("removed_reveal_views", "q10_ep7850_hv02", 27):
            "60a29d2c40fb610c668504567d24690cd30acaa151337e0c227c0ea206ddb5ca"
        }
        source_expected_parsed_status = None

    for path, expected in controls.EXPECTED.items():
        if (not path.is_file() or path.is_symlink()
                or controls.sha256_file(path) != expected):
            raise SystemExit("pinned causal input drift: " + str(path))

    original_rows = []
    for control_type, event_id, prefix, _ in targets:
        expected_sha = source_shas[(control_type, event_id, prefix)]
        path = source_dir / control_type / event_id / ("P%04d.json" % prefix)
        if (not path.is_file() or path.is_symlink()
                or controls.sha256_file(path) != expected_sha):
            raise SystemExit("original invalid response drift: " + str(path))
        row = load_json(path)
        if (row.get("status") != source_expected_status
                or row.get("semantic_closed_under_control") is not False
                or ((row.get("parsed_response") or {}).get("evidence_status")
                    != source_expected_parsed_status)):
            raise SystemExit("retry target is not a format-only invalid response")
        original_rows.append({
            "control_type": control_type,
            "event_id": event_id,
            "prefix_index": prefix,
            "path": str(path.relative_to(ROOT)),
            "sha256": expected_sha,
            "validation_errors": row["validation_errors"],
            "parsed_evidence_status": "NOT_CLOSED",
        })

    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "retry_round": args.round,
            "retry_count": len(targets),
            "original_responses_preserved": True,
            "future_frames_used": 0,
            "panoramas_used": 0,
            "training_authorized": False,
        }, indent=2))
        return 0

    gate = load_json(controls.BASELINE)
    analysis = {row["event_id"]: row for row in
                load_json(controls.ANALYSIS)["events"]}
    media = load_json(controls.MEDIA)
    geometry = {row["event_id"]: row for row in
                load_json(controls.GEOMETRY)["events"]}
    inputs = {row["event_id"]: row for row in
              load_json(controls.INPUTS)["events"]}
    gate_events = {row["event_id"]: row for row in gate["events"]}
    media_by_episode = {}
    for record in media["media_manifest"]:
        media_by_episode.setdefault(record["episode_id"], {})[
            record["prefix_index"]] = record
    event_ranges = media["event_ranges"]
    prompt = controls.PROMPT.read_text()
    prompt_sha = controls.sha256_file(controls.PROMPT)
    key = controls.read_secret()
    controls.CONTROL_DIR = retry_dir

    def run_target(target):
        control_type, event_id, prefix, _ = target
        gate_event = gate_events[event_id]
        if gate_event["status"] != "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED":
            raise RuntimeError("target no longer belongs to the baseline gate")
        event = analysis[event_id]
        reveal_start, reveal_end = gate_event["reveal_interval"]
        if prefix < reveal_start or prefix > reveal_end:
            raise RuntimeError("target prefix is outside the reveal interval")
        prefix_record = {row["prefix_index"]: row
                         for row in event["prefix_records"]}[prefix]
        start = event_ranges[event_id]["history_start_prefix"]
        records = controls.baseline.causal_media(
            media_by_episode, event["episode_id"], start, prefix)
        input_event = inputs[event_id]
        if control_type == "neutral_instruction":
            used_input = controls.replacement_input(input_event)
            used_records = records
        else:
            used_input = input_event
            used_records = controls.masked_records(
                records, event_id, reveal_start, prefix)
        result, path = controls.save_or_run(
            control_type, event, used_input, geometry[event_id],
            prefix_record, used_records, prompt, prompt_sha, key)
        return {
            "control_type": control_type,
            "event_id": event_id,
            "prefix_index": prefix,
            "path": str(path.relative_to(ROOT)),
            "sha256": controls.sha256_file(path),
            "status": result["status"],
            "validation_errors": result["validation_errors"],
            "parsed_evidence_status": (
                (result.get("parsed_response") or {}).get("evidence_status")
            ),
            "semantic_closed_under_control": result[
                "semantic_closed_under_control"],
        }

    retry_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_target, target) for target in targets]
        for future in concurrent.futures.as_completed(futures):
            retry_rows.append(future.result())
    retry_rows.sort(key=lambda row: (
        row["event_id"], row["control_type"], row["prefix_index"]))

    output = {
        "revision": "cr5-queue50-causal-control-retry/%d" % args.round,
        "status": (
            "RETRY_COMPLETE" if all(row["status"] == "VALID_RESPONSE"
                                    for row in retry_rows)
            else "RETRY_INCOMPLETE"
        ),
        "retry_reason": (
            "A prior response was semantically non-closed or truncated but "
            "violated the frozen response schema; retrying provider "
            "formatting only."
        ),
        "original_invalid_responses": original_rows,
        "retry_responses": retry_rows,
        "original_responses_preserved": True,
        "same_evidence_and_intervention": True,
        "future_frames_used": 0,
        "panoramas_used": 0,
        "training_authorized": False,
    }
    controls.atomic_json(output_path, output)
    print(json.dumps({
        "status": output["status"],
        "valid_retries": sum(row["status"] == "VALID_RESPONSE"
                             for row in retry_rows),
        "semantic_closed": sum(row["semantic_closed_under_control"]
                               for row in retry_rows),
        "output": str(output_path.relative_to(ROOT)),
        "sha256": controls.sha256_file(output_path),
    }, indent=2))
    return 0 if output["status"] == "RETRY_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
