#!/usr/bin/env python3
"""Run fail-closed counterfactual controls for CR5 causal-prefix labels."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cr5_causal_prefix_language as baseline  # noqa: E402
from run_phase0c_cr5_hindsight_locator import (  # noqa: E402
    atomic_json,
    parse_json,
    post,
    read_secret,
    response_text,
    sha256_bytes,
    sha256_file,
    stable_bytes,
)


GATE_DIR = ROOT / "artifacts/phase0/phase0c_cr5_causal_gate"
BASELINE = GATE_DIR / "CR5_CAUSAL_PREFIX_LANGUAGE_GATE.json"
ANALYSIS = GATE_DIR / "CR5_CAUSAL_CANDIDATE_ANALYSIS.json"
MEDIA = GATE_DIR / "CR5_CAUSAL_PREFIX_MEDIA_MANIFEST.json"
PROMPT = GATE_DIR / "CR5_CAUSAL_PREFIX_LANGUAGE_PROMPT_V1.md"
GEOMETRY = ROOT / (
    "artifacts/phase0/phase0c_cr5_preflight/multiview_branch/"
    "CR5_DIRECTED_GEOMETRY_PREFLIGHT.json"
)
INPUTS = ROOT / (
    "artifacts/phase0/phase0c_cr5_preflight/multiview_branch/"
    "CR5_MULTIVIEW_PREFLIGHT_INPUTS.json"
)
HUMAN_REVIEW = ROOT / (
    "artifacts/phase0/phase0c_cr5_human_review_v1/reviews/daiyang.jsonl"
)
CONTROL_DIR = GATE_DIR / "negative_control_results"
MASK_DIR = GATE_DIR / "control_media/removed_reveal_views"
OUT = GATE_DIR / "CR5_CAUSAL_NEGATIVE_CONTROLS.json"

EXPECTED = {
    BASELINE: "9fcdce2af6268e19c62b55f8a2d55639a1832a929371c3ddc32e1e3d1d4b63bc",
    ANALYSIS: baseline.EXPECTED_ANALYSIS_SHA256,
    MEDIA: baseline.EXPECTED_MEDIA_SHA256,
    HUMAN_REVIEW: "88eb9934cb8bc0abad3400f295e0bd1527b5d08d11189d0d4f055f61df14f1cb",
}
NEUTRAL_INSTRUCTION = (
    "Continue carefully and choose either one of the available exits."
)
CONTROL_TYPES = ("neutral_instruction", "removed_reveal_views")
PRINT_LOCK = threading.Lock()
MASK_LOCK = threading.Lock()
EXPECTED_CANDIDATE_COUNT = 6
OUTPUT_REVISION = "cr5-causal-negative-controls/1"
OUTPUT_STATUS = "PILOT_CAUSAL_CONTROL_COMPLETE"
OUTPUT_SCOPE = "six-event train-only pilot; not a benchmark claim"


def load_json(path: Path):
    return json.loads(path.read_text())


def atomic_mask(path: Path, event_id: str, prefix: int):
    with MASK_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            image = Image.new("RGB", (448, 496), (48, 52, 58))
            draw = ImageDraw.Draw(image)
            lines = [
                "VISUAL FRAME WITHHELD",
                "NEGATIVE CONTROL",
                event_id,
                "P%04d" % prefix,
            ]
            y = 180
            for line in lines:
                box = draw.textbbox((0, 0), line)
                width = box[2] - box[0]
                draw.text(((448 - width) // 2, y), line,
                          fill=(230, 230, 230))
                y += 32
            temporary = path.with_name(path.stem + ".part.jpg")
            image.save(temporary, format="JPEG", quality=95,
                       optimize=False, progressive=False)
            temporary.replace(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("unsafe control mask")


def masked_records(records, event_id: str, reveal_start: int,
                   current_prefix: int):
    output = []
    for record in records:
        value = copy.deepcopy(record)
        prefix = value["prefix_index"]
        if reveal_start <= prefix <= current_prefix:
            path = MASK_DIR / event_id / ("P%04d.jpg" % prefix)
            atomic_mask(path, event_id, prefix)
            value["path"] = str(path.relative_to(ROOT))
            value["bytes"] = path.stat().st_size
            value["sha256"] = sha256_file(path)
            value["visual_ablation"] = "REVEAL_VIEW_WITHHELD"
        output.append(value)
    return output


def replacement_input(original):
    value = copy.deepcopy(original)
    value["instruction_text"] = NEUTRAL_INSTRUCTION
    value["deterministic_segments"] = [{
        "segment_id": "CTRL_NEUTRAL_00",
        "text": NEUTRAL_INSTRUCTION,
    }]
    return value


def save_or_run(control_type, event, input_event, geometry_event,
                prefix_record, records, prompt, prompt_sha, key):
    prefix = prefix_record["prefix_index"]
    result_path = CONTROL_DIR / control_type / event["event_id"] / (
        "P%04d.json" % prefix)
    evidence = {
        "revision": "cr5-causal-negative-control-request/1",
        "control_type": control_type,
        "baseline_sha256": EXPECTED[BASELINE],
        "analysis_sha256": EXPECTED[ANALYSIS],
        "media_manifest_sha256": EXPECTED[MEDIA],
        "prompt_sha256": prompt_sha,
        "model": baseline.MODEL,
        "temperature": baseline.TEMPERATURE,
        "event_id": event["event_id"],
        "prefix_index": prefix,
        "frame_ids": [row["frame_id"] for row in records],
        "media_sha256": [row["sha256"] for row in records],
        "maximum_media_prefix": max(row["prefix_index"] for row in records),
        "future_frames_in_request": 0,
        "panoramas_in_request": 0,
        "hidden_target_role_in_request": False,
        "neutral_instruction_sha256": (
            sha256_bytes(NEUTRAL_INSTRUCTION.encode())
            if control_type == "neutral_instruction" else None
        ),
        "withheld_prefixes": [
            row["prefix_index"] for row in records
            if row.get("visual_ablation") == "REVEAL_VIEW_WITHHELD"
        ],
    }
    fingerprint = sha256_bytes(stable_bytes(evidence))
    if result_path.is_file():
        prior = load_json(result_path)
        if (prior.get("request_fingerprint_sha256") == fingerprint
                and prior.get("status") in
                {"VALID_RESPONSE", "INVALID_RESPONSE"}):
            return prior, result_path
        raise RuntimeError("drifted existing control result")

    request = baseline.build_request(
        prompt, event, input_event, geometry_event, prefix_record, records)
    started = time.time()
    provider, attempts, request_bytes = post(request, key)
    raw_text = response_text(provider)
    parse_error = None
    try:
        value = parse_json(raw_text)
        errors = baseline.validate_response(
            value, event, input_event, prefix)
    except Exception as exc:
        value = None
        errors = ["parse_error"]
        parse_error = type(exc).__name__ + ": " + str(exc)[:500]
    semantic_closed = bool(
        not errors and value["evidence_status"] == "CLOSED")
    result = {
        "status": "VALID_RESPONSE" if not errors else "INVALID_RESPONSE",
        "request_fingerprint_sha256": fingerprint,
        "request_evidence": evidence,
        "requested_model": baseline.MODEL,
        "provider_model": provider.get("model"),
        "temperature": baseline.TEMPERATURE,
        "usage": provider.get("usage"),
        "request_bytes": request_bytes,
        "http_attempts": attempts,
        "elapsed_seconds": round(time.time() - started, 3),
        "provider_response_sha256": sha256_bytes(stable_bytes(provider)),
        "raw_response_text": raw_text,
        "parsed_response": value,
        "validation_errors": errors,
        "parse_error": parse_error,
        "semantic_closed_under_control": semantic_closed,
        "training_label": False,
    }
    atomic_json(result_path, result)
    return result, result_path


def structural_controls(event, input_event, geometry_event, prefix_record,
                        records, prompt):
    outcomes = {}
    shuffled = list(reversed(records))
    try:
        baseline.build_request(prompt, event, input_event, geometry_event,
                               prefix_record, shuffled)
    except RuntimeError as exc:
        outcomes["temporal_shuffle"] = {
            "status": "REJECTED", "reason": str(exc)}
    else:
        outcomes["temporal_shuffle"] = {"status": "NOT_REJECTED"}

    injected = copy.deepcopy(records)
    fake = copy.deepcopy(injected[-1])
    fake["prefix_index"] = prefix_record["prefix_index"] + 1
    fake["frame_id"] = "P%04d" % fake["prefix_index"]
    injected.append(fake)
    try:
        baseline.build_request(prompt, event, input_event, geometry_event,
                               prefix_record, injected)
    except RuntimeError as exc:
        outcomes["future_frame_injection"] = {
            "status": "REJECTED", "reason": str(exc)}
    else:
        outcomes["future_frame_injection"] = {"status": "NOT_REJECTED"}
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    for path, expected in EXPECTED.items():
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != expected):
            raise SystemExit("pinned causal input drift: " + str(path))
    if not PROMPT.is_file() or PROMPT.is_symlink():
        raise SystemExit("prompt missing")

    gate = load_json(BASELINE)
    analysis = {row["event_id"]: row for row in
                load_json(ANALYSIS)["events"]}
    media = load_json(MEDIA)
    geometry = {row["event_id"]: row for row in
                load_json(GEOMETRY)["events"]}
    inputs = {row["event_id"]: row for row in
              load_json(INPUTS)["events"]}
    review_rows = [json.loads(line) for line in
                   HUMAN_REVIEW.read_text().splitlines() if line.strip()]
    accepted_human = {row["event_id"] for row in review_rows
                      if row["final_label"] == "ACCEPT"}
    media_by_episode = {}
    for record in media["media_manifest"]:
        media_by_episode.setdefault(record["episode_id"], {})[
            record["prefix_index"]] = record
    event_ranges = media["event_ranges"]
    candidates = [row for row in gate["events"] if row["status"] ==
                  "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED"]
    if len(candidates) != EXPECTED_CANDIDATE_COUNT or any(
            row["event_id"] not in accepted_human for row in candidates):
        raise SystemExit("baseline/human candidate contract drift")

    plan = []
    for gate_event in candidates:
        event = analysis[gate_event["event_id"]]
        reveal_start, reveal_end = gate_event["reveal_interval"]
        if reveal_end - reveal_start + 1 != baseline.K:
            raise SystemExit("baseline interval is not exact K=3")
        prefix_map = {row["prefix_index"]: row
                      for row in event["prefix_records"]}
        start = event_ranges[event["event_id"]]["history_start_prefix"]
        for prefix in range(reveal_start, reveal_end + 1):
            records = baseline.causal_media(
                media_by_episode, event["episode_id"], start, prefix)
            for control_type in CONTROL_TYPES:
                plan.append((control_type, event, prefix_map[prefix],
                             records, reveal_start))

    prompt = PROMPT.read_text()
    prompt_sha = sha256_file(PROMPT)
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "candidate_events": len(candidates),
            "mllm_control_requests": len(plan),
            "controls_per_event": {
                "neutral_instruction": baseline.K,
                "removed_reveal_views": baseline.K,
                "temporal_shuffle_structural": 1,
                "future_frame_injection_structural": 1,
            },
            "future_or_panorama_media": 0,
            "training_authorized": False,
        }, indent=2))
        return 0

    key = read_secret()

    def task(item):
        control_type, event, prefix_record, records, reveal_start = item
        input_event = inputs[event["event_id"]]
        if control_type == "neutral_instruction":
            used_input = replacement_input(input_event)
            used_records = records
        else:
            used_input = input_event
            used_records = masked_records(
                records, event["event_id"], reveal_start,
                prefix_record["prefix_index"])
        result, path = save_or_run(
            control_type, event, used_input, geometry[event["event_id"]],
            prefix_record, used_records, prompt, prompt_sha, key)
        with PRINT_LOCK:
            print(control_type, event["event_id"],
                  prefix_record["prefix_index"], result["status"],
                  "closed=" + str(result[
                      "semantic_closed_under_control"]), flush=True)
        return {
            "control_type": control_type,
            "event_id": event["event_id"],
            "prefix_index": prefix_record["prefix_index"],
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "status": result["status"],
            "semantic_closed_under_control": result[
                "semantic_closed_under_control"],
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(args.workers, 4))) as pool:
        for result in concurrent.futures.as_completed(
                [pool.submit(task, item) for item in plan]):
            results.append(result.result())
    results.sort(key=lambda row: (
        row["event_id"], row["control_type"], row["prefix_index"]))

    output_events = []
    for gate_event in candidates:
        event_id = gate_event["event_id"]
        event = analysis[event_id]
        start = event_ranges[event_id]["history_start_prefix"]
        confirmation = gate_event["confirmation_prefix"]
        prefix_record = {row["prefix_index"]: row
                         for row in event["prefix_records"]}[confirmation]
        records = baseline.causal_media(
            media_by_episode, event["episode_id"], start, confirmation)
        structural = structural_controls(
            event, inputs[event_id], geometry[event_id], prefix_record,
            records, prompt)
        controls = {}
        for control_type in CONTROL_TYPES:
            selected = [row for row in results
                        if row["event_id"] == event_id
                        and row["control_type"] == control_type]
            valid = all(row["status"] == "VALID_RESPONSE"
                        for row in selected) and len(selected) == baseline.K
            survives_k3 = valid and all(
                row["semantic_closed_under_control"] for row in selected)
            controls[control_type] = {
                "status": (
                    "CONTROL_BREAKS_K3" if valid and not survives_k3
                    else "CONTROL_SURVIVES_K3" if valid
                    else "CONTROL_INDETERMINATE"
                ),
                "valid_response_count": sum(
                    row["status"] == "VALID_RESPONSE" for row in selected),
                "semantic_closed_count": sum(
                    row["semantic_closed_under_control"] for row in selected),
                "prefix_results": selected,
            }
        structural_pass = all(
            row["status"] == "REJECTED" for row in structural.values())
        mllm_pass = all(row["status"] == "CONTROL_BREAKS_K3"
                        for row in controls.values())
        output_events.append({
            "event_id": event_id,
            "baseline_reveal_interval": gate_event["reveal_interval"],
            "human_review_accept": event_id in accepted_human,
            "mllm_controls": controls,
            "structural_controls": structural,
            "status": ("CAUSAL_CONTROLS_PASS"
                       if structural_pass and mllm_pass
                       else "CAUSAL_CONTROLS_FAIL"),
            "training_label": False,
        })

    pass_count = sum(row["status"] == "CAUSAL_CONTROLS_PASS"
                     for row in output_events)
    mask_manifest = []
    for path in sorted(MASK_DIR.rglob("*.jpg")):
        if (not path.is_file() or path.is_symlink()
                or ROOT.resolve() not in path.resolve().parents):
            raise RuntimeError("unsafe control media discovered")
        mask_manifest.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if len(mask_manifest) != len(candidates) * baseline.K:
        raise RuntimeError("control mask manifest is incomplete")
    output = {
        "revision": OUTPUT_REVISION,
        "status": OUTPUT_STATUS,
        "sources": {
            str(path.relative_to(ROOT)): expected
            for path, expected in EXPECTED.items()
        },
        "prompt_sha256": prompt_sha,
        "model": baseline.MODEL,
        "temperature": baseline.TEMPERATURE,
        "k": baseline.K,
        "events": output_events,
        "control_media_manifest": mask_manifest,
        "counts": {
            "baseline_k3_candidates": len(candidates),
            "causal_controls_pass": pass_count,
            "causal_controls_fail": len(candidates) - pass_count,
            "valid_mllm_responses": sum(
                row["status"] == "VALID_RESPONSE" for row in results),
            "mllm_requests_made_or_reused": len(results),
            "structural_rejections": sum(
                value["status"] == "REJECTED"
                for row in output_events
                for value in row["structural_controls"].values()),
        },
        "control_contract": {
            "neutral_instruction": (
                "Replace the complete instruction with branch-indifferent "
                "language at every baseline K=3 prefix; CLOSED on all three "
                "would invalidate instruction dependence."
            ),
            "removed_reveal_views": (
                "Replace every frame from baseline reveal start through the "
                "tested prefix with a content-free mask; CLOSED on all three "
                "would invalidate reveal-view dependence."
            ),
            "temporal_shuffle": "Fail closed before API request.",
            "future_frame_injection": "Fail closed before API request.",
        },
        "future_frames_used": 0,
        "panoramas_used": 0,
        "training_authorized": False,
        "scope": OUTPUT_SCOPE,
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "counts": output["counts"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
