#!/usr/bin/env python3
"""Assemble queue50 causal controls with separately preserved provider retries."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_phase0c_cr5_hindsight_locator import (  # noqa: E402
    atomic_json,
    sha256_file,
)


CAUSAL = ROOT / "artifacts/phase0/phase0c_cr5_queue50/causal_gate"
ORIGINAL = CAUSAL / "CR5_QUEUE50_CAUSAL_NEGATIVE_CONTROLS.json"
RETRY1 = CAUSAL / "CR5_QUEUE50_CAUSAL_CONTROL_RETRY.json"
RETRY2 = CAUSAL / "CR5_QUEUE50_CAUSAL_CONTROL_RETRY_ROUND2.json"
OUT = CAUSAL / "CR5_QUEUE50_CAUSAL_NEGATIVE_CONTROLS_ACCEPTED.json"
EXPECTED = {
    ORIGINAL: "5091c62a098761be95cd47635818f2c061c043686b1fe7b3eae93e457a417114",
    RETRY1: "1ff78f4a60d5c23c2fbdc06258ef11810692cab0e76858c2c004726758243284",
    RETRY2: "7462678fde69d873b0afd9a633d9131039491a0954949206656fc43752dfaa1b",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    for path, expected in EXPECTED.items():
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != expected):
            raise SystemExit("pinned control evidence drift: " + str(path))

    original = load_json(ORIGINAL)
    retry1 = load_json(RETRY1)
    retry2 = load_json(RETRY2)
    if retry1["status"] != "RETRY_INCOMPLETE":
        raise SystemExit("unexpected first retry status")
    if retry2["status"] != "RETRY_COMPLETE":
        raise SystemExit("second retry did not complete")

    original_invalid = {}
    for event in original["events"]:
        for control_type, control in event["mllm_controls"].items():
            for row in control["prefix_results"]:
                if row["status"] == "INVALID_RESPONSE":
                    key = (control_type, event["event_id"],
                           row["prefix_index"])
                    if key in original_invalid:
                        raise SystemExit("duplicate original invalid result")
                    original_invalid[key] = row
    if set(original_invalid) != {
        ("removed_reveal_views", "q10_ep7850_hv02", 27),
        ("neutral_instruction", "q19_ep31649_hv02", 12),
        ("neutral_instruction", "q37_ep55494_hv03", 36),
    }:
        raise SystemExit("unexpected original invalid result set")

    retries = {}
    for document in (retry1, retry2):
        for row in document["retry_responses"]:
            key = (row["control_type"], row["event_id"],
                   row["prefix_index"])
            if row["status"] == "VALID_RESPONSE":
                retries[key] = row
    if set(retries) != set(original_invalid):
        raise SystemExit("no valid final retry for every format failure")
    if any(row["semantic_closed_under_control"] for row in retries.values()):
        raise SystemExit("a retry changed a format failure into CLOSED")

    events = copy.deepcopy(original["events"])
    replacements = []
    for event in events:
        for control_type, control in event["mllm_controls"].items():
            results = []
            for row in control["prefix_results"]:
                key = (control_type, event["event_id"],
                       row["prefix_index"])
                if key not in retries:
                    results.append(row)
                    continue
                replacement = copy.deepcopy(retries[key])
                replacement["response_provenance"] = "PROVIDER_RETRY"
                replacement["superseded_invalid_response"] = {
                    "path": original_invalid[key]["path"],
                    "sha256": original_invalid[key]["sha256"],
                }
                results.append(replacement)
                replacements.append({
                    "control_type": key[0],
                    "event_id": key[1],
                    "prefix_index": key[2],
                    "invalid_path": original_invalid[key]["path"],
                    "invalid_sha256": original_invalid[key]["sha256"],
                    "accepted_retry_path": replacement["path"],
                    "accepted_retry_sha256": replacement["sha256"],
                })
            results.sort(key=lambda row: row["prefix_index"])
            if len(results) != 3:
                raise SystemExit("control is not K=3")
            valid = all(row["status"] == "VALID_RESPONSE" for row in results)
            survives = valid and all(
                row["semantic_closed_under_control"] for row in results)
            control["prefix_results"] = results
            control["valid_response_count"] = sum(
                row["status"] == "VALID_RESPONSE" for row in results)
            control["semantic_closed_count"] = sum(
                row["semantic_closed_under_control"] for row in results)
            control["status"] = (
                "CONTROL_BREAKS_K3" if valid and not survives
                else "CONTROL_SURVIVES_K3" if valid
                else "CONTROL_INDETERMINATE"
            )
        structural_pass = all(
            row["status"] == "REJECTED"
            for row in event["structural_controls"].values())
        mllm_pass = all(
            row["status"] == "CONTROL_BREAKS_K3"
            for row in event["mllm_controls"].values())
        event["status"] = (
            "CAUSAL_CONTROLS_PASS"
            if structural_pass and mllm_pass
            else "CAUSAL_CONTROLS_FAIL"
        )

    events.sort(key=lambda row: row["event_id"])
    pass_count = sum(row["status"] == "CAUSAL_CONTROLS_PASS"
                     for row in events)
    semantic_failures = []
    for event in events:
        if event["status"] == "CAUSAL_CONTROLS_FAIL":
            semantic_failures.append({
                "event_id": event["event_id"],
                "surviving_controls": [
                    name for name, row in event["mllm_controls"].items()
                    if row["status"] == "CONTROL_SURVIVES_K3"
                ],
                "indeterminate_controls": [
                    name for name, row in event["mllm_controls"].items()
                    if row["status"] == "CONTROL_INDETERMINATE"
                ],
            })

    if (len(events) != 17 or pass_count != 16
            or semantic_failures != [{
                "event_id": "q35_ep50411_hv03",
                "surviving_controls": ["removed_reveal_views"],
                "indeterminate_controls": [],
            }]):
        raise SystemExit("assembled causal-control disposition drift")

    output = {
        "revision": "cr5-queue50-causal-negative-controls-accepted/1",
        "status": "QUEUE50_CAUSAL_CONTROL_ACCEPTANCE_COMPLETE",
        "sources": {str(path.relative_to(ROOT)): expected
                    for path, expected in EXPECTED.items()},
        "events": events,
        "retry_adjudication": {
            "policy": (
                "Only provider responses that failed schema validation may "
                "be retried. Original responses remain immutable; the final "
                "valid provider response replaces no semantic judgment."
            ),
            "replacements": sorted(replacements, key=lambda row: (
                row["event_id"], row["control_type"], row["prefix_index"])),
            "manual_json_repair": False,
            "manual_semantic_override": False,
        },
        "control_media_manifest": original["control_media_manifest"],
        "counts": {
            "baseline_k3_candidates": len(events),
            "causal_controls_pass": pass_count,
            "causal_controls_fail": len(events) - pass_count,
            "format_invalid_first_responses": len(original_invalid),
            "provider_retry_requests": 4,
            "accepted_valid_retries": len(retries),
            "semantic_control_failures": len(semantic_failures),
        },
        "semantic_failures": semantic_failures,
        "future_frames_used": 0,
        "panoramas_used": 0,
        "training_authorized": False,
        "scope": "17 queue50 train-only causal candidates; engineering gate",
    }
    atomic_json(OUT, output)
    print(json.dumps({
        "status": output["status"],
        "counts": output["counts"],
        "semantic_failures": semantic_failures,
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
