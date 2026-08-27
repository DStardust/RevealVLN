#!/usr/bin/env python3
"""Create a non-claiming, reproducible summary of accepted MLLM proposals."""

from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
ARTIFACT_DIR = ROOT / "artifacts/phase0/phase0c_clause_grounding_mllm"
ACCEPTANCE = ARTIFACT_DIR / "MLLM_CLAUSE_GROUNDING_ACCEPTANCE.json"
NORMALIZATION = ARTIFACT_DIR / "MLLM_SEGMENT_ID_NORMALIZATION.json"
PROPOSALS = ARTIFACT_DIR / "proposals"
OUTPUT = ARTIFACT_DIR / "MLLM_CLAUSE_GROUNDING_BATCH_REPORT.json"
EXPECTED_ACCEPTANCE_SHA = (
    "8a014c571b8d8715b057a547ff6c5ee409c358a70244ce0aa94919b485404bfb"
)
EXPECTED_NORMALIZATION_SHA = (
    "0ad3994a0d32da38ea20e011cecdd01ba3305abd2ba4a5ef7bd5be1029ee5f5d"
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def atomic_json(path: Path, value) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def main() -> int:
    if sha256_file(ACCEPTANCE) != EXPECTED_ACCEPTANCE_SHA:
        raise SystemExit("acceptance SHA drift")
    if sha256_file(NORMALIZATION) != EXPECTED_NORMALIZATION_SHA:
        raise SystemExit("normalization SHA drift")
    acceptance = json.loads(ACCEPTANCE.read_text())
    if acceptance.get("status") != "PASS" or \
            acceptance.get("events_passed") != 35:
        raise SystemExit("acceptance is not 35/35 PASS")
    proposal_statuses = collections.Counter()
    selected_counts = collections.Counter()
    evidence_counts = collections.Counter()
    usage = collections.Counter()
    http_statuses = collections.Counter()
    confidences = []
    elapsed = []
    response_ids = set()
    normalized_events = []
    paths = sorted(PROPOSALS.glob("*.json"))
    if len(paths) != 35:
        raise SystemExit("proposal cardinality")
    for path in paths:
        result = json.loads(path.read_text())
        if result.get("status") != "VALID_MLLM_PROPOSAL":
            raise SystemExit(path.name + ": invalid proposal")
        proposal = result["proposal"]
        proposal_statuses[proposal["status"]] += 1
        selected_counts[len(proposal["selected_segment_ids"])] += 1
        evidence_counts[len(proposal["evidence_frame_ids"])] += 1
        confidences.append(float(proposal["confidence"]))
        elapsed.append(float(result["elapsed_seconds"]))
        metadata = result["provider_response_metadata"]
        response_ids.add(metadata["id"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int(metadata["usage"][key])
        for attempt in result["attempts"]:
            http_statuses[str(attempt.get("http_status"))] += 1
        if result.get("lossless_segment_id_normalizations"):
            normalized_events.append({
                "event_id": result["event_id"],
                "changes": result["lossless_segment_id_normalizations"],
                "provider_raw_proposal_preserved":
                    "provider_raw_proposal" in result,
            })
    if len(response_ids) != 35:
        raise SystemExit("provider response ID uniqueness")
    output = {
        "status": "PASS_NON_GROUND_TRUTH_SUMMARY",
        "revision": "phase0c-mllm-batch-report/1",
        "acceptance_sha256": EXPECTED_ACCEPTANCE_SHA,
        "normalization_sha256": EXPECTED_NORMALIZATION_SHA,
        "provider": "DashScope OpenAI-compatible",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "requested_and_returned_model": "qwen3.8-max",
        "accepted_event_count": 35,
        "unique_provider_response_ids": len(response_ids),
        "proposal_status_counts": dict(sorted(proposal_statuses.items())),
        "selected_segment_count_distribution": {
            str(key): value for key, value in sorted(selected_counts.items())},
        "evidence_frame_count_distribution": {
            str(key): value for key, value in sorted(evidence_counts.items())},
        "confidence": {
            "minimum": min(confidences),
            "mean": sum(confidences) / len(confidences),
            "maximum": max(confidences),
        },
        "usage_totals": dict(usage),
        "http_attempt_status_counts": dict(sorted(http_statuses.items())),
        "http_attempt_count": sum(http_statuses.values()),
        "sum_per_event_elapsed_seconds": sum(elapsed),
        "lossless_segment_id_normalization_count": len(normalized_events),
        "normalized_events": normalized_events,
        "monetary_cost_claimed": False,
        "proposal_is_ground_truth": False,
        "human_verification_required": True,
        "verified_language_reveal_events": 0,
        "training_authorized": False,
    }
    atomic_json(OUTPUT, output)
    print(json.dumps({
        "status": output["status"],
        "events": output["accepted_event_count"],
        "usage_totals": output["usage_totals"],
        "http_attempts": output["http_attempt_count"],
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUTPUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
