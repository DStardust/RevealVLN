#!/usr/bin/env python3
"""Apply the frozen lossless S4 -> S04 rule to two pinned provider outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
ARTIFACT_DIR = ROOT / "artifacts/phase0/phase0c_clause_grounding_mllm"
INPUT = ARTIFACT_DIR / "MLLM_CLAUSE_GROUNDING_INPUTS.json"
CLIENT = ROOT / "scripts/run_phase0c_mllm_clause_grounding.py"
OUTPUT = ARTIFACT_DIR / "MLLM_SEGMENT_ID_NORMALIZATION.json"
EXPECTED_INPUT_SHA = (
    "d576b49122b7b3a90c71d6ff6648926afd6b063f433e8feb9a0f85488bb1f4ca"
)
TARGETS = {
    "ep22063_turn02":
        "2986c7d58a527227d1249489eb1c9b17805fa85a6dd973c3eeb04b58386a45a4",
    "ep32770_turn06":
        "3c5e41c81c17bff2423638fd991f72c8d002ce8d29223c0424277b88dfe582dc",
}


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


def load_client():
    spec = importlib.util.spec_from_file_location("phase0c_client", CLIENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if sha256_file(INPUT) != EXPECTED_INPUT_SHA:
        raise SystemExit("input manifest SHA drift")
    manifest = json.loads(INPUT.read_text())
    events = {event["event_id"]: event for event in manifest["events"]}
    client = load_client()
    audit = []
    for event_id, expected_sha in TARGETS.items():
        path = ARTIFACT_DIR / "proposals" / f"{event_id}.json"
        before_sha = sha256_file(path)
        if before_sha != expected_sha:
            raise SystemExit(f"{event_id}: provider-result SHA drift")
        result = json.loads(path.read_text())
        if result.get("status") != "INVALID_MLLM_SCHEMA":
            raise SystemExit(f"{event_id}: unexpected pre-status")
        if result.get("schema_errors") != [
                "UNIQUE_MATCH selection must be 1-3 adjacent segments"]:
            raise SystemExit(f"{event_id}: unexpected pre-error")
        raw = result.get("proposal")
        normalized, changes = client.normalize_unambiguous_segment_ids(
            raw, events[event_id])
        errors = client.validate_proposal(normalized, events[event_id])
        if errors or len(changes) != 1:
            raise SystemExit(f"{event_id}: normalization not uniquely valid")
        change = changes[0]
        if change.get("raw") != "S4" or change.get("normalized") != "S04":
            raise SystemExit(f"{event_id}: normalization outside frozen rule")
        result["provider_raw_proposal"] = raw
        result["proposal"] = normalized
        result["lossless_segment_id_normalizations"] = changes
        result["schema_errors"] = []
        result["status"] = "VALID_MLLM_PROPOSAL"
        result["postprocessing"] = {
            "revision": "lossless-segment-id-normalization/1",
            "semantic_fields_changed": False,
            "provider_output_preserved": True,
            "rule_scope": "unambiguous omitted leading zero only",
        }
        atomic_json(path, result)
        audit.append({
            "event_id": event_id,
            "path": str(path.relative_to(ROOT)),
            "before_sha256": before_sha,
            "after_sha256": sha256_file(path),
            "changes": changes,
            "schema_errors_after": errors,
            "semantic_fields_changed": False,
        })
    output = {
        "status": "PASS",
        "revision": "lossless-segment-id-normalization/1",
        "input_manifest_sha256": EXPECTED_INPUT_SHA,
        "client_sha256": sha256_file(CLIENT),
        "targets": audit,
        "provider_outputs_preserved": True,
        "semantic_fields_changed": False,
        "human_verification_required": True,
        "training_authorized": False,
    }
    atomic_json(OUTPUT, output)
    print(json.dumps({
        "status": output["status"],
        "normalized": len(audit),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUTPUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
