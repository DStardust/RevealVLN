#!/usr/bin/env python3
"""Fail-closed acceptance for the complete 35-event MLLM proposal batch."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
ARTIFACT_DIR = ROOT / "artifacts/phase0/phase0c_clause_grounding_mllm"
INPUT = ARTIFACT_DIR / "MLLM_CLAUSE_GROUNDING_INPUTS.json"
INPUT_ACCEPTANCE = ARTIFACT_DIR / "MLLM_CLAUSE_GROUNDING_INPUTS_ACCEPTANCE.json"
RUN = ARTIFACT_DIR / "MLLM_CLAUSE_GROUNDING_RUN.json"
PROPOSALS = ARTIFACT_DIR / "proposals"
OUTPUT = ARTIFACT_DIR / "MLLM_CLAUSE_GROUNDING_ACCEPTANCE.json"
CLIENT = ROOT / "scripts/run_phase0c_mllm_clause_grounding.py"
EXPECTED_INPUT_SHA = (
    "d576b49122b7b3a90c71d6ff6648926afd6b063f433e8feb9a0f85488bb1f4ca"
)
EXPECTED_INPUT_ACCEPTANCE_SHA = (
    "fb835836eed4bbb56052e68b903274c27ad76de9ef7bb710b879efa62eea2564"
)
EXPECTED_CLIENT_SHA = (
    "03050aa4bc1a634bd5afc10a59f4628611ecb07171cd2754a2765bbd773d3bc8"
)
MODEL = "qwen3.8-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def regular_project_file(path: Path) -> bool:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not path.is_symlink()
        and ROOT.resolve() in resolved.parents
    )


def atomic_json(path: Path, value) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def import_client():
    spec = importlib.util.spec_from_file_location("phase0c_mllm_client", CLIENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    failures = []
    for path, expected, label in (
        (INPUT, EXPECTED_INPUT_SHA, "input"),
        (INPUT_ACCEPTANCE, EXPECTED_INPUT_ACCEPTANCE_SHA, "input_acceptance"),
        (CLIENT, EXPECTED_CLIENT_SHA, "client"),
    ):
        if not regular_project_file(path):
            failures.append(label + ":not_regular_project_file")
        elif sha256_file(path) != expected:
            failures.append(label + ":sha256_drift")
    try:
        input_acceptance = json.loads(INPUT_ACCEPTANCE.read_text())
        if input_acceptance.get("status") != "PASS":
            failures.append("input_acceptance:not_pass")
        manifest = json.loads(INPUT.read_text())
    except Exception as exc:
        failures.append("input_parse:" + type(exc).__name__)
        manifest = {"events": []}

    client = import_client() if not any(
        item.startswith("client:") for item in failures) else None
    event_results = []
    event_ids = [event.get("event_id") for event in manifest.get("events", [])]
    media_by_id = {record.get("frame_id"): record
                   for record in manifest.get("media_manifest", [])}
    if len(event_ids) != 35 or len(set(event_ids)) != 35:
        failures.append("input_event_set")
    for row, event in enumerate(manifest.get("events", [])):
        event_id = event.get("event_id")
        path = PROPOSALS / f"{event_id}.json"
        local_failures = []
        if not regular_project_file(path):
            local_failures.append("missing_or_unsafe_file")
            result = {}
        else:
            try:
                result = json.loads(path.read_text())
            except Exception as exc:
                local_failures.append("json_parse:" + type(exc).__name__)
                result = {}
        if not result:
            event_results.append({
                "row_order": row,
                "event_id": event_id,
                "status": "FAIL",
                "proposal_path": str(path.relative_to(ROOT)),
                "proposal_sha256": sha256_file(path)
                if regular_project_file(path) else None,
                "proposal_status": None,
                "failures": local_failures,
            })
            failures.extend(f"{event_id}:{value}"
                            for value in local_failures)
            continue
        if result.get("status") != "VALID_MLLM_PROPOSAL":
            local_failures.append("status_not_valid")
        if result.get("event_id") != event_id:
            local_failures.append("event_id")
        if result.get("model") != MODEL:
            local_failures.append("requested_model")
        if result.get("base_url") != BASE_URL:
            local_failures.append("base_url")
        metadata = result.get("provider_response_metadata", {})
        if metadata.get("model") != MODEL:
            local_failures.append("provider_model")
        if metadata.get("model_exactly_matches_request") is not True:
            local_failures.append("provider_model_exact_flag")
        if result.get("proposal_is_ground_truth") is not False:
            local_failures.append("proposal_ground_truth")
        if result.get("human_verification_required") is not True:
            local_failures.append("human_verification")
        if result.get("training_authorized") is not False:
            local_failures.append("training_authorized")
        if result.get("schema_errors") != []:
            local_failures.append("schema_errors")
        if client is not None:
            expected_fingerprint, _ = client.request_fingerprint(
                event,
                manifest["model_request_contract"]["system_prompt_sha256"],
                [(media_by_id[frame_id], ROOT / media_by_id[frame_id]["path"])
                 for frame_id in event["sequence_frame_ids"]],
            )
            if result.get("request_fingerprint_sha256") != expected_fingerprint:
                local_failures.append("request_fingerprint")
            local_failures.extend("proposal:" + value for value in
                                  client.validate_proposal(
                                      result.get("proposal"), event))
        attempts = result.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            local_failures.append("attempts")
        else:
            if not any(item.get("http_status") == 200 for item in attempts):
                local_failures.append("no_http_200")
        usage = metadata.get("usage")
        if not isinstance(usage, dict):
            local_failures.append("usage_missing")
        event_results.append({
            "row_order": row,
            "event_id": event_id,
            "status": "PASS" if not local_failures else "FAIL",
            "proposal_path": str(path.relative_to(ROOT)),
            "proposal_sha256": sha256_file(path)
            if regular_project_file(path) else None,
            "proposal_status": result.get("proposal", {}).get("status")
            if isinstance(result.get("proposal"), dict) else None,
            "failures": local_failures,
        })
        failures.extend(f"{event_id}:{value}" for value in local_failures)

    unexpected = []
    if PROPOSALS.is_dir():
        unexpected = sorted(
            str(path.relative_to(ROOT)) for path in PROPOSALS.iterdir()
            if path.is_symlink() or not path.is_file()
            or path.suffix != ".json" or path.stem not in set(event_ids)
        )
    if unexpected:
        failures.append("unexpected_proposal_entries")
    try:
        run = json.loads(RUN.read_text())
    except Exception as exc:
        failures.append("run_summary:" + type(exc).__name__)
        run = {}
    if run.get("model") != MODEL:
        failures.append("run_summary:model")
    if run.get("endpoint") != BASE_URL + "/chat/completions":
        failures.append("run_summary:endpoint")
    if run.get("all_valid") is not True:
        failures.append("run_summary:all_valid")
    if run.get("training_authorized") is not False:
        failures.append("run_summary:training_authorized")
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free_bytes < 8 * 1024**3:
        failures.append("disk_below_8GiB")
    status = "PASS" if not failures else "NO_GO"
    output = {
        "status": status,
        "revision": "phase0c-mllm-clause-acceptance/1",
        "input_manifest_sha256": EXPECTED_INPUT_SHA,
        "input_acceptance_sha256": EXPECTED_INPUT_ACCEPTANCE_SHA,
        "client_sha256": EXPECTED_CLIENT_SHA,
        "model": MODEL,
        "base_url": BASE_URL,
        "events_expected": 35,
        "events_passed": sum(item["status"] == "PASS"
                             for item in event_results),
        "events": event_results,
        "unexpected_proposal_entries": unexpected,
        "run_summary_sha256": sha256_file(RUN)
        if regular_project_file(RUN) else None,
        "free_bytes": free_bytes,
        "failures": failures,
        "proposals_are_ground_truth": False,
        "human_verification_required": True,
        "training_authorized": False,
    }
    atomic_json(OUTPUT, output)
    print(json.dumps({
        "status": status,
        "events_passed": output["events_passed"],
        "failures": len(failures),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUTPUT),
    }, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
