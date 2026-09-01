#!/usr/bin/env python3
"""Execution-only long-DAG transport revision for MF3ZP RevealSkill.

The only semantic-neutral change from v1r1 is ``max_tokens: 2000 -> 8000``.
It is used only for still-missing preannotations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
V1R1_PATH = ROOT / "scripts/annotate_mf3zp_qwen_transport_v1r1.py"
V1R1_PROTOCOL = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_QWEN_TRANSPORT_V1R1_PROTOCOL.json"
PROTOCOL = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_QWEN_TRANSPORT_V1R2_PROTOCOL.json"
MODEL = "qwen3.8-max"
MAX_TOKENS = 8000
MAX_ATTEMPTS = 4
TIMEOUT_SECONDS = 180
BACKOFF = (0.0, 2.0, 5.0, 10.0)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = _load(V1R1_PATH, "mf3zp_qwen_transport_v1r1_parent")
base = v1.base
_base_request_record = base.request_record


def protocol_value():
    v1.verify()
    science = base.verify_protocol()
    return {
        "schema_version": "revealnav-mf3zp-qwen-transport/1",
        "revision": "mf3zp_revealskill_qwen_transport_v1r2",
        "status": "SEALED_BEFORE_QWEN_TRANSPORT_V1R2_RESULTS",
        "science_protocol": v1.inventory(v1.SCIENCE_PROTOCOL),
        "parent_transport_protocol": v1.inventory(V1R1_PROTOCOL),
        "parent_transport_entrypoint": v1.inventory(V1R1_PATH),
        "transport_entrypoint": v1.inventory(Path(__file__).resolve()),
        "model": MODEL,
        "prompt_hashes": {"instruction": science["qwen_annotation"]["instruction_prompt_sha256"], "evidence": science["qwen_annotation"]["evidence_prompt_sha256"]},
        "only_change_from_v1r1": {"max_tokens": {"from": 2000, "to": MAX_TOKENS}},
        "enable_thinking": False,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "model_fallback": False,
        "prompt_change": False,
        "schema_change": False,
        "population_change": False,
        "scientific_gate_change": False,
        "target_payload_read": False,
        "outcome_payload_read": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }


def seal():
    value = protocol_value()
    if PROTOCOL.exists() or PROTOCOL.is_symlink():
        raise RuntimeError("refusing to overwrite v1r2 transport protocol")
    partial = PROTOCOL.with_name(PROTOCOL.name + ".part")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, PROTOCOL)
    return value


def verify():
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if value != protocol_value():
        raise RuntimeError("v1r2 transport protocol/source drift")
    return value


def api_request(api_key: str, payload: dict[str, object]):
    effective = dict(payload)
    effective["enable_thinking"] = False
    effective["max_tokens"] = MAX_TOKENS
    body = json.dumps(effective, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    last_error = "unattempted"
    for attempt in range(MAX_ATTEMPTS):
        if BACKOFF[attempt]: time.sleep(BACKOFF[attempt])
        request = urllib.request.Request(base.QWEN_ENDPOINT, data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                envelope = json.loads(response.read())
            provider_model = str(envelope.get("model", ""))
            if provider_model != MODEL: raise RuntimeError(f"provider model drift: {provider_model!r}")
            return json.loads(envelope["choices"][0]["message"]["content"]), provider_model, str(envelope.get("id", ""))
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, RuntimeError) as error:
            last_error = f"{type(error).__name__}: {error}"
    raise RuntimeError(f"Qwen v1r2 transport failed after {MAX_ATTEMPTS} attempts: {last_error}")


def transport_record(*, stage, payload, response, provider_model):
    record = _base_request_record(stage=stage, payload=payload, response=response, provider_model=provider_model)
    effective = dict(payload); effective["enable_thinking"] = False; effective["max_tokens"] = MAX_TOKENS
    record.update({"effective_request_payload_sha256": base.stable_sha256(effective), "transport_protocol_sha256": v1.sha256_file(PROTOCOL), "enable_thinking": False, "max_tokens": MAX_TOKENS})
    return record


def parallel(items, worker, key, workers):
    counts = {"created": 0, "cached": 0, "failed": 0}; failures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, key, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                _, state = future.result(); counts[state] += 1
            except Exception as error:
                counts["failed"] += 1; failures[str(item if isinstance(item, str) else item.get("request_id", "unknown"))] = f"{type(error).__name__}: {error}"
    return {"counts": counts, "failures": failures}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("seal", "verify", "instruction", "evidence", "all", "status")); parser.add_argument("--workers", type=int, default=8); args = parser.parse_args()
    if args.command == "seal":
        value = seal(); print(json.dumps({"status": value["status"], "sha256": v1.sha256_file(PROTOCOL)}, indent=2)); return 0
    verify()
    if args.command == "verify": print(json.dumps({"status": "MF3ZP_QWEN_TRANSPORT_V1R2_VERIFIED", "sha256": v1.sha256_file(PROTOCOL)}, indent=2)); return 0
    base.api_request = api_request; base.request_record = transport_record
    events = base.read_events(); result = {}; key = None
    if args.command in {"instruction", "all"}:
        key = base.read_api_key(); instructions = sorted({str(event["instruction"]).strip() for event in events}, key=base._instruction_key); result["instruction_run"] = parallel(instructions, base.run_instruction_one, key, args.workers)
    if args.command in {"evidence", "all"}:
        key = key or base.read_api_key(); missing = [event for event in events if not (base.INSTRUCTION_DIR / f"{base._instruction_key(str(event['instruction']))}.json").is_file()]
        if missing: raise RuntimeError(f"instruction preannotation missing for {len(missing)} events")
        result["evidence_run"] = parallel(base.prefix_tasks(events), base.run_evidence_one, key, args.workers)
    result["status"] = base.status(); print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)); return 0 if args.command == "status" or result["status"]["status"] == "MF3ZP_QWEN_PREANNOTATION_READY" else 2


if __name__ == "__main__": raise SystemExit(main())
