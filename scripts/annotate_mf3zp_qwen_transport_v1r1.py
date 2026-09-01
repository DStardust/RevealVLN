#!/usr/bin/env python3
"""Execution-only Qwen transport revision for sealed MF3ZP RevealSkill.

This file changes no prompt, model, population, label schema, or scientific
gate.  It only makes two provider transport controls explicit:
``enable_thinking=False`` and ``max_tokens=2000``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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
BASE_PATH = ROOT / "scripts/annotate_mf3zp_qwen.py"
SCIENCE_PROTOCOL = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_REVEALSKILL_PROTOCOL.json"
TRANSPORT_PROTOCOL = ROOT / "artifacts/training/mf3zp_revealskill_v1/MF3ZP_QWEN_TRANSPORT_V1R1_PROTOCOL.json"
MODEL = "qwen3.8-max"
MAX_TOKENS = 2000
MAX_ATTEMPTS = 4
TIMEOUT_SECONDS = 180
BACKOFF = (0.0, 2.0, 5.0, 10.0)


def _load_base():
    spec = importlib.util.spec_from_file_location("mf3zp_qwen_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sealed base annotation entrypoint")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()
_base_request_record = base.request_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in path.resolve().parents:
        raise RuntimeError(f"invalid project-local file: {path}")
    return {"path": str(path.resolve().relative_to(ROOT.resolve())), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def protocol_value() -> dict[str, object]:
    science = base.verify_protocol()
    return {
        "schema_version": "revealnav-mf3zp-qwen-transport/1",
        "revision": "mf3zp_revealskill_qwen_transport_v1r1",
        "status": "SEALED_BEFORE_QWEN_TRANSPORT_V1R1_RESULTS",
        "science_protocol": inventory(SCIENCE_PROTOCOL),
        "base_annotation_entrypoint": inventory(BASE_PATH),
        "transport_entrypoint": inventory(Path(__file__).resolve()),
        "model": MODEL,
        "prompt_hashes": {
            "instruction": science["qwen_annotation"]["instruction_prompt_sha256"],
            "evidence": science["qwen_annotation"]["evidence_prompt_sha256"],
        },
        "only_changes": {"enable_thinking": False, "max_tokens": MAX_TOKENS},
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_attempts": MAX_ATTEMPTS,
        "backoff_seconds": list(BACKOFF),
        "model_fallback": False,
        "prompt_change": False,
        "schema_change": False,
        "population_change": False,
        "scientific_gate_change": False,
        "target_payload_read": False,
        "outcome_payload_read": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }


def seal() -> dict[str, object]:
    value = protocol_value()
    if TRANSPORT_PROTOCOL.exists() or TRANSPORT_PROTOCOL.is_symlink():
        raise RuntimeError("refusing to overwrite transport protocol")
    partial = TRANSPORT_PROTOCOL.with_name(TRANSPORT_PROTOCOL.name + ".part")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, TRANSPORT_PROTOCOL)
    return value


def verify() -> dict[str, object]:
    value = json.loads(TRANSPORT_PROTOCOL.read_text(encoding="utf-8"))
    expected = protocol_value()
    if value != expected:
        raise RuntimeError("transport protocol/source drift")
    return value


def api_request(api_key: str, payload: dict[str, object]):
    revised = dict(payload)
    revised["enable_thinking"] = False
    revised["max_tokens"] = MAX_TOKENS
    body = json.dumps(revised, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    last_error = "unattempted"
    for attempt in range(MAX_ATTEMPTS):
        if BACKOFF[attempt]:
            time.sleep(BACKOFF[attempt])
        request = urllib.request.Request(
            base.QWEN_ENDPOINT, data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                envelope = json.loads(response.read())
            provider_model = str(envelope.get("model", ""))
            if provider_model != MODEL:
                raise RuntimeError(f"provider model drift: {provider_model!r}")
            content = envelope["choices"][0]["message"]["content"]
            return json.loads(content), provider_model, str(envelope.get("id", ""))
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, RuntimeError) as error:
            last_error = f"{type(error).__name__}: {error}"
    raise RuntimeError(f"Qwen transport failed after {MAX_ATTEMPTS} attempts: {last_error}")


def transport_request_record(*, stage: str, payload: dict[str, object], response: object, provider_model: str):
    record = _base_request_record(stage=stage, payload=payload, response=response, provider_model=provider_model)
    effective = dict(payload)
    effective["enable_thinking"] = False
    effective["max_tokens"] = MAX_TOKENS
    record["effective_request_payload_sha256"] = base.stable_sha256(effective)
    record["transport_protocol_sha256"] = sha256_file(TRANSPORT_PROTOCOL)
    record["enable_thinking"] = False
    record["max_tokens"] = MAX_TOKENS
    return record


def run_parallel(items, worker, api_key: str, workers: int) -> dict[str, object]:
    counts = {"created": 0, "cached": 0, "failed": 0}
    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, api_key, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                _, state = future.result()
                counts[state] += 1
            except Exception as error:
                counts["failed"] += 1
                identity = item if isinstance(item, str) else item.get("request_id", "unknown")
                failures[str(identity)] = f"{type(error).__name__}: {error}"
    return {"counts": counts, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "verify", "instruction", "evidence", "all", "status"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "seal":
        result = seal()
        print(json.dumps({"status": result["status"], "sha256": sha256_file(TRANSPORT_PROTOCOL)}, indent=2))
        return 0
    verify()
    if args.command == "verify":
        print(json.dumps({"status": "MF3ZP_QWEN_TRANSPORT_V1R1_VERIFIED", "sha256": sha256_file(TRANSPORT_PROTOCOL)}, indent=2))
        return 0
    base.api_request = api_request
    base.request_record = transport_request_record
    events = base.read_events()
    result: dict[str, object] = {}
    api_key = None
    if args.command in {"instruction", "all"}:
        api_key = base.read_api_key()
        instructions = sorted({str(event["instruction"]).strip() for event in events}, key=base._instruction_key)
        result["instruction_run"] = run_parallel(instructions, base.run_instruction_one, api_key, args.workers)
    if args.command in {"evidence", "all"}:
        api_key = api_key or base.read_api_key()
        missing = [event for event in events if not (base.INSTRUCTION_DIR / f"{base._instruction_key(str(event['instruction']))}.json").is_file()]
        if missing:
            raise RuntimeError(f"instruction preannotation missing for {len(missing)} events")
        result["evidence_run"] = run_parallel(base.prefix_tasks(events), base.run_evidence_one, api_key, args.workers)
    result["status"] = base.status()
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if args.command == "status" or result["status"]["status"] == "MF3ZP_QWEN_PREANNOTATION_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
