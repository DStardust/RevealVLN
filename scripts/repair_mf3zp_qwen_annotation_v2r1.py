#!/usr/bin/env python3
"""Bounded repair for the failed MF3ZP v2 Qwen annotation requests.

MF3ZP v2 is an immutable historical batch.  This script creates a separate
v2r1 repair protocol, retries only the request/model pairs that v2 recorded as
failed, and can build a read-only merged response index.  It never changes the
v2 response files or protocols and never authorizes a public split.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Mapping
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
V2_SCRIPT = ROOT / "scripts/run_mf3zp_qwen_reference_v2.py"
V2_OUTPUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2"
V2_PROTOCOL = V2_OUTPUT / "MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json"
V2_ANNOTATION = V2_OUTPUT / "MF3ZP_QWEN_ANNOTATION_MANIFEST.json"
V2_REQUESTS = V2_OUTPUT / "MF3ZP_ANNOTATION_REQUESTS.jsonl"
OUTPUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2r1"
PROTOCOL = OUTPUT / "MF3ZP_QWEN_UAD_REFERENCE_V2R1_PROTOCOL.json"
REPAIR_MANIFEST = OUTPUT / "MF3ZP_QWEN_REPAIR_MANIFEST.json"
REPAIR_STATUS = OUTPUT / "MF3ZP_QWEN_REPAIR_STATUS.json"
MERGED_MANIFEST = OUTPUT / "MF3ZP_QWEN_ANNOTATION_MERGED_MANIFEST.json"
MERGED_INDEX = OUTPUT / "MF3ZP_QWEN_ANNOTATION_MERGED_INDEX.jsonl"
METHOD_PATH = ROOT / "METHOD_REVISION_3ZP_QWEN_UAD_REFERENCE_V2R1.md"

REVISION = "mf3zp_qwen_uad_reference_v2r1"
SCHEMA = "revealnav-mf3zp-qwen-uad-reference-repair/1"
STATUS = "SEALED_BEFORE_MF3ZP_V2R1_REPAIR_REQUESTS"
RESPONSE_SCHEMA = "revealnav-mf3zp-semantic-reference/1"
MAX_WORKERS = 3
MAX_ATTEMPTS = 6
BACKOFF_SECONDS = (0.0, 2.0, 4.0, 8.0, 16.0, 30.0)
MAX_TOKENS = 800

# This is deliberately a formatting-only addendum.  Semantic questions and
# the response schema remain those of the sealed v2 prompt.
REPAIR_ADDENDUM = """
FORMAT REPAIR (mandatory): output exactly one JSON object with exactly the
schema keys above.  Do not emit markdown, code fences, or extra keys.  The
rationale must be a concise, evidence-grounded string of 160 characters or
fewer and must not be empty.  Keep all boolean, alias, span, and frame-step
values valid under the supplied schema.
""".strip()


class RepairError(RuntimeError):
    pass


def _load_v2():
    spec = importlib.util.spec_from_file_location("mf3zp_qwen_reference_v2_for_repair", V2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RepairError("cannot load sealed v2 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = _load_v2()
m = v2.m


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise RepairError(f"path escapes project: {path}")
    return str(resolved.relative_to(ROOT.resolve()))


def inventory(path: Path) -> dict:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in resolved.parents:
        raise RepairError(f"invalid project file: {path}")
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: object, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise RepairError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise RepairError(f"stale partial output: {partial}")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def atomic_jsonl(path: Path, rows: list[Mapping[str, object]], *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise RepairError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise RepairError(f"stale partial output: {partial}")
    with partial.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(partial, path)


def strict_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RepairError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RepairError(f"JSON object required: {path}")
    return value


def jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RepairError(f"cannot read JSONL: {path}") from error
    rows = []
    for line_no, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except ValueError as error:
            raise RepairError(f"invalid JSONL {path}:{line_no}") from error
        if not isinstance(value, dict):
            raise RepairError(f"JSONL object required: {path}:{line_no}")
        rows.append(value)
    if not rows:
        raise RepairError(f"empty JSONL: {path}")
    return rows


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_").replace("-", "_")


def request_map() -> dict[str, dict]:
    rows = jsonl(V2_REQUESTS)
    result = {str(row["request_id"]): row for row in rows}
    if len(result) != len(rows):
        raise RepairError("duplicate v2 request IDs")
    return result


def response_path(model: str, request_id: str) -> Path:
    return V2_OUTPUT / "responses" / model_slug(model) / f"{request_id}.json"


def response_errors(value: object, row: Mapping[str, object], model: str) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("response_file_not_object",)
    if value.get("status") != "PASS":
        return (str(value.get("error", "response_status_not_pass")),)
    if value.get("model") != model or value.get("request_id") != row["request_id"]:
        return ("response_identity",)
    if value.get("event_id") != row["event_id"] or value.get("prefix_step") != row["prefix_step"]:
        return ("response_prefix_identity",)
    response = value.get("response")
    return tuple(m.validate_annotation_response(
        response,
        event_id=str(row["event_id"]),
        prefix_step=int(row["prefix_step"]),
        allowed_aliases=[item["alias"] for item in row["contract"]["current_candidates"]],
    ))


def classify_parent() -> tuple[dict, list[dict], list[dict], dict[str, set[str]]]:
    if not V2_PROTOCOL.is_file() or V2_PROTOCOL.is_symlink():
        raise RepairError("sealed v2 protocol missing")
    protocol = strict_json(V2_PROTOCOL)
    v2.verify_protocol_v2(protocol)
    parent = strict_json(V2_ANNOTATION)
    if parent.get("status") != "FAIL":
        raise RepairError("v2 annotation manifest is not the recorded failure batch")
    if parent.get("planned") != 1370 or parent.get("response_files") != 1370:
        raise RepairError("unexpected v2 annotation cardinality")
    if parent.get("target_payload_read") is not False or parent.get("outcome_payload_read") is not False:
        raise RepairError("v2 annotation boundary drift")
    requests = request_map()
    models = [str(value) for value in protocol["annotation"]["models"]]
    if len(requests) * len(models) != int(parent["planned"]):
        raise RepairError("v2 request/model cardinality mismatch")
    manifest_pairs: set[tuple[str, str]] = set()
    for failure in parent.get("failures", []):
        pair = (str(failure.get("model")), str(failure.get("request_id")))
        if pair in manifest_pairs:
            raise RepairError("duplicate v2 failure pair")
        manifest_pairs.add(pair)
    actual_pairs: set[tuple[str, str]] = set()
    checks: dict[str, set[str]] = {model: set() for model in models}
    for model in models:
        for request_id, row in requests.items():
            path = response_path(model, request_id)
            if not path.is_file() or path.is_symlink():
                raise RepairError(f"missing v2 response: {path}")
            errors = response_errors(strict_json(path), row, model)
            if errors:
                actual_pairs.add((model, request_id))
                checks[model].add(request_id)
    if actual_pairs != manifest_pairs:
        missing = sorted(actual_pairs - manifest_pairs)
        extra = sorted(manifest_pairs - actual_pairs)
        raise RepairError(f"v2 failure inventory drift (missing={missing[:2]}, extra={extra[:2]})")
    repair_rows = [
        {"model": model, "request_id": request_id, "event_id": requests[request_id]["event_id"],
         "prefix_step": requests[request_id]["prefix_step"]}
        for model, request_id in sorted(actual_pairs)
    ]
    return protocol, list(requests.values()), repair_rows, checks


def build_protocol() -> dict:
    parent, requests, repair_rows, _ = classify_parent()
    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RepairError("cannot read source commit") from error
    return {
        "schema_version": SCHEMA,
        "revision": REVISION,
        "status": STATUS,
        "scientific_scope": "transport/schema repair for provisional MF3ZP v2 responses; no new method or public evaluation",
        "source_commit": source_commit,
        "parent_v2_protocol": inventory(V2_PROTOCOL),
        "parent_v2_annotation_manifest": inventory(V2_ANNOTATION),
        "parent_v2_request_file": inventory(V2_REQUESTS),
        "implementation": {
            "method": inventory(METHOD_PATH),
            "script": inventory(Path(__file__).resolve()),
            "v2_helper": inventory(V2_SCRIPT),
        },
        "population": {
            "request_count": len(requests),
            "models": list(parent["annotation"]["models"]),
            "repair_pair_count": len(repair_rows),
            "repair_pairs": repair_rows,
            "selection": "exactly the failed model/request pairs in the immutable v2 manifest",
        },
        "repair": {
            "response_schema": RESPONSE_SCHEMA,
            "system_prompt_addendum": REPAIR_ADDENDUM,
            "max_workers": MAX_WORKERS,
            "max_attempts": MAX_ATTEMPTS,
            "backoff_seconds": list(BACKOFF_SECONDS),
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "thinking": False,
            "response_format": {"type": "json_object"},
            "existing_v2_pass_responses_read_only": True,
        },
        "boundary": {
            "target_payload_read": False,
            "outcome_payload_read": False,
            "public_split_access": {
                "val_seen": False, "val_unseen": False,
                "test": False, "test_challenge": False,
            },
            "human_verified": False,
            "formal_probe_a_authorized": False,
            "checkpoint_generated": False,
        },
        "no_post_result_revision": True,
    }


def verify_protocol(protocol: Mapping[str, object] | None = None) -> dict:
    value = dict(protocol) if protocol is not None else strict_json(PROTOCOL)
    if value.get("schema_version") != SCHEMA or value.get("revision") != REVISION or value.get("status") != STATUS:
        raise RepairError("v2r1 protocol identity/status drift")
    access = value.get("boundary", {}).get("public_split_access")
    if access != {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}:
        raise RepairError("public split access is not fail-closed")
    if value.get("boundary", {}).get("target_payload_read") is not False or value.get("boundary", {}).get("outcome_payload_read") is not False:
        raise RepairError("v2r1 boundary drift")
    parent, requests, repair_rows, _ = classify_parent()
    expected_parent = value.get("parent_v2_protocol")
    if expected_parent != inventory(V2_PROTOCOL):
        raise RepairError("v2 parent protocol changed")
    if value.get("parent_v2_annotation_manifest") != inventory(V2_ANNOTATION):
        raise RepairError("v2 annotation manifest changed")
    if value.get("parent_v2_request_file") != inventory(V2_REQUESTS):
        raise RepairError("v2 request file changed")
    impl = value.get("implementation", {})
    for key, path in (("method", METHOD_PATH), ("script", Path(__file__).resolve()), ("v2_helper", V2_SCRIPT)):
        if impl.get(key) != inventory(path):
            raise RepairError(f"sealed implementation changed: {path}")
    pop = value.get("population", {})
    if int(pop.get("request_count", -1)) != len(requests) or int(pop.get("repair_pair_count", -1)) != len(repair_rows):
        raise RepairError("v2r1 population cardinality drift")
    if pop.get("repair_pairs") != repair_rows:
        raise RepairError("v2r1 repair pair inventory drift")
    repair = value.get("repair", {})
    if repair.get("response_schema") != RESPONSE_SCHEMA or repair.get("system_prompt_addendum") != REPAIR_ADDENDUM:
        raise RepairError("repair semantic/schema contract drift")
    if repair.get("max_workers") != MAX_WORKERS or repair.get("max_attempts") != MAX_ATTEMPTS:
        raise RepairError("repair transport constants drift")
    return value


def _image_data(path: Path) -> str:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in resolved.parents:
        raise RepairError(f"invalid annotation image: {path}")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _parse_message(body: dict) -> dict:
    message = body["choices"][0]["message"]["content"]
    if isinstance(message, list):
        message = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in message
        )
    parsed = json.loads(str(message))
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    return parsed


def _request_once(api_key: str, model: str, row: Mapping[str, object]) -> dict:
    content = [
        {"type": "text", "text": str(row["user_text"])},
        {"type": "image_url", "image_url": {"url": _image_data(m.ROOT / str(row["causal_storyboard"]["path"]))}},
        {"type": "image_url", "image_url": {"url": _image_data(m.ROOT / str(row["current_panorama"]["path"]))}},
    ]
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": m.PROMPT.read_text(encoding="utf-8") + "\n\n" + REPAIR_ADDENDUM},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_TOKENS,
        "enable_thinking": False,
    }
    request = urllib.request.Request(
        m.ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    return {"response": _parse_message(body), "provider_model": body.get("model")}


def _repair_one(api_key: str, model: str, row: Mapping[str, object]) -> dict:
    errors: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
        try:
            result = _request_once(api_key, model, row)
            response = result["response"]
            validation = m.validate_annotation_response(
                response,
                event_id=str(row["event_id"]),
                prefix_step=int(row["prefix_step"]),
                allowed_aliases=[item["alias"] for item in row["contract"]["current_candidates"]],
            )
            if validation:
                errors.append("schema:" + ",".join(validation))
                continue
            return {
                "schema_version": "revealnav-mf3zp-qwen-response/1",
                "status": "PASS",
                "model": model,
                "request_id": row["request_id"],
                "event_id": row["event_id"],
                "prefix_step": row["prefix_step"],
                "provider_model": result.get("provider_model"),
                "response": response,
                "attempts": attempt,
                "retry_errors": errors,
            }
        except urllib.error.HTTPError as error:
            errors.append(f"HTTP_{error.code}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as error:
            errors.append(type(error).__name__)
    return {
        "schema_version": "revealnav-mf3zp-qwen-response/1",
        "status": "FAIL",
        "model": model,
        "request_id": row["request_id"],
        "event_id": row["event_id"],
        "prefix_step": row["prefix_step"],
        "error": errors[-1] if errors else "repair_failed",
        "attempts": MAX_ATTEMPTS,
        "retry_errors": errors,
    }


_STATUS_LOCK = threading.Lock()


def _write_status(protocol: Mapping[str, object], summaries: Mapping[tuple[str, str], dict], *, final: bool = False) -> dict:
    pairs = [tuple((str(x["model"]), str(x["request_id"]))) for x in protocol["population"]["repair_pairs"]]
    done = [pair for pair in pairs if pair in summaries]
    passed = [pair for pair in done if summaries[pair].get("status") == "PASS"]
    failed = [pair for pair in done if summaries[pair].get("status") != "PASS"]
    value = {
        "schema_version": "revealnav-mf3zp-qwen-repair-status/1",
        "status": "PASS" if final and len(passed) == len(pairs) else "RUNNING" if not final else "FAIL",
        "planned": len(pairs),
        "completed": len(done),
        "pass": len(passed),
        "fail": len(failed),
        "remaining": len(pairs) - len(done),
        "target_payload_read": False,
        "outcome_payload_read": False,
        "public_split_access": False,
        "summaries": [summaries[pair] for pair in sorted(done)],
    }
    with _STATUS_LOCK:
        atomic_json(REPAIR_STATUS, value)
    return value


def repair(protocol: Mapping[str, object]) -> dict:
    verify_protocol(protocol)
    api_key = m._api_key()
    requests = request_map()
    pairs = [(str(item["model"]), str(item["request_id"])) for item in protocol["population"]["repair_pairs"]]
    response_root = OUTPUT / "responses"
    summaries: dict[tuple[str, str], dict] = {}
    for model, request_id in pairs:
        path = response_root / model_slug(model) / f"{request_id}.json"
        if path.is_file() and not path.is_symlink():
            try:
                existing = strict_json(path)
                if existing.get("status") == "PASS" and not response_errors(existing, requests[request_id], model):
                    summaries[(model, request_id)] = existing
                elif existing.get("status") == "FAIL":
                    summaries[(model, request_id)] = existing
            except RepairError:
                pass
    _write_status(protocol, summaries)

    todo = [pair for pair in pairs if summaries.get(pair, {}).get("status") == "PASS" or pair not in summaries]
    # Existing FAIL files are deliberately retried; a PASS is the only
    # resumable terminal state.
    todo = [pair for pair in pairs if summaries.get(pair, {}).get("status") != "PASS"]

    def run(pair: tuple[str, str]) -> tuple[tuple[str, str], dict]:
        model, request_id = pair
        result = _repair_one(api_key, model, requests[request_id])
        path = response_root / model_slug(model) / f"{request_id}.json"
        atomic_json(path, result)
        return pair, result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(run, pair) for pair in todo]
        for future in as_completed(futures):
            pair, result = future.result()
            summaries[pair] = result
            _write_status(protocol, summaries)
    status = _write_status(protocol, summaries, final=True)
    failures = [value for value in status["summaries"] if value.get("status") != "PASS"]
    manifest = {
        "schema_version": "revealnav-mf3zp-qwen-repair/1",
        "status": "PASS" if not failures and status["completed"] == status["planned"] else "FAIL",
        "parent_v2_protocol": inventory(V2_PROTOCOL),
        "parent_v2_annotation_manifest": inventory(V2_ANNOTATION),
        "planned": status["planned"], "completed": status["completed"],
        "pass": status["pass"], "failures": failures,
        "target_payload_read": False, "outcome_payload_read": False,
        "public_split_access": False,
    }
    atomic_json(REPAIR_MANIFEST, manifest)
    return manifest


def _valid_v2_response(model: str, request_id: str, row: Mapping[str, object]) -> tuple[Path, tuple[str, ...]]:
    path = response_path(model, request_id)
    if not path.is_file() or path.is_symlink():
        return path, ("missing",)
    value = strict_json(path)
    return path, response_errors(value, row, model)


def _valid_repair_response(model: str, request_id: str, row: Mapping[str, object]) -> tuple[Path, tuple[str, ...]]:
    path = OUTPUT / "responses" / model_slug(model) / f"{request_id}.json"
    if not path.is_file() or path.is_symlink():
        return path, ("missing",)
    value = strict_json(path)
    return path, response_errors(value, row, model)


def merge(protocol: Mapping[str, object]) -> dict:
    verify_protocol(protocol)
    manifest = strict_json(REPAIR_MANIFEST)
    if manifest.get("status") != "PASS":
        raise RepairError("repair manifest is not PASS")
    requests = request_map()
    models = [str(value) for value in protocol["population"]["models"]]
    index_rows: list[dict] = []
    failures: list[dict] = []
    for model in models:
        for request_id, row in sorted(requests.items()):
            old_path, old_errors = _valid_v2_response(model, request_id, row)
            if not old_errors:
                index_rows.append({"model": model, "request_id": request_id, "source": "v2", "path": rel(old_path), "sha256": sha256_file(old_path)})
                continue
            new_path, new_errors = _valid_repair_response(model, request_id, row)
            if new_errors:
                failures.append({"model": model, "request_id": request_id, "v2_errors": list(old_errors), "repair_errors": list(new_errors)})
                continue
            index_rows.append({"model": model, "request_id": request_id, "source": "v2r1_repair", "path": rel(new_path), "sha256": sha256_file(new_path)})
    if failures:
        raise RepairError(f"merged response set incomplete: {len(failures)} failures")
    atomic_jsonl(MERGED_INDEX, index_rows, refuse_existing=False)
    value = {
        "schema_version": "revealnav-mf3zp-qwen-merged/1",
        "status": "PASS",
        "models": models,
        "planned": len(requests) * len(models),
        "pass": len(index_rows),
        "source_counts": dict(Counter(row["source"] for row in index_rows)),
        "index": inventory(MERGED_INDEX),
        "parent_v2_protocol": inventory(V2_PROTOCOL),
        "repair_manifest": inventory(REPAIR_MANIFEST),
        "target_payload_read": False, "outcome_payload_read": False,
        "public_split_access": False,
    }
    atomic_json(MERGED_MANIFEST, value, refuse_existing=False)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal")
    sub.add_parser("repair")
    sub.add_parser("merge")
    args = parser.parse_args()
    try:
        if args.command == "seal":
            if PROTOCOL.exists() or PROTOCOL.is_symlink():
                raise RepairError("v2r1 protocol already exists; resealing is forbidden")
            OUTPUT.mkdir(parents=True, exist_ok=True)
            value = build_protocol()
            atomic_json(PROTOCOL, value, refuse_existing=True)
            print(json.dumps({
                "status": value["status"],
                "protocol_sha256": sha256_file(PROTOCOL),
                "repair_pairs": value["population"]["repair_pair_count"],
            }, indent=2))
        elif args.command == "repair":
            print(json.dumps(repair(verify_protocol()), indent=2, ensure_ascii=False))
        elif args.command == "merge":
            print(json.dumps(merge(verify_protocol()), indent=2, ensure_ascii=False))
        return 0
    except BaseException as error:
        print(f"MF3ZP_V2R1_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
