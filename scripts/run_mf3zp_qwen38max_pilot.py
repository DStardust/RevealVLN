#!/usr/bin/env python3
"""Outcome-blind exploratory Qwen3.8-Max annotation pilot.

This pilot is intentionally independent of the sealed MF3ZP v2/v2r1
annotation manifests.  It queries a deterministic 20-event subset, validates
the unchanged semantic response schema, and reports provisional U/A/D
readiness without opening exact outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.error


ROOT = Path(__file__).resolve().parents[1]
V2_SCRIPT = ROOT / "scripts/run_mf3zp_qwen_reference_v2.py"
REPAIR_SCRIPT = ROOT / "scripts/repair_mf3zp_qwen_annotation_v2r1.py"
V2_OUTPUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2"
REPAIR_OUTPUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2r1"
V2_PROTOCOL = V2_OUTPUT / "MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json"
REPAIR_PROTOCOL = REPAIR_OUTPUT / "MF3ZP_QWEN_UAD_REFERENCE_V2R1_PROTOCOL.json"
V2_REQUESTS = V2_OUTPUT / "MF3ZP_ANNOTATION_REQUESTS.jsonl"
OUTPUT = ROOT / "artifacts/training/mf3zp_qwen38max_pilot_v1"
PROTOCOL = OUTPUT / "MF3ZP_QWEN38MAX_PILOT_PROTOCOL.json"
STATUS = OUTPUT / "MF3ZP_QWEN38MAX_PILOT_STATUS.json"
MANIFEST = OUTPUT / "MF3ZP_QWEN38MAX_PILOT_MANIFEST.json"
LABEL_AUDIT = OUTPUT / "MF3ZP_QWEN38MAX_LABEL_AUDIT.json"
METHOD = ROOT / "METHOD_REVISION_3ZP_QWEN38MAX_PILOT.md"

REVISION = "mf3zp_qwen38max_pilot_v1"
SCHEMA = "revealnav-mf3zp-qwen38max-pilot/1"
MODEL = "qwen3.8-max"
EVENTS_PER_DOMAIN = 10
MAX_WORKERS = 2
MAX_ATTEMPTS = 5
BACKOFF = (0.0, 2.0, 4.0, 8.0, 16.0)


class PilotError(RuntimeError):
    pass


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PilotError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = load_module(V2_SCRIPT, "mf3zp_qwen38max_v2")
repair = load_module(REPAIR_SCRIPT, "mf3zp_qwen38max_repair")
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
        raise PilotError(f"path escapes project: {path}")
    return str(resolved.relative_to(ROOT.resolve()))


def inventory(path: Path) -> dict:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in resolved.parents:
        raise PilotError(f"invalid project file: {path}")
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: object, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise PilotError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise PilotError(f"stale partial output: {partial}")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PilotError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise PilotError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except ValueError as error:
            raise PilotError(f"invalid JSONL {path}:{line_no}") from error
        if not isinstance(value, dict):
            raise PilotError(f"JSONL object required: {path}:{line_no}")
        rows.append(value)
    if not rows:
        raise PilotError(f"empty JSONL: {path}")
    return rows


def select_events(parent: dict) -> list[dict]:
    events = [dict(row) for row in parent["population"]["events"]]
    selected: list[dict] = []
    for domain in ("R2R", "RxR"):
        candidates = [row for row in events if str(row["dataset"]) == domain]
        by_scene: dict[str, list[dict]] = defaultdict(list)
        for row in candidates:
            by_scene[str(row["scene_id"])].append(row)
        for rows in by_scene.values():
            rows.sort(key=lambda row: (hashlib.sha256(str(row["event_id"]).encode()).hexdigest(), str(row["event_id"])))
        scene_order = sorted(by_scene, key=lambda scene: (hashlib.sha256(scene.encode()).hexdigest(), scene))
        picked: list[dict] = []
        # First pass gives the fixed pilot broad scene coverage.
        for scene in scene_order:
            if by_scene[scene]:
                picked.append(by_scene[scene][0])
                if len(picked) == EVENTS_PER_DOMAIN:
                    break
        if len(picked) < EVENTS_PER_DOMAIN:
            remaining = [row for scene in scene_order for row in by_scene[scene][1:]]
            remaining.sort(key=lambda row: (hashlib.sha256(str(row["event_id"]).encode()).hexdigest(), str(row["event_id"])))
            picked.extend(remaining[: EVENTS_PER_DOMAIN - len(picked)])
        if len(picked) != EVENTS_PER_DOMAIN:
            raise PilotError(f"insufficient {domain} events")
        selected.extend(picked)
    selected.sort(key=lambda row: (str(row["dataset"]), hashlib.sha256(str(row["event_id"]).encode()).hexdigest(), str(row["event_id"])))
    if len({row["event_id"] for row in selected}) != 2 * EVENTS_PER_DOMAIN:
        raise PilotError("pilot event identity collision")
    return selected


def build_requests(events: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    event_limits = {str(row["event_id"]): int(row["decision_step"]) for row in events}
    all_requests = read_jsonl(V2_REQUESTS)
    selected_ids = set(event_limits)
    requests = [
        row for row in all_requests
        if str(row["event_id"]) in selected_ids and int(row["prefix_step"]) <= event_limits[str(row["event_id"])]
    ]
    by_event: dict[str, list[int]] = defaultdict(list)
    for row in requests:
        by_event[str(row["event_id"])].append(int(row["prefix_step"]))
    for event_id, limit in event_limits.items():
        if sorted(by_event[event_id]) != list(range(limit + 1)):
            raise PilotError(f"incomplete event-local request prefix: {event_id}")
    projection = [row for row in read_jsonl(V2_OUTPUT / "MF3ZP_PROJECTION_MAP.jsonl") if str(row["event_id"]) in selected_ids]
    deterministic = [row for row in read_jsonl(V2_OUTPUT / "MF3ZP_DETERMINISTIC_ORACLE.jsonl") if str(row["event_id"]) in selected_ids and int(row["prefix_step"]) <= event_limits[str(row["event_id"])] ]
    return requests, projection, deterministic


def build_protocol() -> dict:
    parent = read_json(V2_PROTOCOL)
    v2.verify_protocol_v2(parent)
    repair.verify_protocol(read_json(REPAIR_PROTOCOL))
    events = select_events(parent)
    requests, _, _ = build_requests(events)
    return {
        "schema_version": SCHEMA,
        "revision": REVISION,
        "status": "SEALED_BEFORE_QWEN38MAX_RESPONSES",
        "parent_v2_protocol": inventory(V2_PROTOCOL),
        "parent_v2r1_protocol": inventory(REPAIR_PROTOCOL),
        "request_source": inventory(V2_REQUESTS),
        "method": inventory(METHOD),
        "script": inventory(Path(__file__).resolve()),
        "model": MODEL,
        "population": {
            "event_count": len(events),
            "events_per_domain": EVENTS_PER_DOMAIN,
            "events": events,
            "request_count": len(requests),
            "request_ids_sha256": m.stable_sha256([row["request_id"] for row in requests]),
        },
        "annotation": {
            "response_schema": "revealnav-mf3zp-semantic-reference/1",
            "system_prompt": inventory(m.PROMPT),
            "format_addendum": repair.REPAIR_ADDENDUM,
            "temperature": 0.0,
            "thinking": False,
            "response_format": {"type": "json_object"},
            "max_tokens": 800,
            "max_workers": MAX_WORKERS,
            "max_attempts": MAX_ATTEMPTS,
            "backoff_seconds": list(BACKOFF),
        },
        "boundary": {
            "target_payload_read": False,
            "outcome_payload_read": False,
            "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
            "human_verified": False,
            "formal_probe_a_authorized": False,
            "checkpoint_generated": False,
        },
        "selection_rule": "fixed identity/hash order, 10 events per domain, no outcome access",
    }


def verify_protocol(protocol: dict | None = None) -> dict:
    value = protocol if protocol is not None else read_json(PROTOCOL)
    if value.get("schema_version") != SCHEMA or value.get("revision") != REVISION or value.get("status") != "SEALED_BEFORE_QWEN38MAX_RESPONSES":
        raise PilotError("pilot protocol identity/status drift")
    if value.get("model") != MODEL:
        raise PilotError("pilot model drift")
    if value.get("boundary", {}).get("public_split_access") != {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}:
        raise PilotError("public split access is not fail-closed")
    if value.get("boundary", {}).get("target_payload_read") is not False or value.get("boundary", {}).get("outcome_payload_read") is not False:
        raise PilotError("pilot boundary drift")
    if value.get("parent_v2_protocol") != inventory(V2_PROTOCOL) or value.get("parent_v2r1_protocol") != inventory(REPAIR_PROTOCOL) or value.get("request_source") != inventory(V2_REQUESTS):
        raise PilotError("parent source drift")
    if value.get("method") != inventory(METHOD) or value.get("script") != inventory(Path(__file__).resolve()):
        raise PilotError("pilot implementation drift")
    parent = read_json(V2_PROTOCOL)
    v2.verify_protocol_v2(parent)
    repair.verify_protocol(read_json(REPAIR_PROTOCOL))
    events = select_events(parent)
    requests, _, _ = build_requests(events)
    population = value.get("population", {})
    if population.get("events") != events or population.get("request_count") != len(requests) or population.get("request_ids_sha256") != m.stable_sha256([row["request_id"] for row in requests]):
        raise PilotError("pilot population drift")
    annotation = value.get("annotation", {})
    if annotation.get("response_schema") != "revealnav-mf3zp-semantic-reference/1" or annotation.get("format_addendum") != repair.REPAIR_ADDENDUM:
        raise PilotError("pilot annotation contract drift")
    return value


def response_errors(value: object, row: Mapping[str, object]) -> tuple[str, ...]:
    if not isinstance(value, Mapping) or value.get("status") != "PASS":
        return ("status_not_pass",)
    if value.get("model") != MODEL or value.get("request_id") != row["request_id"] or value.get("event_id") != row["event_id"] or value.get("prefix_step") != row["prefix_step"]:
        return ("identity",)
    return tuple(m.validate_annotation_response(
        value.get("response"), event_id=str(row["event_id"]), prefix_step=int(row["prefix_step"]),
        allowed_aliases=[item["alias"] for item in row["contract"]["current_candidates"]],
    ))


def run_one(api_key: str, row: dict) -> dict:
    errors: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)])
        try:
            result = repair._request_once(api_key, MODEL, row)
            response = result["response"]
            validation = m.validate_annotation_response(
                response, event_id=str(row["event_id"]), prefix_step=int(row["prefix_step"]),
                allowed_aliases=[item["alias"] for item in row["contract"]["current_candidates"]],
            )
            if validation:
                errors.append("schema:" + ",".join(validation))
                continue
            return {
                "schema_version": "revealnav-mf3zp-qwen-response/1", "status": "PASS",
                "model": MODEL, "request_id": row["request_id"], "event_id": row["event_id"],
                "prefix_step": row["prefix_step"], "provider_model": result.get("provider_model"),
                "response": response, "attempts": attempt, "retry_errors": errors,
            }
        except urllib.error.HTTPError as error:
            errors.append(f"HTTP_{error.code}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as error:
            errors.append(type(error).__name__)
    return {
        "schema_version": "revealnav-mf3zp-qwen-response/1", "status": "FAIL", "model": MODEL,
        "request_id": row["request_id"], "event_id": row["event_id"], "prefix_step": row["prefix_step"],
        "error": errors[-1] if errors else "pilot_request_failed", "attempts": MAX_ATTEMPTS,
        "retry_errors": errors,
    }


def run(protocol: dict) -> dict:
    verify_protocol(protocol)
    requests, _, _ = build_requests(select_events(read_json(V2_PROTOCOL)))
    api_key = m._api_key()
    response_root = OUTPUT / "responses" / MODEL.replace(".", "_").replace("-", "_")
    summaries: dict[str, dict] = {}
    for row in requests:
        path = response_root / f"{row['request_id']}.json"
        if path.is_file() and not path.is_symlink():
            try:
                existing = read_json(path)
                if not response_errors(existing, row):
                    summaries[row["request_id"]] = existing
            except PilotError:
                pass
    def write_status(final: bool = False) -> None:
        passed = sum(value.get("status") == "PASS" for value in summaries.values())
        failed = sum(value.get("status") != "PASS" for value in summaries.values())
        atomic_json(STATUS, {
            "schema_version": "revealnav-mf3zp-qwen38max-status/1",
            "status": "PASS" if final and passed == len(requests) else "FAIL" if final else "RUNNING",
            "planned": len(requests), "completed": len(summaries), "pass": passed, "fail": failed,
            "remaining": len(requests) - len(summaries), "target_payload_read": False,
            "outcome_payload_read": False, "public_split_access": False,
        })
    write_status()
    todo = [row for row in requests if row["request_id"] not in summaries]
    def worker(row: dict):
        result = run_one(api_key, row)
        path = response_root / f"{row['request_id']}.json"
        atomic_json(path, result)
        return row["request_id"], result
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(worker, row) for row in todo]
        for future in as_completed(futures):
            request_id, result = future.result()
            summaries[request_id] = result
            write_status()
    write_status(final=True)
    failures = [value for value in summaries.values() if value.get("status") != "PASS"]
    result = {
        "schema_version": "revealnav-mf3zp-qwen38max-manifest/1",
        "status": "PASS" if not failures and len(summaries) == len(requests) else "FAIL",
        "model": MODEL, "planned": len(requests), "completed": len(summaries),
        "pass": len(summaries) - len(failures), "failures": failures,
        "target_payload_read": False, "outcome_payload_read": False, "public_split_access": False,
    }
    atomic_json(MANIFEST, result)
    return result


def audit_labels(protocol: dict) -> dict:
    verify_protocol(protocol)
    manifest = read_json(MANIFEST)
    if manifest.get("status") != "PASS":
        raise PilotError("Qwen3.8-Max pilot responses are incomplete")
    parent = read_json(V2_PROTOCOL)
    events = select_events(parent)
    requests, projection_rows, deterministic_rows = build_requests(events)
    projection = {row["event_id"]: row for row in projection_rows}
    deterministic: dict[str, list[dict]] = defaultdict(list)
    for row in deterministic_rows:
        deterministic[row["event_id"]].append(row)
    response_root = OUTPUT / "responses" / MODEL.replace(".", "_").replace("-", "_")
    by_event: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        path = response_root / f"{request['request_id']}.json"
        value = read_json(path)
        errors = response_errors(value, request)
        if errors:
            raise PilotError(f"invalid stored response: {path} ({errors})")
        target_present = next(bool(row["target_in_set"]) for row in deterministic[request["event_id"]] if int(row["prefix_step"]) == int(request["prefix_step"]))
        state = m.derive_semantic_state(value["response"], target_alias=projection[request["event_id"]]["alternative_alias"], native_alias=projection[request["event_id"]]["native_alias"], target_present=target_present)
        by_event[request["event_id"]].append({"prefix_step": int(request["prefix_step"]), **state})
    event_results = {}
    for event_id, rows in by_event.items():
        rows.sort(key=lambda row: row["prefix_step"])
        target = tuple(bool(row["target_in_set"]) for row in deterministic[event_id])
        separated = tuple(bool(row["candidate_separated"]) for row in rows)
        closed = tuple(bool(row["evidence_closed"]) for row in rows)
        states = m.derive_uad(target, separated, closed, stability_k=m.STABILITY_K)
        reveal = m.reveal_interval(states)
        expiry = max((int(row["prefix_step"]) for row in deterministic[event_id] if row["target_in_set"]), default=None)
        event_results[event_id] = {"final_state": states[-1], "states": list(states), "reveal_interval": reveal, "expiry_step": expiry, "complete": reveal is not None and expiry is not None, "any_evidence_closed": any(closed)}
    result = {
        "schema_version": "revealnav-mf3zp-qwen38max-label-audit/1",
        "status": "LABEL_READINESS_PASS" if all(row["complete"] for row in event_results.values()) else "LABEL_READINESS_INCOMPLETE",
        "model": MODEL, "events": len(event_results),
        "domain_counts": dict(Counter(row["dataset"] for row in events)),
        "final_state_counts": dict(Counter(row["final_state"] for row in event_results.values())),
        "complete_events": sum(row["complete"] for row in event_results.values()),
        "events_with_any_evidence_closed": sum(row["any_evidence_closed"] for row in event_results.values()),
        "event_results": event_results,
        "target_payload_read": False, "outcome_payload_read": False, "public_split_access": False,
        "scientific_probe_executed": False,
    }
    atomic_json(LABEL_AUDIT, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal")
    sub.add_parser("run")
    sub.add_parser("audit-labels")
    args = parser.parse_args()
    try:
        if args.command == "seal":
            if PROTOCOL.exists() or PROTOCOL.is_symlink():
                raise PilotError("pilot protocol already exists; resealing is forbidden")
            OUTPUT.mkdir(parents=True, exist_ok=True)
            value = build_protocol()
            atomic_json(PROTOCOL, value, refuse_existing=True)
            print(json.dumps({"status": value["status"], "protocol_sha256": sha256_file(PROTOCOL), "events": value["population"]["event_count"], "requests": value["population"]["request_count"]}, indent=2))
        elif args.command == "run":
            print(json.dumps(run(verify_protocol()), indent=2, ensure_ascii=False))
        else:
            print(json.dumps(audit_labels(verify_protocol()), indent=2, ensure_ascii=False))
        return 0
    except BaseException as error:
        print(f"MF3ZP_QWEN38MAX_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
