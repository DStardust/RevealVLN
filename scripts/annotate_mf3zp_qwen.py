#!/usr/bin/env python3
"""Resumable, outcome-blind Qwen preannotation for MF3ZP RevealSkill."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.evidence_constraints import InstructionEvidenceGraph  # noqa: E402
from revealnav_mf3.qwen_evidence_annotation import (  # noqa: E402
    QWEN_ENDPOINT,
    QWEN_MODEL,
    evidence_request,
    instruction_request,
    parse_instruction_response,
    request_record,
    stable_sha256,
    validate_evidence_response,
)
from revealnav_mf3.revealskill_protocol import OUTPUT, PROTOCOL_PATH, verify_protocol  # noqa: E402


EVENTS = OUTPUT / "MF3ZP_REVEAL_EVENTS.jsonl"
SOURCE_REQUESTS = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2/MF3ZP_ANNOTATION_REQUESTS.jsonl"
INSTRUCTION_DIR = OUTPUT / "qwen_preannotations/instruction"
EVIDENCE_DIR = OUTPUT / "qwen_preannotations/evidence"
STATUS = OUTPUT / "MF3ZP_QWEN_PREANNOTATION_STATUS.json"
KEY_PATH = ROOT / ".secret/qwen_api_key"
MAX_WORKERS = 4
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = (0.0, 2.0, 5.0, 10.0)


class AnnotationError(RuntimeError):
    pass


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise AnnotationError(f"refusing to overwrite annotation: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise AnnotationError(f"stale annotation partial: {partial}")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def replace_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        partial.unlink()
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnnotationError(f"JSON object required: {path}")
    return value


def read_events() -> list[dict[str, object]]:
    values = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
    if len(values) != 300 or any(not isinstance(value, dict) for value in values):
        raise AnnotationError("sealed pilot event population drift")
    return values


def read_source_requests() -> list[dict[str, object]]:
    values = [json.loads(line) for line in SOURCE_REQUESTS.read_text(encoding="utf-8").splitlines()]
    if not values or any(not isinstance(value, dict) for value in values):
        raise AnnotationError("source causal request population drift")
    return values


def prefix_tasks(events: list[dict[str, object]]) -> list[dict[str, object]]:
    maxima: dict[tuple[str, str, str, str], int] = {}
    event_links: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for event in events:
        key = (str(event["dataset"]), str(event["scene_id"]), str(event["episode_id"]), str(event["source_observation_stream_id"]))
        maxima[key] = max(maxima.get(key, -1), int(event["prefix_end"]))
        event_links.setdefault(key, []).append(event)
    tasks: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in read_source_requests():
        key = (str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]), str(row["event_id"]))
        if key not in maxima or int(row["prefix_step"]) > maxima[key]:
            continue
        request_id = str(row["request_id"])
        if request_id in seen:
            raise AnnotationError("duplicate source request identity")
        seen.add(request_id)
        task = dict(row)
        task["pilot_event_ids"] = sorted(
            str(event["event_id"])
            for event in event_links[key]
            if int(row["prefix_step"]) <= int(event["prefix_end"])
        )
        tasks.append(task)
    tasks.sort(key=lambda item: (str(item["dataset"]), str(item["scene_id"]), str(item["episode_id"]), int(item["prefix_step"])))
    expected = {
        (str(event["dataset"]), str(event["scene_id"]), str(event["episode_id"]), str(event["source_observation_stream_id"]), step)
        for event in events
        for step in range(int(event["prefix_start"]), int(event["prefix_end"]) + 1)
    }
    observed = {(str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]), str(row["event_id"]), int(row["prefix_step"])) for row in tasks}
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        raise AnnotationError(f"causal prefix task coverage is incomplete: {missing}")
    return tasks


def read_api_key() -> str:
    if not KEY_PATH.is_file() or KEY_PATH.is_symlink():
        raise AnnotationError("project-local .secret/qwen_api_key is missing or unsafe")
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    if not key.startswith("sk-") or len(key) < 20:
        raise AnnotationError("invalid Qwen API key format")
    return key


def api_request(api_key: str, payload: dict[str, object]) -> tuple[object, str, str]:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    last_error = "unattempted"
    for attempt in range(MAX_ATTEMPTS):
        if BACKOFF_SECONDS[attempt]:
            time.sleep(BACKOFF_SECONDS[attempt])
        request = urllib.request.Request(
            QWEN_ENDPOINT,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read()
            envelope = json.loads(raw)
            provider_model = str(envelope.get("model", ""))
            if provider_model != QWEN_MODEL:
                raise AnnotationError(f"provider model drift: {provider_model!r}")
            content = envelope["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed, provider_model, str(envelope.get("id", ""))
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, AnnotationError) as error:
            last_error = f"{type(error).__name__}: {error}"
    raise AnnotationError(f"Qwen request failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _instruction_key(instruction: str) -> str:
    return stable_sha256({"instruction": instruction.strip()})


def run_instruction_one(api_key: str, instruction: str) -> tuple[str, str]:
    key = _instruction_key(instruction)
    path = INSTRUCTION_DIR / f"{key}.json"
    if path.is_file() and not path.is_symlink():
        saved = read_json(path)
        parse_instruction_response(saved["response"], instruction=instruction)
        return key, "cached"
    payload = instruction_request(instruction)
    response, provider_model, request_id = api_request(api_key, payload)
    graph = parse_instruction_response(response, instruction=instruction)
    record = request_record(stage="instruction", payload=payload, response=response, provider_model=provider_model)
    record.update({
        "schema_version": "revealnav-mf3zp-qwen-instruction-preannotation/1",
        "status": "PROVISIONAL_QWEN_PREANNOTATION",
        "instruction_sha256": key,
        "instruction": instruction.strip(),
        "constraint_graph_sha256": graph.canonical_sha256(),
        "provider_request_id": request_id,
        "human_verified": False,
        "gold": False,
    })
    atomic_json(path, record)
    return key, "created"


def load_graph(instruction: str) -> InstructionEvidenceGraph:
    record = read_json(INSTRUCTION_DIR / f"{_instruction_key(instruction)}.json")
    return parse_instruction_response(record["response"], instruction=instruction)


def run_evidence_one(api_key: str, task: dict[str, object]) -> tuple[str, str]:
    request_id_source = str(task["request_id"])
    path = EVIDENCE_DIR / f"{request_id_source}.json"
    instruction = str(task["contract"]["instruction"])
    graph = load_graph(instruction)
    active = graph.topological_order()
    candidates = [dict(value) for value in task["contract"]["current_candidates"]]
    candidate_ids = [str(value["alias"]) for value in candidates]
    image_paths = [ROOT / str(task["causal_storyboard"]["path"]), ROOT / str(task["current_panorama"]["path"])]
    if path.is_file() and not path.is_symlink():
        saved = read_json(path)
        validate_evidence_response(
            saved["response"], active_constraint_ids=active,
            allowed_candidate_ids=candidate_ids, image_count=len(image_paths),
        )
        return request_id_source, "cached"
    payload = evidence_request(
        instruction=instruction,
        graph=graph,
        active_constraint_ids=active,
        current_candidates=candidates,
        existing_evidence=(),
        causal_image_paths=image_paths,
        prefix_step=int(task["prefix_step"]),
    )
    response, provider_model, request_id = api_request(api_key, payload)
    normalized = validate_evidence_response(
        response, active_constraint_ids=active,
        allowed_candidate_ids=candidate_ids, image_count=len(image_paths),
    )
    record = request_record(stage="evidence", payload=payload, response=response, provider_model=provider_model)
    record.update({
        "schema_version": "revealnav-mf3zp-qwen-evidence-preannotation/1",
        "status": "PROVISIONAL_QWEN_PREANNOTATION",
        "source_request_id": request_id_source,
        "pilot_event_ids": task["pilot_event_ids"],
        "dataset": task["dataset"],
        "scene_id": task["scene_id"],
        "episode_id": task["episode_id"],
        "prefix_step": task["prefix_step"],
        "constraint_graph_sha256": graph.canonical_sha256(),
        "normalized_constraints": normalized,
        "image_sha256": [task["causal_storyboard"]["sha256"], task["current_panorama"]["sha256"]],
        "provider_request_id": request_id,
        "human_verified": False,
        "gold": False,
    })
    atomic_json(path, record)
    return request_id_source, "created"


def _parallel(items: list[object], worker, api_key: str, max_workers: int) -> dict[str, object]:
    counts: Counter[str] = Counter()
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, api_key, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                _, state = future.result()
                counts[state] += 1
            except Exception as error:  # fail each request closed; status retains identity only
                identity = str(item if isinstance(item, str) else item.get("request_id", item.get("event_id", "unknown")))
                failures[identity] = f"{type(error).__name__}: {error}"
                counts["failed"] += 1
    return {"counts": dict(counts), "failures": failures}


def status() -> dict[str, object]:
    events = read_events()
    tasks = prefix_tasks(events)
    instructions = {str(event["instruction"]).strip() for event in events}
    instruction_valid = 0
    evidence_valid = 0
    for instruction in instructions:
        path = INSTRUCTION_DIR / f"{_instruction_key(instruction)}.json"
        if path.is_file() and not path.is_symlink():
            try:
                load_graph(instruction)
                instruction_valid += 1
            except Exception:
                pass
    for task in tasks:
        path = EVIDENCE_DIR / f"{task['request_id']}.json"
        if path.is_file() and not path.is_symlink():
            try:
                record = read_json(path)
                graph = load_graph(str(task["contract"]["instruction"]))
                validate_evidence_response(
                    record["response"], active_constraint_ids=graph.topological_order(),
                    allowed_candidate_ids=[value["alias"] for value in task["contract"]["current_candidates"]], image_count=2,
                )
                evidence_valid += 1
            except Exception:
                pass
    result = {
        "schema_version": "revealnav-mf3zp-qwen-preannotation-status/1",
        "status": "MF3ZP_QWEN_PREANNOTATION_READY" if instruction_valid == len(instructions) and evidence_valid == len(tasks) else "MF3ZP_QWEN_PREANNOTATION_INCOMPLETE",
        "model_identifier": QWEN_MODEL,
        "instruction": {"required": len(instructions), "valid": instruction_valid},
        "evidence": {"required": len(tasks), "valid": evidence_valid, "pilot_events": len(events), "all_j_le_t_prefixes": True},
        "human_verified": False,
        "gold": False,
        "formal_label_validity_pass": False,
        "target_payload_read": False,
        "outcome_payload_read": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }
    replace_json(STATUS, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("instruction", "evidence", "all", "status"))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    verify_protocol()
    events = read_events()
    result: dict[str, object] = {}
    if args.command in {"instruction", "all"}:
        api_key = read_api_key()
        instructions = sorted({str(event["instruction"]).strip() for event in events}, key=_instruction_key)
        result["instruction_run"] = _parallel(instructions, run_instruction_one, api_key, args.workers)
    if args.command in {"evidence", "all"}:
        api_key = read_api_key()
        missing_graphs = [event for event in events if not (INSTRUCTION_DIR / f"{_instruction_key(str(event['instruction']))}.json").is_file()]
        if missing_graphs:
            raise AnnotationError(f"instruction preannotation missing for {len(missing_graphs)} events")
        result["evidence_run"] = _parallel(prefix_tasks(events), run_evidence_one, api_key, args.workers)
    result["status"] = status()
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"]["status"] == "MF3ZP_QWEN_PREANNOTATION_READY" or args.command != "all" else 2


if __name__ == "__main__":
    raise SystemExit(main())
