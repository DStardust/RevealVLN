#!/usr/bin/env python3
"""Versioned MF3ZP evidence-validation correction and exact missing-set rerun."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from functools import lru_cache


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.qwen_evidence_annotation_v1_1 import (  # noqa: E402
    validate_evidence_response_v1_1,
)


OUTPUT = ROOT / "artifacts/training/mf3zp_revealskill_v1"
SCIENCE_PROTOCOL = OUTPUT / "MF3ZP_REVEALSKILL_PROTOCOL.json"
PARENT_TRANSPORT = OUTPUT / "MF3ZP_QWEN_TRANSPORT_V1R2_PROTOCOL.json"
CORRECTNESS_PROTOCOL = OUTPUT / "MF3ZP_REVEALSKILL_V1_1_CORRECTNESS_PROTOCOL.json"
STATUS_PATH = OUTPUT / "MF3ZP_QWEN_EVIDENCE_V1_1_STATUS.json"
V1_1_DIR = OUTPUT / "qwen_preannotations/evidence_v1_1"

TRANSPORT_PATH = ROOT / "scripts/annotate_mf3zp_qwen_transport_v1r2.py"
ERRATUM_PATH = ROOT / "METHOD_REVISION_3ZP_REVEALSKILL_V1_1_CORRECTNESS.md"
VALIDATOR_PATH = ROOT / "revealnav_mf3/qwen_evidence_annotation_v1_1.py"
TEST_PATH = ROOT / "tests/test_mf3zp_evidence_annotation_v1_1.py"
AUDIT_PATH = ROOT / "scripts/audit_mf3zp_labels_v1_1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transport = _load(TRANSPORT_PATH, "mf3zp_qwen_transport_v1r2_for_correctness")
base = transport.base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in path.resolve().parents:
        raise RuntimeError(f"invalid project-local file: {path}")
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _task_context(task: dict[str, object]):
    graph = base.load_graph(str(task["contract"]["instruction"]))
    active = graph.topological_order()
    candidates = [dict(value) for value in task["contract"]["current_candidates"]]
    candidate_ids = [str(value["alias"]) for value in candidates]
    image_paths = [
        ROOT / str(task["causal_storyboard"]["path"]),
        ROOT / str(task["current_panorama"]["path"]),
    ]
    return graph, active, candidates, candidate_ids, image_paths


def _valid_base_record(task: dict[str, object]) -> bool:
    path = base.EVIDENCE_DIR / f"{task['request_id']}.json"
    if not path.is_file() or path.is_symlink():
        return False
    try:
        record = base.read_json(path)
        _, active, _, candidate_ids, image_paths = _task_context(task)
        base.validate_evidence_response(
            record["response"],
            active_constraint_ids=active,
            allowed_candidate_ids=candidate_ids,
            image_count=len(image_paths),
        )
        validate_evidence_response_v1_1(
            record["response"],
            active_constraint_ids=active,
            allowed_candidate_ids=candidate_ids,
            image_count=len(image_paths),
        )
        return True
    except Exception:
        return False


def task_partition() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tasks = base.prefix_tasks(base.read_events())
    valid = []
    selected = []
    for task in tasks:
        (valid if _valid_base_record(task) else selected).append(task)
    if len(tasks) != 538 or len(valid) != 383 or len(selected) != 155:
        raise RuntimeError(
            f"unexpected v1 evidence partition: total={len(tasks)} valid={len(valid)} selected={len(selected)}"
        )
    return valid, selected


def _task_inventory(task: dict[str, object]) -> dict[str, object]:
    return {
        "request_id": str(task["request_id"]),
        "dataset": str(task["dataset"]),
        "scene_id": str(task["scene_id"]),
        "episode_id": str(task["episode_id"]),
        "prefix_step": int(task["prefix_step"]),
        "pilot_event_ids": list(task["pilot_event_ids"]),
        "instruction_sha256": base._instruction_key(str(task["contract"]["instruction"])),
        "candidate_ids": [str(value["alias"]) for value in task["contract"]["current_candidates"]],
        "image_sha256": [
            str(task["causal_storyboard"]["sha256"]),
            str(task["current_panorama"]["sha256"]),
        ],
    }


def protocol_value() -> dict[str, object]:
    transport.verify()
    valid, selected = task_partition()
    base_inventory = [inventory(base.EVIDENCE_DIR / f"{task['request_id']}.json") for task in valid]
    selected_inventory = [_task_inventory(task) for task in selected]
    return {
        "schema_version": "revealnav-mf3zp-evidence-correctness/1",
        "revision": "mf3zp_revealskill_v1_1_annotation_correctness",
        "status": "SEALED_BEFORE_MF3ZP_EVIDENCE_V1_1_RESULTS",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "science_protocol": inventory(SCIENCE_PROTOCOL),
        "parent_transport_protocol": inventory(PARENT_TRANSPORT),
        "erratum": inventory(ERRATUM_PATH),
        "validator": inventory(VALIDATOR_PATH),
        "entrypoint": inventory(Path(__file__).resolve()),
        "audit_entrypoint": inventory(AUDIT_PATH),
        "tests": [inventory(TEST_PATH)],
        "model": "qwen3.8-max",
        "prompt_change": False,
        "model_change": False,
        "event_population_change": False,
        "scientific_gate_change": False,
        "correctness_change": "validate S/G/E independently; D remains derived from S=G=E for K=3",
        "total_prefix_tasks": 538,
        "v1_valid_count": len(valid),
        "v1_valid_record_inventory": base_inventory,
        "v1_1_selected_count": len(selected),
        "v1_1_selected_request_ids": [str(task["request_id"]) for task in selected],
        "v1_1_selected_task_inventory_sha256": base.stable_sha256(selected_inventory),
        "effective_transport": {"enable_thinking": False, "max_tokens": 8000, "temperature": 0.0},
        "target_payload_read": False,
        "outcome_payload_read": False,
        "human_verified": False,
        "gold": False,
        "public_split_access": {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
    }


def seal() -> dict[str, object]:
    value = protocol_value()
    if CORRECTNESS_PROTOCOL.exists() or CORRECTNESS_PROTOCOL.is_symlink():
        raise RuntimeError("refusing to overwrite v1.1 correctness protocol")
    partial = CORRECTNESS_PROTOCOL.with_name(CORRECTNESS_PROTOCOL.name + ".part")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, CORRECTNESS_PROTOCOL)
    return value


def verify() -> dict[str, object]:
    value = json.loads(CORRECTNESS_PROTOCOL.read_text(encoding="utf-8"))
    if value != protocol_value():
        raise RuntimeError("v1.1 correctness protocol/source/input drift")
    return value


@lru_cache(maxsize=1)
def sealed_selected_ids() -> frozenset[str]:
    value = json.loads(CORRECTNESS_PROTOCOL.read_text(encoding="utf-8"))
    return frozenset(str(item) for item in value["v1_1_selected_request_ids"])


def run_one(api_key: str, task: dict[str, object]) -> tuple[str, str]:
    request_id_source = str(task["request_id"])
    if request_id_source not in sealed_selected_ids():
        raise RuntimeError("request is outside the sealed v1.1 correction set")
    path = V1_1_DIR / f"{request_id_source}.json"
    graph, active, candidates, candidate_ids, image_paths = _task_context(task)
    if path.is_file() and not path.is_symlink():
        saved = base.read_json(path)
        validate_evidence_response_v1_1(
            saved["response"], active_constraint_ids=active,
            allowed_candidate_ids=candidate_ids, image_count=len(image_paths),
        )
        return request_id_source, "cached"
    payload = base.evidence_request(
        instruction=str(task["contract"]["instruction"]),
        graph=graph,
        active_constraint_ids=active,
        current_candidates=candidates,
        existing_evidence=(),
        causal_image_paths=image_paths,
        prefix_step=int(task["prefix_step"]),
    )
    response, provider_model, provider_request_id = transport.api_request(api_key, payload)
    normalized = validate_evidence_response_v1_1(
        response, active_constraint_ids=active,
        allowed_candidate_ids=candidate_ids, image_count=len(image_paths),
    )
    record = transport.transport_record(
        stage="evidence", payload=payload, response=response, provider_model=provider_model,
    )
    record.update({
        "schema_version": "revealnav-mf3zp-qwen-evidence-preannotation/1.1",
        "status": "PROVISIONAL_QWEN_PREANNOTATION",
        "validator_revision": "mf3zp_revealskill_v1_1_annotation_correctness",
        "correctness_protocol_sha256": sha256_file(CORRECTNESS_PROTOCOL),
        "source_request_id": request_id_source,
        "pilot_event_ids": task["pilot_event_ids"],
        "dataset": task["dataset"],
        "scene_id": task["scene_id"],
        "episode_id": task["episode_id"],
        "prefix_step": task["prefix_step"],
        "constraint_graph_sha256": graph.canonical_sha256(),
        "normalized_constraints": normalized,
        "image_sha256": [task["causal_storyboard"]["sha256"], task["current_panorama"]["sha256"]],
        "provider_request_id": provider_request_id,
        "human_verified": False,
        "gold": False,
    })
    base.atomic_json(path, record)
    return request_id_source, "created"


def combined_record(task: dict[str, object]) -> dict[str, object] | None:
    if _valid_base_record(task):
        return base.read_json(base.EVIDENCE_DIR / f"{task['request_id']}.json")
    path = V1_1_DIR / f"{task['request_id']}.json"
    if not path.is_file() or path.is_symlink():
        return None
    record = base.read_json(path)
    _, active, _, candidate_ids, image_paths = _task_context(task)
    validate_evidence_response_v1_1(
        record["response"], active_constraint_ids=active,
        allowed_candidate_ids=candidate_ids, image_count=len(image_paths),
    )
    return record


def combined_status(*, write: bool = True) -> dict[str, object]:
    verify()
    tasks = base.prefix_tasks(base.read_events())
    base_valid = sum(_valid_base_record(task) for task in tasks)
    v1_1_valid = 0
    missing = []
    for task in tasks:
        if _valid_base_record(task):
            continue
        try:
            if combined_record(task) is not None:
                v1_1_valid += 1
            else:
                missing.append(str(task["request_id"]))
        except Exception:
            missing.append(str(task["request_id"]))
    valid = base_valid + v1_1_valid
    value = {
        "schema_version": "revealnav-mf3zp-qwen-evidence-v1.1-status/1",
        "status": "MF3ZP_QWEN_PREANNOTATION_READY" if valid == len(tasks) else "MF3ZP_QWEN_PREANNOTATION_INCOMPLETE",
        "model_identifier": "qwen3.8-max",
        "required": len(tasks),
        "valid": valid,
        "v1_valid": base_valid,
        "v1_1_valid": v1_1_valid,
        "missing_request_ids": missing,
        "human_verified": False,
        "gold": False,
        "formal_label_validity_pass": False,
        "target_payload_read": False,
        "outcome_payload_read": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }
    if write:
        base.replace_json(STATUS_PATH, value)
    return value


def run_parallel(tasks: list[dict[str, object]], api_key: str, workers: int) -> dict[str, object]:
    counts = {"created": 0, "cached": 0, "failed": 0}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, api_key, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                _, state = future.result()
                counts[state] += 1
            except Exception as error:
                counts["failed"] += 1
                failures[str(task["request_id"])] = f"{type(error).__name__}: {error}"
    return {"counts": counts, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "verify", "evidence", "status"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.command == "seal":
        value = seal()
        print(json.dumps({"status": value["status"], "sha256": sha256_file(CORRECTNESS_PROTOCOL)}, indent=2))
        return 0
    verify()
    if args.command == "verify":
        print(json.dumps({"status": "MF3ZP_REVEALSKILL_V1_1_CORRECTNESS_VERIFIED", "sha256": sha256_file(CORRECTNESS_PROTOCOL)}, indent=2))
        return 0
    result: dict[str, object] = {}
    if args.command == "evidence":
        _, selected = task_partition()
        result["evidence_run"] = run_parallel(selected, base.read_api_key(), args.workers)
    result["status"] = combined_status()
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"]["status"] == "MF3ZP_QWEN_PREANNOTATION_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
