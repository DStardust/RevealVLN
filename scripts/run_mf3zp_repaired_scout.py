#!/usr/bin/env python3
"""Run the exploratory MF3ZP scout from repaired, event-local prefixes.

The script is deliberately separate from the sealed v2/v2r1 runners.  It
filters out episode-shared request rows that are after an individual event's
decision step, verifies the merged response set, and only then opens the
development exact-outcome source for the provisional Probe-A diagnostic.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V2_SCRIPT = ROOT / "scripts/run_mf3zp_qwen_reference_v2.py"
REPAIR_SCRIPT = ROOT / "scripts/repair_mf3zp_qwen_annotation_v2r1.py"
V2_OUTPUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2"
REPAIR_OUTPUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2r1"
V2_PROTOCOL = V2_OUTPUT / "MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json"
REPAIR_PROTOCOL = REPAIR_OUTPUT / "MF3ZP_QWEN_UAD_REFERENCE_V2R1_PROTOCOL.json"
MERGED_MANIFEST = REPAIR_OUTPUT / "MF3ZP_QWEN_ANNOTATION_MERGED_MANIFEST.json"
SCOUT_PROTOCOL = REPAIR_OUTPUT / "MF3ZP_REPAIRED_SCOUT_PROTOCOL.json"
SCOUT_METHOD = ROOT / "METHOD_REVISION_3ZP_QWEN_UAD_REFERENCE_V2R1_SCOUT.md"
SCOUT_RESULT = REPAIR_OUTPUT / "MF3ZP_REPAIRED_SCOUT_RESULT.json"
SCHEMA = "revealnav-mf3zp-repaired-scout/1"
STATUS = "SEALED_BEFORE_REPAIRED_SCOUT"


class ScoutError(RuntimeError):
    pass


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ScoutError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = _load_module(V2_SCRIPT, "mf3zp_v2_for_repaired_scout")
repair = _load_module(REPAIR_SCRIPT, "mf3zp_v2r1_for_repaired_scout")
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
        raise ScoutError(f"path escapes project: {path}")
    return str(resolved.relative_to(ROOT.resolve()))


def inventory(path: Path) -> dict:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink() or ROOT.resolve() not in resolved.parents:
        raise ScoutError(f"invalid project file: {path}")
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: object, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise ScoutError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ScoutError(f"stale partial output: {partial}")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def atomic_jsonl(path: Path, rows: list[Mapping[str, object]], *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise ScoutError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise ScoutError(f"stale partial output: {partial}")
    with partial.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(partial, path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ScoutError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ScoutError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ScoutError(f"cannot read JSONL: {path}") from error
    for line_no, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except ValueError as error:
            raise ScoutError(f"invalid JSONL {path}:{line_no}") from error
        if not isinstance(value, dict):
            raise ScoutError(f"JSONL object required: {path}:{line_no}")
        rows.append(value)
    if not rows:
        raise ScoutError(f"empty JSONL: {path}")
    return rows


def event_steps(protocol: Mapping[str, object]) -> dict[str, int]:
    events = protocol["population"]["events"]
    result = {str(row["event_id"]): int(row["decision_step"]) for row in events}
    if len(result) != len(events):
        raise ScoutError("duplicate event IDs in protocol")
    return result


def _response_path(model: str, request_id: str, source: str) -> Path:
    if source == "v2":
        return V2_OUTPUT / "responses" / repair.model_slug(model) / f"{request_id}.json"
    if source == "v2r1_repair":
        return REPAIR_OUTPUT / "responses" / repair.model_slug(model) / f"{request_id}.json"
    raise ScoutError(f"unknown response source: {source}")


def _resolve_response(model: str, request: Mapping[str, object]) -> tuple[Path, dict]:
    row = dict(request)
    old_path = _response_path(model, str(row["request_id"]), "v2")
    if old_path.is_file() and not old_path.is_symlink():
        old = read_json(old_path)
        if not repair.response_errors(old, row, model):
            return old_path, old
    new_path = _response_path(model, str(row["request_id"]), "v2r1_repair")
    if not new_path.is_file() or new_path.is_symlink():
        raise ScoutError(f"missing repaired response: {new_path}")
    new = read_json(new_path)
    errors = repair.response_errors(new, row, model)
    if errors:
        raise ScoutError(f"invalid repaired response {new_path}: {errors}")
    return new_path, new


def build_filtered_inputs(protocol: Mapping[str, object]) -> tuple[list[dict], list[dict], list[dict]]:
    limits = event_steps(protocol)
    requests = read_jsonl(V2_OUTPUT / "MF3ZP_ANNOTATION_REQUESTS.jsonl")
    filtered = [row for row in requests if int(row["prefix_step"]) <= limits[str(row["event_id"])] ]
    excluded = [row for row in requests if int(row["prefix_step"]) > limits[str(row["event_id"])] ]
    by_event = defaultdict(list)
    for row in filtered:
        by_event[str(row["event_id"])].append(int(row["prefix_step"]))
    if set(by_event) != set(limits):
        raise ScoutError("filtered request population does not cover all events")
    for event_id, limit in limits.items():
        if sorted(by_event[event_id]) != list(range(limit + 1)):
            raise ScoutError(f"event-local causal prefix is incomplete: {event_id}")
    projection = read_jsonl(V2_OUTPUT / "MF3ZP_PROJECTION_MAP.jsonl")
    deterministic = read_jsonl(V2_OUTPUT / "MF3ZP_DETERMINISTIC_ORACLE.jsonl")
    deterministic = [row for row in deterministic if int(row["prefix_step"]) <= limits[str(row["event_id"])] ]
    det_by_event = defaultdict(list)
    for row in deterministic:
        det_by_event[str(row["event_id"])].append(int(row["prefix_step"]))
    for event_id, limit in limits.items():
        if sorted(det_by_event[event_id]) != list(range(limit + 1)):
            raise ScoutError(f"event-local deterministic oracle is incomplete: {event_id}")
    return filtered, projection, deterministic


def build_scout_protocol() -> dict:
    parent = read_json(V2_PROTOCOL)
    v2.verify_protocol_v2(parent)
    repair_protocol = read_json(REPAIR_PROTOCOL)
    repair.verify_protocol(repair_protocol)
    merged = read_json(MERGED_MANIFEST)
    if merged.get("status") != "PASS":
        raise ScoutError("merged annotation manifest is not PASS")
    filtered, projection, deterministic = build_filtered_inputs(parent)
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "parent_v2_protocol": inventory(V2_PROTOCOL),
        "repair_protocol": inventory(REPAIR_PROTOCOL),
        "merged_manifest": inventory(MERGED_MANIFEST),
        "method": inventory(SCOUT_METHOD),
        "script": inventory(Path(__file__).resolve()),
        "population": {
            "events": len(parent["population"]["events"]),
            "requests_before_event_filter": len(read_jsonl(V2_OUTPUT / "MF3ZP_ANNOTATION_REQUESTS.jsonl")),
            "requests_after_event_filter": len(filtered),
            "excluded_after_event_filter": len(read_jsonl(V2_OUTPUT / "MF3ZP_ANNOTATION_REQUESTS.jsonl")) - len(filtered),
            "event_ids_sha256": m.stable_sha256([row["event_id"] for row in parent["population"]["events"]]),
        },
        "causal_rule": "retain only prefix_step <= that event's sealed decision_step; no future prefix is used",
        "probe": {
            "name": "exploratory_qwen_oracle_relevance",
            "ridge_l2": 1.0,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260831,
            "human_verified": False,
            "formal_authorized": False,
        },
        "boundary": {
            "target_payload_read": False,
            "outcome_payload_read": False,
            "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
            "checkpoint_generated": False,
        },
    }


def verify_scout_protocol(value: Mapping[str, object] | None = None) -> dict:
    protocol = dict(value) if value is not None else read_json(SCOUT_PROTOCOL)
    if protocol.get("schema_version") != SCHEMA or protocol.get("status") != STATUS:
        raise ScoutError("scout protocol identity/status drift")
    access = protocol.get("boundary", {}).get("public_split_access")
    if access != {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}:
        raise ScoutError("public split access is not fail-closed")
    if protocol.get("boundary", {}).get("target_payload_read") is not False or protocol.get("boundary", {}).get("outcome_payload_read") is not False:
        raise ScoutError("scout boundary drift")
    if protocol.get("parent_v2_protocol") != inventory(V2_PROTOCOL) or protocol.get("repair_protocol") != inventory(REPAIR_PROTOCOL) or protocol.get("merged_manifest") != inventory(MERGED_MANIFEST):
        raise ScoutError("scout source artifact drift")
    if protocol.get("method") != inventory(SCOUT_METHOD) or protocol.get("script") != inventory(Path(__file__).resolve()):
        raise ScoutError("scout implementation drift")
    parent = read_json(V2_PROTOCOL)
    v2.verify_protocol_v2(parent)
    repair.verify_protocol(read_json(REPAIR_PROTOCOL))
    filtered, _, _ = build_filtered_inputs(parent)
    pop = protocol.get("population", {})
    if int(pop.get("requests_after_event_filter", -1)) != len(filtered):
        raise ScoutError("filtered population drift")
    return protocol


def run_scout(protocol: Mapping[str, object]) -> dict:
    verify_scout_protocol(protocol)
    parent = read_json(V2_PROTOCOL)
    filtered, projection_rows, deterministic_rows = build_filtered_inputs(parent)
    projection = {row["event_id"]: row for row in projection_rows}
    deterministic: dict[str, list[dict]] = defaultdict(list)
    for row in deterministic_rows:
        deterministic[row["event_id"]].append(row)
    models = [str(value) for value in parent["annotation"]["models"]]
    labels_by_model: dict[str, dict[str, dict]] = {}
    # No exact outcome source is opened until both model label projections are
    # complete and every response has passed validation.
    for model in models:
        by_event: dict[str, list[dict]] = defaultdict(list)
        for request in filtered:
            _, stored = _resolve_response(model, request)
            response = stored["response"]
            projected = m.derive_semantic_state(
                response,
                target_alias=projection[request["event_id"]]["alternative_alias"],
                native_alias=projection[request["event_id"]]["native_alias"],
                target_present=next(
                    bool(value["target_in_set"])
                    for value in deterministic[request["event_id"]]
                    if int(value["prefix_step"]) == int(request["prefix_step"])
                ),
            )
            by_event[request["event_id"]].append({"prefix_step": int(request["prefix_step"]), **projected, "raw_response": response})
        labels: dict[str, dict] = {}
        for event_id, rows in by_event.items():
            rows.sort(key=lambda value: value["prefix_step"])
            target = tuple(bool(value["target_in_set"]) for value in deterministic[event_id])
            separated = tuple(bool(value["candidate_separated"]) for value in rows)
            closed = tuple(bool(value["evidence_closed"]) for value in rows)
            if len(rows) != len(target):
                raise ScoutError(f"response/deterministic prefix mismatch: {event_id}")
            states = m.derive_uad(target, separated, closed, stability_k=m.STABILITY_K)
            reveal = m.reveal_interval(states)
            expiry_values = [int(value["prefix_step"]) for value in deterministic[event_id] if bool(value["target_in_set"])]
            expiry = max(expiry_values) if expiry_values else None
            labels[event_id] = {
                "event_id": event_id,
                "target_in_set": target,
                "candidate_separated": separated,
                "evidence_closed": closed,
                "uad": tuple(value.value for value in states),
                "reveal_interval": reveal,
                "expiry_step": expiry,
                "resolvable": m.resolvable(reveal, expiry),
                "provenance": "qwen_provisional_unverified",
            }
        labels_by_model[model] = labels
        slug = repair.model_slug(model)
        atomic_jsonl(REPAIR_OUTPUT / f"MF3ZP_REPAIRED_PROVISIONAL_LABELS_{slug}.jsonl", [labels[key] for key in sorted(labels)], refuse_existing=False)

    # Explicit boundary transition: the following source is opened only after
    # all provisional labels have been frozen and verified.
    car_path = ROOT / "scripts/train_mf3zm_car.py"
    car = _load_module(car_path, "mf3zp_repaired_scout_car_source")
    car.verify_protocol()
    outcome_rows = car._canonical_rows()
    outcome_by_event = {str(row["event_id"]): float(row["target"]) for row in outcome_rows if "event_id" in row}
    if len(outcome_by_event) != len(outcome_rows):
        from revealnav_mf3.mf3zo_pilot import canonical_event_id
        outcome_by_event = {
            canonical_event_id(row["dataset"], row["scene_id"], row["episode_id"], int(row["decision"]["step"])): float(row["target"])
            for row in outcome_rows
        }
    event_rows = list(parent["population"]["events"])
    event_ids = [row["event_id"] for row in event_rows]
    if not all(event_id in outcome_by_event for event_id in event_ids):
        raise ScoutError("exact outcome identity does not cover filtered events")

    from revealnav_mf3.mf3zo_pilot import load_causal_records
    from revealnav_mf3.mf3zo_probes import current_snapshot_features, oracle_feature_vector, probe_a_oracle_relevance
    from revealnav_mf3.mf3zo_temporal_schema import TemporalOracleLabel
    record_path = ROOT / "artifacts/training/mf3zo_temporal_oracle_gap_v1/MF3ZO_CAUSAL_TEMPORAL_RECORDS.jsonl"
    records = load_causal_records(record_path, ROOT)
    if set(records) != set(event_ids):
        raise ScoutError("temporal record event inventory mismatch")
    for event_id, row in zip(event_ids, event_rows):
        if records[event_id].decision_step != int(row["decision_step"]):
            raise ScoutError(f"temporal record decision step mismatch: {event_id}")
    current = np.stack([current_snapshot_features(records[event_id]) for event_id in event_ids])
    scenes = np.asarray([row["scene_id"] for row in event_rows])
    domains = np.asarray([row["dataset"] for row in event_rows])
    folds = np.asarray([parent["population"]["fold_by_event"][event_id] for event_id in event_ids])
    target = np.asarray([outcome_by_event[event_id] for event_id in event_ids])
    results = {}
    for model in models:
        oracle_features = []
        for event_id, row in zip(event_ids, event_rows):
            label = labels_by_model[model][event_id]
            oracle_label = TemporalOracleLabel(
                event_id=event_id,
                target_in_set=tuple(label["target_in_set"]),
                candidate_separated=tuple(label["candidate_separated"]),
                evidence_closed=tuple(label["evidence_closed"]),
                reveal_interval=label["reveal_interval"],
                expiry_step=label["expiry_step"],
                resolvable=bool(label["resolvable"]),
                unavailable_fields=(),
                provenance="qwen_provisional_unverified",
            )
            oracle_features.append(oracle_feature_vector(oracle_label, int(row["decision_step"])))
        result = probe_a_oracle_relevance(current, np.stack(oracle_features), target, scenes, domains, folds)
        result.update({"annotator_model": model, "exploratory": True, "human_verified": False, "formal_authorized": False, "target_payload_read": True, "event_local_prefix_filter": True})
        results[model] = result
        atomic_json(REPAIR_OUTPUT / f"MF3ZP_REPAIRED_PROBE_A_{repair.model_slug(model)}.json", result, refuse_existing=False)
    positive = all(result["status"] == "ORACLE_RELEVANCE_PASS" for result in results.values())
    final = {
        "schema_version": "revealnav-mf3zp-repaired-scout-result/1",
        "status": "EXPLORATORY_QWEN_ORACLE_SIGNAL_POSITIVE" if positive else "EXPLORATORY_QWEN_ORACLE_SIGNAL_NOT_POSITIVE",
        "models": results,
        "human_verified": False,
        "formal_probe_a_authorized": False,
        "checkpoint_generated": False,
        "public_split_access": False,
        "target_payload_read": True,
        "outcome_payload_read": True,
        "event_local_prefix_filter": True,
        "excluded_request_count": int(protocol["population"]["excluded_after_event_filter"]),
    }
    atomic_json(SCOUT_RESULT, final, refuse_existing=False)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal")
    sub.add_parser("run")
    args = parser.parse_args()
    try:
        if args.command == "seal":
            if SCOUT_PROTOCOL.exists() or SCOUT_PROTOCOL.is_symlink():
                raise ScoutError("scout protocol already exists; resealing is forbidden")
            value = build_scout_protocol()
            atomic_json(SCOUT_PROTOCOL, value, refuse_existing=True)
            print(json.dumps({"status": value["status"], "protocol_sha256": sha256_file(SCOUT_PROTOCOL), "filtered_requests": value["population"]["requests_after_event_filter"], "excluded": value["population"]["excluded_after_event_filter"]}, indent=2))
        else:
            print(json.dumps(run_scout(verify_scout_protocol()), indent=2, ensure_ascii=False))
        return 0
    except BaseException as error:
        print(f"MF3ZP_REPAIRED_SCOUT_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
