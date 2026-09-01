#!/usr/bin/env python3
"""MF3ZP data protocol: seal, collect causal observations, and annotate.

This entrypoint deliberately stops before any formal Probe-A authorization.
The collection side never reads intervention outcomes; the annotation side
only sees role-blinded, prefix-truncated observations.  Outcome data is read
only by the explicit ``scout`` command after every planned response has been
frozen.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Iterable, Mapping

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zp_reference import (  # noqa: E402
    RESPONSE_SCHEMA,
    STABILITY_K,
    ReferenceContractError,
    annotation_user_text,
    branch_aliases,
    build_annotation_contract,
    derive_semantic_state,
    derive_uad,
    resolvable,
    reveal_interval,
    stable_sha256,
    validate_annotation_response,
)


REVISION = "mf3zp_qwen_uad_reference_v1"
SCHEMA = "revealnav-mf3zp-qwen-uad-reference/1"
STATUS = "SEALED_BEFORE_MF3ZP_OBSERVATION_OR_LABEL_RESULTS"
OUTPUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v1"
PROTOCOL = OUTPUT / "MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json"
SELECTION = ROOT / "artifacts/training/mf3zo_temporal_oracle_gap_v1/MF3ZO_PILOT_SELECTION.json"
MF3ZO_PROTOCOL = ROOT / "artifacts/training/mf3zo_temporal_oracle_gap_v1/MF3ZO_TEMPORAL_ORACLE_GAP_PROTOCOL.json"
PARENT_MF3ZN = ROOT / "artifacts/training/mf3zn_tuad_v1/MF3ZN_TUAD_PROTOCOL.json"
MF3ZK_DSR_PROTOCOL = ROOT / "artifacts/training/mf3zk_dsr_v1/MF3ZK_DSR_PROTOCOL.json"
PROMPT = ROOT / "revealnav_mf3/MF3ZP_QWEN_SYSTEM_PROMPT.md"
METHOD_PATH = "METHOD_REVISION_3ZP_QWEN_UAD_REFERENCE.md"
WORKER = ROOT / "scripts/mf3zp_observation_worker.py"
REFERENCE = ROOT / "revealnav_mf3/mf3zp_reference.py"
MODELS = (
    "qwen3-vl-plus-2025-12-19",
    "qwen3.5-plus-2026-02-15",
)
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
EXPECTED_PARENT_MF3ZN_SHA = "b502629d898879c65031a92b91496fd39d640e7c0f09097bd8bce8ebd9118772"
EXPECTED_MF3ZO_SHA = "74f2b0e737f9d5a89cfd1ad63ae9bfc93b4245ddcfa6179d9e2dd6d1471c989f"
CONFIRMATION_BLACKLIST = frozenset({
    "1LXtFkjw3qL", "2n8kARJN3HM", "5LpN3gDmAk7", "759xd9YjKW5",
    "HxpKQynjfin", "JF19kD82Mey", "PuKPg4mmafe", "SN83YJsR3w2",
    "ULsKaCPVFJR", "Uxmj2M2itWa", "V2XKFyX4ASd", "Vvot9Ly1tCj",
    "VzqfbhrpDEA", "XcA2TqTSSAj", "b8cTxDM8gDG", "gTV8FGcVJC9",
    "i5noydFURQK", "jh4fc5c5qoQ", "kEZ7cmS4wCh", "mJXqzFtmKg4",
    "sKLMLpTHeUy",
})
_STATUS_LOCK = threading.Lock()
_LEGACY_TIER_MAP: dict[tuple[str, str, int], str] | None = None


class MF3ZPError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise MF3ZPError(f"path escapes project: {path}")
    return str(resolved.relative_to(ROOT.resolve()))


def inventory(path: Path) -> dict:
    resolved = path.resolve()
    if (
        not path.is_file() or path.is_symlink()
        or ROOT.resolve() not in resolved.parents
    ):
        raise MF3ZPError(f"invalid project-local source: {path}")
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, value: object, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise MF3ZPError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise MF3ZPError(f"stale partial output: {partial}")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, object]], *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise MF3ZPError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise MF3ZPError(f"stale partial output: {partial}")
    with partial.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(partial, path)


def strict_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MF3ZPError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise MF3ZPError(f"JSON object required: {path}")
    return value


def jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except ValueError as error:
            raise MF3ZPError(f"invalid JSONL {path}:{line_no}") from error
        if not isinstance(value, dict):
            raise MF3ZPError(f"JSONL object required: {path}:{line_no}")
        rows.append(value)
    if not rows:
        raise MF3ZPError(f"empty JSONL: {path}")
    return rows


def source_trace_paths(row: Mapping[str, object]) -> tuple[Path, Path, Path | None]:
    feature = (ROOT / str(row["feature_path"])).resolve()
    if not feature.is_file() or feature.is_symlink():
        raise MF3ZPError(f"feature source is not a regular file: {feature}")
    parent = feature.parent
    if str(row["source"]) == "mf3zk_dsr_v1_existing_exact":
        run_dir = parent
    else:
        run_dir = parent.parent
    controller = run_dir / "controller_trace.jsonl"
    base = run_dir / "base_trace.jsonl"
    for value in (controller, base):
        if not value.is_file() or value.is_symlink():
            raise MF3ZPError(f"missing native provenance trace: {value}")
    proposal = run_dir / "proposal_trace.jsonl"
    return controller, base, proposal if proposal.is_file() and not proposal.is_symlink() else None


def trace_rows(path: Path) -> list[dict]:
    return jsonl(path)


def _legacy_tier(dataset: str, episode_id: str, step: int) -> str | None:
    global _LEGACY_TIER_MAP
    if _LEGACY_TIER_MAP is None:
        value = strict_json(MF3ZK_DSR_PROTOCOL)
        rows = value.get("source_inventory", {}).get("canonical_rows", [])
        mapping: dict[tuple[str, str, int], str] = {}
        for row in rows:
            identity = row.get("identity", {})
            tier = row.get("frozen_tier")
            if tier in {"core", "expansion"}:
                mapping[(str(identity.get("dataset")), str(identity.get("episode_id")), int(identity.get("decision_step", -1)))] = str(tier)
        _LEGACY_TIER_MAP = mapping
    return _LEGACY_TIER_MAP.get((dataset, episode_id, step))


def resolve_event(row: Mapping[str, object]) -> dict:
    controller, base, proposal = source_trace_paths(row)
    step = int(row["decision_step"])
    hits = [value for value in trace_rows(controller) if int(value.get("step", -1)) == step]
    if len(hits) != 1:
        raise MF3ZPError(f"event {row['event_id']} has {len(hits)} controller rows at step {step}")
    hit = hits[0]
    # MF3ZL's compact controller trace carries event identity only; its
    # sibling proposal trace carries the executable candidate list and the
    # sealed feature action IDs.  MF3ZK embeds both in controller_trace.
    if proposal is not None:
        proposal_hits = [
            value for value in trace_rows(proposal)
            if int(value.get("step", -1)) == step
        ]
        if len(proposal_hits) != 1:
            raise MF3ZPError(f"event {row['event_id']} has {len(proposal_hits)} proposal rows at step {step}")
        merged = dict(proposal_hits[0])
        merged.update({key: value for key, value in hit.items() if key not in merged})
        # Keep the compact event identity from controller_trace authoritative.
        if isinstance(hit.get("event_identity"), dict):
            merged["event_identity"] = hit["event_identity"]
        hit = merged
    identity = hit.get("event_identity") if isinstance(hit.get("event_identity"), dict) else {}
    native = hit.get("feature_native_action_id") or identity.get("native_action_id") or hit.get("native_action_id")
    runner = hit.get("feature_alternative_action_id") or identity.get("runner_action_id")
    # The earliest MF3ZK collection trace predates explicit feature IDs; its
    # collection-only adapted action is the sealed runner identity.
    if runner is None and str(row["source"]) == "mf3zk_dsr_v1_existing_exact":
        runner = hit.get("adapted_action_id")
    candidates = hit.get("current_local_action_ids")
    if not isinstance(candidates, list):
        raise MF3ZPError(f"event {row['event_id']} has no executable candidate list")
    candidates = tuple(str(value) for value in candidates)
    if (
        not isinstance(native, str) or not native
        or not isinstance(runner, str) or not runner
        or native == runner
        or len(set(candidates)) != len(candidates)
        or native not in candidates
        or runner not in candidates
    ):
        raise MF3ZPError(f"event {row['event_id']} has invalid native/runner identity")
    tier = identity.get("tier")
    if tier not in {"core", "expansion"}:
        feature_name = Path(str(row["feature_path"])).name
        tier = "core" if "event_core_" in feature_name else "expansion" if "event_expansion_" in feature_name else None
    if tier is None:
        tier = _legacy_tier(str(row["dataset"]), str(row["episode_id"]), step)
    if tier is None:
        raise MF3ZPError(f"event {row['event_id']} has no sealed proposal tier")
    if str(identity.get("scene_id", row["scene_id"])) != str(row["scene_id"]):
        raise MF3ZPError(f"event {row['event_id']} scene identity drift")
    return {
        "event_id": str(row["event_id"]),
        "dataset": str(row["dataset"]),
        "scene_id": str(row["scene_id"]),
        "episode_id": str(row["episode_id"]),
        "decision_step": step,
        "tier": str(tier),
        "native_action_id": native,
        "alternative_action_id": runner,
        "sealed_candidate_action_ids": list(candidates),
        "source": str(row["source"]),
        "source_feature": inventory(ROOT / str(row["feature_path"])),
        "source_controller_trace": inventory(controller),
        "source_native_trace": inventory(base),
        "mf3zo_prefix_sha256": str(row.get("prefix_sha256", "")),
    }


def load_population() -> tuple[list[dict], list[dict]]:
    selection = strict_json(SELECTION)
    if selection.get("status") != "SEALED_INPUT_POPULATION_SELECTED_OUTCOME_BLIND":
        raise MF3ZPError("MF3ZO selection status drift")
    events = selection.get("events")
    if not isinstance(events, list) or len(events) != 150:
        raise MF3ZPError("MF3ZO selection must contain 150 events")
    if Counter(str(value.get("dataset")) for value in events) != Counter({"R2R": 75, "RxR": 75}):
        raise MF3ZPError("MF3ZO domain allocation drift")
    if len({str(value.get("event_id")) for value in events}) != 150:
        raise MF3ZPError("MF3ZO event identities are not unique")
    resolved = [resolve_event(row) for row in events]
    if any(value["scene_id"] in CONFIRMATION_BLACKLIST for value in resolved):
        raise MF3ZPError("consumed confirmation scene entered MF3ZP population")
    episodes: dict[tuple[str, str], dict] = {}
    for value in resolved:
        key = (value["dataset"], value["episode_id"])
        prior = episodes.get(key)
        if prior is None:
            episodes[key] = {
                "dataset": value["dataset"], "episode_id": value["episode_id"],
                "scene_id": value["scene_id"], "source_native_trace": value["source_native_trace"],
            }
        elif (prior["scene_id"], prior["source_native_trace"]["path"]) != (
            value["scene_id"], value["source_native_trace"]["path"]
        ):
            raise MF3ZPError(f"same episode has inconsistent native provenance: {key}")
    return resolved, sorted(episodes.values(), key=lambda value: (value["dataset"], value["episode_id"]))


def build_protocol() -> dict:
    events, episodes = load_population()
    mf3zo_sha = sha256_file(MF3ZO_PROTOCOL)
    if mf3zo_sha != EXPECTED_MF3ZO_SHA:
        raise MF3ZPError("MF3ZO parent protocol SHA drift")
    parent_sha = sha256_file(PARENT_MF3ZN)
    if parent_sha != EXPECTED_PARENT_MF3ZN_SHA:
        raise MF3ZPError("MF3ZN parent protocol SHA drift")
    mf3zo = strict_json(MF3ZO_PROTOCOL)
    if mf3zo.get("old_confirmation_reused") is not False:
        raise MF3ZPError("MF3ZO old-confirmation boundary drift")
    selection = strict_json(SELECTION)
    fold_by_event = dict(zip(
        [str(value) for value in mf3zo["pilot"]["event_ids"]],
        [int(value) for value in mf3zo["pilot"]["event_folds"]],
    ))
    if set(fold_by_event) != {value["event_id"] for value in events}:
        raise MF3ZPError("MF3ZO fold inventory does not match population")
    implementation = [
        METHOD_PATH,
        "revealnav_mf3/mf3zp_reference.py",
        "revealnav_mf3/MF3ZP_QWEN_SYSTEM_PROMPT.md",
        "scripts/mf3zp_observation_worker.py",
        "scripts/run_mf3zp_qwen_reference.py",
    ]
    impl_inventory = {path: inventory(ROOT / path) for path in implementation}
    return {
        "schema_version": SCHEMA,
        "revision": REVISION,
        "status": STATUS,
        "scientific_scope": "Qwen-assisted reference observation/label data only; no deployment or public evaluation",
        "source_protocols": {
            "mf3zo": inventory(MF3ZO_PROTOCOL),
            "mf3zn_parent": inventory(PARENT_MF3ZN),
            "mf3zk_dsr_tier_source": inventory(MF3ZK_DSR_PROTOCOL),
            "mf3zo_selection": inventory(SELECTION),
        },
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "implementation_inventory": impl_inventory,
        "population": {
            "events": events,
            "event_count": len(events),
            "episode_count": len(episodes),
            "domain_counts": dict(Counter(value["dataset"] for value in events)),
            "scene_count": len({value["scene_id"] for value in events}),
            "event_ids_sha256": stable_sha256([value["event_id"] for value in events]),
            "fold_by_event": fold_by_event,
            "episodes": episodes,
        },
        "consumed_confirmation_blacklist": sorted(CONFIRMATION_BLACKLIST),
        "observation": {
            "split": "train",
            "native_only": True,
            "panorama_yaws_deg": list(range(0, 360, 30)),
            "prefix_rule": "all replay prefixes j <= native episode terminal step",
            "candidate_identity_rule": "exact frozen GraphMap action ID; mismatch fails closed",
            "geometry_record": "oracle-only, never sent to Qwen or inference tensors",
            "future_observation_input": False,
            "target_or_outcome_input": False,
        },
        "annotation": {
            "endpoint": ENDPOINT,
            "models": list(MODELS),
            "temperature": 0.0,
            "thinking": False,
            "response_format": {"type": "json_object"},
            "system_prompt": inventory(PROMPT),
            "response_schema": RESPONSE_SCHEMA,
            "one_request_per_model_per_event_prefix": True,
            "fixed_transport_retries": 3,
            "human_audit_required_for_formal_probe": True,
        },
        "oracle_derivation": {
            "stability_k": STABILITY_K,
            "target_in_set": "exact sealed target ID in replay executable candidate IDs",
            "expiry": "last replay prefix with exactly one target ID match",
            "reveal": "first stable K-prefix D interval from projected semantic labels",
            "resolvable": "reveal end <= expiry",
            "unavailable_is_not_imputed": True,
        },
        "human_audit": {
            "all_disagreement_events": True,
            "agreement_sample_events": 40,
            "reviewers": 2,
            "adjudicator": 1,
            "uad_cohen_kappa_min": 0.65,
            "evidence_closure_cohen_kappa_min": 0.70,
        },
        "fixed_probe_a_scout": {
            "model": "existing_mf3zo_probe_a_oracle_relevance",
            "ridge_l2": 1.0,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260831,
            "exploratory_only": True,
            "formal_probe_authorized": False,
        },
        "authorization": {
            "observation_replay": True,
            "qwen_provisional_annotation": True,
            "formal_verified_probe_a": False,
            "formal_teal_collection": False,
            "tuad_training": False,
            "checkpoint_generation": False,
            "public_split_access": {
                "val_seen": False, "val_unseen": False,
                "test": False, "test_challenge": False,
            },
        },
        "no_post_result_revision": True,
    }


def verify_protocol(value: Mapping[str, object] | None = None) -> dict:
    protocol = dict(value) if value is not None else strict_json(PROTOCOL)
    if protocol.get("schema_version") != SCHEMA or protocol.get("revision") != REVISION or protocol.get("status") != STATUS:
        raise MF3ZPError("MF3ZP protocol identity/status drift")
    access = protocol.get("authorization", {}).get("public_split_access")
    if access != {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False}:
        raise MF3ZPError("public split access is not fail-closed")
    if protocol.get("authorization", {}).get("formal_verified_probe_a") is not False:
        raise MF3ZPError("formal Probe-A authorization drift")
    if sha256_file(MF3ZO_PROTOCOL) != EXPECTED_MF3ZO_SHA or sha256_file(PARENT_MF3ZN) != EXPECTED_PARENT_MF3ZN_SHA:
        raise MF3ZPError("parent protocol changed")
    for item in protocol.get("implementation_inventory", {}).values():
        path = ROOT / str(item["path"])
        current = inventory(path)
        if current != item:
            raise MF3ZPError(f"sealed implementation source drift: {path}")
    for value in protocol.get("source_protocols", {}).values():
        path = ROOT / str(value["path"])
        if inventory(path) != value:
            raise MF3ZPError(f"sealed source drift: {path}")
    events, episodes = load_population()
    if protocol.get("population", {}).get("events") != events or protocol.get("population", {}).get("episodes") != episodes:
        raise MF3ZPError("sealed event/episode population drift")
    if protocol["population"]["event_count"] != 150 or protocol["population"]["episode_count"] != 149:
        raise MF3ZPError("MF3ZP population cardinality drift")
    return protocol


def _attempt_dir(obs_root: Path, dataset: str, episode: str) -> Path:
    base = obs_root / dataset / f"ep_{episode}"
    attempts = sorted(
        value for value in base.glob("attempt_*")
        if value.is_dir() and not value.is_symlink()
    ) if base.exists() else []
    number = len(attempts) + 1
    return base / f"attempt_{number:03d}"


def _run_one_episode(task: Mapping[str, object], obs_root: Path, gpu_id: int, max_attempts: int) -> dict:
    base = obs_root / str(task["dataset"]) / f"ep_{task['episode_id']}"
    base.mkdir(parents=True, exist_ok=True)
    for existing in sorted(base.glob("attempt_*/RUN_SUMMARY.json")):
        try:
            summary = strict_json(existing)
        except MF3ZPError:
            continue
        if summary.get("status") == "PASS" and summary.get("source_native_replay_exact") is True:
            return {"dataset": task["dataset"], "episode_id": task["episode_id"], "status": "SKIPPED_PASS", "run_dir": rel(existing.parent)}
    last_error = None
    for _ in range(max_attempts):
        run_dir = _attempt_dir(obs_root, str(task["dataset"]), str(task["episode_id"]))
        stdout = run_dir.with_name(run_dir.name + ".stdout")
        stderr = run_dir.with_name(run_dir.name + ".stderr")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(ROOT / ".envs/etpr1/bin/python"), str(WORKER),
            "--dataset", str(task["dataset"]), "--episode-id", str(task["episode_id"]),
            "--scene-id", str(task["scene_id"]),
            "--source-native-trace", str(ROOT / str(task["source_native_trace"]["path"])),
            "--run-dir", str(run_dir), "--gpu-id", str(gpu_id),
        ]
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu_id), "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT)})
        with stdout.open("w") as out, stderr.open("w") as err:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
        summary_path = run_dir / "RUN_SUMMARY.json"
        if summary_path.is_file():
            summary = strict_json(summary_path)
            if summary.get("status") == "PASS" and summary.get("source_native_replay_exact") is True:
                return {"dataset": task["dataset"], "episode_id": task["episode_id"], "status": "PASS", "run_dir": rel(run_dir), "returncode": result.returncode}
            last_error = summary.get("error", f"returncode={result.returncode}")
        else:
            last_error = f"missing RUN_SUMMARY.json (returncode={result.returncode})"
    return {"dataset": task["dataset"], "episode_id": task["episode_id"], "status": "FAIL", "error": str(last_error)}


def collect(protocol: Mapping[str, object], max_workers: int, gpu_ids: list[int], max_attempts: int) -> dict:
    verify_protocol(protocol)
    obs_root = OUTPUT / "observations"
    obs_root.mkdir(parents=True, exist_ok=True)
    tasks = list(protocol["population"]["episodes"])
    if not gpu_ids or any(value < 0 for value in gpu_ids):
        raise MF3ZPError("at least one non-negative GPU ID is required")
    status_path = OUTPUT / "MF3ZP_OBSERVATION_COLLECTION_STATUS.json"
    results: list[dict] = []
    started = time.time()
    def save_status() -> None:
        with _STATUS_LOCK:
            atomic_json(status_path, {
                "schema_version": "revealnav-mf3zp-collection-status/1",
                "status": "RUNNING",
                "planned_episodes": len(tasks),
                "completed_records": len(results),
                "pass": sum(value["status"] in {"PASS", "SKIPPED_PASS"} for value in results),
                "fail": sum(value["status"] == "FAIL" for value in results),
                "results": sorted(results, key=lambda value: (value["dataset"], value["episode_id"])),
                "outcome_payload_read": False,
                "public_split_access": False,
                "elapsed_seconds": round(time.time() - started, 3),
            })
    save_status()
    with ThreadPoolExecutor(max_workers=min(max_workers, len(gpu_ids), len(tasks))) as pool:
        futures = {
            pool.submit(_run_one_episode, task, obs_root, gpu_ids[index % len(gpu_ids)], max_attempts): task
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            save_status()
    failures = [value for value in results if value["status"] == "FAIL"]
    final = {
        "schema_version": "revealnav-mf3zp-observation-collection/1",
        "status": "PASS" if not failures and len(results) == len(tasks) else "FAIL",
        "planned_episodes": len(tasks), "completed_episodes": len(results),
        "failures": failures,
        "results": sorted(results, key=lambda value: (value["dataset"], value["episode_id"])),
        "outcome_payload_read": False, "public_split_access": False,
    }
    atomic_json(OUTPUT / "MF3ZP_OBSERVATION_COLLECTION_MANIFEST.json", final)
    return final


def _load_train_instructions() -> dict[tuple[str, str], str]:
    import gzip
    paths = {
        "R2R": ROOT / "third_party/ETP-R1/data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz",
        "RxR": ROOT / "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz",
    }
    result: dict[tuple[str, str], str] = {}
    for dataset, path in paths.items():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        for episode in payload.get("episodes", []):
            eid = str(episode.get("episode_id"))
            instruction = episode.get("instruction", {}).get("instruction_text")
            if isinstance(instruction, str) and instruction.strip():
                if dataset == "RxR":
                    language = str(episode.get("instruction", {}).get("language", ""))
                    if language not in {"en-US", "en-IN", ""}:
                        continue
                result.setdefault((dataset, eid), instruction)
    return result


def _front_crop(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise MF3ZPError(f"cannot read panorama: {path}")
    height, width = image.shape[:2]
    if width % 4 or height % 3:
        raise MF3ZPError(f"unexpected panorama dimensions: {path}")
    return image[: height // 3, : width // 4]


def _build_storyboard(panorama_paths: list[tuple[int, Path]], destination: Path) -> dict:
    panes: list[np.ndarray] = []
    for step, path in panorama_paths:
        crop = _front_crop(path).copy()
        cv2.putText(crop, f"prefix {step:03d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panes.append(crop)
    if not panes:
        raise MF3ZPError("empty causal storyboard")
    cols = min(4, len(panes))
    rows = [(panes[index:index + cols] + [np.zeros_like(panes[0])] * (cols - len(panes[index:index + cols]))) for index in range(0, len(panes), cols)]
    sheet_rows = [np.concatenate(row, axis=1) for row in rows]
    sheet = np.concatenate(sheet_rows, axis=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part.jpg")
    if not cv2.imwrite(str(partial), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise MF3ZPError(f"failed to write storyboard: {destination}")
    os.replace(partial, destination)
    return {"path": rel(destination), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "steps": [value[0] for value in panorama_paths]}


def assemble(protocol: Mapping[str, object]) -> dict:
    verify_protocol(protocol)
    collection = strict_json(OUTPUT / "MF3ZP_OBSERVATION_COLLECTION_MANIFEST.json")
    if collection.get("status") != "PASS":
        raise MF3ZPError("observation collection is incomplete")
    instructions = _load_train_instructions()
    event_rows = list(protocol["population"]["events"])
    projection_rows: list[dict] = []
    deterministic_rows: list[dict] = []
    request_rows: list[dict] = []
    for event in event_rows:
        key = (event["dataset"], event["episode_id"])
        if key not in instructions:
            raise MF3ZPError(f"missing train instruction: {key}")
        run_candidates = sorted((OUTPUT / "observations" / event["dataset"] / f"ep_{event['episode_id']}").glob("attempt_*/RUN_SUMMARY.json"))
        passing = [path for path in run_candidates if strict_json(path).get("status") == "PASS"]
        if len(passing) != 1:
            raise MF3ZPError(f"expected exactly one passing observation run for {key}")
        run_dir = passing[0].parent
        records = jsonl(run_dir / "causal_prefix_records.jsonl")
        panoramas = {int(value["step"]): ROOT / str(value["path"]) for value in jsonl(run_dir / "panorama_manifest.jsonl")}
        if set(panoramas) != {int(value["step"]) for value in records}:
            raise MF3ZPError(f"observation/panorama prefix mismatch: {key}")
        union = sorted({candidate for row in records for candidate in row["candidate_action_ids"]} | {event["native_action_id"], event["alternative_action_id"]})
        aliases = branch_aliases(event["event_id"], union)
        native_alias = aliases[event["native_action_id"]]
        target_alias = aliases[event["alternative_action_id"]]
        projection_rows.append({
            "event_id": event["event_id"], "dataset": event["dataset"], "scene_id": event["scene_id"],
            "episode_id": event["episode_id"], "decision_step": event["decision_step"],
            "native_alias": native_alias, "alternative_alias": target_alias,
            "aliases": aliases, "observation_run_dir": rel(run_dir),
        })
        expiry_candidates = []
        for row in records:
            step = int(row["step"])
            candidates = set(str(value) for value in row["candidate_action_ids"])
            target_present = event["alternative_action_id"] in candidates
            native_present = event["native_action_id"] in candidates
            if target_present:
                expiry_candidates.append(step)
            deterministic_rows.append({
                "schema_version": "revealnav-mf3zp-deterministic-oracle/1",
                "event_id": event["event_id"], "prefix_step": step,
                "target_in_set": target_present, "native_in_set": native_present,
                "candidate_count": len(candidates), "source": "frozen_native_exact_action_id",
            })
        expiry = max(expiry_candidates) if expiry_candidates else None
        for row in records:
            step = int(row["step"])
            current_candidates = []
            # Candidate cards expose only role-blind aliases and geometry-free
            # heading metadata.  Policy score/rank and raw IDs stay private.
            headings = list(row.get("candidate_relative_heading_rad", []))
            for index, candidate in enumerate(row["candidate_action_ids"]):
                card = {"alias": aliases[str(candidate)]}
                if index < len(headings) and np.isfinite(float(headings[index])):
                    card["relative_heading_rad"] = round(float(headings[index]), 6)
                current_candidates.append(card)
            prefix_panels = [(int(value["step"]), panoramas[int(value["step"])]) for value in records if int(value["step"]) <= step]
            storyboard = _build_storyboard(prefix_panels, run_dir / "storyboards" / f"prefix_{step:03d}.jpg")
            frames = [{"step": int(value["step"]), "frame_id": f"P{int(value['step']):03d}", "storyboard_path": storyboard["path"]} for value in records if int(value["step"]) <= step]
            contract = build_annotation_contract(
                event_id=event["event_id"], prefix_step=step, instruction=instructions[key],
                chronological_frames=frames,
                current_candidates=current_candidates,
            )
            request_rows.append({
                "request_id": stable_sha256({"event_id": event["event_id"], "prefix_step": step}),
                "event_id": event["event_id"], "dataset": event["dataset"], "scene_id": event["scene_id"],
                "episode_id": event["episode_id"], "prefix_step": step,
                "contract": contract,
                "user_text": annotation_user_text(contract),
                "current_panorama": {"path": rel(panoramas[step]), "bytes": panoramas[step].stat().st_size, "sha256": sha256_file(panoramas[step])},
                "causal_storyboard": storyboard,
            })
    if len({value["request_id"] for value in request_rows}) != len(request_rows):
        raise MF3ZPError("annotation request identity collision")
    # A static recursive scan is intentionally performed before any network
    # call.  These request rows must contain no target/outcome vocabulary.
    forbidden = ("delta_utility", "counterfactual_outcome", "catastrophic", "treatment_result", "model_prediction", "fold")
    encoded = json.dumps(request_rows, sort_keys=True, ensure_ascii=False).lower()
    if any(token in encoded for token in forbidden):
        raise MF3ZPError("forbidden outcome field entered annotation queue")
    atomic_jsonl(OUTPUT / "MF3ZP_DETERMINISTIC_ORACLE.jsonl", deterministic_rows, refuse_existing=True)
    atomic_jsonl(OUTPUT / "MF3ZP_PROJECTION_MAP.jsonl", projection_rows, refuse_existing=True)
    atomic_jsonl(OUTPUT / "MF3ZP_ANNOTATION_REQUESTS.jsonl", request_rows, refuse_existing=True)
    manifest = {
        "schema_version": "revealnav-mf3zp-annotation-input-manifest/1",
        "status": "SEALED_BEFORE_QWEN_RESPONSES",
        "events": len(event_rows), "request_count": len(request_rows),
        "models": list(MODELS), "target_payload_read": False,
        "outcome_payload_read": False, "public_split_access": False,
        "request_file": inventory(OUTPUT / "MF3ZP_ANNOTATION_REQUESTS.jsonl"),
        "deterministic_oracle": inventory(OUTPUT / "MF3ZP_DETERMINISTIC_ORACLE.jsonl"),
        "projection_map": inventory(OUTPUT / "MF3ZP_PROJECTION_MAP.jsonl"),
    }
    atomic_json(OUTPUT / "MF3ZP_ANNOTATION_INPUT_MANIFEST.json", manifest, refuse_existing=True)
    return manifest


def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_").replace("-", "_")


def _api_key() -> str:
    path = ROOT / ".secret/qwen_api_key"
    if not path.is_file() or path.is_symlink():
        raise MF3ZPError("project-local .secret/qwen_api_key is missing")
    key = path.read_text(encoding="utf-8").strip()
    if not key or any(ch.isspace() for ch in key):
        raise MF3ZPError("invalid Qwen API key file")
    return key


def _image_data(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _request_qwen(api_key: str, model: str, row: Mapping[str, object]) -> dict:
    import urllib.error
    import urllib.request
    content = [
        {"type": "text", "text": str(row["user_text"])},
        {"type": "image_url", "image_url": {"url": _image_data(ROOT / str(row["causal_storyboard"]["path"]))}},
        {"type": "image_url", "image_url": {"url": _image_data(ROOT / str(row["current_panorama"]["path"]))}},
    ]
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": PROMPT.read_text(encoding="utf-8")},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 1200,
        "enable_thinking": False,
    }
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body["choices"][0]["message"]["content"]
            if isinstance(message, list):
                message = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in message)
            parsed = json.loads(str(message))
            return {"provider_model": body.get("model"), "response": parsed, "usage": body.get("usage")}
        except (OSError, ValueError, KeyError, TypeError, urllib.error.HTTPError) as error:
            # Do not persist response bodies or API headers; they may contain
            # credentials or provider diagnostics unrelated to the experiment.
            last = type(error).__name__
            if attempt < 2:
                time.sleep(2.0)
    raise MF3ZPError(f"fixed Qwen request failed: {last}")


def annotate(protocol: Mapping[str, object], max_workers: int) -> dict:
    verify_protocol(protocol)
    manifest = strict_json(OUTPUT / "MF3ZP_ANNOTATION_INPUT_MANIFEST.json")
    if manifest.get("status") != "SEALED_BEFORE_QWEN_RESPONSES":
        raise MF3ZPError("annotation input manifest is not sealed")
    requests = jsonl(OUTPUT / "MF3ZP_ANNOTATION_REQUESTS.jsonl")
    api_key = _api_key()
    response_root = OUTPUT / "responses"
    response_root.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for model in MODELS:
        model_dir = response_root / _model_slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        todo = []
        for row in requests:
            path = model_dir / f"{row['request_id']}.json"
            if path.is_file() and not path.is_symlink():
                try:
                    existing = strict_json(path)
                    if existing.get("status") == "PASS":
                        continue
                except MF3ZPError:
                    pass
            todo.append((row, path))
        def one(item):
            row, path = item
            try:
                result = _request_qwen(api_key, model, row)
                errors = validate_annotation_response(
                    result["response"], event_id=str(row["event_id"]),
                    prefix_step=int(row["prefix_step"]),
                    allowed_aliases=[value["alias"] for value in row["contract"]["current_candidates"]],
                )
                if errors:
                    raise MF3ZPError("schema validation: " + ",".join(errors))
                output = {
                    "schema_version": "revealnav-mf3zp-qwen-response/1",
                    "status": "PASS", "model": model,
                    "request_id": row["request_id"], "event_id": row["event_id"],
                    "prefix_step": row["prefix_step"], "provider_model": result.get("provider_model"),
                    "response": result["response"], "usage": result.get("usage"),
                }
            except BaseException as error:
                output = {
                    "schema_version": "revealnav-mf3zp-qwen-response/1",
                    "status": "FAIL", "model": model,
                    "request_id": row["request_id"], "event_id": row["event_id"],
                    "prefix_step": row["prefix_step"], "error": str(error),
                }
            atomic_json(path, output)
            return output
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(one, item) for item in todo]
            for future in as_completed(futures):
                all_summaries.append(future.result())
    failures = [value for value in all_summaries if value.get("status") != "PASS"]
    # Include already-existing responses in the aggregate count.
    total = 0
    passed = 0
    for model in MODELS:
        for path in (response_root / _model_slug(model)).glob("*.json"):
            total += 1
            try:
                passed += strict_json(path).get("status") == "PASS"
            except MF3ZPError:
                pass
    result = {
        "schema_version": "revealnav-mf3zp-qwen-annotation/1",
        "status": "PASS" if not failures and total == len(requests) * len(MODELS) and passed == total else "FAIL",
        "models": list(MODELS), "planned": len(requests) * len(MODELS),
        "response_files": total, "pass": passed, "failures": failures,
        "target_payload_read": False, "outcome_payload_read": False, "public_split_access": False,
    }
    atomic_json(OUTPUT / "MF3ZP_QWEN_ANNOTATION_MANIFEST.json", result)
    return result


def scout(protocol: Mapping[str, object]) -> dict:
    """Run the exploratory Probe-A scout only after all Qwen responses exist."""
    verify_protocol(protocol)
    annotation = strict_json(OUTPUT / "MF3ZP_QWEN_ANNOTATION_MANIFEST.json")
    if annotation.get("status") != "PASS":
        raise MF3ZPError("all fixed Qwen responses are required before scout")
    requests = jsonl(OUTPUT / "MF3ZP_ANNOTATION_REQUESTS.jsonl")
    projection = {row["event_id"]: row for row in jsonl(OUTPUT / "MF3ZP_PROJECTION_MAP.jsonl")}
    deterministic = {}
    for row in jsonl(OUTPUT / "MF3ZP_DETERMINISTIC_ORACLE.jsonl"):
        deterministic.setdefault(row["event_id"], []).append(row)
    labels_by_model = {}
    for model in MODELS:
        model_dir = OUTPUT / "responses" / _model_slug(model)
        by_event: dict[str, list[dict]] = {}
        for request in requests:
            response_path = model_dir / f"{request['request_id']}.json"
            response = strict_json(response_path)["response"]
            aliases = [value["alias"] for value in request["contract"]["current_candidates"]]
            errors = validate_annotation_response(response, event_id=request["event_id"], prefix_step=int(request["prefix_step"]), allowed_aliases=aliases)
            if errors:
                raise MF3ZPError(f"stored response failed validation: {response_path}")
            projected = derive_semantic_state(
                response,
                target_alias=projection[request["event_id"]]["alternative_alias"],
                native_alias=projection[request["event_id"]]["native_alias"],
                target_present=next(
                    value["target_in_set"]
                    for value in deterministic[request["event_id"]]
                    if int(value["prefix_step"]) == int(request["prefix_step"])
                ),
            )
            by_event.setdefault(request["event_id"], []).append({"prefix_step": int(request["prefix_step"]), **projected, "raw_response": response})
        labels = {}
        for event_id, rows in by_event.items():
            rows.sort(key=lambda value: value["prefix_step"])
            target = tuple(value["target_in_set"] for value in deterministic[event_id])
            separated = tuple(value["candidate_separated"] for value in rows)
            closed = tuple(value["evidence_closed"] for value in rows)
            if len(target) != len(rows):
                raise MF3ZPError(f"deterministic/semantic prefix mismatch: {event_id}")
            states = derive_uad(target, separated, closed, stability_k=STABILITY_K)
            reveal = reveal_interval(states)
            expiry_values = [int(value["prefix_step"]) for value in deterministic[event_id] if value["target_in_set"]]
            expiry = max(expiry_values) if expiry_values else None
            labels[event_id] = {
                "event_id": event_id, "target_in_set": target,
                "candidate_separated": separated, "evidence_closed": closed,
                "uad": tuple(value.value for value in states),
                "reveal_interval": reveal, "expiry_step": expiry,
                "resolvable": resolvable(reveal, expiry),
                "provenance": "qwen_provisional_unverified",
            }
        labels_by_model[model] = labels
        atomic_jsonl(OUTPUT / f"MF3ZP_PROVISIONAL_LABELS_{_model_slug(model)}.jsonl", (labels[key] for key in sorted(labels)), refuse_existing=True)
    # Only now, after all provisional labels are frozen, open exact outcomes.
    car = ROOT / "scripts/train_mf3zm_car.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("mf3zp_car_source", car)
    if spec is None or spec.loader is None:
        raise MF3ZPError("cannot load exact-outcome source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_protocol()
    outcome_rows = module._canonical_rows()
    outcome_by_event = {str(row["event_id"]): float(row["target"]) for row in outcome_rows if "event_id" in row}
    # Older source rows derive identity; use the project helper when needed.
    if len(outcome_by_event) != len(outcome_rows):
        from revealnav_mf3.mf3zo_pilot import canonical_event_id
        outcome_by_event = {
            canonical_event_id(row["dataset"], row["scene_id"], row["episode_id"], int(row["decision"]["step"])): float(row["target"])
            for row in outcome_rows
        }
    if not all(event_id in outcome_by_event for event_id in labels_by_model[MODELS[0]]):
        raise MF3ZPError("exact outcome identity does not cover MF3ZP events")
    from revealnav_mf3.mf3zo_probes import current_snapshot_features, probe_a_oracle_relevance, oracle_feature_vector
    from revealnav_mf3.mf3zo_pilot import load_causal_records
    records = load_causal_records(ROOT / "artifacts/training/mf3zo_temporal_oracle_gap_v1/MF3ZO_CAUSAL_TEMPORAL_RECORDS.jsonl", ROOT)
    event_rows = list(protocol["population"]["events"])
    event_ids = [value["event_id"] for value in event_rows]
    current = np.stack([current_snapshot_features(records[event_id]) for event_id in event_ids])
    scenes = np.asarray([value["scene_id"] for value in event_rows])
    domains = np.asarray([value["dataset"] for value in event_rows])
    folds = np.asarray([protocol["population"]["fold_by_event"][event_id] for event_id in event_ids])
    scout_results = {}
    for model in MODELS:
        from revealnav_mf3.mf3zo_temporal_schema import TemporalOracleLabel
        oracle_features = []
        for event_id in event_ids:
            label = labels_by_model[model][event_id]
            oracle_label = TemporalOracleLabel(
                event_id=event_id,
                target_in_set=tuple(label["target_in_set"]),
                candidate_separated=tuple(label["candidate_separated"]),
                evidence_closed=tuple(label["evidence_closed"]),
                reveal_interval=label["reveal_interval"], expiry_step=label["expiry_step"],
                resolvable=bool(label["resolvable"]), unavailable_fields=(),
                provenance="qwen_provisional_unverified",
            )
            oracle_features.append(oracle_feature_vector(oracle_label, int(next(value["decision_step"] for value in event_rows if value["event_id"] == event_id))))
        result = probe_a_oracle_relevance(current, np.stack(oracle_features), np.asarray([outcome_by_event[event_id] for event_id in event_ids]), scenes, domains, folds)
        result.update({"annotator_model": model, "exploratory": True, "human_verified": False, "formal_authorized": False, "target_payload_read": True})
        scout_results[model] = result
        atomic_json(OUTPUT / f"MF3ZP_EXPLORATORY_PROBE_A_{_model_slug(model)}.json", result, refuse_existing=True)
    positive = all(result["status"] == "ORACLE_RELEVANCE_PASS" for result in scout_results.values())
    final = {
        "schema_version": "revealnav-mf3zp-final/1",
        "status": "EXPLORATORY_QWEN_ORACLE_SIGNAL_POSITIVE" if positive else "EXPLORATORY_QWEN_ORACLE_SIGNAL_NOT_POSITIVE",
        "models": scout_results, "human_verified": False,
        "formal_probe_a_authorized": False, "checkpoint_generated": False,
        "public_split_access": False, "target_payload_read": True,
        "stop_rule": False,
    }
    atomic_json(OUTPUT / "MF3ZP_EXPLORATORY_SCOUT_RESULT.json", final, refuse_existing=True)
    return final


def status() -> dict:
    if not PROTOCOL.is_file():
        return {"status": "NOT_SEALED", "output": rel(OUTPUT)}
    protocol = strict_json(PROTOCOL)
    result = {"status": protocol.get("status"), "protocol_sha256": sha256_file(PROTOCOL), "output": rel(OUTPUT)}
    collection = OUTPUT / "MF3ZP_OBSERVATION_COLLECTION_MANIFEST.json"
    if collection.is_file():
        result["collection"] = strict_json(collection)
    annotation = OUTPUT / "MF3ZP_QWEN_ANNOTATION_MANIFEST.json"
    if annotation.is_file():
        result["annotation"] = strict_json(annotation)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal")
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--max-workers", type=int, default=2)
    collect_parser.add_argument("--gpu-ids", default="0,1")
    collect_parser.add_argument("--max-attempts", type=int, default=2)
    sub.add_parser("assemble")
    annotate_parser = sub.add_parser("annotate")
    annotate_parser.add_argument("--max-workers", type=int, default=8)
    sub.add_parser("scout")
    sub.add_parser("status")
    args = parser.parse_args()
    try:
        if args.command == "seal":
            if PROTOCOL.exists() or PROTOCOL.is_symlink():
                raise MF3ZPError("MF3ZP protocol already exists; resealing is forbidden")
            OUTPUT.mkdir(parents=True, exist_ok=True)
            value = build_protocol()
            atomic_json(PROTOCOL, value, refuse_existing=True)
            print(json.dumps({"status": value["status"], "protocol_sha256": sha256_file(PROTOCOL), "events": value["population"]["event_count"], "episodes": value["population"]["episode_count"]}, indent=2))
        elif args.command == "collect":
            print(json.dumps(collect(strict_json(PROTOCOL), args.max_workers, [int(value) for value in args.gpu_ids.split(",") if value.strip()], args.max_attempts), indent=2))
        elif args.command == "assemble":
            print(json.dumps(assemble(strict_json(PROTOCOL)), indent=2))
        elif args.command == "annotate":
            print(json.dumps(annotate(strict_json(PROTOCOL), args.max_workers), indent=2))
        elif args.command == "scout":
            print(json.dumps(scout(strict_json(PROTOCOL)), indent=2))
        else:
            print(json.dumps(status(), indent=2))
        return 0
    except BaseException as error:
        print(f"MF3ZP_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
