#!/usr/bin/env python3
"""MF3ZP v2: prefix-safe Qwen reference annotation pipeline.

This is a new, independently sealed protocol.  MF3ZP v1 is intentionally
left untouched as an engineering-failure record: several legacy ``base``
traces contained the one switched action.  v2 uses an independent native
baseline where available and otherwise treats the old trace only as a
pre-decision prefix witness.  All annotation, assembly, and scout operations
are delegated to the sealed v1 implementation after its paths and verifier
are replaced by the v2 boundary.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import as_completed
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = ROOT / "scripts/run_mf3zp_qwen_reference.py"
V1_PROTOCOL = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v1/MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json"
V2_WORKER = ROOT / "scripts/mf3zp_observation_worker_v2.py"
V2_METHOD = "METHOD_REVISION_3ZP_QWEN_UAD_REFERENCE_V2.md"
V2_OUTPUT = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2"
V2_PROTOCOL = V2_OUTPUT / "MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json"
V2_REVISION = "mf3zp_qwen_uad_reference_v2"
V2_SCHEMA = "revealnav-mf3zp-qwen-uad-reference/2"
V2_STATUS = "SEALED_BEFORE_MF3ZP_V2_OBSERVATION_OR_LABEL_RESULTS"


def _load_v1():
    spec = importlib.util.spec_from_file_location("mf3zp_qwen_reference_v1_helper", V1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sealed MF3ZP v1 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = _load_v1()

# Replace only operational paths.  The v1 protocol and implementation files
# remain immutable inputs and are explicitly inventoried by v2.
m.OUTPUT = V2_OUTPUT
m.PROTOCOL = V2_PROTOCOL
m.REVISION = V2_REVISION
m.SCHEMA = V2_SCHEMA
m.STATUS = V2_STATUS
m.WORKER = V2_WORKER
m.METHOD_PATH = V2_METHOD


def _regular_project_file(path: Path) -> Path:
    resolved = path.resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or ROOT.resolve() not in resolved.parents
    ):
        raise m.MF3ZPError(f"invalid v2 source trace: {path}")
    return resolved


def _native_candidate_path(event: dict, old_path: Path) -> tuple[Path, str]:
    """Choose a trace without using any outcome or target metric.

    The R2R baseline-completion runs are explicitly native-only.  The
    mf3zl native collection is also native-only.  Legacy mf3zk RxR runs may
    contain a treatment action; those paths are retained only as prefix
    witnesses and are never compared at the target step.
    """

    source = str(event["source"])
    dataset = str(event["dataset"])
    episode = str(event["episode_id"])
    if source == "mf3zk_dsr_v1_existing_exact" and dataset == "R2R":
        candidate = (
            ROOT / "artifacts/training/mf3zk_joint_v1/r2r_baseline_completion/runs"
            / f"ep_{episode}/base_trace.jsonl"
        )
        if candidate.is_file() and not candidate.is_symlink():
            return _regular_project_file(candidate), "native_reference"
    if source == "mf3zk_dsr_v1_existing_exact" and dataset == "RxR":
        candidate = (
            ROOT / "artifacts/training/mf3zl_rcsp_v1/runs/native/rxr"
            / f"ep_{episode}/attempt_001/base_trace.jsonl"
        )
        if candidate.is_file() and not candidate.is_symlink():
            return _regular_project_file(candidate), "native_reference"
    if source in {"mf3zl_parent_dense_exact", "mf3zl_v1r1_variant_exact"}:
        return _regular_project_file(old_path), "native_reference"
    return _regular_project_file(old_path), "prefix_witness"


def load_population_v2() -> tuple[list[dict], list[dict]]:
    selection = m.strict_json(m.SELECTION)
    if selection.get("status") != "SEALED_INPUT_POPULATION_SELECTED_OUTCOME_BLIND":
        raise m.MF3ZPError("MF3ZO selection status drift")
    raw_events = selection.get("events")
    if not isinstance(raw_events, list) or len(raw_events) != 150:
        raise m.MF3ZPError("MF3ZO fixed population drift")
    if Counter(str(value.get("dataset")) for value in raw_events) != Counter({"R2R": 75, "RxR": 75}):
        raise m.MF3ZPError("MF3ZO domain allocation drift")
    events: list[dict] = []
    for raw in raw_events:
        event = m.resolve_event(raw)
        old_path = ROOT / str(event["source_native_trace"]["path"])
        chosen, mode = _native_candidate_path(event, old_path)
        event["source_trace_original"] = dict(event["source_native_trace"])
        event["source_native_trace"] = m.inventory(chosen)
        event["source_trace_mode"] = mode
        event["prefix_validation_rule"] = (
            "actual native actions equal source witness for indices < decision_step; target action is validated from observation"
        )
        events.append(event)
    if len({value["event_id"] for value in events}) != 150:
        raise m.MF3ZPError("MF3ZP event identities are not unique")
    if any(value["scene_id"] in m.CONFIRMATION_BLACKLIST for value in events):
        raise m.MF3ZPError("consumed confirmation scene entered v2 population")
    episodes: dict[tuple[str, str], dict] = {}
    for event in events:
        key = (event["dataset"], event["episode_id"])
        prior = episodes.get(key)
        if prior is None:
            episodes[key] = {
                "dataset": event["dataset"],
                "episode_id": event["episode_id"],
                "scene_id": event["scene_id"],
                "decision_step": int(event["decision_step"]),
                "source_native_trace": event["source_native_trace"],
                "source_trace_mode": event["source_trace_mode"],
            }
        else:
            if (
                prior["scene_id"], prior["source_native_trace"]["path"],
                prior["source_trace_mode"],
            ) != (
                event["scene_id"], event["source_native_trace"]["path"],
                event["source_trace_mode"],
            ):
                raise m.MF3ZPError(f"inconsistent v2 episode provenance: {key}")
            prior["decision_step"] = max(
                int(prior["decision_step"]), int(event["decision_step"])
            )
    return events, sorted(episodes.values(), key=lambda value: (value["dataset"], value["episode_id"]))


def _implementation_inventory() -> dict:
    paths = (
        V2_METHOD,
        "revealnav_mf3/mf3zp_reference.py",
        "revealnav_mf3/MF3ZP_QWEN_SYSTEM_PROMPT.md",
        "scripts/mf3zp_observation_worker_v2.py",
        "scripts/run_mf3zp_qwen_reference_v2.py",
        # Imported helpers are sealed dependencies, not mutable fallbacks.
        "scripts/mf3zp_observation_worker.py",
        "scripts/run_mf3zp_qwen_reference.py",
    )
    return {path: m.inventory(ROOT / path) for path in paths}


def build_protocol_v2() -> dict:
    events, episodes = load_population_v2()
    # Reuse the parent construction only for the already-audited source
    # inventory and fixed annotation constants, then replace all v2-specific
    # fields before sealing.
    old_loader = m.load_population
    try:
        m.load_population = load_population_v2
        base = m.build_protocol()
    finally:
        m.load_population = old_loader
    base.update({
        "schema_version": V2_SCHEMA,
        "revision": V2_REVISION,
        "status": V2_STATUS,
        "scientific_scope": "prefix-safe Qwen-assisted reference observation/label data only; no deployment or public evaluation",
        "implementation_inventory": _implementation_inventory(),
    })
    base["source_protocols"]["mf3zp_v1_engineering_attempt"] = m.inventory(V1_PROTOCOL)
    base["population"] = dict(base["population"])
    base["population"]["events"] = events
    base["population"]["episodes"] = episodes
    base["population"]["event_count"] = len(events)
    base["population"]["episode_count"] = len(episodes)
    base["observation"].update({
        "prefix_rule": "all replay prefixes j <= maximum sealed decision_step for the episode; no post-decision records are exported",
        "source_trace_policy": "native_reference when independently native; prefix_witness otherwise; source actions are compared only at indices < decision_step",
        "target_step_action_validation": "assembler checks observed native identity and executable candidate IDs against the sealed event",
        "post_decision_observation_exported": False,
    })
    base["authorization"].update({
        "formal_verified_probe_a": False,
        "formal_teal_collection": False,
        "tuad_training": False,
        "checkpoint_generation": False,
    })
    return base


_ORIGINAL_VERIFY = m.verify_protocol


def verify_protocol_v2(value=None) -> dict:
    protocol = dict(value) if value is not None else m.strict_json(V2_PROTOCOL)
    verified = _ORIGINAL_VERIFY(protocol)
    events, episodes = load_population_v2()
    if verified["population"]["events"] != events or verified["population"]["episodes"] != episodes:
        raise m.MF3ZPError("v2 sealed population drift")
    if any(event.get("source_trace_mode") not in {"native_reference", "prefix_witness"} for event in events):
        raise m.MF3ZPError("invalid v2 source trace mode")
    return verified


# All delegated operations call their module-global verifier; replace it only
# in the imported helper namespace, never in the historical v1 source file.
m.verify_protocol = verify_protocol_v2
m.load_population = load_population_v2


def _run_one_episode_v2(task: dict, obs_root: Path, gpu_id: int, max_attempts: int) -> dict:
    base_dir = obs_root / str(task["dataset"]) / f"ep_{task['episode_id']}"
    base_dir.mkdir(parents=True, exist_ok=True)
    for existing in sorted(base_dir.glob("attempt_*/RUN_SUMMARY.json")):
        try:
            summary = m.strict_json(existing)
        except m.MF3ZPError:
            continue
        if summary.get("status") == "PASS" and summary.get("source_prefix_replay_exact") is True:
            return {
                "dataset": task["dataset"], "episode_id": task["episode_id"],
                "status": "SKIPPED_PASS", "run_dir": m.rel(existing.parent),
            }
    last_error = None
    for _ in range(max_attempts):
        run_dir = m._attempt_dir(obs_root, str(task["dataset"]), str(task["episode_id"]))
        stdout = run_dir.with_name(run_dir.name + ".stdout")
        stderr = run_dir.with_name(run_dir.name + ".stderr")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(ROOT / ".envs/etpr1/bin/python"), str(V2_WORKER),
            "--dataset", str(task["dataset"]),
            "--episode-id", str(task["episode_id"]),
            "--scene-id", str(task["scene_id"]),
            "--source-native-trace", str(ROOT / str(task["source_native_trace"]["path"])),
            "--run-dir", str(run_dir),
            "--decision-step", str(int(task["decision_step"])),
            "--gpu-id", str(gpu_id),
            "--source-trace-mode", str(task["source_trace_mode"]),
        ]
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(ROOT),
        })
        with stdout.open("w") as out, stderr.open("w") as err:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
        summary_path = run_dir / "RUN_SUMMARY.json"
        if summary_path.is_file():
            summary = m.strict_json(summary_path)
            if summary.get("status") == "PASS" and summary.get("source_prefix_replay_exact") is True:
                return {
                    "dataset": task["dataset"], "episode_id": task["episode_id"],
                    "status": "PASS", "run_dir": m.rel(run_dir),
                    "returncode": result.returncode,
                }
            last_error = summary.get("error", f"returncode={result.returncode}")
        else:
            last_error = f"missing RUN_SUMMARY.json (returncode={result.returncode})"
    return {
        "dataset": task["dataset"], "episode_id": task["episode_id"],
        "status": "FAIL", "error": str(last_error),
    }


m._run_one_episode = _run_one_episode_v2


def _sealed_protocol() -> dict:
    if not V2_PROTOCOL.is_file() or V2_PROTOCOL.is_symlink():
        raise m.MF3ZPError("v2 protocol is not sealed")
    return m.strict_json(V2_PROTOCOL)


def _run_delegated(command: str, args: argparse.Namespace) -> dict:
    protocol = _sealed_protocol()
    if command == "collect":
        gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
        return m.collect(protocol, args.max_workers, gpu_ids, args.max_attempts)
    if command == "assemble":
        return m.assemble(protocol)
    if command == "annotate":
        return m.annotate(protocol, args.max_workers)
    if command == "scout":
        return m.scout(protocol)
    if command == "status":
        return m.status()
    raise AssertionError(command)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    collect = sub.add_parser("collect")
    collect.add_argument("--max-workers", type=int, default=2)
    collect.add_argument("--gpu-ids", default="0,1")
    collect.add_argument("--max-attempts", type=int, default=2)
    sub.add_parser("assemble")
    annotate = sub.add_parser("annotate")
    annotate.add_argument("--max-workers", type=int, default=8)
    sub.add_parser("scout")
    sub.add_parser("status")
    args = parser.parse_args()
    try:
        if args.command == "seal":
            if V2_PROTOCOL.exists() or V2_PROTOCOL.is_symlink():
                raise m.MF3ZPError("v2 protocol already exists; resealing is forbidden")
            V2_OUTPUT.mkdir(parents=True, exist_ok=True)
            value = build_protocol_v2()
            m.atomic_json(V2_PROTOCOL, value, refuse_existing=True)
            print(json.dumps({
                "status": value["status"],
                "protocol_sha256": m.sha256_file(V2_PROTOCOL),
                "events": value["population"]["event_count"],
                "episodes": value["population"]["episode_count"],
            }, indent=2))
        else:
            print(json.dumps(_run_delegated(args.command, args), indent=2))
        return 0
    except BaseException as error:
        print(f"MF3ZP_V2_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
