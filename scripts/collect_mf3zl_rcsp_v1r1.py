#!/usr/bin/env python3
"""Outcome-blind R2R instruction-variant expansion for MF3ZL-RCSP v1."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts/training/mf3zl_rcsp_v1r1"
PROTOCOL = OUT / "MF3ZL_RCSP_V1R1_PROTOCOL.json"
SELECTION = OUT / "MF3ZL_R2R_VARIANT_SELECTION.json"
TARGETS = OUT / "MF3ZL_R2R_VARIANT_TARGETS.json"
MANIFEST = OUT / "MF3ZL_R2R_VARIANT_MANIFEST.json"
AUDIT = OUT / "MF3ZL_V1R1_DATA_SUPPORT_AUDIT.json"
NATIVE_PROGRESS = OUT / "MF3ZL_R2R_VARIANT_NATIVE_PROGRESS.json"
TARGET_PROGRESS = OUT / "MF3ZL_R2R_VARIANT_TARGET_PROGRESS.json"
R2R_DATA = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "R2R_VLNCE_v1-3_preprocessed_xlmr/train/train.json.gz"
)
PARENT_OUT = ROOT / "artifacts/training/mf3zl_rcsp_v1"
PARENT_PROTOCOL = PARENT_OUT / "MF3ZL_RCSP_PROTOCOL.json"
PARENT_SELECTION = PARENT_OUT / "MF3ZL_EXACT_REPLAY_SELECTION.json"
PARENT_MANIFEST = PARENT_OUT / "MF3ZL_EXACT_REPLAY_MANIFEST.json"
PARENT_AUDIT = PARENT_OUT / "MF3ZL_DATA_SUPPORT_AUDIT.json"
DSR_PROTOCOL = ROOT / "artifacts/training/mf3zk_dsr_v1/MF3ZK_DSR_PROTOCOL.json"
OLD_R2R_SELECTION = ROOT / (
    "artifacts/training/mf3zk_joint_v1/r2r_collection/"
    "MF3ZK_R2R_COLLECTION_SELECTION.json"
)
WORKER = ROOT / "scripts/mf3zl_exact_replay_worker.py"
LEGACY_COLLECTOR = ROOT / "scripts/collect_mf3zl_exact_replay.py"
SCHEMA = "revealnav-mf3zl-r2r-variant-expansion"
PUBLIC_TOKENS = {"val_seen", "val_unseen", "test", "test_challenge"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if part.exists() or part.is_symlink():
        raise RuntimeError(f"stale atomic partial: {part}")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def inventory(path: Path) -> dict:
    resolved = path.resolve()
    if ROOT not in resolved.parents or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"invalid project-local source: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _scene(episode: dict) -> str:
    parts = str(episode["scene_id"]).split("/")
    if len(parts) < 2 or len(parts[-2]) != 11:
        raise RuntimeError("R2R scene identity drift")
    return parts[-2]


def _sort_id(value: object) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _load_json_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as stream:
        value = json.load(stream)
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError(f"invalid episode payload: {path}")
    return episodes


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location(
        "mf3zl_parent_collector", LEGACY_COLLECTOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load parent exact-replay collector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_legacy(module) -> None:
    """Reuse the audited worker/orchestration mechanics with new output roots."""
    module.OUT = OUT
    module.PROTOCOL = PROTOCOL
    module.SELECTION = SELECTION
    module.TARGETS = TARGETS
    module.MANIFEST = MANIFEST
    module.AUDIT = AUDIT
    module.NATIVE_PROGRESS = NATIVE_PROGRESS
    module.TARGET_PROGRESS = TARGET_PROGRESS
    module.WORKER = WORKER
    module.SCHEMA = SCHEMA
    module.verify_protocol = verify_protocol


def _parent_route_keys() -> tuple[set[tuple[str, str]], set[str]]:
    parent = json.loads(PARENT_SELECTION.read_text())
    if (
        parent.get("status") != "SEALED_COMPLETE_UNUSED_TRAIN_POPULATION"
        or parent.get("public_split_access") is not False
    ):
        raise RuntimeError("parent v1 selection is not a sealed train population")
    routes = [row for row in parent["routes"] if row["dataset"] == "R2R"]
    if not routes:
        raise RuntimeError("parent v1 has no R2R routes")
    keys = {(str(row["scene_id"]), str(row["trajectory_id"])) for row in routes}
    canonical_ids = {str(row["episode_id"]) for row in routes}
    return keys, canonical_ids


def _historical_episode_ids() -> set[str]:
    """Collect only already sealed exact-replay episode identities."""
    result: set[str] = set()
    parent = json.loads(PARENT_PROTOCOL.read_text())
    for item in parent.get("source_inventory", {}).get("canonical_rows", []):
        identity = item.get("identity", {})
        if identity.get("dataset") == "R2R":
            result.add(str(identity["episode_id"]))
    return result


def build_selection() -> dict:
    if not PARENT_SELECTION.is_file() or not PARENT_PROTOCOL.is_file():
        raise RuntimeError("parent v1 seal is unavailable")
    allowed_keys, canonical_ids = _parent_route_keys()
    historical_ids = _historical_episode_ids()
    episodes = _load_json_gz(R2R_DATA)
    candidates = []
    seen_ids: set[str] = set()
    excluded_reasons = Counter()
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        scene = _scene(episode)
        key = (scene, str(episode["trajectory_id"]))
        if scene in PUBLIC_TOKENS or any(token in scene for token in PUBLIC_TOKENS):
            raise RuntimeError("public split token entered R2R train payload")
        if key not in allowed_keys:
            continue
        if len(episode.get("reference_path", [])) < 4:
            excluded_reasons["short_reference_path"] += 1
            continue
        if episode_id in canonical_ids:
            excluded_reasons["parent_canonical_episode"] += 1
            continue
        if episode_id in historical_ids:
            excluded_reasons["historical_exact_episode"] += 1
            continue
        if episode_id in seen_ids:
            raise RuntimeError("duplicate R2R episode identity in train payload")
        seen_ids.add(episode_id)
        candidates.append(episode)
    candidates.sort(key=lambda row: (
        _scene(row), _sort_id(row["trajectory_id"]), _sort_id(row["episode_id"])
    ))
    rows = []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for episode in candidates:
        key = (_scene(episode), str(episode["trajectory_id"]))
        grouped.setdefault(key, []).append(episode)
    for key, variants in sorted(grouped.items()):
        parent_digest = stable_hash({
            "scene_id": key[0], "trajectory_id": key[1], "revision": "mf3zl_rcsp_v1r1"
        })
        for rank, episode in enumerate(variants, start=1):
            instruction = episode.get("instruction", {})
            instruction_digest = hashlib.sha256(
                str(instruction.get("instruction_text", "")).encode()
            ).hexdigest()
            row = {
                "dataset": "R2R",
                "split": "train",
                "scene_id": key[0],
                "trajectory_id": key[1],
                "episode_id": str(episode["episode_id"]),
                "reference_points": len(episode["reference_path"]),
                "instruction_sha256": instruction_digest,
                "parent_route_digest": parent_digest,
                "variant_rank": rank,
            }
            row["selection_digest"] = stable_hash(row)
            rows.append(row)
    if not rows:
        raise RuntimeError("R2R variant population is empty")
    identities = [(row["scene_id"], row["trajectory_id"], row["episode_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate variant selection identity")
    if any(row["split"] != "train" for row in rows):
        raise RuntimeError("non-train route entered variant selection")
    return {
        "schema_version": f"{SCHEMA}-selection/1",
        "status": "SEALED_COMPLETE_R2R_INSTRUCTION_VARIANT_POPULATION",
        "revision": "mf3zl_rcsp_v1r1",
        "parent_revision": "mf3zl_rcsp_v1",
        "allowed_scene_ids": sorted({row["scene_id"] for row in rows}),
        "excluded_consumed_scene_ids": sorted(
            json.loads(DSR_PROTOCOL.read_text()).get(
                "known_consumed_scene_ids", []
            )
        ),
        "selection_rule": (
            "all remaining R2R train instruction episodes for the 1350 parent "
            "v1 unused trajectory representatives; canonical parent episodes "
            "and historical exact episodes excluded"
        ),
        "outcome_fields_used_for_selection": [],
        "adaptive_stopping_allowed": False,
        "public_split_access": False,
        "untouched_scenes_consumed": False,
        "counts": {
            "selected_episodes": len(rows),
            "selected_trajectories": len({
                (row["scene_id"], row["trajectory_id"]) for row in rows
            }),
            "selected_scenes": len({row["scene_id"] for row in rows}),
            "excluded_reasons": dict(excluded_reasons),
        },
        "sources": {
            "r2r_train": inventory(R2R_DATA),
            "parent_protocol": inventory(PARENT_PROTOCOL),
            "parent_selection": inventory(PARENT_SELECTION),
            "parent_manifest": inventory(PARENT_MANIFEST),
            "parent_audit": inventory(PARENT_AUDIT),
            "dsr_protocol": inventory(DSR_PROTOCOL),
            "old_r2r_selection": inventory(OLD_R2R_SELECTION),
        },
        "routes": rows,
    }


def _implementation_paths() -> tuple[Path, ...]:
    return (
        ROOT / "METHOD_REVISION_3ZL_RCSP_V1R1.md",
        ROOT / "scripts/collect_mf3zl_rcsp_v1r1.py",
        ROOT / "scripts/collect_mf3zl_exact_replay.py",
        ROOT / "scripts/mf3zl_exact_replay_worker.py",
        ROOT / "revealnav_mf3/exact_replay.py",
    )


def build_protocol(selection: dict) -> dict:
    return {
        "schema_version": "revealnav-mf3zl-r2r-expansion-protocol/1",
        "status": "SEALED_BEFORE_MF3ZL_R2R_EXPANSION",
        "revision": "mf3zl_rcsp_v1r1",
        "parent_revision": "mf3zl_rcsp_v1",
        "purpose": "outcome_blind R2R train instruction-variant support expansion",
        "parent_v1_status": "TRAIN_DATA_SUPPORT_FAIL_RETAINED_IMMUTABLE",
        "frozen_algorithm": {
            "name": "Risk-Constrained Counterfactual Switch Policy",
            "proposal_revision": "mf3zg",
            "utility": {"ndtw": 0.50, "sdtw": 0.25, "spl": 0.25},
            "catastrophic_threshold": -0.10,
            "decision_rule": "collection always abstains; one sealed runner-up switch in treatment",
        },
        "selection": {
            "path": str(SELECTION.relative_to(ROOT)),
            "bytes": SELECTION.stat().st_size,
            "sha256": sha256_file(SELECTION),
            "complete_population_required": True,
            "adaptive_stopping": False,
            "allowed_scene_ids": selection["allowed_scene_ids"],
            "parent_route_count": 1350,
            "selected_variant_count": selection["counts"]["selected_episodes"],
        },
        "data_gate": {
            "minimum_combined_exact_events_r2r": 300,
            "minimum_development_scenes_r2r": 30,
            "maximum_conflicting_identities": 0,
            "lower_gate_after_outcomes": False,
        },
        "public_split_access": {
            "val_seen": False, "val_unseen": False,
            "test": False, "test_challenge": False,
        },
        "authorization": {
            "trainer_may_authorize_confirmation": False,
            "trainer_may_authorize_public_unseen": False,
        },
        "source_files": {
            "r2r_train": inventory(R2R_DATA),
            "parent_protocol": inventory(PARENT_PROTOCOL),
            "parent_selection": inventory(PARENT_SELECTION),
            "parent_manifest": inventory(PARENT_MANIFEST),
            "parent_audit": inventory(PARENT_AUDIT),
            "dsr_protocol": inventory(DSR_PROTOCOL),
            "old_r2r_selection": inventory(OLD_R2R_SELECTION),
        },
        "implementation_files": {
            str(path.relative_to(ROOT)): inventory(path)
            for path in _implementation_paths()
        },
    }


def seal() -> int:
    if PROTOCOL.exists() or SELECTION.exists():
        raise RuntimeError("v1r1 protocol/selection already exists; refusing reseal")
    OUT.mkdir(parents=True, exist_ok=True)
    selection = build_selection()
    atomic_json(SELECTION, selection)
    protocol = build_protocol(selection)
    atomic_json(PROTOCOL, protocol)
    print(json.dumps({
        "status": protocol["status"],
        "selected_episodes": selection["counts"]["selected_episodes"],
        "selected_scenes": selection["counts"]["selected_scenes"],
        "selection_sha256": sha256_file(SELECTION),
        "protocol_sha256": sha256_file(PROTOCOL),
    }, indent=2, sort_keys=True))
    return 0


def verify_protocol() -> tuple[dict, dict]:
    if not PROTOCOL.is_file() or PROTOCOL.is_symlink():
        raise RuntimeError("v1r1 protocol is unavailable")
    protocol = json.loads(PROTOCOL.read_text())
    selection = json.loads(SELECTION.read_text())
    if (
        protocol.get("status") != "SEALED_BEFORE_MF3ZL_R2R_EXPANSION"
        or selection.get("status") != "SEALED_COMPLETE_R2R_INSTRUCTION_VARIANT_POPULATION"
        or protocol["selection"]["sha256"] != sha256_file(SELECTION)
        or protocol["selection"]["bytes"] != SELECTION.stat().st_size
        or protocol.get("public_split_access") != {
            "test": False, "test_challenge": False,
            "val_seen": False, "val_unseen": False,
        }
    ):
        raise RuntimeError("v1r1 sealed protocol semantics drift")
    for section in ("source_files", "implementation_files"):
        for item in protocol[section].values():
            path = ROOT / item["path"]
            if (
                path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]
            ):
                raise RuntimeError(f"v1r1 sealed file drift: {item['path']}")
    for row in selection["routes"]:
        if row.get("split") != "train":
            raise RuntimeError("v1r1 selection contains a non-train split")
        if any(token in str(row.get("scene_id", "")) for token in PUBLIC_TOKENS):
            raise RuntimeError("v1r1 selection contains a public split token")
    return protocol, selection


def _legacy() :
    module = _load_legacy_module()
    _configure_legacy(module)
    return module


def run_native(args) -> int:
    module = _legacy()
    _, selection = verify_protocol()
    jobs = [{
        "job_id": f"r2r_variant_ep_{row['episode_id']}",
        "dataset": "R2R",
        "episode_id": row["episode_id"],
        "scene_id": row["scene_id"],
        "mode": "native_shadow",
        "job_root": OUT / "runs/native/r2r" / f"ep_{row['episode_id']}",
    } for row in selection["routes"]]
    return module._run_jobs(
        "r2r_variant_native", jobs, NATIVE_PROGRESS,
        tuple(args.gpus), args.workers_per_gpu, args.retry_failed,
    )


def _native_jobs(module, selection: dict) -> list[dict]:
    return [{
        "job_id": f"r2r_variant_ep_{row['episode_id']}",
        "dataset": "R2R",
        "episode_id": row["episode_id"],
        "scene_id": row["scene_id"],
        "mode": "native_shadow",
        "job_root": OUT / "runs/native/r2r" / f"ep_{row['episode_id']}",
    } for row in selection["routes"]]


def run_targets(args) -> int:
    module = _legacy()
    _, selection = verify_protocol()
    native = json.loads(NATIVE_PROGRESS.read_text())
    if native.get("status") != "COMPLETE":
        raise RuntimeError("all v1r1 native shadows must pass before targets")
    if TARGETS.exists():
        targets = json.loads(TARGETS.read_text())
        if targets.get("source_selection_sha256") != sha256_file(SELECTION):
            raise RuntimeError("v1r1 target source drift")
    else:
        # The audited parent routine seals exactly the first core and first
        # expansion event observed in each shadow, without reading outcomes.
        targets = module.seal_targets(selection)
    return module._run_jobs(
        "r2r_variant_target", module._target_jobs(targets), TARGET_PROGRESS,
        tuple(args.gpus), args.workers_per_gpu, args.retry_failed,
    )


def _passed(module, job_root: Path) -> tuple[Path, dict]:
    state, attempt, summary = module._job_state(job_root)
    if state != "pass" or attempt is None or summary is None:
        raise RuntimeError(f"required v1r1 rollout is not PASS: {job_root}")
    return attempt, summary


def assemble() -> int:
    module = _legacy()
    verify_protocol()
    if (
        not TARGETS.is_file()
        or json.loads(TARGET_PROGRESS.read_text()).get("status") != "COMPLETE"
    ):
        raise RuntimeError("all v1r1 targeted treatments must pass before assembly")
    targets = json.loads(TARGETS.read_text())
    selection = json.loads(SELECTION.read_text())
    native_jobs = {
        (job["dataset"], job["episode_id"]): job
        for job in _native_jobs(module, selection)
    }
    records = []
    seen = {}
    for item, job in zip(
        targets["targets"], module._target_jobs(targets), strict=True
    ):
        identity = module.ProposalEventIdentity(**item["event_identity"])
        native_attempt, native_summary = _passed(
            module, native_jobs[(identity.dataset, identity.episode_id)]["job_root"]
        )
        treatment_attempt, treatment_summary = _passed(module, job["job_root"])
        if (
            treatment_summary.get("mode") != "targeted_switch"
            or treatment_summary.get("target") != item["event_identity"]
            or treatment_summary.get("changed_actions") != 1
            or treatment_summary.get("task_metric_payload_read") is not False
            or treatment_summary.get("public_split_access") is not False
        ):
            raise RuntimeError("v1r1 targeted treatment boundary drift")
        native_controller = module._read_trace(native_summary["controller_trace"])
        treatment_controller = module._read_trace(treatment_summary["controller_trace"])
        native_event_records = [
            row for row in native_controller
            if row.get("event_identity") == item["event_identity"]
        ]
        if len(native_event_records) != 1:
            raise RuntimeError("v1r1 native event trace cardinality drift")
        module.validate_shadow_event(native_event_records[0])
        module.validate_forced_switch(treatment_controller, identity)
        native_actions = module._read_trace(native_summary["base_trace"])
        treatment_actions = module._read_trace(treatment_summary["base_trace"])
        module.validate_exact_prefix(native_actions, treatment_actions, identity.step)
        treatment_events = [
            event for event in treatment_summary["proposal_events"]
            if event["event_identity"] == item["event_identity"]
        ]
        if len(treatment_events) != 1:
            raise RuntimeError("v1r1 treatment event feature cardinality drift")
        native_feature = item["native_feature"]
        treatment_feature = treatment_events[0]["feature"]
        for feature in (native_feature, treatment_feature):
            path = ROOT / feature["path"]
            if (
                path.stat().st_size != feature["bytes"]
                or sha256_file(path) != feature["sha256"]
            ):
                raise RuntimeError("v1r1 feature provenance drift")
        if (
            native_feature["bytes"] != treatment_feature["bytes"]
            or native_feature["sha256"] != treatment_feature["sha256"]
        ):
            raise RuntimeError("v1r1 causal feature changed on replay")
        baseline, baseline_stats = module._metrics(native_summary, identity.episode_id)
        treatment, treatment_stats = module._metrics(treatment_summary, identity.episode_id)
        delta = {
            key: float(treatment[key]) - float(baseline[key])
            for key in ("success", "spl", "ndtw", "sdtw")
        }
        delta["utility"] = module._utility(treatment) - module._utility(baseline)
        record = {
            "dataset": identity.dataset,
            "episode_id": identity.episode_id,
            "scene_id": identity.scene_id,
            "decision_step": identity.step,
            "tier": identity.tier,
            "native_action_id": identity.native_action_id,
            "runner_action_id": identity.runner_action_id,
            "target": delta["utility"],
            "catastrophic": delta["utility"] <= -0.10,
            "delta": delta,
            "decision": item["decision"],
            "baseline_metrics": baseline,
            "treatment_metrics": treatment,
            "feature": native_feature,
            "baseline_stats": baseline_stats,
            "treatment_stats": treatment_stats,
            "native_run_summary": inventory(native_attempt / "RUN_SUMMARY.json"),
            "treatment_run_summary": inventory(treatment_attempt / "RUN_SUMMARY.json"),
            "exact_prefix_verified": True,
            "exact_one_switch_verified": True,
        }
        key = stable_hash(item["event_identity"])
        signature = stable_hash(record)
        if key in seen and seen[key] != signature:
            raise RuntimeError("conflicting v1r1 exact event identity")
        seen[key] = signature
        records.append(record)
    value = {
        "schema_version": f"{SCHEMA}-manifest/1",
        "status": "R2R_VARIANT_EXACT_REPLAY_READY",
        "revision": "mf3zl_rcsp_v1r1",
        "parent_revision": "mf3zl_rcsp_v1",
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "source_selection_sha256": sha256_file(SELECTION),
        "source_targets_sha256": sha256_file(TARGETS),
        "complete_population_executed": True,
        "task_metric_payload_read_only_during_assembly": True,
        "public_split_access": False,
        "counts": {
            "exact_events": len(records),
            "datasets": dict(Counter(row["dataset"] for row in records)),
            "scenes": len({row["scene_id"] for row in records}),
            "positive": sum(row["target"] > 0 for row in records),
            "catastrophic": sum(row["catastrophic"] for row in records),
            "conflicting_identities": 0,
        },
        "records": records,
    }
    if MANIFEST.exists():
        raise RuntimeError("refusing to overwrite v1r1 manifest")
    atomic_json(MANIFEST, value)
    print(json.dumps(value["counts"], indent=2, sort_keys=True))
    return 0


def _parent_rows() -> list[dict]:
    dsr = json.loads(DSR_PROTOCOL.read_text())
    return list(dsr["source_inventory"]["canonical_rows"])


def _dense_rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    return list(value.get("records", []))


def audit() -> int:
    protocol, _ = verify_protocol()
    if not MANIFEST.is_file():
        raise RuntimeError("v1r1 manifest is unavailable")
    manifest = json.loads(MANIFEST.read_text())
    if (
        manifest.get("status") != "R2R_VARIANT_EXACT_REPLAY_READY"
        or manifest.get("complete_population_executed") is not True
        or manifest.get("public_split_access") is not False
        or manifest.get("source_protocol_sha256") != sha256_file(PROTOCOL)
    ):
        raise RuntimeError("v1r1 manifest boundary drift")
    parent_manifest = json.loads(PARENT_MANIFEST.read_text())
    existing = _parent_rows()
    parent_dense = _dense_rows(PARENT_MANIFEST)
    extension = list(manifest["records"])
    failures = []
    domains = {}
    seen = {}
    for row in existing:
        identity = (
            str(row["identity"]["dataset"]),
            str(row["identity"]["episode_id"]),
            int(row["identity"]["step"]),
        )
        seen[identity] = stable_hash(row)
    for row in parent_dense + extension:
        identity = (
            str(row["dataset"]), str(row["episode_id"]), int(
                row.get("decision_step", row["event_identity"]["step"])
            ),
        )
        signature = stable_hash(row)
        if identity in seen and seen[identity] != signature:
            failures.append("conflicting_exact_identity")
        seen[identity] = signature
    for domain in ("RxR", "R2R"):
        old = [row for row in existing if row["identity"]["dataset"] == domain]
        old_dense = [row for row in parent_dense if row["dataset"] == domain]
        new = [row for row in extension if row["dataset"] == domain]
        scenes = {
            str(row["identity"]["scene_id"]) for row in old
        } | {str(row["scene_id"]) for row in old_dense + new}
        events = len(old) + len(old_dense) + len(new)
        domains[domain] = {
            "existing_exact_events": len(old),
            "parent_dense_exact_events": len(old_dense),
            "v1r1_variant_exact_events": len(new),
            "combined_unique_exact_events": events,
            "combined_development_scenes": len(scenes),
        }
        if events < 300:
            failures.append(f"{domain}:fewer_than_300_exact_events")
        if len(scenes) < 30:
            failures.append(f"{domain}:fewer_than_30_development_scenes")
    extension_episode_ids = {str(row["episode_id"]) for row in extension}
    parent_episode_ids = {
        str(row["identity"]["episode_id"]) for row in existing
    } | {str(row["episode_id"]) for row in parent_dense}
    if extension_episode_ids & parent_episode_ids:
        failures.append("historical_episode_overlap")
    if int(manifest["counts"]["conflicting_identities"]) != 0:
        failures.append("manifest_conflicting_exact_identity")
    value = {
        "schema_version": "revealnav-mf3zl-r2r-expansion-audit/1",
        "status": "TRAIN_DATA_SUPPORT_PASS" if not failures else "TRAIN_DATA_SUPPORT_FAIL",
        "revision": "mf3zl_rcsp_v1r1",
        "parent_revision": "mf3zl_rcsp_v1",
        "source_protocol_sha256": sha256_file(PROTOCOL),
        "source_variant_manifest": inventory(MANIFEST),
        "source_parent_manifest": inventory(PARENT_MANIFEST),
        "complete_population_executed": True,
        "adaptive_stopping_used": False,
        "untouched_scenes_consumed": False,
        "public_split_access": False,
        "domains": domains,
        "failure_reasons": sorted(set(failures)),
        "rcsp_training_authorized": not failures,
        "confirmation_authorized": False,
        "public_unseen_authorized": False,
    }
    if AUDIT.exists():
        raise RuntimeError("refusing to overwrite v1r1 audit")
    atomic_json(AUDIT, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if not failures else 2


def monitor() -> int:
    values = {}
    for name, path in (
        ("native_shadow", NATIVE_PROGRESS),
        ("targeted_switch", TARGET_PROGRESS),
        ("targets", TARGETS),
        ("manifest", MANIFEST),
        ("audit", AUDIT),
    ):
        if path.is_file():
            value = json.loads(path.read_text())
            values[name] = value.get("counts", value) if name in {"targets", "manifest"} else value
    print(json.dumps(values, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal")
    for name in ("run-native-shadow", "run-targeted-switches"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
        cmd.add_argument("--workers-per-gpu", type=int, default=4)
        cmd.add_argument("--retry-failed", action="store_true")
    sub.add_parser("assemble")
    sub.add_parser("audit")
    sub.add_parser("monitor")
    args = parser.parse_args()
    if args.command == "seal":
        return seal()
    if args.command == "run-native-shadow":
        return run_native(args)
    if args.command == "run-targeted-switches":
        return run_targets(args)
    if args.command == "assemble":
        return assemble()
    if args.command == "audit":
        return audit()
    return monitor()


if __name__ == "__main__":
    raise SystemExit(main())
