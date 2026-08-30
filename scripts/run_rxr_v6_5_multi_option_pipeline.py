#!/usr/bin/env python3
"""Seal, collect, and assemble RxR-train V6.5 multi-option groups."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".envs/etpr1/bin/python"
PIPELINE = Path(__file__).resolve()
WORKER = ROOT / "scripts/rxr_v6_5_all_live_worker.py"
DESIGN = ROOT / "artifacts/design/MF2_MULTI_OPTION_LISTWISE_ADVANTAGE_V6_5.md"
CORRECTION = ROOT / (
    "artifacts/design/MF2_V6_5_0_1_TRAJECTORY_DISJOINT_SCENE_CORRECTION.md"
)
STATISTICAL_COMPLETION = ROOT / (
    "artifacts/design/MF2_V6_5_0_2_STATISTICAL_COMPLETION.md"
)
TASK_SCOPE_REVISION = ROOT / (
    "artifacts/design/METHOD_FREEZE_3A_TASK_SCOPE_REVISION.md"
)
FROZEN_SPEC = ROOT / "FROZEN_SPEC.md"
DATASET = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
PRIOR_SELECTION_ROOT = ROOT / "artifacts/phase1/rxr_v6"
BASE = PRIOR_SELECTION_ROOT / "v6_5_multi_option"
ROLES = ("development", "holdout")
ROLE_EPISODES = {"development": 220, "holdout": 140}
# These are frozen ETP controller/replay seeds. V6.5 model-training seeds are
# separately locked to 20260901/02/03 by the method design.
COLLECTION_SEEDS = (20260826, 20260827, 20260828)
MAX_GROUPS_PER_EPISODE = 2
HOLDOUT_SCENES = 15
EXPECTED_SOURCE_SCENES = 59
EXPECTED_ELIGIBLE_SCENES = 56
DEVELOPMENT_GATE = BASE / "development/RXR_V6_5_DEVELOPMENT_GATE.json"
SCOPE = {
    "auxiliary_mechanism_diagnostic_only": True,
    "not_vln_mainline": True,
    "cannot_gate_uad_mainline": True,
    "intended_downstream_task": "open_vocabulary_object_search",
    "public_rxr_r2r_unseen_authorized": False,
    "uad_mainline_training_authorized": False,
}
METRICS = (
    "ndtw", "sdtw", "spl", "success", "distance_to_goal",
    "path_length", "steps_taken", "oracle_success",
)
FEATURE_KEYS = {
    "instruction", "post_observation", "temporal_history", "checkpoint",
    "native", "option_embeddings", "option_scalars", "option_ids",
}
CAPACITY = {
    "development": {
        "complete_groups": 360,
        "episodes": 180,
        "scenes": 40,
        "positive_groups": 72,
        "positive_groups_per_fold": 8,
    },
    "holdout": {
        "complete_groups": 200,
        "episodes": 100,
        "scenes": 15,
        "positive_groups": 40,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def update_array_hash(
    digest: "hashlib._Hash", name: str, value: np.ndarray,
) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode())
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes())


def recompute_feature_hashes(
    group: dict, values: dict[str, np.ndarray],
) -> tuple[str, str, list[str]]:
    embeddings = {group["native_branch_id"]: values["native"]}
    for index, option in enumerate(group["options"]):
        embeddings[option["alternative_branch_id"]] = values[
            "option_embeddings"
        ][index]
    if sorted(embeddings) != group["candidate_branch_ids"]:
        raise RuntimeError("V6.5 feature candidate identities drift")
    candidate_digest = hashlib.sha256()
    for branch_id in sorted(embeddings):
        candidate_digest.update(json.dumps(str(branch_id)).encode())
        update_array_hash(candidate_digest, "embedding", embeddings[branch_id])
    candidate_hash = candidate_digest.hexdigest()
    shared_digest = hashlib.sha256()
    shared_digest.update(candidate_hash.encode())
    for key in sorted({
        "instruction", "post_observation", "temporal_history", "checkpoint",
        "native",
    }):
        update_array_hash(shared_digest, key, values[key])
    shared_hash = shared_digest.hexdigest()
    option_hashes = []
    for index, option in enumerate(group["options"]):
        digest = hashlib.sha256()
        digest.update(shared_hash.encode())
        digest.update(json.dumps(str(option["alternative_branch_id"])).encode())
        update_array_hash(digest, "option_embedding", values["option_embeddings"][index])
        update_array_hash(digest, "option_scalars", values["option_scalars"][index])
        option_hashes.append(digest.hexdigest())
    return candidate_hash, shared_hash, option_hashes


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def scoped(value: dict) -> dict:
    """Attach the immutable V6.5 auxiliary-only scope to an artifact."""
    return {**value, **SCOPE}


def seal_json(path: Path, value: dict) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"sealed path is not a regular file: {path}")
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"sealed artifact drift: {path}")
        return
    atomic_json(path, value)


def atomic_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite assembled arrays: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(part, path)


def source_record(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular source: {path}")
    resolved = path.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise RuntimeError(f"source resolves outside project: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def role_paths(role: str) -> dict[str, Path]:
    root = BASE / role
    return {
        "root": root,
        "selection": root / "RXR_V6_5_EPISODE_SELECTION.json",
        "group_selection": root / "RXR_V6_5_GROUP_SELECTION.json",
        "progress": root / "RXR_V6_5_PROGRESS.json",
        "runs": root / "runs",
        "targets": root / "targets",
        "logs": root / "logs",
        "arrays": root / "RXR_V6_5_MULTI_OPTION_DATASET.npz",
        "manifest": root / "RXR_V6_5_MULTI_OPTION_DATASET_MANIFEST.json",
    }


def roles_path() -> Path:
    return BASE / "RXR_V6_5_SCENE_ROLES.json"


def protocol_path() -> Path:
    return BASE / "RXR_V6_5_PROTOCOL.json"


def load_english_episodes() -> list[dict]:
    if DATASET.parent.name != "train" or DATASET.name != "train_guide.json.gz":
        raise RuntimeError("V6.5 dataset path is not the sealed RxR train guide")
    with gzip.open(DATASET, "rt", encoding="utf-8") as stream:
        episodes = json.load(stream)["episodes"]
    rows = []
    for episode in episodes:
        language = episode.get("instruction", {}).get("language")
        if language not in ("en-US", "en-IN"):
            continue
        trajectory_id = str(episode.get("trajectory_id", ""))
        if not trajectory_id or trajectory_id == "None":
            raise RuntimeError("RxR English episode lacks trajectory identity")
        rows.append({
            "episode_id": str(episode["episode_id"]),
            "trajectory_id": trajectory_id,
            "scene_id": Path(episode["scene_id"]).parts[-2],
            "language": language,
        })
    if not rows or len({row["episode_id"] for row in rows}) != len(rows):
        raise RuntimeError("RxR English train episode identity drift")
    if len({row["scene_id"] for row in rows}) != EXPECTED_SOURCE_SCENES:
        raise RuntimeError("RxR English train scene count drift")
    return rows


def prior_selections() -> tuple[set[str], set[str], dict[str, dict]]:
    episode_ids: set[str] = set()
    trajectory_ids: set[str] = set()
    evidence: dict[str, dict] = {}
    paths = sorted(PRIOR_SELECTION_ROOT.glob("**/RXR_V6_EPISODE_SELECTION.json"))
    for path in paths:
        if BASE in path.parents:
            continue
        value = json.loads(path.read_text())
        rows = value.get("episodes")
        if not isinstance(rows, list):
            raise RuntimeError(f"invalid prior selection: {path}")
        for row in rows:
            episode_ids.add(str(row["episode_id"]))
            trajectory_id = str(row.get("trajectory_id", ""))
            if not trajectory_id or trajectory_id == "None":
                raise RuntimeError(f"prior selection lacks trajectory id: {path}")
            trajectory_ids.add(trajectory_id)
        evidence[relative(path)] = source_record(path)
    if not evidence:
        raise RuntimeError("no prior V6 episode selections found")
    return episode_ids, trajectory_ids, evidence


def scene_roles(episodes: list[dict]) -> tuple[dict, set[str], set[str]]:
    scenes = sorted({row["scene_id"] for row in episodes})
    ordered = sorted(
        scenes,
        key=lambda scene: stable_hash({"v6_5_holdout_scene": scene}),
    )
    holdout = set(ordered[:HOLDOUT_SCENES])
    development = set(ordered[HOLDOUT_SCENES:])
    final_order = sorted(
        development,
        key=lambda scene: stable_hash({
            "v6_5_final_calibration_scene": scene,
        }),
    )
    final_calibration = set(final_order[:10])
    final_fit = set(final_order[10:])
    value = scoped({
        "schema_version": "revealnav-rxr-v6.5-scene-roles/1",
        "status": "SEALED_BEFORE_V6_5_COLLECTION",
        "rule": (
            "after all prior episode/trajectory exclusions, sort the 56 "
            "eligible RxR-train-English scene IDs by "
            "sha256(canonical_json({v6_5_holdout_scene: scene_id})); "
            "first 15 holdout, remaining 41 development"
        ),
        "scene_hashes": {
            scene: stable_hash({"v6_5_holdout_scene": scene})
            for scene in sorted(scenes)
        },
        "holdout_scenes": sorted(holdout),
        "development_scenes": sorted(development),
        "holdout_scene_count": len(holdout),
        "development_scene_count": len(development),
        "final_calibration_rule": (
            "sort 41 development scenes by sha256(canonical_json("
            "{v6_5_final_calibration_scene: scene_id})); first 10 final "
            "calibration, remaining 31 final fit"
        ),
        "final_calibration_scene_hashes": {
            scene: stable_hash({"v6_5_final_calibration_scene": scene})
            for scene in sorted(development)
        },
        "final_calibration_scenes": sorted(final_calibration),
        "final_fit_scenes": sorted(final_fit),
        "final_calibration_scene_count": len(final_calibration),
        "final_fit_scene_count": len(final_fit),
        "split": "RxR-train-English-only",
        "unseen_or_test_read": False,
        "paper_result": False,
    })
    if (
        len(holdout) != 15
        or len(development) != 41
        or len(final_calibration) != 10
        or len(final_fit) != 31
    ):
        raise RuntimeError("V6.5 scene role cardinality drift")
    return value, development, holdout


def select_role(
    role: str,
    episodes: list[dict],
    role_scenes: set[str],
    excluded_episode_ids: set[str],
    excluded_trajectory_ids: set[str],
) -> list[dict]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in episodes:
        if row["scene_id"] not in role_scenes:
            continue
        if row["episode_id"] in excluded_episode_ids:
            continue
        if row["trajectory_id"] in excluded_trajectory_ids:
            continue
        by_scene[row["scene_id"]].append(row)
    if set(by_scene) != role_scenes:
        raise RuntimeError(f"{role} has a scene with no eligible trajectory")

    # Multiple English instructions can share a trajectory. The lowest sealed
    # hash supplies the one episode, so no trajectory can enter twice.
    for scene, rows in by_scene.items():
        rows.sort(key=lambda row: stable_hash({
            "v6_5_episode": row["episode_id"],
            "trajectory": row["trajectory_id"],
            "role": role,
        }))
        unique = {}
        for row in rows:
            unique.setdefault(row["trajectory_id"], row)
        by_scene[scene] = sorted(
            unique.values(),
            key=lambda row: stable_hash({
                "v6_5_role": role,
                "scene": scene,
                "episode": row["episode_id"],
                "trajectory": row["trajectory_id"],
            }),
        )

    scene_order = sorted(
        role_scenes,
        key=lambda scene: stable_hash({"v6_5_selection_scene": scene}),
    )
    selected = []
    depth = 0
    target = ROLE_EPISODES[role]
    while len(selected) < target:
        added = False
        for scene in scene_order:
            if depth >= len(by_scene[scene]):
                continue
            row = dict(by_scene[scene][depth])
            row["controller_seed"] = COLLECTION_SEEDS[
                int(stable_hash({
                    "v6_5_controller_seed": row["episode_id"],
                    "trajectory": row["trajectory_id"],
                }), 16) % len(COLLECTION_SEEDS)
            ]
            row["scene_fold"] = (
                int(stable_hash({"v6_5_outer_fold": scene}), 16) % 5
                if role == "development" else None
            )
            row["final_role"] = None
            selected.append(row)
            added = True
            if len(selected) == target:
                break
        if not added:
            raise RuntimeError(f"insufficient eligible episodes for {role}")
        depth += 1
    if len({row["episode_id"] for row in selected}) != target:
        raise RuntimeError(f"duplicate episode in {role} selection")
    if len({row["trajectory_id"] for row in selected}) != target:
        raise RuntimeError(f"duplicate trajectory in {role} selection")
    return selected


def selection_value(
    role: str,
    rows: list[dict],
    prior_episode_ids: set[str],
    prior_trajectory_ids: set[str],
    prior_evidence: dict[str, dict],
) -> dict:
    return scoped({
        "schema_version": "revealnav-rxr-v6.5-episode-selection/1",
        "status": "SEALED_BEFORE_V6_5_COLLECTION",
        "role": role,
        "split": "RxR-train-English-only",
        "selection_rule": (
            "scene-balanced deterministic hash round-robin after prior V6 "
            "episode and trajectory exclusion; one episode per trajectory"
        ),
        "episodes": rows,
        "episode_count": len(rows),
        "trajectory_count": len({row["trajectory_id"] for row in rows}),
        "scene_count": len({row["scene_id"] for row in rows}),
        "fold_counts": dict(sorted(Counter(
            str(row["scene_fold"]) for row in rows
            if row["scene_fold"] is not None
        ).items())),
        "prior_selection_sources": prior_evidence,
        "excluded_prior_episode_count": len(prior_episode_ids),
        "excluded_prior_trajectory_count": len(prior_trajectory_ids),
        "excluded_prior_episode_ids_sha256": stable_hash(
            sorted(prior_episode_ids)
        ),
        "excluded_prior_trajectory_ids_sha256": stable_hash(
            sorted(prior_trajectory_ids)
        ),
        "unseen_or_test_read": False,
        "paper_result": False,
    })


def seal() -> dict:
    episodes = load_english_episodes()
    old_episode_ids, old_trajectory_ids, old_evidence = prior_selections()
    eligible = [
        row for row in episodes
        if row["episode_id"] not in old_episode_ids
        and row["trajectory_id"] not in old_trajectory_ids
    ]
    if len({row["scene_id"] for row in eligible}) != EXPECTED_ELIGIBLE_SCENES:
        raise RuntimeError("trajectory-disjoint RxR scene count drift")
    role_value, development_scenes, holdout_scenes = scene_roles(eligible)
    development = select_role(
        "development", episodes, development_scenes,
        old_episode_ids, old_trajectory_ids,
    )
    holdout = select_role(
        "holdout", episodes, holdout_scenes,
        old_episode_ids, old_trajectory_ids,
    )
    final_calibration_scenes = set(role_value["final_calibration_scenes"])
    for row in development:
        row["final_role"] = (
            "final_calibration"
            if row["scene_id"] in final_calibration_scenes
            else "final_fit"
        )
    if {row["episode_id"] for row in development} & {
        row["episode_id"] for row in holdout
    }:
        raise RuntimeError("development/holdout episode overlap")
    if {row["trajectory_id"] for row in development} & {
        row["trajectory_id"] for row in holdout
    }:
        raise RuntimeError("development/holdout trajectory overlap")

    selections = {
        "development": selection_value(
            "development", development, old_episode_ids,
            old_trajectory_ids, old_evidence,
        ),
        "holdout": selection_value(
            "holdout", holdout, old_episode_ids,
            old_trajectory_ids, old_evidence,
        ),
    }
    seal_json(roles_path(), role_value)
    for role in ROLES:
        seal_json(role_paths(role)["selection"], selections[role])

    source_paths = (
        FROZEN_SPEC, DESIGN, CORRECTION, STATISTICAL_COMPLETION,
        TASK_SCOPE_REVISION,
        PIPELINE, WORKER, DATASET,
    )
    protocol = scoped({
        "schema_version": "revealnav-rxr-v6.5-multi-option-protocol/1",
        "status": "SEALED_BEFORE_V6_5_COLLECTION",
        "split": "RxR-train-English-only",
        "roles": {role: ROLE_EPISODES[role] for role in ROLES},
        "collection_seeds": list(COLLECTION_SEEDS),
        "maximum_groups_per_episode": MAX_GROUPS_PER_EPISODE,
        "eligible_candidate_width_including_native": [3, 4],
        "group_rule": (
            "earliest two causal groups; candidate set is the native UNTRIED "
            "branch plus at most three UNTRIED alternatives ordered by "
            "descending frontier age then branch ID; retain only width 3/4; "
            "outcome-independent"
        ),
        "worker": {
            "path": relative(WORKER),
            "modes": ["shadow", "macro"],
            "shadow_group_required_fields": [
                "group_id", "decision_index", "prefix_action_count",
                "checkpoint_id", "native_branch_id", "candidate_branch_ids",
                "candidate_set_sha256", "shared_state_sha256",
                "feature_path", "feature_bytes", "feature_sha256", "options",
            ],
            "shadow_option_required_fields": [
                "option_id", "alternative_branch_id", "option_causal_sha256",
            ],
            "feature_npz": {
                "shared_768d": [
                    "instruction", "post_observation", "temporal_history",
                    "checkpoint", "native",
                ],
                "option_embeddings_shape": "[K,768]",
                "option_scalars_shape": "[K,16]",
                "option_ids_shape": "[K] string",
            },
            "macro_summary": (
                "must echo the exact target object; independently report "
                "observed_group_id, observed_checkpoint_id, "
                "observed_native_branch_id, observed_candidate_branch_ids, "
                "observed_candidate_set_sha256, observed_shared_state_sha256, "
                "observed_option_causal_sha256, committed_alternative_branch_id; "
                "and report "
                "target_physical_return_verified, target_topology_restored, "
                "target_alternative_committed, and finite terminal metrics"
            ),
        },
        "utility": "0.50*ndtw+0.25*sdtw+0.25*spl",
        "main_dataset_completeness": (
            "if any pre-enumerated option is explicitly rejected, exclude the "
            "entire group and retain its rejection evidence"
        ),
        "capacity": CAPACITY,
        "holdout_unlock": {
            "artifact": relative(DEVELOPMENT_GATE),
            "required_status": "PASS",
            "note": (
                "internal auxiliary diagnostic holdout only; dataset capacity "
                "alone does not unlock it and it cannot authorize any public "
                "RxR/R2R evaluation or UAD-mainline action"
            ),
        },
        "sources": {
            relative(path): source_record(path) for path in source_paths
        },
        "sealed_artifacts": {
            relative(roles_path()): source_record(roles_path()),
            **{
                relative(role_paths(role)["selection"]): source_record(
                    role_paths(role)["selection"]
                )
                for role in ROLES
            },
        },
        "forbidden_online_inputs": [
            "goal", "reference_path", "task_metric", "future_frame",
            "counterfactual_outcome", "branch_id_as_numeric_input",
            "post_return_state", "MLLM_label",
        ],
        "unseen_or_test_read": False,
        "paper_result": False,
        "public_rxr_r2r_unseen_authorized": False,
        "uad_mainline_training_authorized": False,
    })
    seal_json(protocol_path(), protocol)
    return {
        "protocol": protocol,
        "roles": role_value,
        "selections": selections,
    }


def verify_seal() -> dict[str, dict]:
    required = [protocol_path(), roles_path()] + [
        role_paths(role)["selection"] for role in ROLES
    ]
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise RuntimeError("V6.5 protocol, scene roles, and both selections must be sealed")
    protocol = json.loads(protocol_path().read_text())
    if (
        protocol.get("status") != "SEALED_BEFORE_V6_5_COLLECTION"
        or any(protocol.get(key) != value for key, value in SCOPE.items())
        or protocol.get("public_rxr_r2r_unseen_authorized") is not False
        or protocol.get("uad_mainline_training_authorized") is not False
    ):
        raise RuntimeError("invalid V6.5 protocol status")
    for name, record in {
        **protocol["sources"], **protocol["sealed_artifacts"],
    }.items():
        path = ROOT / name
        if source_record(path) != record:
            raise RuntimeError(f"sealed V6.5 source drift: {name}")

    roles = json.loads(roles_path().read_text())
    if (
        len(roles.get("development_scenes", [])) != 41
        or len(roles.get("holdout_scenes", [])) != 15
        or len(roles.get("final_fit_scenes", [])) != 31
        or len(roles.get("final_calibration_scenes", [])) != 10
    ):
        raise RuntimeError("invalid sealed V6.5 scene roles")
    selections = {
        role: json.loads(role_paths(role)["selection"].read_text())
        for role in ROLES
    }
    for role in ROLES:
        rows = selections[role].get("episodes", [])
        if (
            selections[role].get("status") != "SEALED_BEFORE_V6_5_COLLECTION"
            or selections[role].get("role") != role
            or len(rows) != ROLE_EPISODES[role]
            or len({row["episode_id"] for row in rows}) != len(rows)
            or len({row["trajectory_id"] for row in rows}) != len(rows)
        ):
            raise RuntimeError(f"invalid sealed {role} selection")
        allowed = set(roles[f"{role}_scenes"])
        if {row["scene_id"] for row in rows} - allowed:
            raise RuntimeError(f"{role} selection crosses sealed scene roles")
        if role == "development":
            final_calibration = set(roles["final_calibration_scenes"])
            if any(
                row.get("final_role") != (
                    "final_calibration"
                    if row["scene_id"] in final_calibration else "final_fit"
                )
                for row in rows
            ):
                raise RuntimeError("development final-fit/calibration role drift")
        elif any(row.get("final_role") is not None for row in rows):
            raise RuntimeError("holdout episode has a development final role")
    development = selections["development"]["episodes"]
    holdout = selections["holdout"]["episodes"]
    if {row["episode_id"] for row in development} & {
        row["episode_id"] for row in holdout
    }:
        raise RuntimeError("sealed development/holdout episode overlap")
    if {row["trajectory_id"] for row in development} & {
        row["trajectory_id"] for row in holdout
    }:
        raise RuntimeError("sealed development/holdout trajectory overlap")
    return selections


def require_holdout_unlock(role: str) -> None:
    if role != "holdout":
        return
    if not DEVELOPMENT_GATE.is_file() or DEVELOPMENT_GATE.is_symlink():
        raise RuntimeError("holdout is sealed until the development gate exists")
    gate = json.loads(DEVELOPMENT_GATE.read_text())
    if (
        gate.get("status") != "PASS"
        or gate.get("role") != "development"
        or any(gate.get(key) != value for key, value in SCOPE.items())
        or gate.get("public_rxr_r2r_unseen_authorized") is not False
        or gate.get("uad_mainline_training_authorized") is not False
    ):
        raise RuntimeError("holdout is sealed until development status is PASS")


def finite_metrics(value: object) -> bool:
    return isinstance(value, dict) and all(
        key in value and math.isfinite(float(value[key])) for key in METRICS
    )


def valid_shadow_summary(path: Path, role: str, episode: dict) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        value.get("status") == "PASS"
        and value.get("mode") == "shadow"
        and value.get("split") == "train"
        and value.get("role") == role
        and str(value.get("episode_id")) == episode["episode_id"]
        and str(value.get("trajectory_id")) == episode["trajectory_id"]
        and value.get("scene_id") == episode["scene_id"]
        and value.get("decision_groups") is not None
        and isinstance(value.get("decision_groups"), list)
        and finite_metrics(value.get("metrics"))
        and not value.get("unseen_or_test_read")
        and all(value.get(key) == expected for key, expected in SCOPE.items())
    )


def macro_state(value: dict, role: str, target: dict) -> str | None:
    common = (
        value.get("mode") == "macro"
        and value.get("split") == "train"
        and value.get("role") == role
        and value.get("target") == target
        and value.get("observed_group_id") == target["group_id"]
        and value.get("observed_checkpoint_id") == target["checkpoint_id"]
        and value.get("observed_native_branch_id") == target["native_branch_id"]
        and value.get("observed_candidate_branch_ids") == target["candidate_branch_ids"]
        and value.get("observed_candidate_set_sha256") == target["candidate_set_sha256"]
        and value.get("observed_shared_state_sha256") == target["shared_state_sha256"]
        and value.get("observed_option_causal_sha256") == target["option_causal_sha256"]
        and finite_metrics(value.get("metrics"))
        and not value.get("unseen_or_test_read")
        and all(value.get(key) == expected for key, expected in SCOPE.items())
    )
    if not common:
        return None
    if (
        value.get("status") == "PASS"
        and value.get("target_physical_return_verified") is True
        and value.get("target_topology_restored") is True
        and value.get("target_alternative_committed") is True
        and value.get("committed_alternative_branch_id")
        == target["alternative_branch_id"]
    ):
        return "accepted"
    if (
        value.get("status") == "REJECTED_UNEXECUTABLE_OPTION"
        and value.get("target_alternative_committed") is False
        and value.get("committed_alternative_branch_id") is None
        and isinstance(value.get("rejection_reason"), str)
        and bool(value["rejection_reason"])
    ):
        return "rejected"
    return None


def valid_macro_summary(path: Path, role: str, target: dict) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        return macro_state(json.loads(path.read_text()), role, target)
    except (OSError, ValueError):
        return None


def progress(
    role: str,
    stage: str,
    total: int,
    completed: int,
    active: dict,
    failures: list[dict],
    rejections: list[dict],
) -> None:
    atomic_json(role_paths(role)["progress"], scoped({
        "schema_version": "revealnav-rxr-v6.5-progress/1",
        "updated_at_utc": utc_timestamp(),
        "role": role,
        "stage": stage,
        "total": total,
        "completed": completed,
        "remaining": total - completed,
        "active": active,
        "failures": failures,
        "rejections": rejections,
    }))


def run_worker(command: list[str], gpu: int, stdout: Path, stderr: Path) -> dict:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONNOUSERSITE": "1",
    })
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w") as out, stderr.open("w") as err:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=err,
            text=True,
            check=False,
        )
    return {"returncode": completed.returncode, "gpu": gpu}


def parallel_jobs(
    role: str,
    stage: str,
    jobs: list[dict],
    gpus: tuple[int, ...],
    total: int,
    already_completed: int,
    prior_rejections: list[dict],
) -> None:
    active: dict[str, dict] = {}
    failures: list[dict] = []
    rejections = list(prior_rejections)
    completed = already_completed
    progress(role, stage, total, completed, active, failures, rejections)
    if not jobs:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        running = {}
        iterator = iter(jobs)
        for slot, gpu in enumerate(gpus):
            try:
                job = next(iterator)
            except StopIteration:
                break
            future = pool.submit(
                run_worker, job["command"], gpu, job["stdout"], job["stderr"]
            )
            running[future] = (slot, gpu, job)
            active[str(slot)] = {"gpu": gpu, "job_id": job["id"]}
        progress(role, stage, total, completed, active, failures, rejections)
        while running:
            done, _ = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                slot, gpu, job = running.pop(future)
                result = future.result()
                state = job["validate"]()
                if result["returncode"] or state not in job["accepted_states"]:
                    failures.append({
                        "job_id": job["id"],
                        "returncode": result["returncode"],
                        "stderr": relative(job["stderr"]),
                    })
                elif state == "rejected":
                    rejections.append({
                        "job_id": job["id"],
                        "reason": "unexecutable_option",
                    })
                completed += 1
                active.pop(str(slot), None)
                try:
                    next_job = next(iterator)
                except StopIteration:
                    next_job = None
                if next_job is not None:
                    future = pool.submit(
                        run_worker,
                        next_job["command"],
                        gpu,
                        next_job["stdout"],
                        next_job["stderr"],
                    )
                    running[future] = (slot, gpu, next_job)
                    active[str(slot)] = {
                        "gpu": gpu, "job_id": next_job["id"],
                    }
                progress(
                    role, stage, total, completed, active, failures, rejections
                )
    if failures:
        raise RuntimeError(f"{role} {stage} has {len(failures)} failed workers")


def shadow_run_dir(role: str, episode: dict) -> Path:
    return role_paths(role)["runs"] / (
        f"shadow_ep{episode['episode_id']}_s{episode['controller_seed']}"
    )


def collect_shadows(role: str, gpus: tuple[int, ...]) -> None:
    selections = verify_seal()
    require_holdout_unlock(role)
    episodes = selections[role]["episodes"]
    jobs = []
    complete = 0
    for episode in episodes:
        run_dir = shadow_run_dir(role, episode)
        summary = run_dir / "RUN_SUMMARY.json"
        if valid_shadow_summary(summary, role, episode):
            complete += 1
            continue
        output = role_paths(role)["logs"] / f"shadow_ep{episode['episode_id']}.out"
        error = role_paths(role)["logs"] / f"shadow_ep{episode['episode_id']}.err"
        jobs.append({
            "id": f"shadow:{episode['episode_id']}",
            "stdout": output,
            "stderr": error,
            "accepted_states": {"accepted"},
            "validate": lambda p=summary, r=role, e=episode: (
                "accepted" if valid_shadow_summary(p, r, e) else None
            ),
            "command": [
                str(PYTHON), str(WORKER),
                "--mode", "shadow",
                "--role", role,
                "--episode-id", episode["episode_id"],
                "--seed", str(episode["controller_seed"]),
                "--run-dir", str(run_dir),
            ],
        })
    parallel_jobs(
        role, "shadow", jobs, gpus, len(episodes), complete, []
    )


def normalized_group(group: dict, episode: dict, run_dir: Path) -> dict | None:
    required = {
        "group_id", "decision_index", "prefix_action_count", "checkpoint_id",
        "native_branch_id", "candidate_branch_ids", "candidate_set_sha256",
        "shared_state_sha256", "feature_path", "feature_bytes",
        "feature_sha256", "options",
    }
    if not required.issubset(group):
        raise RuntimeError("V6.5 shadow decision group misses required fields")
    candidate_ids = group["candidate_branch_ids"]
    options = group["options"]
    if not isinstance(candidate_ids, list) or not isinstance(options, list):
        raise RuntimeError("V6.5 candidate or option set is not a list")
    if len(candidate_ids) not in (3, 4):
        return None
    if len(options) != len(candidate_ids) - 1:
        raise RuntimeError("V6.5 option count does not match candidate width")
    if any(not isinstance(item, str) or not item for item in candidate_ids):
        raise RuntimeError("V6.5 branch IDs must be nonempty strings")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("duplicate V6.5 candidate branch ID")
    if candidate_ids != sorted(candidate_ids):
        raise RuntimeError("V6.5 candidate branch IDs are not canonical")
    native = group["native_branch_id"]
    if native not in candidate_ids:
        raise RuntimeError("V6.5 native branch absent from candidate set")
    option_fields = {"option_id", "alternative_branch_id", "option_causal_sha256"}
    if any(not option_fields.issubset(option) for option in options):
        raise RuntimeError("V6.5 option misses required fields")
    alternatives = [option["alternative_branch_id"] for option in options]
    if alternatives != [item for item in candidate_ids if item != native]:
        raise RuntimeError("V6.5 option order/identity is not canonical")
    option_ids = [option["option_id"] for option in options]
    if (
        any(not isinstance(item, str) or not item for item in option_ids)
        or len(set(option_ids)) != len(option_ids)
    ):
        raise RuntimeError("invalid V6.5 option identity")
    if not all(is_sha256(option["option_causal_sha256"]) for option in options):
        raise RuntimeError("invalid V6.5 option causal hash")
    if not is_sha256(group["candidate_set_sha256"]) or not is_sha256(
        group["shared_state_sha256"]
    ) or not is_sha256(group["feature_sha256"]):
        raise RuntimeError("invalid V6.5 group hash")
    if (
        not isinstance(group["decision_index"], int)
        or group["decision_index"] < 0
        or not isinstance(group["prefix_action_count"], int)
        or group["prefix_action_count"] < 0
        or not isinstance(group["feature_bytes"], int)
        or group["feature_bytes"] <= 0
    ):
        raise RuntimeError("invalid V6.5 group integer field")
    summary_path = run_dir / "RUN_SUMMARY.json"
    trace_path = run_dir / "base_trace.jsonl"
    summary_record = source_record(summary_path)
    trace_record = source_record(trace_path)
    return {
        "group_id": str(group["group_id"]),
        "decision_index": group["decision_index"],
        "prefix_action_count": group["prefix_action_count"],
        "checkpoint_id": str(group["checkpoint_id"]),
        "native_branch_id": native,
        "candidate_branch_ids": candidate_ids,
        "candidate_set_sha256": group["candidate_set_sha256"],
        "shared_state_sha256": group["shared_state_sha256"],
        "feature_path": group["feature_path"],
        "feature_bytes": group["feature_bytes"],
        "feature_sha256": group["feature_sha256"],
        "options": options,
        "episode_id": episode["episode_id"],
        "trajectory_id": episode["trajectory_id"],
        "scene_id": episode["scene_id"],
        "language": episode["language"],
        "controller_seed": episode["controller_seed"],
        "scene_fold": episode["scene_fold"],
        "shadow_run_dir": relative(run_dir),
        "shadow_summary_path": relative(summary_path),
        "shadow_summary_bytes": summary_record["bytes"],
        "shadow_summary_sha256": summary_record["sha256"],
        "shadow_trace_path": relative(trace_path),
        "shadow_trace_bytes": trace_record["bytes"],
        "shadow_trace_sha256": trace_record["sha256"],
    }


def seal_group_selection(role: str) -> dict:
    selections = verify_seal()
    require_holdout_unlock(role)
    episodes = selections[role]["episodes"]
    selected = []
    group_ids: set[str] = set()
    for episode in episodes:
        run_dir = shadow_run_dir(role, episode)
        summary_path = run_dir / "RUN_SUMMARY.json"
        if not valid_shadow_summary(summary_path, role, episode):
            raise RuntimeError(f"missing valid shadow for episode {episode['episode_id']}")
        summary = json.loads(summary_path.read_text())
        eligible = []
        for raw_group in summary["decision_groups"]:
            group = normalized_group(raw_group, episode, run_dir)
            if group is not None:
                eligible.append(group)
        eligible.sort(key=lambda group: (group["decision_index"], group["group_id"]))
        for group in eligible[:MAX_GROUPS_PER_EPISODE]:
            if not group["group_id"] or group["group_id"] in group_ids:
                raise RuntimeError("empty or duplicate V6.5 group ID")
            group_ids.add(group["group_id"])
            selected.append(group)
    value = scoped({
        "schema_version": "revealnav-rxr-v6.5-group-selection/1",
        "status": "SEALED_AFTER_COMPLETE_SHADOWS_BEFORE_MACROS",
        "role": role,
        "rule": "earliest two eligible width-3/4 groups per sealed episode",
        "groups": selected,
        "group_count": len(selected),
        "episode_count": len({group["episode_id"] for group in selected}),
        "option_count": sum(len(group["options"]) for group in selected),
        "outcome_used_for_selection": False,
        "unseen_or_test_read": False,
        "paper_result": False,
    })
    seal_json(role_paths(role)["group_selection"], value)
    return value


def option_key(group: dict, option: dict) -> str:
    return stable_hash({
        "v6_5_group": group["group_id"],
        "option": option["option_id"],
    })


def option_target(group: dict, option: dict) -> dict:
    return scoped({
        "schema_version": "revealnav-rxr-v6.5-option-target/1",
        "group_id": group["group_id"],
        "decision_index": group["decision_index"],
        "prefix_action_count": group["prefix_action_count"],
        "checkpoint_id": group["checkpoint_id"],
        "native_branch_id": group["native_branch_id"],
        "candidate_branch_ids": group["candidate_branch_ids"],
        "candidate_set_sha256": group["candidate_set_sha256"],
        "shared_state_sha256": group["shared_state_sha256"],
        "option_id": option["option_id"],
        "alternative_branch_id": option["alternative_branch_id"],
        "option_causal_sha256": option["option_causal_sha256"],
    })


def macro_run_dir(role: str, group: dict, option: dict) -> Path:
    return role_paths(role)["runs"] / f"macro_{option_key(group, option)}"


def collect_macros(role: str, gpus: tuple[int, ...]) -> None:
    verify_seal()
    require_holdout_unlock(role)
    selection = seal_group_selection(role)
    jobs = []
    complete = 0
    prior_rejections = []
    for group in selection["groups"]:
        for option in group["options"]:
            key = option_key(group, option)
            target = option_target(group, option)
            target_path = role_paths(role)["targets"] / f"{key}.json"
            seal_json(target_path, target)
            run_dir = macro_run_dir(role, group, option)
            summary = run_dir / "RUN_SUMMARY.json"
            state = valid_macro_summary(summary, role, target)
            if state is not None:
                complete += 1
                if state == "rejected":
                    prior_rejections.append({
                        "job_id": f"macro:{group['group_id']}:{option['option_id']}",
                        "reason": "unexecutable_option",
                    })
                continue
            output = role_paths(role)["logs"] / f"macro_{key}.out"
            error = role_paths(role)["logs"] / f"macro_{key}.err"
            jobs.append({
                "id": f"macro:{group['group_id']}:{option['option_id']}",
                "stdout": output,
                "stderr": error,
                "accepted_states": {"accepted", "rejected"},
                "validate": lambda p=summary, r=role, t=target: (
                    valid_macro_summary(p, r, t)
                ),
                "command": [
                    str(PYTHON), str(WORKER),
                    "--mode", "macro",
                    "--role", role,
                    "--episode-id", group["episode_id"],
                    "--seed", str(group["controller_seed"]),
                    "--target", str(target_path),
                    "--run-dir", str(run_dir),
                ],
            })
    total = selection["option_count"]
    parallel_jobs(
        role, "macro", jobs, gpus, total, complete, prior_rejections
    )


def project_file(path_text: str, expected_bytes: int, expected_sha: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        raise RuntimeError("V6.5 provenance path must be project-relative")
    unresolved = ROOT / path
    if unresolved.is_symlink() or not unresolved.is_file():
        raise RuntimeError("V6.5 provenance is not a regular project file")
    resolved = unresolved.resolve()
    if ROOT not in resolved.parents:
        raise RuntimeError("V6.5 provenance resolves outside project")
    if resolved.stat().st_size != expected_bytes or sha256_file(resolved) != expected_sha:
        raise RuntimeError("V6.5 feature provenance drift")
    return resolved


def load_group_features(group: dict) -> dict[str, np.ndarray]:
    path = project_file(
        group["feature_path"], group["feature_bytes"], group["feature_sha256"]
    )
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != FEATURE_KEYS:
            raise RuntimeError("V6.5 feature NPZ schema drift")
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    option_count = len(group["options"])
    for key in (
        "instruction", "post_observation", "temporal_history", "checkpoint", "native",
    ):
        if values[key].shape != (768,) or not np.isfinite(values[key]).all():
            raise RuntimeError(f"invalid V6.5 shared feature {key}")
    if (
        values["option_embeddings"].shape != (option_count, 768)
        or not np.isfinite(values["option_embeddings"]).all()
        or values["option_scalars"].shape != (option_count, 16)
        or not np.isfinite(values["option_scalars"]).all()
        or values["option_ids"].shape != (option_count,)
        or values["option_ids"].dtype.kind not in "SU"
    ):
        raise RuntimeError("invalid V6.5 option feature shape or value")
    expected_ids = [option["option_id"] for option in group["options"]]
    if [str(item) for item in values["option_ids"].tolist()] != expected_ids:
        raise RuntimeError("V6.5 option feature ordering drift")
    candidate_hash, shared_hash, option_hashes = recompute_feature_hashes(
        group, values
    )
    if candidate_hash != group["candidate_set_sha256"]:
        raise RuntimeError("V6.5 feature candidate-set hash mismatch")
    if shared_hash != group["shared_state_sha256"]:
        raise RuntimeError("V6.5 feature shared-state hash mismatch")
    if option_hashes != [
        option["option_causal_sha256"] for option in group["options"]
    ]:
        raise RuntimeError("V6.5 feature option causal hash mismatch")
    return values


def trace_prefix_exact(shadow_dir: Path, macro_dir: Path, prefix: int) -> bool:
    shadow_path = shadow_dir / "base_trace.jsonl"
    macro_path = macro_dir / "base_trace.jsonl"
    if any(
        path.is_symlink() or not path.is_file() or ROOT not in path.resolve().parents
        for path in (shadow_path, macro_path)
    ):
        raise RuntimeError("missing regular V6.5 trace evidence")
    shadow = shadow_path.read_text().splitlines()
    macro = macro_path.read_text().splitlines()
    return (
        len(shadow) >= prefix
        and len(macro) >= prefix
        and shadow[:prefix] == macro[:prefix]
    )


def task_utility(metrics: dict) -> float:
    return (
        0.50 * float(metrics["ndtw"])
        + 0.25 * float(metrics["sdtw"])
        + 0.25 * float(metrics["spl"])
    )


def capacity_result(role: str, records: list[dict]) -> dict:
    observed = {
        "complete_groups": len(records),
        "episodes": len({record["episode_id"] for record in records}),
        "scenes": len({record["scene_id"] for record in records}),
        "positive_groups": sum(record["positive_group"] for record in records),
    }
    if role == "development":
        per_fold = Counter(
            str(record["scene_fold"])
            for record in records if record["positive_group"]
        )
        observed["positive_groups_per_fold"] = {
            str(fold): per_fold[str(fold)] for fold in range(5)
        }
    requirements = CAPACITY[role]
    failures = []
    for key in ("complete_groups", "episodes", "scenes", "positive_groups"):
        if observed[key] < requirements[key]:
            failures.append(
                f"{key}: observed {observed[key]} < required {requirements[key]}"
            )
    if role == "development":
        minimum = requirements["positive_groups_per_fold"]
        for fold in range(5):
            count = observed["positive_groups_per_fold"][str(fold)]
            if count < minimum:
                failures.append(
                    f"fold {fold} positive_groups: observed {count} < required {minimum}"
                )
    return {
        "passed": not failures,
        "requirements": requirements,
        "observed": observed,
        "failures": failures,
    }


def assemble(role: str) -> dict:
    verify_seal()
    require_holdout_unlock(role)
    layout = role_paths(role)
    if layout["arrays"].exists() or layout["manifest"].exists():
        raise RuntimeError("refusing to overwrite an existing V6.5 assembly")
    group_selection = seal_group_selection(role)
    arrays: dict[str, list] = defaultdict(list)
    records = []
    rejections = []
    for group in group_selection["groups"]:
        shadow_dir = ROOT / group["shadow_run_dir"]
        shadow_path = shadow_dir / "RUN_SUMMARY.json"
        episode = {
            key: group[key]
            for key in ("episode_id", "trajectory_id", "scene_id")
        }
        if not valid_shadow_summary(shadow_path, role, episode):
            raise RuntimeError("invalid V6.5 shadow evidence during assembly")
        shadow = json.loads(shadow_path.read_text())
        if source_record(shadow_path) != {
            "bytes": group["shadow_summary_bytes"],
            "sha256": group["shadow_summary_sha256"],
        }:
            raise RuntimeError("sealed V6.5 shadow summary drift")
        shadow_trace_path = shadow_dir / "base_trace.jsonl"
        if source_record(shadow_trace_path) != {
            "bytes": group["shadow_trace_bytes"],
            "sha256": group["shadow_trace_sha256"],
        }:
            raise RuntimeError("sealed V6.5 shadow trace drift")
        native_metrics = shadow["metrics"]
        features = load_group_features(group)
        option_evidence = []
        incomplete = False
        for option_index, option in enumerate(group["options"]):
            target = option_target(group, option)
            macro_dir = macro_run_dir(role, group, option)
            summary_path = macro_dir / "RUN_SUMMARY.json"
            state = valid_macro_summary(summary_path, role, target)
            if state is None:
                raise RuntimeError("missing or invalid V6.5 option macro evidence")
            macro = json.loads(summary_path.read_text())
            if not trace_prefix_exact(
                shadow_dir, macro_dir, group["prefix_action_count"]
            ):
                raise RuntimeError("V6.5 exact-prefix evidence drift")
            if macro["observed_shared_state_sha256"] != group["shared_state_sha256"]:
                raise RuntimeError("V6.5 shared-state hash drift")
            if macro["observed_candidate_set_sha256"] != group["candidate_set_sha256"]:
                raise RuntimeError("V6.5 candidate-set hash drift")
            if macro["observed_option_causal_sha256"] != option["option_causal_sha256"]:
                raise RuntimeError("V6.5 option causal hash drift")
            if macro["observed_candidate_branch_ids"] != group["candidate_branch_ids"]:
                raise RuntimeError("V6.5 candidate branch identity drift")
            if (
                state == "accepted"
                and macro["committed_alternative_branch_id"]
                != option["alternative_branch_id"]
            ):
                raise RuntimeError("V6.5 committed option identity drift")
            summary_record = source_record(summary_path)
            trace_record = source_record(macro_dir / "base_trace.jsonl")
            option_evidence.append({
                "option": option,
                "option_index": option_index,
                "state": state,
                "macro": macro,
                "summary_path": relative(summary_path),
                "summary_bytes": summary_record["bytes"],
                "summary_sha256": summary_record["sha256"],
                "trace_path": relative(macro_dir / "base_trace.jsonl"),
                "trace_bytes": trace_record["bytes"],
                "trace_sha256": trace_record["sha256"],
                "prefix_exact": True,
            })
            if state == "rejected":
                incomplete = True
        if incomplete:
            rejections.append({
                "group_id": group["group_id"],
                "episode_id": group["episode_id"],
                "reason": "incomplete_pre_enumerated_option_set",
                "included_in_main_dataset": False,
                "options": [
                    {
                        "option_id": item["option"]["option_id"],
                        "alternative_branch_id": item["option"]["alternative_branch_id"],
                        "state": item["state"],
                        "summary_path": item["summary_path"],
                        "summary_bytes": item["summary_bytes"],
                        "summary_sha256": item["summary_sha256"],
                        "trace_path": item["trace_path"],
                        "trace_bytes": item["trace_bytes"],
                        "trace_sha256": item["trace_sha256"],
                        "rejection_reason": item["macro"].get("rejection_reason"),
                        "prefix_exact": item["prefix_exact"],
                        "shared_state_sha256": group["shared_state_sha256"],
                        "candidate_set_sha256": group["candidate_set_sha256"],
                        "option_causal_sha256": item["option"]["option_causal_sha256"],
                    }
                    for item in option_evidence
                ],
            })
            continue

        group_index = len(records)
        group_options = []
        native_utility = task_utility(native_metrics)
        for item in option_evidence:
            option = item["option"]
            option_index = item["option_index"]
            macro_metrics = item["macro"]["metrics"]
            relative_advantage = task_utility(macro_metrics) - native_utility
            row_index = len(arrays["target"])
            for name in (
                "instruction", "post_observation", "temporal_history",
                "checkpoint", "native",
            ):
                arrays[name].append(features[name])
            arrays["option"].append(features["option_embeddings"][option_index])
            arrays["scalars"].append(features["option_scalars"][option_index])
            arrays["target"].append(relative_advantage)
            arrays["group_index"].append(group_index)
            arrays["option_index"].append(option_index)
            group_options.append({
                "row_index": row_index,
                "option_index": option_index,
                "option_id": option["option_id"],
                "alternative_branch_id": option["alternative_branch_id"],
                "option_causal_sha256": option["option_causal_sha256"],
                "macro_summary_path": item["summary_path"],
                "macro_summary_bytes": item["summary_bytes"],
                "macro_summary_sha256": item["summary_sha256"],
                "macro_trace_path": item["trace_path"],
                "macro_trace_bytes": item["trace_bytes"],
                "macro_trace_sha256": item["trace_sha256"],
                "macro_metrics": {
                    key: float(macro_metrics[key]) for key in METRICS
                },
                "metric_delta_macro_minus_native": {
                    key: float(macro_metrics[key]) - float(native_metrics[key])
                    for key in METRICS
                },
                "relative_advantage": relative_advantage,
                "macro_better_than_native": relative_advantage > 1e-6,
                "prefix_exact": True,
                "physical_return_verified": True,
                "topology_restored": True,
                "alternative_committed": True,
            })
        advantages = [item["relative_advantage"] for item in group_options]
        best = max([0.0, *advantages])
        teacher_best = [group["native_branch_id"]] if abs(best) <= 1e-6 else []
        teacher_best.extend(
            item["alternative_branch_id"]
            for item in group_options
            if abs(item["relative_advantage"] - best) <= 1e-6
        )
        records.append({
            "group_index": group_index,
            **{key: group[key] for key in (
                "group_id", "episode_id", "trajectory_id", "scene_id",
                "language", "controller_seed", "decision_index",
                "prefix_action_count", "checkpoint_id", "native_branch_id",
                "candidate_branch_ids", "candidate_set_sha256",
                "shared_state_sha256", "feature_path", "feature_bytes",
                "feature_sha256", "scene_fold", "shadow_summary_path",
                "shadow_summary_bytes", "shadow_summary_sha256",
                "shadow_trace_path", "shadow_trace_bytes", "shadow_trace_sha256",
            )},
            "candidate_width": len(group["candidate_branch_ids"]),
            "native_metrics": {
                key: float(native_metrics[key]) for key in METRICS
            },
            "options": group_options,
            "positive_group": max(advantages) > 1e-6,
            "teacher_best_branch_ids": teacher_best,
            "complete_option_set": True,
        })
    if not records:
        raise RuntimeError("no complete V6.5 multi-option groups")

    tensor_arrays = {
        name: np.asarray(values, dtype=(
            np.float16 if name in {
                "instruction", "post_observation", "temporal_history",
                "checkpoint", "native", "option",
            } else np.float32 if name in {"scalars", "target"} else np.int64
        ))
        for name, values in arrays.items()
    }
    if any(not np.isfinite(value).all() for value in tensor_arrays.values()):
        raise RuntimeError("non-finite value in assembled V6.5 arrays")
    capacity = capacity_result(role, records)
    atomic_npz_once(layout["arrays"], tensor_arrays)
    manifest = scoped({
        "schema_version": "revealnav-rxr-v6.5-multi-option-dataset/1",
        "status": (
            "RXR_V6_5_AUXILIARY_DIAGNOSTIC_DATA_READY"
            if capacity["passed"] else "AUXILIARY_DIAGNOSTIC_DATA_INSUFFICIENT"
        ),
        "role": role,
        "split": "RxR-train-English-only",
        "groups": records,
        "group_rejections": rejections,
        "metadata": {
            "attempted_groups": group_selection["group_count"],
            "complete_groups": len(records),
            "rejected_incomplete_groups": len(rejections),
            "options": len(tensor_arrays["target"]),
            "episodes": len({record["episode_id"] for record in records}),
            "trajectories": len({record["trajectory_id"] for record in records}),
            "scenes": len({record["scene_id"] for record in records}),
            "positive_groups": sum(record["positive_group"] for record in records),
            "future_information_used_for_online_input": 0,
            "unseen_or_test_read": False,
            "paper_result": False,
        },
        "capacity_gate": capacity,
        "arrays": {
            "path": relative(layout["arrays"]),
            "bytes": layout["arrays"].stat().st_size,
            "sha256": sha256_file(layout["arrays"]),
            "flattening": "one row per alternative; group_index preserves sets",
            "shapes": {
                name: list(value.shape) for name, value in tensor_arrays.items()
            },
            "dtypes": {
                name: str(value.dtype) for name, value in tensor_arrays.items()
            },
        },
        "protocol": {
            "path": relative(protocol_path()),
            "sha256": sha256_file(protocol_path()),
        },
        "group_selection": {
            "path": relative(layout["group_selection"]),
            "sha256": sha256_file(layout["group_selection"]),
        },
        "assembler": {
            "path": relative(PIPELINE),
            "sha256": sha256_file(PIPELINE),
        },
        "public_rxr_r2r_unseen_authorized": False,
        "uad_mainline_training_authorized": False,
    })
    seal_json(layout["manifest"], manifest)
    return manifest


def parse_gpus(value: str) -> tuple[int, ...]:
    try:
        gpus = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpus must be comma-separated integers") from error
    if not gpus or any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus must contain unique non-negative indices")
    return gpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=("seal", "shadow", "macro", "assemble", "all"),
    )
    parser.add_argument("--role", choices=ROLES, default="development")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5,6,7"))
    args = parser.parse_args()

    if args.stage in ("seal", "all"):
        value = seal()
        print(json.dumps({
            "status": value["protocol"]["status"],
            "development_episodes": value["selections"]["development"]["episode_count"],
            "holdout_episodes": value["selections"]["holdout"]["episode_count"],
            "protocol": relative(protocol_path()),
        }, sort_keys=True))
    if args.stage == "seal":
        return 0
    verify_seal()
    require_holdout_unlock(args.role)
    if args.stage in ("shadow", "all"):
        collect_shadows(args.role, args.gpus)
    if args.stage in ("macro", "all"):
        collect_macros(args.role, args.gpus)
    if args.stage in ("assemble", "all"):
        manifest = assemble(args.role)
        print(json.dumps({
            "status": manifest["status"],
            "role": manifest["role"],
            "capacity_gate": manifest["capacity_gate"],
            "manifest": relative(role_paths(args.role)["manifest"]),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
