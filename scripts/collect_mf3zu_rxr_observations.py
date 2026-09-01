#!/usr/bin/env python3
"""Collect and bit-exactly audit MF3ZU RxR causal panoramas.

The source cohort is the 154 episodes covered by the separately frozen,
sanitized 1,428-decision population. The script never opens the physically
separate ranking-label table. Each selected native episode is replayed through
its last physical step and all compact feature rows are checked against MF3B.
"""

from __future__ import annotations

import argparse
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zu_evidence_memory import stable_sha256  # noqa: E402


REVISION = "mf3zu_rxr_evidence_memory_feasibility_v1"
DEFAULT_OUTPUT = (
    ROOT / "artifacts/training/mf3zu_rxr_evidence_memory_feasibility_v1"
)
SOURCE_MANIFEST = (
    ROOT / "artifacts/phase1/mf3b_uad_online/dataset_v1/"
    "MF3B_ONLINE_DATA_MANIFEST.json"
)
WORKER = ROOT / "scripts/mf3zp_observation_worker_v2.py"
PYTHON = ROOT / ".envs/etpr1/bin/python"
_WRITE_LOCK = threading.Lock()


class MF3ZUCollectionError(RuntimeError):
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
        raise MF3ZUCollectionError(f"path escapes project: {path}")
    return str(resolved.relative_to(ROOT.resolve()))


def inventory(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise MF3ZUCollectionError(f"invalid regular file: {path}")
    return {
        "path": rel(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def strict_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MF3ZUCollectionError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise MF3ZUCollectionError(f"JSON object required: {path}")
    return value


def jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MF3ZUCollectionError(f"cannot read JSONL: {path}") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as error:
            raise MF3ZUCollectionError(
                f"invalid JSONL at {path}:{number}"
            ) from error
        if not isinstance(value, dict):
            raise MF3ZUCollectionError(
                f"JSONL object required at {path}:{number}"
            )
        rows.append(value)
    if not rows:
        raise MF3ZUCollectionError(f"empty JSONL: {path}")
    return rows


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise MF3ZUCollectionError(f"stale partial output: {partial}")
    partial.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise MF3ZUCollectionError(f"stale partial output: {partial}")
    with partial.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n")
    os.replace(partial, path)


def _project_file(path: Path) -> Path:
    resolved = path.resolve()
    if (
        ROOT.resolve() not in resolved.parents
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise MF3ZUCollectionError(f"invalid project-local source: {path}")
    return resolved


def _sanitized_population(output: Path) -> tuple[list[dict], dict]:
    manifest_path = output / "MF3ZU_RXR_DECISION_POPULATION_MANIFEST.json"
    manifest = strict_json(manifest_path)
    if (
        manifest.get("revision") != REVISION
        or manifest.get("status") != "MF3ZU_RXR_EXACT_SUPPORT_POPULATION_FROZEN"
        or manifest.get("population_rows") != 1428
        or manifest.get("episodes") != 154
        or manifest.get("raw_scenes") != 59
    ):
        raise MF3ZUCollectionError("sanitized population is not frozen")
    source = manifest.get("population")
    if not isinstance(source, Mapping):
        raise MF3ZUCollectionError("sanitized population inventory is missing")
    path = ROOT / str(source.get("path"))
    current = inventory(path)
    if any(current.get(key) != source.get(key) for key in ("path", "bytes", "sha256")):
        raise MF3ZUCollectionError("sanitized population changed")
    rows = jsonl(path)
    forbidden = {
        "target", "target_index", "target_feature_slot", "teacher",
        "teacher_action_id_label_only", "teacher_action_index_label_only",
        "reward", "utility", "outcome", "success", "correct_candidate",
    }
    if (
        len(rows) != 1428
        or len({str(row.get("event_id")) for row in rows}) != 1428
        or any(
            ({str(key).casefold() for key in row} & forbidden)
            for row in rows
        )
    ):
        raise MF3ZUCollectionError("sanitized population schema/count drift")
    return rows, manifest


def load_tasks(output: Path = DEFAULT_OUTPUT) -> list[dict]:
    population, _ = _sanitized_population(output)
    selected_episodes = {str(row["episode_id"]) for row in population}
    if len(selected_episodes) != 154:
        raise MF3ZUCollectionError("sanitized population episode count drift")
    manifest = strict_json(SOURCE_MANIFEST)
    if manifest.get("failures") != []:
        raise MF3ZUCollectionError("MF3B source manifest contains failures")
    if manifest.get("public_unseen_authorized") is not False:
        raise MF3ZUCollectionError("MF3B source boundary drift")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 156:
        raise MF3ZUCollectionError("MF3B source must contain 156 episodes")
    tasks: list[dict] = []
    seen: set[str] = set()
    population_by_episode: dict[str, list[dict]] = {}
    for row in population:
        population_by_episode.setdefault(str(row["episode_id"]), []).append(row)
    for record in records:
        if not isinstance(record, Mapping):
            raise MF3ZUCollectionError("MF3B source record is not an object")
        episode_id = str(record.get("episode_id"))
        scene_id = str(record.get("scene_id"))
        if not episode_id or not scene_id or episode_id in seen:
            raise MF3ZUCollectionError("MF3B episode identity drift")
        seen.add(episode_id)
        if episode_id not in selected_episodes:
            continue
        feature = _project_file(ROOT / str(record.get("path")))
        if sha256_file(feature) != str(record.get("sha256")):
            raise MF3ZUCollectionError(f"MF3B feature SHA drift: {episode_id}")
        run_dir = feature.parent
        summary = strict_json(_project_file(run_dir / "RUN_SUMMARY.json"))
        if (
            summary.get("status") != "SHADOW_PASS"
            or str(summary.get("episode_id")) != episode_id
            or summary.get("split") != "train"
            or summary.get("actions_changed") != 0
            or summary.get("public_unseen_authorized") is not False
        ):
            raise MF3ZUCollectionError(
                f"MF3B run is not a frozen train-only pass: {episode_id}"
            )
        source_trace = _project_file(run_dir / "base_trace.jsonl")
        source_actions = jsonl(source_trace)
        with np.load(feature, allow_pickle=False) as arrays:
            mask = np.asarray(arrays["candidate_mask"])
            history = np.asarray(arrays["history_embeddings"])
            if mask.ndim != 2 or history.ndim != 2 or mask.shape[0] != history.shape[0]:
                raise MF3ZUCollectionError(
                    f"MF3B feature cardinality drift: {episode_id}"
                )
            feature_rows = int(mask.shape[0])
            eligible = int(np.sum(mask.sum(axis=1) >= 2))
        if feature_rows != int(record.get("steps", -1)) or feature_rows < 1:
            raise MF3ZUCollectionError(f"MF3B step count drift: {episode_id}")
        if len(source_actions) < feature_rows:
            raise MF3ZUCollectionError(
                f"MF3B physical trace shorter than compact features: {episode_id}"
            )
        tasks.append({
            "dataset": "RxR",
            "scene_id": scene_id,
            "episode_id": episode_id,
            "feature_rows": feature_rows,
            "physical_steps": len(source_actions),
            "decision_step": len(source_actions) - 1,
            "ranking_eligible_decisions": eligible,
            "source_feature": inventory(feature),
            "source_native_trace": inventory(source_trace),
            "population_decisions": sorted(
                population_by_episode[episode_id],
                key=lambda row: (
                    int(row["decision_step"]),
                    int(row["feature_row_index"]),
                ),
            ),
        })
    if len(tasks) != 154 or {str(row["episode_id"]) for row in tasks} != selected_episodes:
        raise MF3ZUCollectionError(
            "sanitized population/source episode coverage drift"
        )
    return sorted(tasks, key=lambda row: int(row["episode_id"]))


def _action_prefix_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "act": int(row["act"]),
        "ghost_vp": None if row.get("ghost_vp") is None else str(row["ghost_vp"]),
        "front_vp": None if row.get("front_vp") is None else str(row["front_vp"]),
        "back_path_len": int(row.get("back_path_len", 0)),
        "tryout": bool(row.get("tryout", False)),
    }


def causal_prefix_sha256(
    *,
    scene_id: str,
    episode_id: str,
    decision_step: int,
    source_actions: list[dict],
) -> str:
    if len(source_actions) <= decision_step:
        raise MF3ZUCollectionError("native source is shorter than decision")
    return stable_sha256({
        "dataset": "RxR",
        "scene_id": scene_id,
        "episode_id": episode_id,
        "decision_step": decision_step,
        "native_action_prefix_strictly_before_decision": [
            _action_prefix_row(row)
            for row in source_actions[:decision_step]
        ],
    })


def _array_equal(left: np.ndarray, right: np.ndarray, *, name: str) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
        raise MF3ZUCollectionError(f"bit-exact replay mismatch: {name}")


def _scalar_bit_equal(left: np.generic, right: np.generic) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def match_replay_candidates_to_feature_slots(
    *,
    replay_embeddings: np.ndarray,
    replay_scores: np.ndarray,
    source_embeddings: np.ndarray,
    source_scores: np.ndarray,
    active_slots: np.ndarray,
) -> tuple[list[int], bool]:
    """Find the exact candidate-ID order in compact MF3B slot coordinates.

    Embedding plus policy-score bytes are the identity.  GraphMap indices from
    the shadow trace are a different coordinate system and are never used as
    feature-slot indices.  Non-unique byte matches fail closed.
    """

    if replay_embeddings.shape != (len(active_slots), 768) or replay_scores.shape != (len(active_slots),):
        raise MF3ZUCollectionError("replay candidate array shape drift")
    choices: list[list[int]] = []
    for index in range(len(active_slots)):
        matches = []
        for slot in active_slots:
            slot_i = int(slot)
            if (
                np.asarray(replay_embeddings[index]).dtype
                == np.asarray(source_embeddings[slot_i]).dtype
                and np.asarray(replay_embeddings[index]).shape
                == np.asarray(source_embeddings[slot_i]).shape
                and np.asarray(replay_embeddings[index]).tobytes()
                == np.asarray(source_embeddings[slot_i]).tobytes()
                and _scalar_bit_equal(
                    replay_scores[index], source_scores[slot_i]
                )
            ):
                matches.append(slot_i)
        choices.append(matches)
    direct = [int(value) for value in active_slots]
    if not all(len(value) == 1 for value in choices):
        raise MF3ZUCollectionError("candidate slot mapping is not unique")
    mapped = [value[0] for value in choices]
    if len(set(mapped)) != len(mapped) or set(mapped) != set(direct):
        raise MF3ZUCollectionError("candidate slot mapping is not bijective")
    return mapped, mapped == direct


def audit_replay(task: Mapping[str, object], run_dir: Path) -> dict:
    """Audit one worker result and materialize label-free decision rows."""

    summary = strict_json(run_dir / "RUN_SUMMARY.json")
    if (
        summary.get("status") != "PASS"
        or summary.get("dataset") != "RxR"
        or summary.get("split") != "train"
        or summary.get("source_prefix_replay_exact") is not True
        or summary.get("source_target_action_compared") is not False
        or summary.get("no_outcome_or_target_input") is not True
        or summary.get("public_split_access") is not False
    ):
        raise MF3ZUCollectionError("observation worker boundary/status drift")
    feature_path = _project_file(ROOT / str(task["source_feature"]["path"]))
    source_trace_path = _project_file(
        ROOT / str(task["source_native_trace"]["path"])
    )
    if inventory(feature_path) != task["source_feature"]:
        raise MF3ZUCollectionError("source feature changed during replay")
    if inventory(source_trace_path) != task["source_native_trace"]:
        raise MF3ZUCollectionError("source native trace changed during replay")

    records = jsonl(run_dir / "causal_prefix_records.jsonl")
    panoramas = jsonl(run_dir / "panorama_manifest.jsonl")
    source_actions = jsonl(source_trace_path)
    physical_steps = int(task["physical_steps"])
    feature_rows = int(task["feature_rows"])
    expected_steps = list(range(physical_steps))
    for values, name in ((records, "records"), (panoramas, "panoramas")):
        if [int(row.get("step", -1)) for row in values] != expected_steps:
            raise MF3ZUCollectionError(f"replay {name} step sequence drift")
    if len(source_actions) != physical_steps:
        raise MF3ZUCollectionError("source native physical trace length drift")
    replay_actions = jsonl(run_dir / "base_trace.jsonl")
    if len(replay_actions) < physical_steps or [
        _action_prefix_row(row) for row in replay_actions[:physical_steps]
    ] != [_action_prefix_row(row) for row in source_actions]:
        raise MF3ZUCollectionError("full native physical action replay differs")
    panorama_by_step = {int(row["step"]): row for row in panoramas}
    sealed_population: dict[tuple[int, int], dict] = {}
    for row in task["population_decisions"]:
        coordinate = (
            int(row["decision_step"]),
            int(row["feature_row_index"]),
        )
        if coordinate in sealed_population:
            raise MF3ZUCollectionError("duplicate sealed population coordinate")
        if (
            str(row["episode_id"]) != str(task["episode_id"])
            or str(row["scene_id"]) != str(task["scene_id"])
            or str(row["source_feature_path"])
            != str(task["source_feature"]["path"])
            or str(row["source_feature_sha256"])
            != str(task["source_feature"]["sha256"])
            or str(row["source_native_trace_path"])
            != str(task["source_native_trace"]["path"])
            or str(row["source_native_trace_sha256"])
            != str(task["source_native_trace"]["sha256"])
        ):
            raise MF3ZUCollectionError("sealed population provenance drift")
        sealed_population[coordinate] = row
    matched_population: set[tuple[int, int]] = set()
    enriched: list[dict] = []
    with np.load(feature_path, allow_pickle=False) as source:
        instruction = np.asarray(source["instruction_embedding"])
        history = np.asarray(source["history_embeddings"])
        candidate_embeddings = np.asarray(source["candidate_embeddings"])
        candidate_mask = np.asarray(source["candidate_mask"])
        native_scores = np.asarray(source["native_scores"])
        if (
            instruction.shape != (768,)
            or history.shape != (feature_rows, 768)
            or candidate_embeddings.shape[:2] != candidate_mask.shape
            or candidate_embeddings.shape[0] != feature_rows
            or candidate_embeddings.shape[2] != 768
            or native_scores.shape != candidate_mask.shape
            or candidate_mask.dtype != np.bool_
        ):
            raise MF3ZUCollectionError("MF3B source array schema drift")
        feature_row = 0
        direct_slot_order_equal = 0
        for step, replay in enumerate(records):
            arrays_path = _project_file(ROOT / str(replay["arrays"]["path"]))
            if inventory(arrays_path) != replay["arrays"]:
                raise MF3ZUCollectionError("replay array inventory drift")
            replay_ids = tuple(
                str(value) for value in replay["candidate_action_ids"]
            )
            with np.load(arrays_path, allow_pickle=False) as actual:
                actual_instruction = np.asarray(actual["instruction"])
                actual_checkpoint = np.asarray(actual["checkpoint"])
                actual_embeddings = np.asarray(actual["action_embeddings"])
                actual_scores = np.asarray(actual["policy_scores"])
            source_slots_in_id_order: list[int] = []
            compact_index: int | None = None
            if replay_ids:
                if feature_row >= feature_rows:
                    raise MF3ZUCollectionError("replay has excess nonempty decisions")
                compact_index = feature_row
                slots = np.flatnonzero(candidate_mask[feature_row])
                if len(slots) != len(replay_ids):
                    raise MF3ZUCollectionError(
                        f"candidate mask cardinality mismatch at physical step {step}"
                    )
                candidate_graph_indices: list[int] = []
                source_slots_in_id_order, direct = match_replay_candidates_to_feature_slots(
                    replay_embeddings=actual_embeddings,
                    replay_scores=actual_scores,
                    source_embeddings=candidate_embeddings[feature_row],
                    source_scores=native_scores[feature_row],
                    active_slots=slots,
                )
                direct_slot_order_equal += int(direct)
                _array_equal(
                    actual_instruction,
                    instruction,
                    name=f"instruction[compact={feature_row},physical={step}]",
                )
                _array_equal(
                    actual_checkpoint,
                    history[feature_row],
                    name=f"history[compact={feature_row},physical={step}]",
                )
                _array_equal(
                    actual_embeddings,
                    candidate_embeddings[feature_row, source_slots_in_id_order],
                    name=f"candidate_embeddings[compact={feature_row},physical={step}]",
                )
                _array_equal(
                    actual_scores,
                    native_scores[feature_row, source_slots_in_id_order],
                    name=f"native_scores[compact={feature_row},physical={step}]",
                )
                sealed = sealed_population.get((step, feature_row))
                if sealed is not None:
                    sealed_ids = tuple(
                        str(value) for value in sealed["candidate_action_ids"]
                    )
                    sealed_active_slots = {
                        int(value)
                        for value in sealed["active_candidate_feature_slots"]
                    }
                    candidate_graph_indices = [
                        int(value) for value in sealed["candidate_graph_indices"]
                    ]
                    if (
                        replay_ids != sealed_ids
                        or set(source_slots_in_id_order) != sealed_active_slots
                        or len(candidate_graph_indices) != len(replay_ids)
                        or replay.get("native_action_id")
                        != sealed.get("native_action_id")
                    ):
                        raise MF3ZUCollectionError(
                            f"sealed population/replay mismatch at physical step {step}"
                        )
                    matched_population.add((step, feature_row))
                feature_row += 1
            elif (
                actual_embeddings.shape != (0, 768)
                or actual_scores.shape != (0,)
            ):
                raise MF3ZUCollectionError(
                    f"empty candidate replay array drift at physical step {step}"
                )
            pano_path = _project_file(
                ROOT / str(panorama_by_step[step]["path"])
            )
            if inventory(pano_path) != {
                key: panorama_by_step[step][key]
                for key in ("path", "bytes", "sha256")
            }:
                raise MF3ZUCollectionError("panorama inventory drift")
            source_node = str(source_actions[step].get("cur_vp", ""))
            if not source_node:
                raise MF3ZUCollectionError("source topology node is missing")
            event_id = stable_sha256({
                "revision": REVISION,
                "dataset": "RxR",
                "scene_id": task["scene_id"],
                "episode_id": task["episode_id"],
                "decision_step": step,
            })
            sealed = (
                sealed_population.get((step, compact_index))
                if compact_index is not None else None
            )
            if sealed is not None:
                event_id = str(sealed["event_id"])
            headings = replay.get("candidate_relative_heading_rad", [])
            if not isinstance(headings, list) or len(headings) != len(replay_ids):
                raise MF3ZUCollectionError("candidate heading cardinality drift")
            enriched.append({
                "schema_version": "revealnav-mf3zu-causal-decision/1",
                "revision": REVISION,
                "event_id": event_id,
                "dataset": "RxR",
                "split": "train",
                "scene_id": str(task["scene_id"]),
                "episode_id": str(task["episode_id"]),
                "decision_step": step,
                "source_node_id": source_node,
                "prefix_sha256": causal_prefix_sha256(
                    scene_id=str(task["scene_id"]),
                    episode_id=str(task["episode_id"]),
                    decision_step=step,
                    source_actions=source_actions,
                ),
                "candidate_action_ids": list(replay_ids),
                "candidate_relative_heading_rad": [float(value) for value in headings],
                "feature_row_index": compact_index,
                "source_candidate_slots": source_slots_in_id_order,
                "active_feature_slots": (
                    sorted(source_slots_in_id_order)
                    if source_slots_in_id_order else []
                ),
                "candidate_id_to_feature_slot": {
                    identity: slot
                    for identity, slot in zip(
                        replay_ids, source_slots_in_id_order, strict=True
                    )
                },
                "candidate_graph_indices": (
                    candidate_graph_indices if replay_ids else []
                ),
                "sealed_shadow_provenance": (
                    {
                        "source_sha256": str(sealed["source_shadow_sha256"]),
                        "record_hash": str(sealed["source_shadow_record_hash"]),
                    }
                    if sealed is not None else None
                ),
                "candidate_count": len(replay_ids),
                "ranking_eligible": len(replay_ids) >= 2,
                "source_feature": dict(task["source_feature"]),
                "replay_arrays": inventory(arrays_path),
                "full_panorama": inventory(pano_path),
                "observation_action_changed": False,
                "ranking_label_read": False,
                "task_metric_read": False,
            })
        if feature_row != feature_rows:
            raise MF3ZUCollectionError(
                f"nonempty replay/MF3B compact row mismatch: {feature_row}!={feature_rows}"
            )
        if matched_population != set(sealed_population):
            raise MF3ZUCollectionError(
                "sealed population is not completely covered by replay"
            )
    if sum(bool(row["ranking_eligible"]) for row in enriched) != int(task["ranking_eligible_decisions"]):
        raise MF3ZUCollectionError("eligible decision count changed in replay")
    decisions_path = run_dir / "MF3ZU_CAUSAL_DECISIONS.jsonl"
    atomic_jsonl(decisions_path, enriched)
    audit = {
        "schema_version": "revealnav-mf3zu-rxr-bit-exact-replay-audit/1",
        "revision": REVISION,
        "status": "PASS",
        "dataset": "RxR",
        "split": "train",
        "scene_id": str(task["scene_id"]),
        "episode_id": str(task["episode_id"]),
        "physical_steps": physical_steps,
        "compact_feature_rows": feature_rows,
        "ranking_eligible_decisions": int(task["ranking_eligible_decisions"]),
        "bit_exact": {
            "instruction_embedding": True,
            "history_embeddings": True,
            "candidate_embeddings": True,
            "candidate_mask_and_ids": True,
            "native_scores": True,
            "sealed_population_native_action_ids": True,
        },
        "teacher_bearing_source_opened": False,
        "direct_mask_slot_order_equal_rows": direct_slot_order_equal,
        "prefix_sha256_rebuilt": True,
        "source_node_id_rebuilt": True,
        "decision_rows": inventory(decisions_path),
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
    }
    atomic_json(run_dir / "MF3ZU_REPLAY_AUDIT.json", audit)
    return audit


def _attempt_dir(output: Path, episode_id: str) -> Path:
    parent = output / "observations/RxR" / f"ep_{episode_id}"
    parent.mkdir(parents=True, exist_ok=True)
    attempts = sorted(
        value for value in parent.glob("attempt_*")
        if value.is_dir() and not value.is_symlink()
    )
    return parent / f"attempt_{len(attempts) + 1:03d}"


def _passing_attempt(output: Path, task: Mapping[str, object]) -> Path | None:
    parent = output / "observations/RxR" / f"ep_{task['episode_id']}"
    if not parent.is_dir() or parent.is_symlink():
        return None
    passing: list[Path] = []
    for path in sorted(parent.glob("attempt_*/MF3ZU_REPLAY_AUDIT.json")):
        try:
            value = strict_json(path)
        except MF3ZUCollectionError:
            continue
        if (
            value.get("status") == "PASS"
            and value.get("episode_id") == str(task["episode_id"])
            and value.get("bit_exact", {}).get("candidate_mask_and_ids") is True
        ):
            passing.append(path.parent)
    if len(passing) > 1:
        raise MF3ZUCollectionError(
            f"multiple passing attempts for episode {task['episode_id']}"
        )
    return passing[0] if passing else None


def run_one(
    task: Mapping[str, object],
    *,
    output: Path,
    gpu_id: int,
    max_attempts: int,
) -> dict:
    prior = _passing_attempt(output, task)
    if prior is not None:
        audit = audit_replay(task, prior)
        return {
            "status": "SKIPPED_PASS",
            "episode_id": str(task["episode_id"]),
            "scene_id": str(task["scene_id"]),
            "run_dir": rel(prior),
            "audit": inventory(prior / "MF3ZU_REPLAY_AUDIT.json"),
            "decision_rows": audit["decision_rows"],
        }
    last_error = "no attempt"
    for _ in range(max_attempts):
        run_dir = _attempt_dir(output, str(task["episode_id"]))
        stdout = run_dir.with_name(run_dir.name + ".stdout.log")
        stderr = run_dir.with_name(run_dir.name + ".stderr.log")
        command = [
            str(PYTHON), str(WORKER),
            "--dataset", "RxR",
            "--episode-id", str(task["episode_id"]),
            "--scene-id", str(task["scene_id"]),
            "--source-native-trace",
            str(ROOT / str(task["source_native_trace"]["path"])),
            "--run-dir", str(run_dir),
            "--decision-step", str(task["decision_step"]),
            "--gpu-id", str(gpu_id),
            "--source-trace-mode", "native_reference",
        ]
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(ROOT),
        })
        try:
            with stdout.open("w", encoding="utf-8") as out, stderr.open(
                "w", encoding="utf-8"
            ) as err:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=out,
                    stderr=err,
                    check=False,
                )
            if result.returncode != 0:
                last_error = f"worker returncode={result.returncode}"
                continue
            audit = audit_replay(task, run_dir)
            return {
                "status": "PASS",
                "episode_id": str(task["episode_id"]),
                "scene_id": str(task["scene_id"]),
                "run_dir": rel(run_dir),
                "audit": inventory(run_dir / "MF3ZU_REPLAY_AUDIT.json"),
                "decision_rows": audit["decision_rows"],
            }
        except BaseException as error:
            last_error = f"{type(error).__name__}: {error}"
    return {
        "status": "FAIL",
        "episode_id": str(task["episode_id"]),
        "scene_id": str(task["scene_id"]),
        "error": last_error,
    }


def collect(
    *,
    output: Path,
    max_workers: int,
    gpu_ids: list[int],
    max_attempts: int,
) -> dict:
    population_rows, population_manifest = _sanitized_population(output)
    tasks = load_tasks(output)
    if (
        max_workers < 1
        or max_attempts < 1
        or not gpu_ids
        or any(value < 0 for value in gpu_ids)
    ):
        raise MF3ZUCollectionError("invalid worker/GPU/attempt configuration")
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "MF3ZU_OBSERVATION_COLLECTION_STATUS.json"
    results: list[dict] = []
    started = time.time()

    def save_status(status: str = "RUNNING") -> None:
        with _WRITE_LOCK:
            atomic_json(status_path, {
                "schema_version": "revealnav-mf3zu-observation-status/1",
                "revision": REVISION,
                "status": status,
                "planned_episodes": len(tasks),
                "completed_episodes": len(results),
                "pass": sum(
                    row["status"] in {"PASS", "SKIPPED_PASS"}
                    for row in results
                ),
                "fail": sum(row["status"] == "FAIL" for row in results),
                "elapsed_seconds": round(time.time() - started, 3),
                "results": sorted(
                    results, key=lambda row: int(row["episode_id"])
                ),
                "ranking_label_read": False,
                "task_metric_read": False,
                "public_split_access": False,
            })

    save_status()
    workers = min(max_workers, len(gpu_ids), len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_one,
                task,
                output=output,
                gpu_id=gpu_ids[index % len(gpu_ids)],
                max_attempts=max_attempts,
            ): task
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            try:
                result = future.result()
            except BaseException as error:
                task = futures[future]
                result = {
                    "status": "FAIL",
                    "episode_id": str(task["episode_id"]),
                    "scene_id": str(task["scene_id"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            results.append(result)
            save_status()
    failures = [row for row in results if row["status"] == "FAIL"]
    final = {
        "schema_version": "revealnav-mf3zu-observation-collection/1",
        "revision": REVISION,
        "status": "PASS" if not failures and len(results) == 154 else "FAIL",
        "source_manifest": inventory(SOURCE_MANIFEST),
        "source_population_manifest": inventory(
            output / "MF3ZU_RXR_DECISION_POPULATION_MANIFEST.json"
        ),
        "source_population": {
            key: population_manifest["population"][key]
            for key in ("path", "bytes", "sha256")
        },
        "planned_episodes": 154,
        "source_population_decisions": len(population_rows),
        "completed_episodes": len(results),
        "planned_physical_steps": sum(int(row["physical_steps"]) for row in tasks),
        "compact_feature_rows": sum(int(row["feature_rows"]) for row in tasks),
        "ranking_eligible_decisions": sum(
            int(row["ranking_eligible_decisions"]) for row in tasks
        ),
        "bit_exact_episode_audits": sum(
            row["status"] in {"PASS", "SKIPPED_PASS"} for row in results
        ),
        "failures": failures,
        "results": sorted(results, key=lambda row: int(row["episode_id"])),
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
    }
    atomic_json(output / "MF3ZU_OBSERVATION_COLLECTION_MANIFEST.json", final)
    save_status(final["status"])
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    try:
        tasks = load_tasks(args.output_root.resolve())
        if args.list_only:
            print(json.dumps({
                "status": "PASS",
                "episodes": len(tasks),
                "physical_steps": sum(int(row["physical_steps"]) for row in tasks),
                "compact_feature_rows": sum(int(row["feature_rows"]) for row in tasks),
                "ranking_eligible_decisions": sum(
                    int(row["ranking_eligible_decisions"]) for row in tasks
                ),
                "scenes": len({row["scene_id"] for row in tasks}),
            }, indent=2))
            return 0
        result = collect(
            output=args.output_root.resolve(),
            max_workers=args.max_workers,
            gpu_ids=[
                int(value) for value in args.gpu_ids.split(",") if value.strip()
            ],
            max_attempts=args.max_attempts,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    except BaseException as error:
        print(
            f"MF3ZU_COLLECTION_ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
