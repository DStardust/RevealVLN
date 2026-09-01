#!/usr/bin/env python3
"""Build the outcome-blind 300-event MF3ZP RevealSkill pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.reveal_event_data import RevealEvent, canonical_event_id  # noqa: E402


REVISION = "mf3zp_revealskill_v1"
SOURCE_DIR = ROOT / "artifacts/training/mf3zp_qwen_uad_reference_v2"
SOURCE_PROTOCOL = SOURCE_DIR / "MF3ZP_QWEN_UAD_REFERENCE_PROTOCOL.json"
SOURCE_REQUESTS = SOURCE_DIR / "MF3ZP_ANNOTATION_REQUESTS.jsonl"
OUTPUT = ROOT / "artifacts/training/mf3zp_revealskill_v1"
EVENTS_PATH = OUTPUT / "MF3ZP_REVEAL_EVENTS.jsonl"
SELECTION_PATH = OUTPUT / "MF3ZP_REVEAL_PILOT_SELECTION.json"
AUDIT_PATH = OUTPUT / "MF3ZP_REVEAL_PILOT_DATA_AUDIT.json"
EVENTS_PER_DOMAIN = 150
FOLD_COUNT = 5
FOLD_SALT = "mf3zp-revealskill-v1-whole-mp3d-scene"

FORBIDDEN_KEYS = {
    "delta_utility", "reward", "utility", "success", "spl", "ndtw", "sdtw",
    "catastrophe", "outcome", "target", "future_frame", "future_candidate_set",
    "navmesh", "pose", "correct_action", "better_action",
}


class PilotBuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def inventory(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not path.is_file() or path.is_symlink() or resolved != ROOT.resolve() and ROOT.resolve() not in resolved.parents:
        raise PilotBuildError(f"invalid project-local source: {path}")
    return {
        "path": str(resolved.relative_to(ROOT.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: object, *, refuse_existing: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise PilotBuildError(f"refusing to overwrite: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise PilotBuildError(f"stale partial: {partial}")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, object]], *, refuse_existing: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise PilotBuildError(f"refusing to overwrite: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise PilotBuildError(f"stale partial: {partial}")
    with partial.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(partial, path)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotBuildError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PilotBuildError(f"JSON object required: {path}:{line_no}")
        rows.append(value)
    if not rows:
        raise PilotBuildError(f"empty JSONL: {path}")
    return rows


def reject_outcome_keys(value: object, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered in {"outcome_disclosed", "target_or_outcome_input", "future_observation_input"} and child is False:
                continue
            if lowered in FORBIDDEN_KEYS or lowered.startswith(("outcome_", "reward_", "future_", "treatment_")):
                raise PilotBuildError(f"outcome-bearing source field at {path}.{key}")
            reject_outcome_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_outcome_keys(child, f"{path}[{index}]")


def scene_fold(scene_id: str) -> int:
    digest = hashlib.sha256(f"{FOLD_SALT}:{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % FOLD_COUNT


def _candidate_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    contract = row.get("contract")
    if not isinstance(contract, Mapping) or not isinstance(contract.get("current_candidates"), list):
        raise PilotBuildError("source request lacks current candidates")
    result = tuple(str(item["alias"]) for item in contract["current_candidates"])
    if len(result) != len(set(result)):
        raise PilotBuildError("candidate aliases are not unique")
    return result


def candidate_events(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    streams: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        reject_outcome_keys(row)
        key = (str(row["dataset"]), str(row["scene_id"]), str(row["episode_id"]), str(row["event_id"]))
        streams[key].append(row)
    by_episode: dict[tuple[str, str, str], list[tuple[str, list[dict[str, object]]]]] = defaultdict(list)
    for (dataset, scene_id, episode_id, stream_id), stream_rows in streams.items():
        by_episode[(dataset, scene_id, episode_id)].append((stream_id, stream_rows))
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for episode_key, candidates in by_episode.items():
        # Source streams differ only by old event-local role-blinded aliases.
        # Retain exactly one without consulting outcomes.
        stream_id, stream_rows = min(candidates, key=lambda item: (stable_sha256({"source_stream": item[0]}), item[0]))
        grouped[(*episode_key, stream_id)] = stream_rows
    events: list[dict[str, object]] = []
    for key, episode_rows in grouped.items():
        episode_rows.sort(key=lambda item: int(item["prefix_step"]))
        prior: tuple[str, ...] | None = None
        for row in episode_rows:
            current = _candidate_ids(row)
            triggers: list[str] = []
            if prior is None:
                triggers.append("PREFIX_START")
            else:
                if len(current) > len(prior):
                    triggers.append("CANDIDATE_COUNT_INCREASE")
                if set(current) != set(prior):
                    triggers.append("CANDIDATE_IDENTITY_CHANGE")
                if current != prior:
                    triggers.append("CANDIDATE_RANK_CHANGE")
            prior = current
            if not triggers:
                continue
            dataset, scene_id, episode_id, source_stream_id = key
            step = int(row["prefix_step"])
            causal_payload = {
                "dataset": dataset,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "prefix_step": step,
                "source_observation_stream_id": source_stream_id,
                "instruction": row["contract"]["instruction"],
                "current_candidates": row["contract"]["current_candidates"],
                "causal_storyboard_sha256": row["causal_storyboard"]["sha256"],
                "causal_storyboard_steps": row["causal_storyboard"]["steps"],
                "current_panorama_sha256": row["current_panorama"]["sha256"],
            }
            event_id = canonical_event_id(dataset, scene_id, episode_id, step, triggers)
            event = RevealEvent(
                dataset=dataset,
                scene_id=scene_id,
                episode_id=episode_id,
                event_id=event_id,
                instruction=str(row["contract"]["instruction"]),
                constraint_graph_sha256=None,
                prefix_start=0,
                prefix_end=step,
                causal_prefix_sha256=stable_sha256(causal_payload),
                option_ids=current,
                source_request_id=str(row["request_id"]),
                observation_path=str(row["causal_storyboard"]["path"]),
                current_panorama_path=str(row["current_panorama"]["path"]),
                trigger_types=tuple(triggers),
            )
            mapping = event.as_mapping()
            mapping["source_observation_stream_id"] = source_stream_id
            mapping["scene_fold"] = scene_fold(scene_id)
            mapping["source_image_inventory"] = {
                "causal_storyboard": {
                    "bytes": int(row["causal_storyboard"]["bytes"]),
                    "sha256": str(row["causal_storyboard"]["sha256"]),
                },
                "current_panorama": {
                    "bytes": int(row["current_panorama"]["bytes"]),
                    "sha256": str(row["current_panorama"]["sha256"]),
                },
            }
            events.append(mapping)
    return events


def select_balanced(events: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for domain in ("R2R", "RxR"):
        by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
        for event in events:
            if event["dataset"] == domain:
                by_scene[str(event["scene_id"])].append(event)
        for values in by_scene.values():
            values.sort(key=lambda item: (stable_sha256(item["event_id"]), str(item["event_id"])))
        scene_order = sorted(by_scene, key=lambda scene: (stable_sha256({"scene": scene}), scene))
        picked: list[dict[str, object]] = []
        offset = 0
        while len(picked) < EVENTS_PER_DOMAIN:
            before = len(picked)
            for scene in scene_order:
                if offset < len(by_scene[scene]):
                    picked.append(by_scene[scene][offset])
                    if len(picked) == EVENTS_PER_DOMAIN:
                        break
            if len(picked) == before:
                raise PilotBuildError(f"insufficient outcome-blind events for {domain}")
            offset += 1
        selected.extend(picked)
    selected.sort(key=lambda item: (str(item["dataset"]), int(item["scene_fold"]), str(item["scene_id"]), str(item["event_id"])))
    if len(selected) != 2 * EVENTS_PER_DOMAIN or len({item["event_id"] for item in selected}) != len(selected):
        raise PilotBuildError("pilot event count/identity failure")
    return selected


def verify_source_protocol() -> dict[str, object]:
    protocol = read_json(SOURCE_PROTOCOL)
    if protocol.get("status") != "SEALED_BEFORE_MF3ZP_V2_OBSERVATION_OR_LABEL_RESULTS":
        raise PilotBuildError("source protocol status drift")
    auth = protocol.get("authorization", {})
    if auth.get("public_split_access") != {"test": False, "test_challenge": False, "val_seen": False, "val_unseen": False}:
        raise PilotBuildError("source public split boundary is not closed")
    observation = protocol.get("observation", {})
    if observation.get("split") != "train" or observation.get("target_or_outcome_input") is not False or observation.get("future_observation_input") is not False:
        raise PilotBuildError("source observation boundary drift")
    return protocol


def build() -> dict[str, object]:
    source_protocol = verify_source_protocol()
    rows = read_jsonl(SOURCE_REQUESTS)
    blacklisted = set(str(value) for value in source_protocol["consumed_confirmation_blacklist"])
    raw_events = candidate_events(rows)
    selected = select_balanced(raw_events)
    if {str(item["scene_id"]) for item in selected} & blacklisted:
        raise PilotBuildError("consumed confirmation scene entered pilot")
    for event in selected:
        for field in ("observation_path", "current_panorama_path"):
            path = ROOT / str(event[field])
            key = "causal_storyboard" if field == "observation_path" else "current_panorama"
            expected = event["source_image_inventory"][key]
            if inventory(path)["bytes"] != expected["bytes"] or sha256_file(path) != expected["sha256"]:
                raise PilotBuildError(f"causal image provenance mismatch: {path}")
    domain_counts = Counter(str(item["dataset"]) for item in selected)
    scene_counts = {domain: len({str(item["scene_id"]) for item in selected if item["dataset"] == domain}) for domain in domain_counts}
    fold_counts = Counter((str(item["dataset"]), int(item["scene_fold"])) for item in selected)
    selection = {
        "schema_version": "revealnav-mf3zp-reveal-pilot-selection/1",
        "revision": REVISION,
        "status": "OUTCOME_BLIND_PILOT_SELECTED",
        "selection_rule": "all prefix-start/candidate-count/candidate-identity/candidate-rank events; deterministic scene-round-robin; no outcome access",
        "source_protocol": inventory(SOURCE_PROTOCOL),
        "source_requests": inventory(SOURCE_REQUESTS),
        "events": selected,
        "event_count": len(selected),
        "domain_counts": dict(sorted(domain_counts.items())),
        "scene_counts": scene_counts,
        "fold_counts": {f"{domain}:{fold}": count for (domain, fold), count in sorted(fold_counts.items())},
        "event_ids_sha256": stable_sha256([item["event_id"] for item in selected]),
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
        "target_payload_read": False,
        "outcome_payload_read": False,
    }
    atomic_jsonl(EVENTS_PATH, selected)
    atomic_json(SELECTION_PATH, selection)
    audit = {
        "schema_version": "revealnav-mf3zp-reveal-pilot-audit/1",
        "status": "MF3ZP_REVEAL_PILOT_DATA_PASS",
        "event_count": len(selected),
        "domain_counts": dict(sorted(domain_counts.items())),
        "scene_counts": scene_counts,
        "raw_scene_count": len({str(item["scene_id"]) for item in selected}),
        "source_candidate_event_count": len(raw_events),
        "blacklist_intersection": [],
        "identity_conflicts": 0,
        "causal_image_provenance_verified": True,
        "scene_fold_rule": {"folds": FOLD_COUNT, "salt": FOLD_SALT, "shared_raw_scene_same_fold": True},
        "target_payload_read": False,
        "outcome_payload_read": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
        "selection": inventory(SELECTION_PATH),
        "events": inventory(EVENTS_PATH),
    }
    atomic_json(AUDIT_PATH, audit)
    return audit


def verify() -> dict[str, object]:
    audit = read_json(AUDIT_PATH)
    selection = read_json(SELECTION_PATH)
    events = read_jsonl(EVENTS_PATH)
    if audit.get("status") != "MF3ZP_REVEAL_PILOT_DATA_PASS" or len(events) != 300:
        raise PilotBuildError("pilot artifact status/count failure")
    if selection.get("events") != events:
        raise PilotBuildError("selection/event JSONL drift")
    if Counter(str(item["dataset"]) for item in events) != Counter({"R2R": 150, "RxR": 150}):
        raise PilotBuildError("pilot domain balance drift")
    if audit.get("selection") != inventory(SELECTION_PATH) or audit.get("events") != inventory(EVENTS_PATH):
        raise PilotBuildError("pilot inventory drift")
    verify_source_protocol()
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    result = build() if args.command == "build" else verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
