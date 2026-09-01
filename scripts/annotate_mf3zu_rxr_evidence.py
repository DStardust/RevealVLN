#!/usr/bin/env python3
"""Prepare and run the fixed two-stage MF3ZU RxR Qwen annotation.

Stage 1 creates one instruction-atom graph per selected train episode. Stage 2
sends the 1,428 exact-supported sanitized population decisions a complete
historical 12-view panorama storyboard plus a separate current 12-view
panorama. Only qwen3.8-max at temperature zero is used. Replay is restricted to
the 154 episodes represented by the sealed population; additional physical
prefix rows in those episodes serve only as its causal-history audit boundary.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Iterable, Mapping
import urllib.error
import urllib.request

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revealnav_mf3.mf3zu_evidence_memory import (  # noqa: E402
    EVIDENCE_SYSTEM_PROMPT,
    QWEN_ENABLE_THINKING,
    QWEN_ENDPOINT,
    QWEN_MAX_TOKENS,
    QWEN_MODEL,
    QWEN_TEMPERATURE,
    REVISION,
    evidence_contract,
    instruction_request,
    parse_instruction_response,
    reject_sensitive_mapping,
    stable_sha256,
    validate_evidence_response,
)


DEFAULT_OUTPUT = (
    ROOT / "artifacts/training/mf3zu_rxr_evidence_memory_feasibility_v1"
)
INSTRUCTION_DATA = (
    ROOT / "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
_WRITE_LOCK = threading.Lock()


class MF3ZUAnnotationError(RuntimeError):
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
        raise MF3ZUAnnotationError(f"path escapes project: {path}")
    return str(resolved.relative_to(ROOT.resolve()))


def inventory(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise MF3ZUAnnotationError(f"invalid regular file: {path}")
    return {
        "path": rel(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def strict_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MF3ZUAnnotationError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise MF3ZUAnnotationError(f"JSON object required: {path}")
    return value


def jsonl(path: Path, *, allow_empty: bool = False) -> list[dict]:
    rows: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MF3ZUAnnotationError(f"cannot read JSONL: {path}") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as error:
            raise MF3ZUAnnotationError(
                f"invalid JSONL at {path}:{number}"
            ) from error
        if not isinstance(value, dict):
            raise MF3ZUAnnotationError(
                f"JSONL object required at {path}:{number}"
            )
        rows.append(value)
    if not rows and not allow_empty:
        raise MF3ZUAnnotationError(f"empty JSONL: {path}")
    return rows


def atomic_json(path: Path, value: object, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise MF3ZUAnnotationError(f"refusing to overwrite: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise MF3ZUAnnotationError(f"stale partial output: {partial}")
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


def atomic_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    refuse_existing: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise MF3ZUAnnotationError(f"refusing to overwrite: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise MF3ZUAnnotationError(f"stale partial output: {partial}")
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


def _load_collection(output: Path) -> dict:
    path = output / "MF3ZU_OBSERVATION_COLLECTION_MANIFEST.json"
    value = strict_json(path)
    if (
        value.get("revision") != REVISION
        or value.get("status") != "PASS"
        or value.get("planned_episodes") != 154
        or value.get("completed_episodes") != 154
        or value.get("source_population_decisions") != 1428
        or int(value.get("ranking_eligible_decisions", 0)) < 1428
        or value.get("ranking_label_read") is not False
        or value.get("task_metric_read") is not False
        or value.get("public_split_access") is not False
    ):
        raise MF3ZUAnnotationError("observation collection is not a fixed pass")
    return value


def _population_rows(output: Path) -> list[dict]:
    manifest = strict_json(
        output / "MF3ZU_RXR_DECISION_POPULATION_MANIFEST.json"
    )
    if (
        manifest.get("revision") != REVISION
        or not str(manifest.get("status", "")).endswith("FROZEN")
        or manifest.get("population_rows") != 1428
        or manifest.get("episodes") != 154
        or manifest.get("raw_scenes") != 59
    ):
        raise MF3ZUAnnotationError("fixed exact-supported population is not frozen")
    source = manifest.get("population")
    if not isinstance(source, Mapping):
        raise MF3ZUAnnotationError("population inventory is missing")
    path = ROOT / str(source.get("path"))
    current = inventory(path)
    if any(current.get(key) != source.get(key) for key in ("path", "bytes", "sha256")):
        raise MF3ZUAnnotationError("frozen population file changed")
    rows = jsonl(path)
    forbidden = {
        "target", "target_index", "target_slot", "teacher",
        "teacher_action_id_label_only", "teacher_action_index_label_only",
        "correct_candidate", "reward", "utility", "outcome", "success",
    }
    for row in rows:
        lowered = {str(key).casefold() for key in row}
        if lowered & forbidden or any(
            key.startswith(("target_", "teacher_", "outcome_", "utility_"))
            for key in lowered
        ):
            raise MF3ZUAnnotationError(
                "ranking label field entered annotation population"
            )
    if (
        len(rows) != 1428
        or len({str(row.get("event_id")) for row in rows}) != 1428
        or len({str(row.get("episode_id")) for row in rows}) != 154
        or len({str(row.get("scene_id")) for row in rows}) != 59
    ):
        raise MF3ZUAnnotationError("frozen exact-supported population drift")
    return rows


def load_decisions(output: Path) -> list[dict]:
    collection = _load_collection(output)
    replay_rows: list[dict] = []
    for episode in collection["results"]:
        if episode.get("status") not in {"PASS", "SKIPPED_PASS"}:
            raise MF3ZUAnnotationError("non-passing episode in collection")
        source = ROOT / str(episode["decision_rows"]["path"])
        if inventory(source) != episode["decision_rows"]:
            raise MF3ZUAnnotationError("causal decision source changed")
        replay_rows.extend(
            row for row in jsonl(source) if row["ranking_eligible"]
        )
    expected_replay_rows = int(collection["ranking_eligible_decisions"])
    if (
        len(replay_rows) != expected_replay_rows
        or any(
            row.get("dataset") != "RxR"
            or row.get("split") != "train"
            or int(row.get("candidate_count", 0)) < 2
            or row.get("ranking_label_read") is not False
            or row.get("task_metric_read") is not False
            for row in replay_rows
        )
    ):
        raise MF3ZUAnnotationError("eligible replay universe drift")
    replay_by_identity = {
        (
            str(row["scene_id"]),
            str(row["episode_id"]),
            int(row["decision_step"]),
            int(row["feature_row_index"]),
        ): row
        for row in replay_rows
    }
    if len(replay_by_identity) != expected_replay_rows:
        raise MF3ZUAnnotationError("eligible replay identity collision")
    joined = []
    for population in _population_rows(output):
        identity = (
            str(population["scene_id"]),
            str(population["episode_id"]),
            int(population["decision_step"]),
            int(population["feature_row_index"]),
        )
        replay = replay_by_identity.get(identity)
        if replay is None:
            raise MF3ZUAnnotationError("population decision is absent from replay")
        action_ids = [str(value) for value in population["candidate_action_ids"]]
        active_slots = [
            int(value)
            for value in population["active_candidate_feature_slots"]
        ]
        if (
            action_ids != replay["candidate_action_ids"]
            or set(active_slots) != set(replay["active_feature_slots"])
            or len(active_slots) != len(action_ids)
            or str(population["source_feature_path"])
            != str(replay["source_feature"]["path"])
            or str(population["source_feature_sha256"])
            != str(replay["source_feature"]["sha256"])
        ):
            raise MF3ZUAnnotationError("population/replay candidate identity drift")
        row = dict(replay)
        row.update({
            "event_id": str(population["event_id"]),
            "scene_fold": int(population["scene_fold"]),
            "active_candidate_feature_slots": active_slots,
            "source_feature_path": str(population["source_feature_path"]),
            "source_feature_sha256": str(population["source_feature_sha256"]),
        })
        joined.append(row)
    joined.sort(key=lambda row: (
        str(row["scene_id"]), str(row["episode_id"]),
        int(row["decision_step"]), int(row["feature_row_index"]),
    ))
    return joined


def load_instructions() -> dict[str, str]:
    import gzip

    with gzip.open(INSTRUCTION_DATA, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    result: dict[str, str] = {}
    for episode in payload.get("episodes", []):
        instruction = episode.get("instruction", {})
        language = str(instruction.get("language", ""))
        text = instruction.get("instruction_text")
        episode_id = str(episode.get("episode_id"))
        if language not in {"en-US", "en-IN", ""}:
            continue
        if isinstance(text, str) and text.strip():
            result.setdefault(episode_id, text.strip())
    return result


def prepare_instructions(output: Path) -> dict:
    decisions = load_decisions(output)
    instructions = load_instructions()
    episodes = sorted(
        {str(row["episode_id"]) for row in decisions}, key=int
    )
    if len(episodes) != 154 or any(value not in instructions for value in episodes):
        raise MF3ZUAnnotationError("RxR train instruction coverage drift")
    rows = []
    for episode_id in episodes:
        payload = instruction_request(instructions[episode_id])
        reject_sensitive_mapping(payload)
        rows.append({
            "schema_version": "revealnav-mf3zu-instruction-request/1",
            "request_id": stable_sha256({
                "revision": REVISION,
                "stage": "instruction_graph",
                "episode_id": episode_id,
                "instruction": instructions[episode_id],
            }),
            "episode_id": episode_id,
            "instruction": instructions[episode_id],
            "payload": payload,
        })
    request_path = output / "MF3ZU_INSTRUCTION_REQUESTS.jsonl"
    atomic_jsonl(request_path, rows, refuse_existing=True)
    manifest = {
        "schema_version": "revealnav-mf3zu-instruction-input/1",
        "revision": REVISION,
        "status": "SEALED_BEFORE_INSTRUCTION_RESPONSES",
        "model": QWEN_MODEL,
        "temperature": QWEN_TEMPERATURE,
        "max_tokens": QWEN_MAX_TOKENS,
        "thinking": QWEN_ENABLE_THINKING,
        "episodes": len(rows),
        "requests": inventory(request_path),
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
    }
    atomic_json(
        output / "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json",
        manifest,
        refuse_existing=True,
    )
    return manifest


def _api_key() -> str:
    path = ROOT / ".secret/qwen_api_key"
    if not path.is_file() or path.is_symlink():
        raise MF3ZUAnnotationError("project-local Qwen key is unavailable")
    value = path.read_text(encoding="utf-8").strip()
    if not value or any(char.isspace() for char in value):
        raise MF3ZUAnnotationError("project-local Qwen key is invalid")
    return value


def _image_data(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise MF3ZUAnnotationError(f"invalid causal image: {path}")
    mime = "image/png" if path.suffix.casefold() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _provider_request(api_key: str, payload: Mapping[str, object]) -> dict:
    reject_sensitive_mapping(payload)
    request = urllib.request.Request(
        QWEN_ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error = "unknown"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body["choices"][0]["message"]["content"]
            if isinstance(message, list):
                message = "".join(
                    str(item.get("text", ""))
                    if isinstance(item, Mapping) else str(item)
                    for item in message
                )
            parsed = json.loads(str(message))
            return {
                "provider_model": body.get("model"),
                "response": parsed,
                "usage": body.get("usage"),
            }
        except (
            OSError, ValueError, KeyError, TypeError, urllib.error.HTTPError
        ) as error:
            last_error = type(error).__name__
            if attempt < 2:
                time.sleep(2.0)
    raise MF3ZUAnnotationError(f"fixed Qwen request failed: {last_error}")


def _run_response_stage(
    *,
    rows: list[dict],
    response_root: Path,
    build_payload: Callable[[dict], dict],
    validate: Callable[[dict, object], None],
    max_workers: int,
    stage: str,
) -> dict:
    if max_workers < 1:
        raise MF3ZUAnnotationError("max-workers must be positive")
    response_root.mkdir(parents=True, exist_ok=True)
    api_key = _api_key()
    todo: list[tuple[dict, Path]] = []
    for row in rows:
        path = response_root / f"{row['request_id']}.json"
        if path.is_file() and not path.is_symlink():
            try:
                stored = strict_json(path)
                if stored.get("status") == "PASS":
                    validate(row, stored.get("response"))
                    continue
            except BaseException:
                pass
        todo.append((row, path))
    completed = 0
    lock = threading.Lock()

    def one(item: tuple[dict, Path]) -> dict:
        nonlocal completed
        row, path = item
        try:
            payload = build_payload(row)
            if (
                payload.get("model") != QWEN_MODEL
                or payload.get("temperature") != 0.0
                or payload.get("max_tokens") != QWEN_MAX_TOKENS
                or payload.get("enable_thinking") is not False
            ):
                raise MF3ZUAnnotationError("fixed Qwen configuration drift")
            reject_sensitive_mapping(payload)
            result = _provider_request(api_key, payload)
            if str(result.get("provider_model")) != QWEN_MODEL:
                raise MF3ZUAnnotationError("provider model identity drift")
            validate(row, result["response"])
            value = {
                "schema_version": f"revealnav-mf3zu-{stage}-response/1",
                "revision": REVISION,
                "status": "PASS",
                "stage": stage,
                "request_id": row["request_id"],
                "model_requested": QWEN_MODEL,
                "provider_model": result.get("provider_model"),
                "temperature": QWEN_TEMPERATURE,
                "max_tokens": QWEN_MAX_TOKENS,
                "thinking": QWEN_ENABLE_THINKING,
                "request_payload_sha256": stable_sha256(payload),
                "response": result["response"],
                "usage": result.get("usage"),
                "ranking_label_read": False,
                "task_metric_read": False,
                "public_split_access": False,
            }
        except BaseException as error:
            value = {
                "schema_version": f"revealnav-mf3zu-{stage}-response/1",
                "revision": REVISION,
                "status": "FAIL",
                "stage": stage,
                "request_id": row["request_id"],
                "model_requested": QWEN_MODEL,
                "error": f"{type(error).__name__}: {error}",
                "ranking_label_read": False,
                "task_metric_read": False,
                "public_split_access": False,
            }
        atomic_json(path, value)
        with lock:
            completed += 1
        return value

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(todo)))) as pool:
        outputs = [future.result() for future in as_completed(
            [pool.submit(one, item) for item in todo]
        )]
    stored_rows = []
    for row in rows:
        path = response_root / f"{row['request_id']}.json"
        if not path.is_file() or path.is_symlink():
            continue
        try:
            value = strict_json(path)
            if value.get("status") == "PASS":
                validate(row, value.get("response"))
            stored_rows.append(value)
        except BaseException:
            continue
    passed = sum(value.get("status") == "PASS" for value in stored_rows)
    failures = [
        value for value in stored_rows if value.get("status") != "PASS"
    ]
    return {
        "status": (
            "PASS"
            if len(stored_rows) == len(rows) and passed == len(rows)
            else "FAIL"
        ),
        "planned": len(rows),
        "response_files": len(stored_rows),
        "pass": passed,
        "newly_attempted": completed,
        "new_failures": [
            value for value in outputs if value.get("status") != "PASS"
        ],
        "failures": failures,
    }


def annotate_instructions(output: Path, *, max_workers: int) -> dict:
    manifest_path = output / "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json"
    manifest = strict_json(manifest_path)
    if manifest.get("status") != "SEALED_BEFORE_INSTRUCTION_RESPONSES":
        raise MF3ZUAnnotationError("instruction inputs are not sealed")
    request_path = ROOT / str(manifest["requests"]["path"])
    if inventory(request_path) != manifest["requests"]:
        raise MF3ZUAnnotationError("instruction request file changed")
    rows = jsonl(request_path)

    def validate(row: dict, response: object) -> None:
        parse_instruction_response(
            response,
            instruction=str(row["instruction"]),
        )

    result = _run_response_stage(
        rows=rows,
        response_root=output / "responses/instruction",
        build_payload=lambda row: dict(row["payload"]),
        validate=validate,
        max_workers=max_workers,
        stage="instruction",
    )
    result.update({
        "schema_version": "revealnav-mf3zu-instruction-annotation/1",
        "revision": REVISION,
        "model": QWEN_MODEL,
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
    })
    atomic_json(output / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json", result)
    return result


def _read_full_panorama(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise MF3ZUAnnotationError(f"cannot read panorama: {path}")
    # Preserve the complete 12-view contact sheet.  Only deterministic spatial
    # downscaling is allowed before chronological tiling.
    return cv2.resize(image, (448, 387), interpolation=cv2.INTER_AREA)


def build_full_panorama_storyboard(
    panorama_rows: list[dict],
    destination: Path,
) -> dict[str, object]:
    if not panorama_rows:
        raise MF3ZUAnnotationError("historical storyboard cannot be empty")
    panes = []
    for row in panorama_rows:
        path = ROOT / str(row["full_panorama"]["path"])
        if inventory(path) != row["full_panorama"]:
            raise MF3ZUAnnotationError("causal panorama inventory drift")
        image = _read_full_panorama(path)
        banner = np.zeros((26, image.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            banner,
            f"prior step {int(row['decision_step']):03d}",
            (7, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panes.append(np.concatenate((banner, image), axis=0))
    columns = min(4, len(panes))
    blank = np.zeros_like(panes[0])
    grid_rows = []
    for index in range(0, len(panes), columns):
        row = panes[index:index + columns]
        grid_rows.append(np.concatenate(
            row + [blank] * (columns - len(row)), axis=1
        ))
    storyboard = np.concatenate(grid_rows, axis=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.stem + ".part.jpg")
    if not cv2.imwrite(
        str(partial), storyboard, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    ):
        raise MF3ZUAnnotationError("failed to write causal storyboard")
    os.replace(partial, destination)
    result = inventory(destination)
    result["steps"] = [int(row["decision_step"]) for row in panorama_rows]
    result["source_full_panorama_sha256"] = [
        str(row["full_panorama"]["sha256"]) for row in panorama_rows
    ]
    return result


def _instruction_graphs(output: Path) -> tuple[dict[str, object], dict[str, str]]:
    annotation = strict_json(
        output / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json"
    )
    if annotation.get("status") != "PASS" or annotation.get("pass") != 154:
        raise MF3ZUAnnotationError("instruction annotation is incomplete")
    request_manifest = strict_json(
        output / "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json"
    )
    request_path = ROOT / str(request_manifest["requests"]["path"])
    if inventory(request_path) != request_manifest["requests"]:
        raise MF3ZUAnnotationError("sealed instruction input changed")
    graphs: dict[str, object] = {}
    instructions: dict[str, str] = {}
    for row in jsonl(request_path):
        response_path = (
            output / "responses/instruction" / f"{row['request_id']}.json"
        )
        response = strict_json(response_path)
        if response.get("status") != "PASS":
            raise MF3ZUAnnotationError("instruction response is not a pass")
        graph = parse_instruction_response(
            response.get("response"), instruction=str(row["instruction"])
        )
        episode_id = str(row["episode_id"])
        graphs[episode_id] = graph
        instructions[episode_id] = str(row["instruction"])
    return graphs, instructions


def prepare_evidence(output: Path) -> dict:
    decisions = load_decisions(output)
    graphs, instructions = _instruction_graphs(output)
    by_episode: dict[str, list[dict]] = {}
    for row in decisions:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
    # Full causal episode rows include the 105 one-candidate steps omitted from
    # the ranking population.  Recover them from the already-audited files so
    # historical storyboards are never gapped.
    collection = _load_collection(output)
    full_by_episode: dict[str, list[dict]] = {}
    for episode in collection["results"]:
        source = ROOT / str(episode["decision_rows"]["path"])
        full_by_episode[str(episode["episode_id"])] = jsonl(source)

    rows = []
    for decision in decisions:
        episode_id = str(decision["episode_id"])
        step = int(decision["decision_step"])
        full_rows = full_by_episode[episode_id]
        if [int(row["decision_step"]) for row in full_rows] != list(range(len(full_rows))):
            raise MF3ZUAnnotationError("episode causal step sequence drift")
        historical = full_rows[:step]
        storyboard = None
        if historical:
            storyboard = build_full_panorama_storyboard(
                historical,
                output / "storyboards/RxR" / f"ep_{episode_id}"
                / f"history_before_{step:03d}.jpg",
            )
        raw_candidates = [str(value) for value in decision["candidate_action_ids"]]
        aliases = {
            # C aliases are deliberately distinct from the worker panorama's
            # local-waypoint L markers.  GraphMap ranking candidates can be a
            # strict subset of local waypoints, so equating the two identities
            # would create false visual bindings.  Relative heading is the
            # only role-blind candidate cue sent to the annotator.
            f"C{index:02d}": value
            for index, value in enumerate(raw_candidates)
        }
        headings = list(decision["candidate_relative_heading_rad"])
        cards = [
            {
                "candidate_id": alias,
                "relative_heading_rad": round(float(headings[index]), 6),
            }
            for index, alias in enumerate(aliases)
        ]
        contract = evidence_contract(
            instruction=instructions[episode_id],
            graph=graphs[episode_id],
            decision_step=step,
            current_candidates=cards,
            historical_steps=list(range(step)),
        )
        reject_sensitive_mapping(contract)
        current = decision["full_panorama"]
        current_path = ROOT / str(current["path"])
        if inventory(current_path) != current:
            raise MF3ZUAnnotationError("current causal panorama changed")
        rows.append({
            "schema_version": "revealnav-mf3zu-evidence-request/1",
            "request_id": stable_sha256({
                "revision": REVISION,
                "stage": "decision_evidence",
                "event_id": decision["event_id"],
                "prefix_sha256": decision["prefix_sha256"],
            }),
            "event_id": decision["event_id"],
            "scene_id": decision["scene_id"],
            "episode_id": episode_id,
            "decision_step": step,
            "prefix_sha256": decision["prefix_sha256"],
            "contract": contract,
            "candidate_alias_to_action_id": aliases,
            "historical_full_panorama_storyboard": storyboard,
            "current_full_panorama": current,
        })
    if len(rows) != 1428 or len({row["request_id"] for row in rows}) != 1428:
        raise MF3ZUAnnotationError("evidence request population drift")
    # Structural scan occurs before any provider call.  The per-event payload
    # excludes the raw action IDs stored only in the local projection map.
    for row in rows:
        reject_sensitive_mapping(row["contract"])
    request_path = output / "MF3ZU_EVIDENCE_REQUESTS.jsonl"
    atomic_jsonl(request_path, rows, refuse_existing=True)
    manifest = {
        "schema_version": "revealnav-mf3zu-evidence-input/1",
        "revision": REVISION,
        "status": "SEALED_BEFORE_EVIDENCE_RESPONSES",
        "model": QWEN_MODEL,
        "temperature": QWEN_TEMPERATURE,
        "max_tokens": QWEN_MAX_TOKENS,
        "thinking": QWEN_ENABLE_THINKING,
        "request_count": 1428,
        "episodes": 154,
        "historical_visual_rule": (
            "all complete 12-view panorama contact sheets at steps j<t; "
            "deterministically downscaled and chronologically tiled"
        ),
        "current_visual_rule": "complete 12-view panorama at step t, separate",
        "requests": inventory(request_path),
        "instruction_input_manifest": inventory(
            output / "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json"
        ),
        "instruction_annotation_manifest": inventory(
            output / "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json"
        ),
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
    }
    atomic_json(
        output / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json",
        manifest,
        refuse_existing=True,
    )
    return manifest


def _evidence_payload(row: Mapping[str, object]) -> dict:
    content: list[dict[str, object]] = [{
        "type": "text",
        "text": json.dumps(
            row["contract"], ensure_ascii=False, sort_keys=True
        ),
    }]
    historical = row.get("historical_full_panorama_storyboard")
    if isinstance(historical, Mapping):
        path = ROOT / str(historical["path"])
        if inventory(path) != {
            key: historical[key] for key in ("path", "bytes", "sha256")
        }:
            raise MF3ZUAnnotationError("historical storyboard changed")
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_data(path)},
        })
    current = row["current_full_panorama"]
    path = ROOT / str(current["path"])
    if inventory(path) != current:
        raise MF3ZUAnnotationError("current panorama changed")
    content.append({
        "type": "image_url",
        "image_url": {"url": _image_data(path)},
    })
    payload = {
        "model": QWEN_MODEL,
        "temperature": QWEN_TEMPERATURE,
        "max_tokens": QWEN_MAX_TOKENS,
        "enable_thinking": QWEN_ENABLE_THINKING,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    }
    reject_sensitive_mapping(payload)
    return payload


def annotate_evidence(output: Path, *, max_workers: int) -> dict:
    manifest = strict_json(output / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json")
    if manifest.get("status") != "SEALED_BEFORE_EVIDENCE_RESPONSES":
        raise MF3ZUAnnotationError("evidence inputs are not sealed")
    request_path = ROOT / str(manifest["requests"]["path"])
    if inventory(request_path) != manifest["requests"]:
        raise MF3ZUAnnotationError("sealed evidence request file changed")
    rows = jsonl(request_path)
    graphs, _ = _instruction_graphs(output)

    def validate(row: dict, response: object) -> None:
        validate_evidence_response(
            response,
            graph=graphs[str(row["episode_id"])],
            decision_step=int(row["decision_step"]),
            allowed_candidate_ids=list(row["candidate_alias_to_action_id"]),
        )

    result = _run_response_stage(
        rows=rows,
        response_root=output / "responses/evidence",
        build_payload=_evidence_payload,
        validate=validate,
        max_workers=max_workers,
        stage="evidence",
    )
    result.update({
        "schema_version": "revealnav-mf3zu-evidence-annotation/1",
        "revision": REVISION,
        "model": QWEN_MODEL,
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
    })
    atomic_json(output / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json", result)
    return result


def status(output: Path) -> dict:
    result = {"revision": REVISION, "output": rel(output)}
    for name in (
        "MF3ZU_OBSERVATION_COLLECTION_MANIFEST.json",
        "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json",
        "MF3ZU_INSTRUCTION_ANNOTATION_MANIFEST.json",
        "MF3ZU_EVIDENCE_INPUT_MANIFEST.json",
        "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json",
    ):
        path = output / name
        if path.is_file() and not path.is_symlink():
            result[name] = strict_json(path).get("status")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-instructions")
    instruction = sub.add_parser("annotate-instructions")
    instruction.add_argument("--max-workers", type=int, default=8)
    sub.add_parser("prepare-evidence")
    evidence = sub.add_parser("annotate-evidence")
    evidence.add_argument("--max-workers", type=int, default=8)
    run = sub.add_parser("run")
    run.add_argument("--max-workers", type=int, default=8)
    sub.add_parser("status")
    args = parser.parse_args()
    output = args.output_root.resolve()
    try:
        if args.command == "prepare-instructions":
            value = prepare_instructions(output)
        elif args.command == "annotate-instructions":
            value = annotate_instructions(output, max_workers=args.max_workers)
        elif args.command == "prepare-evidence":
            value = prepare_evidence(output)
        elif args.command == "annotate-evidence":
            value = annotate_evidence(output, max_workers=args.max_workers)
        elif args.command == "run":
            if not (output / "MF3ZU_INSTRUCTION_INPUT_MANIFEST.json").exists():
                prepare_instructions(output)
            first = annotate_instructions(output, max_workers=args.max_workers)
            if first["status"] != "PASS":
                value = first
            else:
                if not (output / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json").exists():
                    prepare_evidence(output)
                value = annotate_evidence(output, max_workers=args.max_workers)
        else:
            value = status(output)
        print(json.dumps(value, indent=2))
        return 0 if value.get("status", "PASS") != "FAIL" else 2
    except BaseException as error:
        print(
            f"MF3ZU_ANNOTATION_ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
