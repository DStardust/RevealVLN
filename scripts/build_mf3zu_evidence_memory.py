#!/usr/bin/env python3
"""Freeze MF3ZU outcome-blind semantic memories and fixed 78D features."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for value in (ROOT, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import annotate_mf3zu_rxr_evidence as annotation  # noqa: E402
from revealnav_mf3.mf3zu_evidence_memory import (  # noqa: E402
    CANDIDATE_EVIDENCE_FEATURE_DIM,
    ConfidenceClass,
    EvidenceJudgement,
    K_MEM,
    REVISION,
    build_evidence_record,
    candidate_memory_feature,
    memory_required,
    retrieve_evidence,
    validate_evidence_response,
)


DEFAULT_OUTPUT = (
    ROOT / "artifacts/training/mf3zu_rxr_evidence_memory_feasibility_v1"
)
MEMORY_PATH_NAME = "MF3ZU_RXR_EVIDENCE_MEMORY.jsonl"
MANIFEST_PATH_NAME = "MF3ZU_RXR_EVIDENCE_MEMORY_MANIFEST.json"


class MF3ZUMemoryError(RuntimeError):
    pass


def atomic_json(path: Path, value: object, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and (path.exists() or path.is_symlink()):
        raise MF3ZUMemoryError(f"refusing to overwrite immutable artifact: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise MF3ZUMemoryError(f"stale partial output: {partial}")
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
        raise MF3ZUMemoryError(f"refusing to overwrite immutable artifact: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise MF3ZUMemoryError(f"stale partial output: {partial}")
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


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _map_judgement_candidates(
    judgement: EvidenceJudgement,
    aliases: Mapping[str, object],
) -> EvidenceJudgement:
    try:
        candidate_ids = tuple(str(aliases[value]) for value in judgement.candidate_ids)
    except KeyError as error:
        raise MF3ZUMemoryError("unknown local candidate alias") from error
    return EvidenceJudgement(
        instruction_atom_id=judgement.instruction_atom_id,
        evidence_type=judgement.evidence_type,
        active_for_current_ranking=judgement.active_for_current_ranking,
        relevant_to_current_ranking=judgement.relevant_to_current_ranking,
        historical_status=judgement.historical_status,
        current_status=judgement.current_status,
        source_step=judgement.source_step,
        candidate_ids=candidate_ids,
        semantic_value=judgement.semantic_value,
    )


def build(output: Path) -> dict:
    evidence_manifest_path = output / "MF3ZU_EVIDENCE_ANNOTATION_MANIFEST.json"
    evidence_manifest = annotation.strict_json(evidence_manifest_path)
    if (
        evidence_manifest.get("revision") != REVISION
        or evidence_manifest.get("status") != "PASS"
        or evidence_manifest.get("planned") != 1428
        or evidence_manifest.get("pass") != 1428
        or evidence_manifest.get("ranking_label_read") is not False
        or evidence_manifest.get("task_metric_read") is not False
        or evidence_manifest.get("public_split_access") is not False
    ):
        raise MF3ZUMemoryError("complete fixed evidence annotation is required")
    input_manifest_path = output / "MF3ZU_EVIDENCE_INPUT_MANIFEST.json"
    input_manifest = annotation.strict_json(input_manifest_path)
    if (
        input_manifest.get("revision") != REVISION
        or input_manifest.get("status") != "SEALED_BEFORE_EVIDENCE_RESPONSES"
        or input_manifest.get("request_count") != 1428
    ):
        raise MF3ZUMemoryError("sealed evidence request boundary is missing")
    request_path = ROOT / str(input_manifest["requests"]["path"])
    if annotation.inventory(request_path) != input_manifest["requests"]:
        raise MF3ZUMemoryError("sealed evidence requests changed")
    requests = annotation.jsonl(request_path)
    decisions = annotation.load_decisions(output)
    decision_by_event = {str(row["event_id"]): row for row in decisions}
    if (
        len(decision_by_event) != 1428
        or {str(row["event_id"]) for row in requests} != set(decision_by_event)
    ):
        raise MF3ZUMemoryError("decision/request identity mismatch")
    graphs, _ = annotation._instruction_graphs(output)

    collection = annotation._load_collection(output)
    full_by_episode: dict[str, dict[int, dict]] = {}
    for episode in collection["results"]:
        source = ROOT / str(episode["decision_rows"]["path"])
        if annotation.inventory(source) != episode["decision_rows"]:
            raise MF3ZUMemoryError("causal replay rows changed")
        full_by_episode[str(episode["episode_id"])] = {
            int(row["decision_step"]): row for row in annotation.jsonl(source)
        }

    rows: list[dict] = []
    evidence_type_counts: Counter[str] = Counter()
    record_counts: list[int] = []
    retrieved_counts: list[int] = []
    ages: list[int] = []
    current_absent_or_ambiguous = 0
    retrieved_from_two_or_more = 0
    retrieved_total = 0
    for request in requests:
        event_id = str(request["event_id"])
        decision = decision_by_event[event_id]
        episode_id = str(decision["episode_id"])
        step = int(decision["decision_step"])
        response_path = output / "responses/evidence" / f"{request['request_id']}.json"
        response = annotation.strict_json(response_path)
        if (
            response.get("status") != "PASS"
            or response.get("request_id") != request["request_id"]
            or response.get("ranking_label_read") is not False
            or response.get("task_metric_read") is not False
            or response.get("public_split_access") is not False
        ):
            raise MF3ZUMemoryError(f"invalid evidence response: {event_id}")
        aliases = request["candidate_alias_to_action_id"]
        judgements = tuple(
            _map_judgement_candidates(value, aliases)
            for value in validate_evidence_response(
                response.get("response"),
                graph=graphs[episode_id],
                decision_step=step,
                allowed_candidate_ids=list(aliases),
            )
        )
        if any(
            value not in decision["candidate_action_ids"]
            for judgement in judgements for value in judgement.candidate_ids
        ):
            raise MF3ZUMemoryError("evidence candidate binding escaped event")
        full_steps = full_by_episode[episode_id]
        records = []
        for judgement in judgements:
            if judgement.historical_status is not ConfidenceClass.OBSERVED:
                continue
            assert judgement.source_step is not None
            source = full_steps.get(judgement.source_step)
            if source is None or judgement.source_step >= step:
                raise MF3ZUMemoryError("evidence source is not causal")
            record = build_evidence_record(
                event_id=event_id,
                judgement=judgement,
                source_node_id=str(source["source_node_id"]),
                source_observation_sha256=str(
                    source["full_panorama"]["sha256"]
                ),
            )
            records.append(record)
            evidence_type_counts[record.evidence_type.value] += 1
            ages.append(step - record.source_step)
            current_absent_or_ambiguous += int(
                record.current_status is not ConfidenceClass.OBSERVED
            )
        active = tuple(
            judgement.instruction_atom_id
            for judgement in judgements
            if judgement.active_for_current_ranking
        )
        retrieved = retrieve_evidence(
            records,
            active_instruction_atom_ids=active,
            budget=K_MEM,
        )
        retrieved_total += len(retrieved)
        retrieved_from_two_or_more += sum(
            step - record.source_step >= 2 for record in retrieved
        )
        source_slots = [int(value) for value in decision["source_candidate_slots"]]
        action_ids = [str(value) for value in decision["candidate_action_ids"]]
        mapping = {
            str(key): int(value)
            for key, value in decision["candidate_id_to_feature_slot"].items()
        }
        if (
            len(source_slots) != len(action_ids)
            or set(mapping) != set(action_ids)
            or [mapping[value] for value in action_ids] != source_slots
            or len(set(source_slots)) != len(source_slots)
        ):
            raise MF3ZUMemoryError("candidate ID/feature-slot mapping drift")
        candidate_features = []
        for action_id in action_ids:
            feature = candidate_memory_feature(
                records,
                active_instruction_atom_ids=active,
                decision_step=step,
                candidate_id=action_id,
            )
            if feature.shape != (CANDIDATE_EVIDENCE_FEATURE_DIM,):
                raise MF3ZUMemoryError("candidate evidence feature shape drift")
            candidate_features.append({
                "candidate_action_id": action_id,
                "feature_slot": mapping[action_id],
                "feature": [float(value) for value in feature],
            })
        is_required = memory_required(judgements)
        record_counts.append(len(records))
        retrieved_counts.append(len(retrieved))
        rows.append({
            "schema_version": "revealnav-mf3zu-rxr-evidence-memory-row/1",
            "revision": REVISION,
            "event_id": event_id,
            "dataset": "RxR",
            "scene_id": str(decision["scene_id"]),
            "episode_id": episode_id,
            "decision_step": step,
            "feature_row_index": int(decision["feature_row_index"]),
            "scene_fold": int(decision["scene_fold"]),
            "prefix_sha256": str(decision["prefix_sha256"]),
            "source_feature_path": str(decision["source_feature_path"]),
            "source_feature_sha256": str(decision["source_feature_sha256"]),
            "candidate_action_ids": action_ids,
            "source_candidate_slots": source_slots,
            "active_candidate_feature_slots": [
                int(value)
                for value in decision["active_candidate_feature_slots"]
            ],
            "candidate_id_to_feature_slot": mapping,
            "active_instruction_atom_ids": list(active),
            "memory_required": is_required,
            "judgements": [value.as_mapping() for value in judgements],
            "records": [value.as_mapping() for value in records],
            "retrieved_records": [value.as_mapping() for value in retrieved],
            "retrieved_count": len(retrieved),
            "retrieval_budget": K_MEM,
            "candidate_memory_features_by_slot": candidate_features,
            "candidate_memory_feature_dim": CANDIDATE_EVIDENCE_FEATURE_DIM,
            "exact_target_artifact_opened": False,
            "ranking_label_read": False,
            "task_metric_read": False,
            "public_split_access": False,
        })
    rows.sort(key=lambda row: (
        str(row["scene_id"]), str(row["episode_id"]),
        int(row["decision_step"]), int(row["feature_row_index"]),
    ))
    if len(rows) != 1428 or len({row["event_id"] for row in rows}) != 1428:
        raise MF3ZUMemoryError("evidence memory cardinality drift")
    required_rows = [row for row in rows if row["memory_required"]]
    required_scenes = {row["scene_id"] for row in required_rows}
    support_pass = len(required_rows) >= 50 and len(required_scenes) >= 10
    memory_path = output / MEMORY_PATH_NAME
    atomic_jsonl(memory_path, rows, refuse_existing=True)
    total_records = sum(record_counts)
    diagnostics = {
        "decisions": len(rows),
        "memory_required_decisions": len(required_rows),
        "memory_required_raw_scenes": len(required_scenes),
        "memory_not_required_decisions": len(rows) - len(required_rows),
        "evidence_records": total_records,
        "records_per_decision_mean": (
            float(statistics.mean(record_counts)) if record_counts else 0.0
        ),
        "retrieved_per_decision_mean": (
            float(statistics.mean(retrieved_counts)) if retrieved_counts else 0.0
        ),
        "retrieved_per_decision_max": max(retrieved_counts, default=0),
        "evidence_type_distribution": dict(sorted(evidence_type_counts.items())),
        "historical_evidence_age": {
            "mean": float(statistics.mean(ages)) if ages else None,
            "median": float(statistics.median(ages)) if ages else None,
            "p90": _percentile(ages, 0.90),
        },
        "percentage_current_frame_absent_or_ambiguous": (
            100.0 * current_absent_or_ambiguous / total_records
            if total_records else 0.0
        ),
        "percentage_retrieved_from_at_least_two_steps_ago": (
            100.0 * retrieved_from_two_or_more / retrieved_total
            if retrieved_total else 0.0
        ),
    }
    manifest = {
        "schema_version": "revealnav-mf3zu-rxr-evidence-memory-manifest/1",
        "revision": REVISION,
        "status": (
            "MF3ZU_RXR_EVIDENCE_MEMORY_FROZEN"
            if support_pass else "MF3ZU_RXR_MEMORY_REQUIRED_SUPPORT_FAIL"
        ),
        "evidence_memory": annotation.inventory(memory_path),
        "source_evidence_input_manifest": annotation.inventory(
            input_manifest_path
        ),
        "source_evidence_annotation_manifest": annotation.inventory(
            evidence_manifest_path
        ),
        "rows": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "raw_scenes": len({row["scene_id"] for row in rows}),
        "memory_required_support": {
            "minimum_decisions": 50,
            "minimum_raw_scenes": 10,
            "observed_decisions": len(required_rows),
            "observed_raw_scenes": len(required_scenes),
            "pass": support_pass,
        },
        "feature_contract": {
            "record_feature_dim": 77,
            "candidate_binding_dim": 1,
            "candidate_feature_dim": CANDIDATE_EVIDENCE_FEATURE_DIM,
            "pooling": "mean_after_deterministic_K8_retrieval",
            "empty_memory": "all-zero vector",
        },
        "diagnostics": diagnostics,
        "human_review_performed": False,
        "exact_target_artifact_opened": False,
        "candidate_target_accessed": False,
        "outcome_or_utility_accessed": False,
        "ranking_label_read": False,
        "task_metric_read": False,
        "public_split_access": False,
        "full_navigation_run": False,
        "checkpoint_generated": False,
    }
    atomic_json(
        output / MANIFEST_PATH_NAME,
        manifest,
        refuse_existing=True,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        value = build(args.output_root.resolve())
        print(json.dumps(value, indent=2))
        return 0 if value["status"] == "MF3ZU_RXR_EVIDENCE_MEMORY_FROZEN" else 3
    except BaseException as error:
        print(
            f"MF3ZU_MEMORY_ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
