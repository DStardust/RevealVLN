#!/usr/bin/env python3
"""Prepare blinded human review from combined MF3ZP v1/v1.1 Qwen records."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


correction = _load(ROOT / "scripts/repair_mf3zp_qwen_evidence_v1_1.py", "mf3zp_evidence_v1_1_for_audit")
legacy = _load(ROOT / "scripts/audit_mf3zp_labels.py", "mf3zp_legacy_label_audit")
base = correction.base
OUTPUT = correction.OUTPUT
PREAUDIT = OUTPUT / "MF3ZP_QWEN_PREANNOTATION_AUDIT_V1_1.json"
REVIEW_TEMPLATE = OUTPUT / "MF3ZP_HUMAN_REVIEW_TEMPLATE_V1_1.jsonl"
REVIEW_HTML = OUTPUT / "MF3ZP_HUMAN_REVIEW_V1_1.html"


def build_review_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    status = correction.combined_status(write=False)
    if status["status"] != "MF3ZP_QWEN_PREANNOTATION_READY":
        raise RuntimeError("combined Qwen preannotation is incomplete")
    events = base.read_events()
    tasks = base.prefix_tasks(events)
    task_by_key = {
        (str(task["dataset"]), str(task["scene_id"]), str(task["episode_id"]), str(task["event_id"]), int(task["prefix_step"])): task
        for task in tasks
    }
    records: list[dict[str, object]] = []
    kind_counts: Counter[str] = Counter()
    constraint_counts: list[int] = []
    for event in events:
        graph = base.load_graph(str(event["instruction"]))
        constraint_counts.append(len(graph.constraints))
        kind_counts.update(constraint.kind.value for constraint in graph.constraints)
        prefixes = []
        for step in range(int(event["prefix_start"]), int(event["prefix_end"]) + 1):
            key = (
                str(event["dataset"]), str(event["scene_id"]), str(event["episode_id"]),
                str(event["source_observation_stream_id"]), step,
            )
            task = task_by_key[key]
            qwen = correction.combined_record(task)
            if qwen is None:
                raise RuntimeError(f"missing combined evidence record: {task['request_id']}")
            prefixes.append({
                "step": step,
                "causal_storyboard_path": task["causal_storyboard"]["path"],
                "current_panorama_path": task["current_panorama"]["path"],
                "current_candidate_ids": [item["alias"] for item in task["contract"]["current_candidates"]],
                "qwen_preannotation": qwen["normalized_constraints"],
                "human_constraints": {
                    constraint.constraint_id: {
                        "instantiated": None,
                        "distinguishable": None,
                        "resolved": None,
                        "correction_note": "",
                    }
                    for constraint in graph.constraints
                },
            })
        records.append({
            "schema_version": "revealnav-mf3zp-human-review/1.1",
            "reviewer_id": "FILL_REVIEWER_ID",
            "reviewer_blinded_to_outcomes": True,
            "event_id": event["event_id"],
            "dataset": event["dataset"],
            "scene_id": event["scene_id"],
            "episode_id": event["episode_id"],
            "instruction": event["instruction"],
            "constraint_graph": [constraint.as_mapping() for constraint in graph.constraints],
            "constraint_graph_sha256": graph.canonical_sha256(),
            "prefixes": prefixes,
            "review_complete": False,
        })
    stats = {
        "events": len(records),
        "constraint_count_total_event_weighted": sum(constraint_counts),
        "constraint_count_min": min(constraint_counts),
        "constraint_count_max": max(constraint_counts),
        "constraint_count_mean": sum(constraint_counts) / len(constraint_counts),
        "constraint_kind_counts_event_weighted": dict(sorted(kind_counts.items())),
        "prefix_reviews": sum(len(record["prefixes"]) for record in records),
    }
    return records, stats


def prepare() -> dict[str, object]:
    correction.verify()
    records, stats = build_review_records()
    legacy.atomic_text(
        REVIEW_TEMPLATE,
        "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records),
    )
    legacy.atomic_text(REVIEW_HTML, legacy._review_html(records))
    result = {
        "schema_version": "revealnav-mf3zp-qwen-preannotation-audit/1.1",
        "status": "MF3ZP_QWEN_PREANNOTATION_READY",
        "scientific_gate": "LABEL_VALIDITY_NOT_RUN",
        "model_identifier": "qwen3.8-max",
        "statistics": stats,
        "qwen_coverage": correction.combined_status(write=False),
        "correctness_protocol_sha256": correction.sha256_file(correction.CORRECTNESS_PROTOCOL),
        "reviewers_required": 3,
        "adjudicator_required_on_disagreement": True,
        "uad_kappa_min": 0.65,
        "evidence_kappa_min": 0.70,
        "review_template_sha256": base.stable_sha256(records),
        "human_verified": False,
        "gold": False,
        "oracle_headroom_authorized": False,
        "checkpoint_generated": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }
    legacy.atomic_json(PREAUDIT, result)
    return result


def main() -> int:
    result = prepare()
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
