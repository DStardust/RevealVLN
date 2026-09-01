#!/usr/bin/env python3
"""Audit MF3ZP Qwen preannotations and prepare blinded human review."""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revealnav_mf3.evidence_uad import derive_constraint_uad  # noqa: E402
from revealnav_mf3.qwen_evidence_annotation import parse_instruction_response, stable_sha256  # noqa: E402
from revealnav_mf3.revealskill_protocol import OUTPUT, verify_protocol  # noqa: E402

import annotate_mf3zp_qwen as annotation  # noqa: E402


PREAUDIT = OUTPUT / "MF3ZP_QWEN_PREANNOTATION_AUDIT.json"
REVIEW_TEMPLATE = OUTPUT / "MF3ZP_HUMAN_REVIEW_TEMPLATE.jsonl"
REVIEW_HTML = OUTPUT / "MF3ZP_HUMAN_REVIEW.html"
FORMAL_RESULT = OUTPUT / "MF3ZP_LABEL_VALIDITY_RESULT.json"


class LabelAuditError(RuntimeError):
    pass


def atomic_text(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise LabelAuditError(f"refusing to overwrite: {path}")
    partial = path.with_name(path.name + ".part")
    if partial.exists() or partial.is_symlink():
        raise LabelAuditError(f"stale partial: {partial}")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LabelAuditError(f"JSON object required: {path}")
    return value


def _graph_for_instruction(instruction: str):
    record = read_json(annotation.INSTRUCTION_DIR / f"{annotation._instruction_key(instruction)}.json")
    return parse_instruction_response(record["response"], instruction=instruction)


def build_review_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    status = annotation.status()
    if status["status"] != "MF3ZP_QWEN_PREANNOTATION_READY":
        raise LabelAuditError("Qwen preannotation is incomplete")
    events = annotation.read_events()
    tasks = annotation.prefix_tasks(events)
    task_by_key = {
        (str(task["dataset"]), str(task["scene_id"]), str(task["episode_id"]), str(task["event_id"]), int(task["prefix_step"])): task
        for task in tasks
    }
    records: list[dict[str, object]] = []
    kind_counts: Counter[str] = Counter()
    constraint_counts: list[int] = []
    for event in events:
        graph = _graph_for_instruction(str(event["instruction"]))
        constraint_counts.append(len(graph.constraints))
        kind_counts.update(constraint.kind.value for constraint in graph.constraints)
        prefixes: list[dict[str, object]] = []
        for step in range(int(event["prefix_start"]), int(event["prefix_end"]) + 1):
            task = task_by_key[(str(event["dataset"]), str(event["scene_id"]), str(event["episode_id"]), str(event["source_observation_stream_id"]), step)]
            qwen = read_json(annotation.EVIDENCE_DIR / f"{task['request_id']}.json")
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
            "schema_version": "revealnav-mf3zp-human-review/1",
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


def _review_html(records: list[dict[str, object]]) -> str:
    cards: list[str] = []
    for index, record in enumerate(records):
        final = record["prefixes"][-1]
        graph_rows = "".join(
            f"<tr><td>{html.escape(item['constraint_id'])}</td><td>{html.escape(item['kind'])}</td><td>{html.escape(item['subject'])}</td><td>{html.escape(', '.join(item['dependencies']) or '-')}</td></tr>"
            for item in record["constraint_graph"]
        )
        cards.append(f"""
<section id="event-{index}">
  <header><b>{index:03d}</b> · {html.escape(record['dataset'])} · {html.escape(record['scene_id'])} · ep {html.escape(record['episode_id'])}</header>
  <div class="grid">
    <div class="visual">
      <img src="../../../{html.escape(final['causal_storyboard_path'])}" alt="causal storyboard">
      <img src="../../../{html.escape(final['current_panorama_path'])}" alt="current panorama">
    </div>
    <aside>
      <h3>指令</h3><p>{html.escape(record['instruction'])}</p>
      <h3>审核规则</h3><p>逐 prefix 核对 Qwen 的 S/G/E。只看当前及过去图像；不推测未来、结果或正确动作。D 必须由连续 3 个 prefix 的 S=G=E 派生。</p>
      <table><tr><th>ID</th><th>类型</th><th>内容</th><th>依赖</th></tr>{graph_rows}</table>
      <p>实际填写文件：<code>MF3ZP_HUMAN_REVIEW_TEMPLATE.jsonl</code> 的 human_constraints。</p>
    </aside>
  </div>
</section>""")
    return """<!doctype html><meta charset="utf-8"><title>MF3ZP 人工审核</title>
<style>
body{font-family:system-ui,sans-serif;background:#eef1f5;margin:0;padding:12px;color:#17202a}
section{background:white;margin:0 0 18px;padding:14px;border-radius:10px;box-shadow:0 1px 5px #ccd}
header{font-size:18px;margin-bottom:10px}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(360px,1fr);gap:14px}
.visual img{display:block;width:100%;max-height:62vh;object-fit:contain;background:#111;margin-bottom:10px}
aside{overflow:auto;max-height:78vh}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd;padding:5px;text-align:left}
@media(max-width:900px){.grid{grid-template-columns:1fr}aside{max-height:none}}
</style>""" + "\n".join(cards)


def prepare() -> dict[str, object]:
    verify_protocol()
    records, stats = build_review_records()
    atomic_text(REVIEW_TEMPLATE, "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records))
    atomic_text(REVIEW_HTML, _review_html(records))
    result = {
        "schema_version": "revealnav-mf3zp-qwen-preannotation-audit/1",
        "status": "MF3ZP_QWEN_PREANNOTATION_READY",
        "scientific_gate": "LABEL_VALIDITY_NOT_RUN",
        "statistics": stats,
        "qwen_coverage": annotation.status(),
        "reviewers_required": 3,
        "adjudicator_required_on_disagreement": True,
        "uad_kappa_min": 0.65,
        "evidence_kappa_min": 0.70,
        "review_template_sha256": stable_sha256(records),
        "human_verified": False,
        "gold": False,
        "oracle_headroom_authorized": False,
        "checkpoint_generated": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }
    atomic_json(PREAUDIT, result)
    return result


def fleiss_kappa(items: list[list[str]], categories: tuple[str, ...]) -> float:
    if not items or any(len(ratings) != 3 for ratings in items):
        raise LabelAuditError("Fleiss kappa requires three ratings per item")
    n = 3
    counts = [[ratings.count(category) for category in categories] for ratings in items]
    p_bar = sum((sum(count * count for count in row) - n) / (n * (n - 1)) for row in counts) / len(counts)
    category_totals = [sum(row[index] for row in counts) for index in range(len(categories))]
    p = [total / (len(items) * n) for total in category_totals]
    p_e = sum(value * value for value in p)
    return 1.0 if math.isclose(p_e, 1.0) and math.isclose(p_bar, 1.0) else (p_bar - p_e) / (1.0 - p_e)


def read_reviews(paths: list[Path]) -> list[dict[str, dict[str, object]]]:
    if len(paths) != 3:
        raise LabelAuditError("exactly three independent reviewer files are required")
    outputs = []
    reviewer_ids: set[str] = set()
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by_event = {str(row["event_id"]): row for row in rows}
        if len(by_event) != 300 or any(row.get("review_complete") is not True or row.get("reviewer_blinded_to_outcomes") is not True for row in rows):
            raise LabelAuditError(f"incomplete/non-blinded review file: {path}")
        ids = {str(row["reviewer_id"]) for row in rows}
        if len(ids) != 1 or "FILL_REVIEWER_ID" in ids:
            raise LabelAuditError(f"reviewer identity invalid: {path}")
        reviewer_id = next(iter(ids))
        if reviewer_id in reviewer_ids:
            raise LabelAuditError("reviewers must be distinct")
        reviewer_ids.add(reviewer_id)
        outputs.append(by_event)
    if any(set(output) != set(outputs[0]) for output in outputs[1:]):
        raise LabelAuditError("review event populations differ")
    return outputs


def formal_audit(paths: list[Path]) -> dict[str, object]:
    verify_protocol()
    reviews = read_reviews(paths)
    evidence_items: list[list[str]] = []
    uad_items: list[list[str]] = []
    for event_id in sorted(reviews[0]):
        rows = [review[event_id] for review in reviews]
        graph_ids = [item["constraint_id"] for item in rows[0]["constraint_graph"]]
        if any([item["constraint_id"] for item in row["constraint_graph"]] != graph_ids for row in rows[1:]):
            raise LabelAuditError("review graph mismatch")
        per_reviewer_factors = []
        for row in rows:
            factors = {cid: ([], [], []) for cid in graph_ids}
            for prefix in row["prefixes"]:
                for cid in graph_ids:
                    item = prefix["human_constraints"][cid]
                    values = (item["instantiated"], item["distinguishable"], item["resolved"])
                    if any(type(value) is not bool for value in values):
                        raise LabelAuditError("human factor is not boolean")
                    for target, value in zip(factors[cid], values, strict=True):
                        target.append(value)
            per_reviewer_factors.append(factors)
        for prefix_index in range(len(rows[0]["prefixes"])):
            for cid in graph_ids:
                evidence_items.append(["1" if factors[cid][2][prefix_index] else "0" for factors in per_reviewer_factors])
        for cid in graph_ids:
            uad_items.append([
                derive_constraint_uad(*factors[cid])[-1].value
                for factors in per_reviewer_factors
            ])
    evidence_kappa = fleiss_kappa(evidence_items, ("0", "1"))
    uad_kappa = fleiss_kappa(uad_items, ("U", "A", "D"))
    passed = evidence_kappa >= 0.70 and uad_kappa >= 0.65
    result = {
        "schema_version": "revealnav-mf3zp-label-validity-result/1",
        "status": "MF3ZP_LABEL_VALIDITY_PASS" if passed else "MF3ZP_LABEL_VALIDITY_FAIL",
        "reviewer_count": 3,
        "uad_fleiss_kappa": uad_kappa,
        "evidence_closure_fleiss_kappa": evidence_kappa,
        "thresholds": {"uad": 0.65, "evidence_closure": 0.70},
        "oracle_headroom_authorized": passed,
        "checkpoint_generated": False,
        "public_split_access": {"val_seen": False, "val_unseen": False, "test": False, "test_challenge": False},
    }
    atomic_json(FORMAL_RESULT, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "formal"))
    parser.add_argument("--reviews", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else formal_audit(args.reviews)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"] != "MF3ZP_LABEL_VALIDITY_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
