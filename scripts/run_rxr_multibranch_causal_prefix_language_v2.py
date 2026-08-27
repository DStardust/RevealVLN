#!/usr/bin/env python3
"""Run full-set causal language closure over verified RxR-train branches."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_cr5_causal_prefix_language as gate  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V2 = BASE / "multibranch_v2"
V1 = BASE / "causal_frontend"
gate.ANALYSIS = V2 / "RXR_MULTIBRANCH_CAUSAL_CANDIDATE_ANALYSIS_V2.json"
gate.MEDIA = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_MEDIA_MANIFEST_V2.json"
gate.PROMPT = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_LANGUAGE_PROMPT_V2.md"
gate.GEOMETRY = V2 / "RXR_MULTIBRANCH_DIRECTED_GEOMETRY_V2.json"
gate.INPUTS = BASE / "multiview_factory/RXR_PRIMARY_MULTIVIEW_INPUTS.json"
gate.RESULT_DIR = V2 / "prefix_language_results"
gate.OUT = V2 / "RXR_MULTIBRANCH_CAUSAL_PREFIX_LANGUAGE_GATE_V2.json"
gate.OUTPUT_REVISION = "rxr-multibranch-causal-prefix-language-gate/2"
gate.USE_ALL_BRANCHES = True
gate.RESPONSE_SCHEMA_VERSION = "revealnav-fullset-causal-prefix-language-v2"
gate.REQUEST_REVISION = "rxr-multibranch-causal-prefix-language-request/2"


def verified_pairwise_reuse(analysis):
    old_analysis_path = V1 / "RXR_EXPANSION_CAUSAL_CANDIDATE_ANALYSIS.json"
    old_geometry_path = BASE / "geometry/RXR_EXPANSION_DIRECTED_GEOMETRY.json"
    old_gate_path = V1 / "RXR_EXPANSION_CAUSAL_PREFIX_LANGUAGE_GATE.json"
    old_analysis = {row["event_id"]: row for row in
                    json.loads(old_analysis_path.read_text())["events"]}
    old_geometry = {row["event_id"]: row for row in
                    json.loads(old_geometry_path.read_text())["events"]}
    new_geometry = {row["event_id"]: row for row in
                    json.loads(gate.GEOMETRY.read_text())["events"]}
    old_gate = {row["event_id"]: row for row in
                json.loads(old_gate_path.read_text())["events"]}
    reuse = {}
    for event in analysis["events"]:
        if (event["status"] != "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
                or len(event["candidate_branch_ids"]) != 2
                or event["event_id"] not in old_gate):
            continue
        event_id = event["event_id"]
        prior = old_analysis.get(event_id)
        old_geo = old_geometry.get(event_id)
        new_geo = new_geometry[event_id]
        alternative = next(branch_id for branch_id in
                           event["candidate_branch_ids"]
                           if branch_id != event["target_branch_id"])
        if prior is None or old_geo is None:
            continue
        checks = [
            prior["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED",
            prior["target_branch_id"] == event["target_branch_id"],
            prior["alternative_branch_id"] == alternative,
            prior["Q_prefix"] == event["Q_prefix"],
            prior["D_prefix"] == event["D_prefix"],
            prior["branch_current_runs"] == event["branch_current_runs"],
            prior["stable_geometric_ready_runs"] ==
                event["stable_geometric_ready_runs"],
            old_geo["target"] == new_geo["target"],
            old_geo["alternative"] == new_geo["alternatives"][0],
        ]
        old_prefix = {row["prefix_index"]: row
                      for row in prior["prefix_records"]}
        for row in event["prefix_records"]:
            old = old_prefix.get(row["prefix_index"])
            checks.extend([
                old is not None,
                old is not None and old["branch_current"] ==
                    row["branch_current"],
                old is not None and old["alternative_in_causal_history"] ==
                    row["competition_in_causal_history"][alternative],
            ])
        if not all(checks):
            continue
        reuse[event_id] = old_gate[event_id]
    return reuse, {
        "analysis": {"path": str(old_analysis_path.relative_to(ROOT)),
                     "sha256": gate.sha256_file(old_analysis_path)},
        "geometry": {"path": str(old_geometry_path.relative_to(ROOT)),
                     "sha256": gate.sha256_file(old_geometry_path)},
        "language_gate": {"path": str(old_gate_path.relative_to(ROOT)),
                          "sha256": gate.sha256_file(old_gate_path)},
        "proof": (
            "two-candidate branch ids, geometry, causal runs, per-prefix "
            "current availability, and competition history are exactly equal"
        ),
    }


def main() -> int:
    analysis = json.loads(gate.ANALYSIS.read_text())
    media = json.loads(gate.MEDIA.read_text())
    available_event_ids = set(media["event_ranges"])
    required = {
        row["event_id"] for row in analysis["events"]
        if row["status"] == "FRONTEND_CAUSAL_READY_LANGUAGE_REQUIRED"
    }
    missing = sorted(required - available_event_ids)
    if missing:
        raise SystemExit(
            f"causal media missing for {len(missing)} full-set events"
        )
    gate.EXPECTED_ANALYSIS_SHA256 = gate.sha256_file(gate.ANALYSIS)
    gate.EXPECTED_MEDIA_SHA256 = gate.sha256_file(gate.MEDIA)
    gate.EXPECTED_EVENT_COUNT = len(required)
    gate.PAIRWISE_EQUIVALENCE_REUSE, gate.PAIRWISE_REUSE_SOURCE = \
        verified_pairwise_reuse(analysis)
    if len(gate.PAIRWISE_EQUIVALENCE_REUSE) != 571:
        raise SystemExit(
            "unexpected verified pairwise reuse count: "
            + str(len(gate.PAIRWISE_EQUIVALENCE_REUSE))
        )
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
