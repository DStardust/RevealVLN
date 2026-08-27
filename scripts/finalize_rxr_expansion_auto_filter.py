#!/usr/bin/env python3
"""Summarize the frozen automatic funnel and enforce the 300-event floor."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
SOURCES = {
    "queue": BASE / "RXR_TRAIN_UNBIASED_EXPANSION_QUEUE.json",
    "hindsight": BASE / "hindsight_factory/RXR_HINDSIGHT_EVENT_CANDIDATES.json",
    "multiview": BASE / "multiview_factory/RXR_PRIMARY_MULTIVIEW_INPUTS.json",
    "prescreen": BASE / "branch_factory/RXR_MULTIVIEW_MACHINE_PRESCREEN.json",
    "geometry": BASE / "geometry/RXR_EXPANSION_DIRECTED_GEOMETRY.json",
    "controller": BASE / "geometry/RXR_EXPANSION_CONTROLLER_EXECUTION.json",
    "causal_analysis": BASE / "causal_frontend/RXR_EXPANSION_CAUSAL_CANDIDATE_ANALYSIS.json",
    "causal_language": BASE / "causal_frontend/RXR_EXPANSION_CAUSAL_PREFIX_LANGUAGE_GATE.json",
}
OUT = BASE / "RXR_EXPANSION_AUTOMATIC_FILTER_ACCEPTANCE.json"
EVENT_FLOOR = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str):
    path = SOURCES[name]
    if not path.is_file() or path.is_symlink():
        raise SystemExit("missing automatic-filter source: " + name)
    return json.loads(path.read_text())


def main() -> int:
    queue = load("queue")
    hindsight = load("hindsight")
    multiview = load("multiview")
    prescreen = load("prescreen")
    geometry = load("geometry")
    controller = load("controller")
    analysis = load("causal_analysis")
    language = load("causal_language")
    eligible = [row for row in language["events"] if row["status"] ==
                "CAUSAL_LANGUAGE_K3_PASS_CONTROLS_REQUIRED"]
    analysis_by_id = {row["event_id"]: row for row in analysis["events"]}
    scenes = {analysis_by_id[row["event_id"]]["scene_id"] for row in eligible}
    episodes = {analysis_by_id[row["event_id"]]["episode_id"] for row in eligible}
    passed = len(eligible) >= EVENT_FLOOR
    output = {
        "manifest": "RevealNav RxR expansion automatic filter acceptance",
        "revision": "rxr-expansion-automatic-filter-acceptance/1",
        "status": "PASS_READY_FOR_300_HUMAN_PILOT" if passed else
                  "FAIL_BELOW_300_EVENT_FLOOR",
        "source_manifest": {name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        } for name, path in SOURCES.items()},
        "funnel": {
            "unbiased_trajectories": queue["candidate_count"],
            "hindsight_raw_candidates": hindsight["raw_candidate_count"],
            "primary_candidates": multiview["selected_primary_count"],
            "multiview_rendered": multiview["event_count"],
            "machine_prescreen_dispositions": prescreen["disposition_counts"],
            "geometry_statuses": geometry["status_counts"],
            "controller_statuses": controller["status_counts"],
            "frontend_statuses": analysis["status_counts"],
            "language_counts": language["counts"],
            "strict_causal_event_count": len(eligible),
            "strict_causal_episode_count": len(episodes),
            "strict_causal_scene_count": len(scenes),
        },
        "event_floor": EVENT_FLOOR,
        "event_floor_pass": passed,
        "eligible_event_ids": [row["event_id"] for row in eligible],
        "selection_policy": {
            "mllm_branch_selection": "first provider response only",
            "semantic_invalid_response": "fail closed",
            "provider_retries_used_for_branch_selection": 0,
            "replacement_samples_created": 0,
            "future_frames_used_online": 0,
        },
        "human_labels_created": 0,
        "training_authorized": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_name(OUT.name + ".part")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({
        "status": output["status"],
        "strict_causal_event_count": len(eligible),
        "strict_causal_scene_count": len(scenes),
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
