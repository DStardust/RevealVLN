#!/usr/bin/env python3
"""Print compact live progress for the secondary train-data pipeline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/secondary_expansion_v1"
)
TRAINING_STATUS = ROOT / (
    "artifacts/evaluation/mf2_secondary_augmentation_v1/"
    "RXR_SECONDARY_TRAINING_STATUS.json"
)
COMPARISON = ROOT / (
    "artifacts/evaluation/mf2_secondary_augmentation_v1/"
    "RXR_SECONDARY_AUGMENTATION_COMPARISON_V1.json"
)


def load(relative: str):
    path = BASE / relative
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> int:
    output = {"selected_secondary_events": 903}
    status = load("RXR_SECONDARY_AUTOMATIC_PIPELINE_STATUS.json")
    if status:
        output["supervisor"] = status
    multiview = load("multiview_factory/RXR_SECONDARY_MULTIVIEW_INPUTS.json")
    if multiview:
        output["multiview"] = {
            key: multiview[key]
            for key in ("status", "event_count", "failure_count", "media_file_count")
        }
    language_results = list(
        (BASE / "multibranch/prefix_language_results").glob("*/*.json")
    )
    if language_results:
        output["causal_language_live"] = {
            "prefix_responses": len(language_results),
            "events_with_responses": len({path.parent.name for path in language_results}),
            "expected_events": 206,
        }
    results = list((BASE / "branch_factory/results").glob("*/attempt_001.json"))
    if results:
        counts = Counter()
        for path in results:
            counts[json.loads(path.read_text()).get("status", "UNKNOWN")] += 1
        output["branch_first_response"] = {
            "completed": len(results),
            "expected": multiview["event_count"] if multiview else None,
            "status_counts": dict(sorted(counts.items())),
        }
    for key, relative in (
        ("branch_prescreen", "branch_factory/RXR_SECONDARY_MACHINE_PRESCREEN.json"),
        ("geometry", "multibranch/RXR_SECONDARY_MULTIBRANCH_GEOMETRY.json"),
        ("controller", "multibranch/RXR_SECONDARY_MULTIBRANCH_CONTROLLER.json"),
        ("causal_analysis", "multibranch/RXR_SECONDARY_CAUSAL_CANDIDATE_ANALYSIS.json"),
        ("causal_language", "multibranch/RXR_SECONDARY_CAUSAL_PREFIX_LANGUAGE_GATE.json"),
        ("training_index", "multibranch/RXR_SECONDARY_TRAINING_INDEX.json"),
        ("resource_labels", "multibranch/RXR_SECONDARY_TX_GATE.json"),
        ("features", "multibranch/RXR_SECONDARY_FEATURE_GATE.json"),
    ):
        document = load(relative)
        if document:
            output[key] = {
                name: document[name]
                for name in ("status", "counts", "status_counts", "disposition_counts")
                if name in document
            }
    if TRAINING_STATUS.is_file():
        output["training_supervisor"] = json.loads(TRAINING_STATUS.read_text())
    if COMPARISON.is_file():
        comparison = json.loads(COMPARISON.read_text())
        output["development_ablation"] = {
            "status": comparison.get("status"),
            "predeclared_signal_criteria": comparison.get(
                "predeclared_signal_criteria"
            ),
        }
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
