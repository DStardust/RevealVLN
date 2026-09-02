#!/usr/bin/env python3
"""Discover MF3ZV lexical candidates without reading outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from revealnav_mf3.progress_language_filter import (
    earliest_atom,
    load_train_instructions,
    propose_progress_atoms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2r", type=Path, required=True)
    parser.add_argument("--rxr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = []
    total = {}
    excluded_languages = {"RxR": 0}
    family_episode_counts = Counter()
    family_scene_sets: dict[str, set[str]] = {"ORDINAL": set(), "PASSED_LANDMARK": set()}
    for dataset, path in (("R2R", args.r2r), ("RxR", args.rxr)):
        rows = list(load_train_instructions(path, dataset))
        total[dataset] = len(rows)
        for row in rows:
            proposals = propose_progress_atoms(row)
            families = {item.atom.family for item in proposals}
            for family in families:
                family_episode_counts[(dataset, family)] += 1
                family_scene_sets[family].add(row.scene_id)
            first = earliest_atom(row)
            if first is not None:
                records.append(first.to_dict())
    payload = {
        "schema_version": "revealnav-mf3zv-language-discovery/1",
        "revision": "mf3zv_minimal_progress_support_v1",
        "status": "LANGUAGE_DISCOVERY_COMPLETE",
        "source_split": "train-development",
        "outcome_payload_read": False,
        "public_split_access": {
            "val_seen": False,
            "val_unseen": False,
            "test": False,
            "test_challenge": False,
        },
        "english_instruction_counts": total,
        "candidate_instruction_count": len(records),
        "family_episode_counts": {
            dataset: {
                family: family_episode_counts[(dataset, family)]
                for family in ("ORDINAL", "PASSED_LANDMARK")
            }
            for dataset in ("R2R", "RxR")
        },
        "family_scene_counts": {
            family: len(scenes) for family, scenes in family_scene_sets.items()
        },
        "candidates": sorted(
            records,
            key=lambda row: (row["dataset"], row["scene_id"], row["episode_id"]),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

