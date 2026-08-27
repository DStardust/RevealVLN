#!/usr/bin/env python3
"""Screen official train/val-seen VLN-CE annotations for Reveal candidates."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from toporeveal.provenance import canonical_phase0_asset
from toporeveal.screening import (
    iter_vlnce_episodes,
    pilot_sample,
    screen_vlnce,
    screening_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_json_gz")
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="language tag to retain; repeat for multiple tags",
    )
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.sample_count < 0:
        parser.error("--sample-count must be non-negative")
    project_root = Path(__file__).resolve().parents[1]
    try:
        asset = canonical_phase0_asset(Path(args.annotation_json_gz), project_root)
    except ValueError as error:
        parser.error(str(error))
    if args.output is not None:
        resolved_output = args.output.resolve()
        if not resolved_output.is_relative_to(project_root):
            parser.error("--output must stay inside the project")
    requested_languages = (
        set(args.languages)
        if args.languages
        else ({"en-US", "en-IN"} if asset.dataset == "rxr-ce" else {"en"})
    )
    try:
        candidates = list(
            screen_vlnce(
                iter_vlnce_episodes(asset.path),
                dataset=asset.dataset,
                split=asset.split,
                languages=requested_languages,
            )
        )
    except (OSError, UnicodeError, gzip.BadGzipFile, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        parser.error(f"invalid annotation payload: {error}")
    output = screening_summary(candidates)
    output["source"] = {
        "dataset": asset.dataset,
        "split": asset.split,
        "role": asset.role,
        "path": str(asset.path.relative_to(project_root)),
        "bytes": asset.byte_count,
        "sha256": asset.sha256,
        "manifest_sha256": asset.manifest_sha256,
        "language_filter": sorted(requested_languages),
    }
    output["sampling"] = {
        "requested": args.sample_count,
        "seed": args.sample_seed,
        "actual": 0,
        "design": "uniform_unique_trajectory_v1" if args.sample_count else "none",
    }
    if args.sample_count:
        samples = [
            {
                "dataset": candidate.dataset,
                "split": candidate.split,
                "episode_id": candidate.episode_id,
                "instruction_id": candidate.instruction_id,
                "trajectory_id": candidate.trajectory_id,
                "scene_id": candidate.scene_id,
                "language": candidate.language,
                "triggers": candidate.triggers,
                "instruction": candidate.instruction,
            }
            for candidate in pilot_sample(
                candidates, args.sample_count, seed=args.sample_seed
            )
        ]
        output["samples"] = samples
        output["sampling"]["actual"] = len(samples)
    serialized = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
