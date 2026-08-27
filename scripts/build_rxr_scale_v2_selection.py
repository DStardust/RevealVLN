#!/usr/bin/env python3
"""Freeze every eligible scale-v2 hindsight candidate for automatic closure."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
BASE = ROOT / "artifacts/phase1/rxr_train_expansion/scale_v2"
QUEUE = BASE / "RXR_SCALE_V2_ROUTE_CENSUS.json"
HINDSIGHT = BASE / "hindsight_factory/RXR_SCALE_V2_HINDSIGHT_EVENT_CANDIDATES.json"
OUT = BASE / "RXR_SCALE_V2_SELECTION.json"
EXPECTED_QUEUE = "3a5e1d03620b1e993a1039c95d55ba423338490ca36fc1f681586e38d43fd6b6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not QUEUE.is_file() or QUEUE.is_symlink() or sha256_file(QUEUE) != EXPECTED_QUEUE:
        raise RuntimeError("scale-v2 route census drift")
    queue = json.loads(QUEUE.read_text())
    hindsight = json.loads(HINDSIGHT.read_text())
    if not (
        queue.get("status") == "SCALE_V2_ROUTE_CENSUS_FROZEN"
        and hindsight.get("status") == "PASS_PENDING_MULTIVIEW_AND_3D_GATES"
        and hindsight.get("route_count") == len(queue["candidates"])
    ):
        raise RuntimeError("scale-v2 hindsight precondition failed")
    routes = {row["expansion_order"]: row for row in queue["candidates"]}
    eligible = [
        row
        for row in hindsight["candidates"]
        if row["candidate_kind"] != "LIKELY_NO_CHOICE_HARD_NEGATIVE"
        and not row["conflicting_kind_votes"]
    ]
    eligible.sort(
        key=lambda row: (
            row["scale_v2_order"],
            int(row["hindsight_candidate_id"].rsplit("hv", 1)[1]),
        )
    )
    items = []
    for scale_order, candidate in enumerate(eligible):
        route = routes[candidate["expansion_order"]]
        if not (
            candidate["episode_id"] == route["episode_id"]
            and candidate["scene_id"] == route["scene_id"]
            and candidate["scene_split"] == route["scene_split"]
            and route["scene_split"] in {"train", "development"}
        ):
            raise RuntimeError("scale-v2 candidate route drift")
        items.append(
            {
                "scale_order": scale_order,
                "scale_v2_order": route["scale_v2_order"],
                "event_id": candidate["hindsight_candidate_id"],
                "episode_id": candidate["episode_id"],
                "trajectory_id": candidate["trajectory_id"],
                "instruction_id": candidate["instruction_id"],
                "expansion_order": candidate["expansion_order"],
                "scene_id": candidate["scene_id"],
                "scene_split": candidate["scene_split"],
                "lane": "automatic",
                "candidate_kind": candidate["candidate_kind"],
                "candidate_interval": candidate["interval"],
                "source": candidate["source"],
                "processing_status": "PENDING_SCALE_MULTIVIEW",
            }
        )
    if not items:
        raise RuntimeError("scale-v2 produced no eligible candidates")
    output = {
        "schema_version": "revealnav-rxr-scale-v2-selection/1",
        "status": "SCALE_V2_SELECTION_FROZEN",
        "scope": "all eligible candidates from the remaining train/development route census",
        "sources": {
            str(QUEUE.relative_to(ROOT)): EXPECTED_QUEUE,
            str(HINDSIGHT.relative_to(ROOT)): sha256_file(HINDSIGHT),
        },
        "selection_rule": {
            "route_census": True,
            "eligible_hindsight_candidates_only": True,
            "candidate_subsampling": False,
            "post_failure_replacement": False,
            "outcome_fields_used": [],
        },
        "counts": {
            "routes": len(queue["candidates"]),
            "automatic_candidates": len(items),
            "train": sum(row["scene_split"] == "train" for row in items),
            "development": sum(
                row["scene_split"] == "development" for row in items
            ),
        },
        "automatic": items,
        "old_gold_payload_read": False,
        "human_labels_created": 0,
        "training_authorized": False,
        "paper_result": False,
    }
    part = OUT.with_name(OUT.name + ".part")
    part.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(part, OUT)
    print(
        json.dumps(
            {
                "status": output["status"],
                "counts": output["counts"],
                "output": str(OUT.relative_to(ROOT)),
                "sha256": sha256_file(OUT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
