#!/usr/bin/env python3
"""Freeze the unprocessed RxR train/development route census for scale-v2."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from toporeveal.screening import iter_vlnce_episodes, screen_vlnce


ROOT = Path("/mnt/daiyang/vla")
CANONICAL = ROOT / "data/phase0/raw/rxr_vlnce_v0/train/train_guide.json.gz"
RUNTIME = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
QUEUE50 = ROOT / "artifacts/phase0/rxr_train_screening_seed20260822.json"
QUEUE2303 = ROOT / (
    "artifacts/phase1/rxr_train_expansion/"
    "RXR_TRAIN_UNBIASED_EXPANSION_QUEUE.json"
)
SCALE_V1 = ROOT / (
    "artifacts/phase1/rxr_train_expansion/scale_v1/"
    "RXR_SCALE_V1_SELECTION.json"
)
MP3D_MANIFEST = ROOT / (
    "artifacts/upstream/matterport3d/MP3D_90_SCENE_MANIFEST.json"
)
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
OUT = ROOT / (
    "artifacts/phase1/rxr_train_expansion/scale_v2/"
    "RXR_SCALE_V2_ROUTE_CENSUS.json"
)
EXPECTED = {
    CANONICAL: "fe127cfe9350123e7ff511c858f866b2d538564bac910da698bfff4ee46be07e",
    RUNTIME: "f06b2ef4dc947ca15d6c4a5a3d629c9212328f4cbdd38a13bed9c5c1fc224a94",
    QUEUE50: "9571a8a03489abe0998e69f7179e58a029b3210aa3dec396c7d14121d261a73a",
    QUEUE2303: "7b3578afae71dc35327c9ad31b4a97df1a3ccd4960109a2e1fd78f4fa4facbab",
    SCALE_V1: "2845a6f0ae5de87fb939e5b397a93354be6bc35aae6060c6896fa9bd3cbf4ec2",
    MP3D_MANIFEST: "f89f8693d1ac06dbc5b17406136c3418e3a00b56c596d7b2bd6759545a876ed9",
}
LANGUAGES = {"en-US", "en-IN"}
ORDER_SEED = "revealnav-rxr-scale-v2-route-census/1"
ORDER_OFFSET = 10_000
EXPECTED_REMAINING = 2_971


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def rank(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def main() -> int:
    for path, expected in EXPECTED.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise RuntimeError("scale-v2 source drift: " + str(path))

    queue50 = json.loads(QUEUE50.read_text())
    queue2303 = json.loads(QUEUE2303.read_text())
    scale_v1 = json.loads(SCALE_V1.read_text())
    if scale_v1.get("status") != "SCALE_V1_SELECTION_FROZEN":
        raise RuntimeError("scale-v1 selection is not frozen")

    scene_split = {}
    for row in scale_v1["automatic"]:
        scene = row["scene_id"]
        split = row["scene_split"]
        if split not in {"train", "development"}:
            raise RuntimeError("unexpected automatic scene split")
        if scene in scene_split and scene_split[scene] != split:
            raise RuntimeError("scene split drift")
        scene_split[scene] = split
    gold_scenes = {row["scene_id"] for row in scale_v1["new_gold"]}
    if set(scene_split) & gold_scenes:
        raise RuntimeError("automatic and Gold scenes overlap")

    screened = list(
        screen_vlnce(
            iter_vlnce_episodes(CANONICAL),
            dataset="rxr-ce",
            split="train",
            languages=LANGUAGES,
        )
    )
    groups = {}
    for candidate in screened:
        groups.setdefault((candidate.scene_id, candidate.trajectory_id), []).append(
            candidate
        )
    for candidates in groups.values():
        candidates.sort(
            key=lambda row: (
                row.instruction_id,
                row.episode_id,
                row.language,
                row.instruction,
            )
        )
    if len(screened) != 11_487 or len(groups) != 6_219:
        raise RuntimeError("screened population drift")

    prior = {
        (row["scene_id"], str(row["trajectory_id"]))
        for row in queue50["samples"]
    }
    prior.update(
        (row["scene_id"], str(row["trajectory_id"]))
        for row in queue2303["candidates"]
    )
    if len(prior) != 2_353 or not prior <= set(groups):
        raise RuntimeError("prior route closure failure")

    remaining = [
        key for key in set(groups) - prior if key[0] in scene_split
    ]
    remaining.sort(key=lambda key: rank(ORDER_SEED, key[0], key[1]))
    if len(remaining) != EXPECTED_REMAINING:
        raise RuntimeError(
            f"expected {EXPECTED_REMAINING} remaining automatic routes, got {len(remaining)}"
        )

    with gzip.open(RUNTIME, "rt", encoding="utf-8") as stream:
        runtime = {
            str(row["episode_id"]): row for row in json.load(stream)["episodes"]
        }

    records = []
    for local_order, key in enumerate(remaining):
        candidates = groups[key]
        candidate = min(
            candidates,
            key=lambda row: rank(
                ORDER_SEED,
                "instruction",
                key[0],
                key[1],
                row.episode_id,
                row.instruction_id,
            ),
        )
        episode = runtime.get(candidate.episode_id)
        if episode is None:
            raise RuntimeError("runtime episode missing: " + candidate.episode_id)
        instruction = episode["instruction"]
        scene = Path(episode["scene_id"]).stem
        if not all(
            (
                str(episode["trajectory_id"]) == candidate.trajectory_id,
                str(instruction["instruction_id"]) == candidate.instruction_id,
                instruction["language"] == candidate.language,
                instruction["instruction_text"] == candidate.instruction,
                scene == candidate.scene_id,
            )
        ):
            raise RuntimeError("canonical/runtime identity mismatch")
        for suffix in (".glb", ".navmesh"):
            asset = MP3D / scene / (scene + suffix)
            if not asset.is_file() or asset.is_symlink():
                raise RuntimeError("scene asset missing: " + str(asset))
        records.append(
            {
                "scale_v2_order": local_order,
                "expansion_order": ORDER_OFFSET + local_order,
                "episode_id": candidate.episode_id,
                "instruction_id": candidate.instruction_id,
                "trajectory_id": candidate.trajectory_id,
                "scene_id": scene,
                "scene_split": scene_split[scene],
                "language": candidate.language,
                "triggers": list(candidate.triggers),
                "instruction_text": candidate.instruction,
                "instruction_sha256": hashlib.sha256(
                    candidate.instruction.encode()
                ).hexdigest(),
                "reference_path_points": len(episode.get("reference_path") or []),
                "selection_stratum": "remaining_unprocessed_train_development_route_census",
                "selection_probability": 1.0,
                "processing_status": "PENDING_HINDSIGHT_EVENT_FACTORY",
            }
        )

    commitment = [
        {
            key: row[key]
            for key in (
                "scale_v2_order",
                "expansion_order",
                "episode_id",
                "instruction_id",
                "trajectory_id",
                "scene_id",
                "scene_split",
                "instruction_sha256",
            )
        }
        for row in records
    ]
    output = {
        "schema_version": "revealnav-rxr-scale-v2-route-census/1",
        "status": "SCALE_V2_ROUTE_CENSUS_FROZEN",
        "scope": "all previously unprocessed RxR routes in frozen train/development scenes",
        "sources": {
            str(path.relative_to(ROOT)): expected for path, expected in EXPECTED.items()
        },
        "selection": {
            "design": "route census after frozen queue50 and queue2303",
            "selection_probability": 1.0,
            "one_instruction_hash_selected_per_route": True,
            "route_order_seed": ORDER_SEED,
            "route_order_committed_before_scale_v2_outcomes": True,
            "outcome_fields_used": [],
            "post_failure_replacement_forbidden": True,
        },
        "counts": {
            "screened_candidates": len(screened),
            "screened_unique_routes": len(groups),
            "prior_routes_excluded": len(prior),
            "remaining_train_development_routes": len(records),
            "train": sum(row["scene_split"] == "train" for row in records),
            "development": sum(
                row["scene_split"] == "development" for row in records
            ),
            "scenes": len({row["scene_id"] for row in records}),
        },
        "language_counts": dict(sorted(Counter(row["language"] for row in records).items())),
        "selection_commitment_sha256": stable_sha(commitment),
        "candidates": records,
        "old_gold_payload_read": False,
        "human_labels_created": 0,
        "training_authorized": False,
        "paper_result": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    part = OUT.with_name(OUT.name + ".part")
    part.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, OUT)
    print(
        json.dumps(
            {
                "status": output["status"],
                "counts": output["counts"],
                "selection_commitment_sha256": output[
                    "selection_commitment_sha256"
                ],
                "output": str(OUT.relative_to(ROOT)),
                "sha256": sha256_file(OUT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
