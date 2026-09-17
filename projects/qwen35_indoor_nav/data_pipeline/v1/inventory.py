"""CPU-only source inventory. Produces source candidates, NOT training labels."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
LINE = OUT.parents[1]
ROOT = LINE.parents[1]


def guarded(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"OUTSIDE_PROJECT: {path}")
    return resolved


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def main():
    source = guarded(ROOT / "data/phase0/raw/r2r_vlnce_v1-3/train/train.json.gz")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = "f411066b53f96d1241c045fd05a6a9e01b484c2ed9369f5b6f41806969056a34"
    if source_hash != expected:
        raise ValueError("SOURCE_CHANGED_SINCE_P2")
    with gzip.open(source, "rt") as handle:
        episodes = json.load(handle)["episodes"]
    scenes = collections.Counter()
    trajectories = set()
    physical_routes = set()
    ids = set()
    rows = []
    for episode in episodes:
        eid = str(episode["episode_id"])
        if eid in ids:
            raise ValueError("DUPLICATE_EPISODE_ID")
        ids.add(eid)
        relative_scene = Path(episode["scene_id"])
        scene = relative_scene.parent.name
        if relative_scene.parts != ("mp3d", scene, f"{scene}.glb"):
            raise ValueError("UNEXPECTED_SCENE_PATH")
        assert episode["reference_path"] and episode["instruction"]["instruction_text"].strip()
        physical = {key: episode[key] for key in
                    ("scene_id", "start_position", "start_rotation", "reference_path", "goals")}
        route_hash = hashlib.sha256(canonical(physical)).hexdigest()
        scenes[scene] += 1
        trajectories.add((scene, str(episode["trajectory_id"])))
        physical_routes.add(route_hash)
        rows.append({
            "record_type": "SOURCE_CANDIDATE_NOT_TRAINING_SAMPLE",
            "source_split": "train", "source_sha256": source_hash,
            "source_relative_path": str(source.relative_to(ROOT)),
            "episode_id": eid, "trajectory_id": str(episode["trajectory_id"]),
            "scene_id": scene, "physical_source_route_sha256": route_hash,
            "reference_waypoints": len(episode["reference_path"]),
            "instruction": episode["instruction"]["instruction_text"],
            "split_group": scene, "q35n_split": "UNASSIGNED_PENDING_EXPOSURE_AUDIT",
            "legacy_exposure": "UNKNOWN_NOT_CLAIMED_UNSEEN",
            "runtime_replay_pass": None, "action_labels_exported": False,
        })
    assets = []
    asset_root = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
    for scene, count in sorted(scenes.items()):
        files = []
        for suffix in (".glb", ".navmesh", ".house", "_semantic.ply"):
            path = guarded(asset_root / scene / f"{scene}{suffix}")
            files.append({"path": str(path.relative_to(ROOT)), "present": path.is_file(),
                          "bytes": path.stat().st_size if path.is_file() else None,
                          "content_hash_verified_this_inventory": False})
        assets.append({"scene_id": scene, "source_episodes": count, "files": files,
                       "all_four_files_present": all(f["present"] for f in files),
                       "runtime_accepted": scene == "17DRP5sb8fy",
                       "legacy_exposure": "KNOWN_EXPOSED" if scene == "17DRP5sb8fy" else "UNKNOWN"})
    report = {
        "node": "Q35N_DATA_PIPELINE_SOURCE_INVENTORY_V1", "date": "2026-09-09",
        "mode": "CPU_ONLY_SOURCE_MANIFEST_GENERATION",
        "source_sha256": source_hash, "source_hash_matches_p2": True,
        "source_episodes": len(episodes), "source_scenes": len(scenes),
        "distinct_scene_trajectory_ids": len(trajectories),
        "distinct_physical_source_routes": len(physical_routes),
        "scenes_with_four_files_present": sum(a["all_four_files_present"] for a in assets),
        "source_instruction_records": len(rows),
        "new_rgb_frames": 0, "new_runtime_replays": 0,
        "certified_mechanism_families": 0, "exported_training_labels": 0,
        "training_allowed": False, "scientific_pass": False,
        "asset_presence_is_not_runtime_or_semantic_acceptance": True,
        "source_episode_is_not_independent_physical_route": True,
        "official_validation_or_test_content_read": False,
    }
    products = {
        "SOURCE_INVENTORY.json": json.dumps(report, indent=2) + "\n",
        "SCENE_ASSET_PRESENCE.json": json.dumps(assets, indent=2) + "\n",
        "SOURCE_CANDIDATES.jsonl": "".join(canonical(row).decode() + "\n" for row in
            sorted(rows, key=lambda r: (r["scene_id"], int(r["episode_id"])))),
    }
    # Never overwrite a differing previous inventory or lose its provenance.
    for name, content in products.items():
        path = OUT / name
        if path.exists() and path.read_text() != content:
            raise ValueError(f"VERSION_REQUIRED: {name}")
    for name, content in products.items():
        path = OUT / name
        if not path.exists():
            with path.open("x") as handle:
                handle.write(content)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
