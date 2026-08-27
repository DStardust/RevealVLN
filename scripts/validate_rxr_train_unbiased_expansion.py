#!/usr/bin/env python3
"""Validate the frozen RxR-train expansion without creating event labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
QUEUE = ROOT / (
    "artifacts/phase1/rxr_train_expansion/"
    "RXR_TRAIN_UNBIASED_EXPANSION_QUEUE.json")
ORIGINAL = ROOT / "artifacts/phase0/rxr_train_screening_seed20260822.json"
AUTH = ROOT / (
    "artifacts/upstream/matterport3d/"
    "MP3D_ACCESS_AUTHORIZATION_ATTESTATION.json")
OUT = QUEUE.with_name("RXR_TRAIN_UNBIASED_EXPANSION_ACCEPTANCE.json")
EXPECTED = {
    QUEUE: "7b3578afae71dc35327c9ad31b4a97df1a3ccd4960109a2e1fd78f4fa4facbab",
    ORIGINAL:
        "9571a8a03489abe0998e69f7179e58a029b3210aa3dec396c7d14121d261a73a",
    AUTH: "d840d2edde2049c1dccdf3c4bc696deed4bd79354cf49931087086a03900fcad",
    ROOT / "FROZEN_SPEC.md":
        "cff97bac8741a94f41827fbceb6a7947d2ff7508fe5e8cda6f6b6268350b3d81",
    ROOT / "PHASE0_PROTOCOL.md":
        "7fb096b0e39a19dfc92c47b25270c670403d02d36edac7816d5c1b4c2601f96d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    checks = []

    def check(condition: bool, name: str) -> None:
        if not condition:
            raise SystemExit("expansion acceptance failure: " + name)
        checks.append(name)

    for path, expected in EXPECTED.items():
        check(path.is_file() and not path.is_symlink(),
              "safe source: " + str(path.relative_to(ROOT)))
        check(sha256_file(path) == expected,
              "pinned source: " + str(path.relative_to(ROOT)))
    queue = json.loads(QUEUE.read_text())
    rows = queue["candidates"]
    design = queue["sampling_design"]
    basis = queue["sample_size_basis"]
    check(queue["status"] == "FROZEN_READY_FOR_HINDSIGHT_EVENT_FACTORY",
          "queue status")
    check(len(rows) == queue["candidate_count"] == 2303,
          "2303 candidate closure")
    check([row["expansion_order"] for row in rows] == list(range(2303)),
          "contiguous frozen processing order")
    check(len({row["episode_id"] for row in rows}) == 2303,
          "unique episodes")
    keys = {(row["scene_id"], row["trajectory_id"]) for row in rows}
    check(len(keys) == 2303, "unique trajectories")
    original = json.loads(ORIGINAL.read_text())
    original_keys = {(row["scene_id"], str(row["trajectory_id"]))
                     for row in original["samples"]}
    check(not keys & original_keys, "no overlap with frozen queue50")
    check(all(row["split"] == "train" for row in rows),
          "RxR train only")
    check({row["language"] for row in rows} <= {"en-US", "en-IN"},
          "English language scope")
    check(all(row["processing_status"] ==
              "PENDING_HINDSIGHT_EVENT_FACTORY" for row in rows),
          "no post-selection outcome written")
    check(all(set(row) == {
        "expansion_order", "episode_id", "instruction_id", "trajectory_id",
        "scene_id", "language", "split", "triggers", "instruction_text",
        "instruction_sha256", "reference_path_points", "selection_stratum",
        "selection_probability", "analysis_weight", "processing_status"}
              for row in rows), "candidate schema contains no label field")
    probability = 2303 / 6169
    weight = 6169 / 2303
    check(all(math.isclose(row["selection_probability"], probability,
                           rel_tol=0.0, abs_tol=1e-15)
              and math.isclose(row["analysis_weight"], weight,
                               rel_tol=0.0, abs_tol=1e-15)
              for row in rows), "uniform remaining-stratum weights")
    check(design["name"] ==
          "stratified_queue50_census_plus_remaining_srswor/1"
          and design["seed"] == 20260825
          and design["outcome_fields_used_for_item_selection"] == []
          and design["post_selection_resampling_for_failures_forbidden"] is
          True, "outcome-blind fixed sampling design")
    check(design["population_unique_candidate_trajectories"] == 6219
          and design["frozen_queue50_census_stratum"] == 50
          and design["remaining_population"] == 6169
          and design["remaining_sample"] == 2303,
          "population and stratum closure")
    check(basis["strict_events"] == 11
          and basis["reviewed_candidates"] == 50
          and basis["required_combined_candidates"] == 2353
          and basis["new_candidates"] == 2303
          and basis["lower_bound_projected_events"] >= 300.0,
          "conservative 300-event sample-size basis")
    identity = [{key: row[key] for key in (
        "expansion_order", "episode_id", "instruction_id", "trajectory_id",
        "scene_id", "language", "instruction_sha256")} for row in rows]
    check(stable_sha(identity) == queue["selection_commitment_sha256"] ==
          "f2e7ce5aa7bde1ebb3af6d113bf4970d25a9c5afea1d3e43a86516163305ea3d",
          "selection identity commitment")
    mp3d = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
    check(all((mp3d / scene / (scene + ".glb")).is_file()
              and not (mp3d / scene / (scene + ".glb")).is_symlink()
              and (mp3d / scene / (scene + ".navmesh")).is_file()
              and not (mp3d / scene / (scene + ".navmesh")).is_symlink()
              for scene in {row["scene_id"] for row in rows}),
          "all selected scenes close to project-local MP3D")
    check(json.loads(AUTH.read_text())["status"] ==
          "USER_CONFIRMED_AUTHORIZED", "MP3D authorization")
    check(queue["network_calls_made"] == 0
          and queue["forbidden_split_payloads_opened"] == 0
          and queue["new_labels_created"] == 0
          and queue["feature_generation_authorized"] is False
          and queue["training_authorized"] is False,
          "execution boundary")
    reserves = sorted((ROOT / ".disk_reserve").glob("reserve_10G_*.bin"))
    check(len(reserves) == 19 and all(
        path.is_file() and not path.is_symlink()
        and path.stat().st_size == 10_737_418_240 for path in reserves),
        "19 reserve files untouched")

    output = {
        "manifest": "RevealNav unbiased RxR-train expansion acceptance",
        "revision": "rxr-train-unbiased-expansion-acceptance/1",
        "status": "PASS",
        "sources": {str(path.relative_to(ROOT)): expected
                    for path, expected in EXPECTED.items()},
        "checks_passed": len(checks),
        "candidate_count": len(rows),
        "selection_commitment_sha256": queue[
            "selection_commitment_sha256"],
        "estimated_lower_bound_event_coverage": basis[
            "lower_bound_projected_events"],
        "next_gate": (
            "streaming hindsight event factory and sealed automatic gates; "
            "no failed item may be replaced"),
        "feature_generation_authorized": False,
        "training_authorized": False,
    }
    temporary = OUT.with_name(OUT.name + ".part")
    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({
        "status": output["status"],
        "checks_passed": output["checks_passed"],
        "candidate_count": output["candidate_count"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
