#!/usr/bin/env python3
"""Freeze an outcome-blind RxR-train expansion queue for the 300-event pilot."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from random import Random

from toporeveal.screening import iter_vlnce_episodes, screen_vlnce


ROOT = Path("/mnt/daiyang/vla")
CANONICAL = ROOT / "data/phase0/raw/rxr_vlnce_v0/train/train_guide.json.gz"
RUNTIME = ROOT / (
    "third_party/ETP-R1/data/datasets/"
    "RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz")
ORIGINAL = ROOT / "artifacts/phase0/rxr_train_screening_seed20260822.json"
ADJUDICATION = ROOT / (
    "artifacts/phase0/phase0c_cr5_queue50/tx_gate/"
    "CR5_QUEUE50_TX_SCIENTIFIC_ADJUDICATION.json")
MP3D_MANIFEST = ROOT / (
    "artifacts/upstream/matterport3d/MP3D_90_SCENE_MANIFEST.json")
MP3D = ROOT / "third_party/ETP-R1/data/scene_datasets/mp3d"
OUT_DIR = ROOT / "artifacts/phase1/rxr_train_expansion"
OUT = OUT_DIR / "RXR_TRAIN_UNBIASED_EXPANSION_QUEUE.json"
EXPECTED = {
    CANONICAL:
        "fe127cfe9350123e7ff511c858f866b2d538564bac910da698bfff4ee46be07e",
    RUNTIME:
        "f06b2ef4dc947ca15d6c4a5a3d629c9212328f4cbdd38a13bed9c5c1fc224a94",
    ORIGINAL:
        "9571a8a03489abe0998e69f7179e58a029b3210aa3dec396c7d14121d261a73a",
    ADJUDICATION:
        "c25b59861a9f3ffa0911dc28b302e246b82be39ee2b0483488c6df3016337396",
    MP3D_MANIFEST:
        "f89f8693d1ac06dbc5b17406136c3418e3a00b56c596d7b2bd6759545a876ed9",
}
LANGUAGES = {"en-US", "en-IN"}
SEED = 20260825
PILOT_TARGET = 300
OBSERVED_SUCCESSES = 11
OBSERVED_TOTAL = 50
WILSON_Z = 1.959963984540054


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def wilson_lower(successes: int, total: int) -> float:
    rate = successes / total
    denominator = 1.0 + WILSON_Z * WILSON_Z / total
    center = (rate + WILSON_Z * WILSON_Z / (2.0 * total)) / denominator
    margin = WILSON_Z * math.sqrt(
        rate * (1.0 - rate) / total
        + WILSON_Z * WILSON_Z / (4.0 * total * total)) / denominator
    return center - margin


def stable_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    for path, expected in EXPECTED.items():
        if (not path.is_file() or path.is_symlink()
                or sha256_file(path) != expected):
            raise SystemExit("pinned expansion source drift: " + str(path))
    adjudication = json.loads(ADJUDICATION.read_text())
    if (adjudication["positive_findings"]["tx_admitted_events"] !=
            OBSERVED_SUCCESSES
            or adjudication["risk_findings"][
                "end_to_end_queue50_tx_yield"] !=
            OBSERVED_SUCCESSES / OBSERVED_TOTAL
            or adjudication["automated_event_expansion_authorized"] is not
            True):
        raise SystemExit("expansion authorization drift")

    lower = wilson_lower(OBSERVED_SUCCESSES, OBSERVED_TOTAL)
    total_required = math.ceil(PILOT_TARGET / lower)
    expansion_count = total_required - OBSERVED_TOTAL
    if total_required != 2353 or expansion_count != 2303:
        raise SystemExit("sample-size derivation drift")

    candidates = list(screen_vlnce(
        iter_vlnce_episodes(CANONICAL), dataset="rxr-ce", split="train",
        languages=LANGUAGES))
    groups = {}
    for candidate in candidates:
        groups.setdefault((candidate.scene_id, candidate.trajectory_id), []).append(
            candidate)
    for key in groups:
        groups[key].sort(key=lambda row: (
            row.instruction_id, row.episode_id, row.language,
            row.triggers, row.instruction))
    if len(candidates) != 11487 or len(groups) != 6219:
        raise SystemExit("screened candidate population drift")

    original_doc = json.loads(ORIGINAL.read_text())
    original_keys = {(row["scene_id"], str(row["trajectory_id"]))
                     for row in original_doc["samples"]}
    if len(original_keys) != OBSERVED_TOTAL or not original_keys <= set(groups):
        raise SystemExit("original queue trajectory closure failure")
    remaining_keys = sorted(set(groups) - original_keys)
    if len(remaining_keys) != 6169:
        raise SystemExit("remaining population closure failure")

    random = Random(SEED)
    selected_keys = random.sample(remaining_keys, expansion_count)
    selected = [random.choice(groups[key]) for key in selected_keys]

    with gzip.open(RUNTIME, "rt", encoding="utf-8") as handle:
        runtime = {str(row["episode_id"]): row
                   for row in json.load(handle)["episodes"]}
    records = []
    for order, candidate in enumerate(selected):
        episode = runtime.get(candidate.episode_id)
        if episode is None:
            raise SystemExit("runtime episode missing: " + candidate.episode_id)
        instruction = episode["instruction"]
        scene = Path(episode["scene_id"]).stem
        identity_ok = all((
            str(episode["trajectory_id"]) == candidate.trajectory_id,
            str(instruction["instruction_id"]) == candidate.instruction_id,
            instruction["language"] == candidate.language,
            instruction["instruction_text"] == candidate.instruction,
            scene == candidate.scene_id,
        ))
        if not identity_ok:
            raise SystemExit("canonical/runtime identity mismatch: "
                             + candidate.episode_id)
        for suffix in (".glb", ".navmesh"):
            asset = MP3D / scene / (scene + suffix)
            if not asset.is_file() or asset.is_symlink():
                raise SystemExit("scene asset closure failure: " + str(asset))
        records.append({
            "expansion_order": order,
            "episode_id": candidate.episode_id,
            "instruction_id": candidate.instruction_id,
            "trajectory_id": candidate.trajectory_id,
            "scene_id": candidate.scene_id,
            "language": candidate.language,
            "split": candidate.split,
            "triggers": list(candidate.triggers),
            "instruction_text": candidate.instruction,
            "instruction_sha256": sha256_text(candidate.instruction),
            "reference_path_points": len(episode.get("reference_path") or []),
            "selection_stratum": "remaining_after_frozen_queue50",
            "selection_probability": expansion_count / len(remaining_keys),
            "analysis_weight": len(remaining_keys) / expansion_count,
            "processing_status": "PENDING_HINDSIGHT_EVENT_FACTORY",
        })

    identity_commitment = [{
        key: row[key] for key in (
            "expansion_order", "episode_id", "instruction_id",
            "trajectory_id", "scene_id", "language", "instruction_sha256")
    } for row in records]
    output = {
        "manifest": "RevealNav unbiased RxR-train expansion queue",
        "revision": "rxr-train-unbiased-expansion/1",
        "status": "FROZEN_READY_FOR_HINDSIGHT_EVENT_FACTORY",
        "sources": {str(path.relative_to(ROOT)): expected
                    for path, expected in EXPECTED.items()},
        "sampling_design": {
            "name": "stratified_queue50_census_plus_remaining_srswor/1",
            "selection_was_committed_before_new_outcomes": True,
            "outcome_fields_used_for_item_selection": [],
            "outcome_used_only_for_total_sample_size":
                "queue50 strict T_X Wilson lower bound",
            "seed": SEED,
            "population_unique_candidate_trajectories": len(groups),
            "frozen_queue50_census_stratum": OBSERVED_TOTAL,
            "remaining_population": len(remaining_keys),
            "remaining_sample": expansion_count,
            "combined_review_target": total_required,
            "remaining_inclusion_probability":
                expansion_count / len(remaining_keys),
            "remaining_analysis_weight": len(remaining_keys) / expansion_count,
            "no_replacement": True,
            "one_instruction_uniformly_selected_within_each_trajectory": True,
            "post_selection_resampling_for_failures_forbidden": True,
        },
        "sample_size_basis": {
            "strict_events": OBSERVED_SUCCESSES,
            "reviewed_candidates": OBSERVED_TOTAL,
            "observed_rate": OBSERVED_SUCCESSES / OBSERVED_TOTAL,
            "wilson_z": WILSON_Z,
            "wilson_95_lower": lower,
            "pilot_event_target": PILOT_TARGET,
            "required_combined_candidates": total_required,
            "new_candidates": expansion_count,
            "lower_bound_projected_events": lower * total_required,
        },
        "candidate_count": len(records),
        "unique_trajectory_count": len({
            (row["scene_id"], row["trajectory_id"]) for row in records}),
        "unique_episode_count": len({row["episode_id"] for row in records}),
        "scene_count": len({row["scene_id"] for row in records}),
        "language_counts": dict(sorted(Counter(
            row["language"] for row in records).items())),
        "trigger_counts": dict(sorted(Counter(
            trigger for row in records for trigger in row["triggers"]).items())),
        "selection_commitment_sha256": stable_sha(identity_commitment),
        "candidates": records,
        "network_calls_made": 0,
        "forbidden_split_payloads_opened": 0,
        "new_labels_created": 0,
        "feature_generation_authorized": False,
        "training_authorized": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_name(OUT.name + ".part")
    temporary.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({
        "status": output["status"],
        "candidate_count": output["candidate_count"],
        "scene_count": output["scene_count"],
        "language_counts": output["language_counts"],
        "selection_commitment_sha256": output[
            "selection_commitment_sha256"],
        "output": str(OUT.relative_to(ROOT)),
        "sha256": sha256_file(OUT),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
