#!/usr/bin/env python3
"""Extend frozen causal features through observed/censored T_X horizons."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/mnt/daiyang/vla").resolve()
SCRIPTS = ROOT / "scripts"
ETPR1 = ROOT / "third_party/ETP-R1"
for path in (ROOT, SCRIPTS, ETPR1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from automatic_semantic_candidate_worker import (  # noqa: E402
    build_models,
    build_sim,
    install_network_guard,
    make_observations,
    set_state,
)
from phase0c_oracle_lowlevel_probe import build_lowlevel_trace  # noqa: E402
from revealnav_cr1.causal_frontend import (  # noqa: E402
    apply_raw_view_mask,
    causal_vp_feature_variable,
    filter_waypoint_outputs,
)
from revealnav_mf2.data import ARRAY_KEYS  # noqa: E402
from revealnav_mf2r3.data import R3_ARRAY_KEYS  # noqa: E402
from rxr_multibranch_feature_v2_lane import route_labels  # noqa: E402


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
SOURCE_MANIFEST = BASE / "RXR_SECONDARY_AUGMENTED_FEATURE_MANIFEST_V1.json"
PRIMARY = BASE / "multibranch_v2"
SECONDARY = BASE / "secondary_expansion_v1/multibranch"
RXR_TRAIN = ETPR1 / (
    "data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(part, path)


def tx_path(record: dict) -> Path:
    base = (
        PRIMARY if record["label_source"] == "primary_human_audited"
        else SECONDARY
    )
    return base / "tx_runs/round1" / f"{record['event_id']}.json"


def expiry_contract(tx: dict, source_steps: int) -> dict:
    target = tx["target_branch_id"]
    rows = tx["branches"][target]["controllers"][
        "frozen_shortest_path_compat"
    ]["prefix_costs"]
    q_prefix = tx["checkpoint"]["prefix_index"]
    if rows[0]["prefix_index"] != q_prefix:
        raise RuntimeError("T_X rows do not begin at the checkpoint")
    safe = [row["prefix_index"] for row in rows
            if row["cstar_action_count"] is not None]
    if not safe:
        raise RuntimeError("T_X evidence contains no safe target prefix")
    last_safe = max(safe)
    final_prefix = rows[-1]["prefix_index"]
    observed = last_safe < final_prefix
    horizon = last_safe if observed else final_prefix
    source_end = q_prefix + source_steps - 1
    return {
        "q_prefix": q_prefix,
        "source_end": source_end,
        "horizon": max(source_end, horizon),
        "expiry_prefix": last_safe,
        "expiry_observed": observed,
        "rows": rows,
    }


def load_source(record: dict) -> dict[str, np.ndarray]:
    path = (SOURCE_MANIFEST.parent / record["path"]).resolve()
    if (
        SOURCE_MANIFEST.parent not in path.parents
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record["bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise RuntimeError("R2 source feature provenance failure")
    with np.load(path, allow_pickle=False) as shard:
        if set(shard.files) != ARRAY_KEYS:
            raise RuntimeError("R2 source feature schema drift")
        return {key: shard[key] for key in shard.files}


def existing_record(output: Path, record: dict, contract: dict) -> dict | None:
    if not output.is_file() or output.is_symlink():
        return None
    try:
        with np.load(output, allow_pickle=False) as shard:
            if set(shard.files) != R3_ARRAY_KEYS:
                return None
            steps = shard["history_embeddings"].shape[0]
            expiry = shard["expiry_hazard"]
            if steps != contract["horizon"] - contract["q_prefix"] + 1:
                return None
            if int((expiry == 1.0).sum()) != int(contract["expiry_observed"]):
                return None
            if contract["expiry_observed"] and expiry[
                contract["expiry_prefix"] - contract["q_prefix"]
            ] != 1.0:
                return None
            candidate_count = shard["candidate_embeddings"].shape[1]
            feature_dim = shard["history_embeddings"].shape[1]
    except (OSError, ValueError, KeyError):
        return None
    return {
        "event_id": record["event_id"],
        "scene_id": record["scene_id"],
        "split": record["split"],
        "label_source": record["label_source"],
        "path": str(output.relative_to(ROOT)),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "steps": steps,
        "candidate_count": candidate_count,
        "feature_dim": feature_dim,
        "expiry_observed": contract["expiry_observed"],
        "expiry_prefix": (
            contract["expiry_prefix"] if contract["expiry_observed"] else None
        ),
        "censor_prefix": (
            None if contract["expiry_observed"] else contract["horizon"]
        ),
        "source_feature_sha256": record["sha256"],
    }


def extend_event(
    policy, predictor, episode: dict, record: dict, tx: dict,
    output: Path, device: torch.device,
) -> dict:
    source = load_source(record)
    contract = expiry_contract(tx, source["history_embeddings"].shape[0])
    reused = existing_record(output, record, contract)
    if reused is not None:
        return reused
    q_prefix = contract["q_prefix"]
    source_end = contract["source_end"]
    horizon = contract["horizon"]
    branch_ids = tx["candidate_branch_ids"]
    target_index = branch_ids.index(tx["target_branch_id"])
    final_candidates = source["candidate_embeddings"][-1]
    final_mask = source["candidate_mask"][-1]
    if not final_mask.all() or source["target_index"][-1] != target_index:
        raise RuntimeError("R2 terminal branch set is not decision-complete")

    additions = {
        "history_embeddings": [],
        "candidate_embeddings": [],
        "candidate_mask": [],
        "target_index": [],
        "target_in_set": [],
        "separation": [],
        "evidence_complete": [],
        "reveal_hazard": [],
        "option_cost": [],
        "current_feasibility": [],
        "checkpoint_value": [],
    }
    if horizon > source_end:
        sim = build_sim(record["scene_id"])
        try:
            trace = build_lowlevel_trace(sim.pathfinder, episode)
            if horizon >= len(trace):
                raise RuntimeError("R3 expiry horizon exceeds causal trace")
            for prefix in range(source_end + 1, horizon + 1):
                state = trace[prefix]
                set_state(sim, state["position"], state["heading"])
                observations = make_observations(sim, device)
                acquired = torch.zeros((1, 12), dtype=torch.bool, device=device)
                acquired[:, 0] = True
                observations = apply_raw_view_mask(observations, acquired)
                with torch.no_grad():
                    raw = policy.net(
                        mode="waypoint", waypoint_predictor=predictor,
                        observations=observations, in_train=False,
                    )
                    filtered = filter_waypoint_outputs(raw, acquired)
                    pano_inputs = causal_vp_feature_variable(filtered, device)
                    pano_inputs["mode"] = "panorama"
                    pano, pano_mask = policy.net(**pano_inputs)
                history = (
                    pano[0] * pano_mask[0].unsqueeze(-1)
                ).sum(0) / pano_mask[0].sum().clamp_min(1)
                costs, feasible, value = route_labels(
                    tx, prefix, branch_ids, final_mask
                )
                additions["history_embeddings"].append(
                    history.detach().cpu().float().numpy()
                )
                additions["candidate_embeddings"].append(final_candidates)
                additions["candidate_mask"].append(final_mask)
                additions["target_index"].append(target_index)
                additions["target_in_set"].append(1.0)
                additions["separation"].append(1.0)
                additions["evidence_complete"].append(1.0)
                additions["reveal_hazard"].append(0.0)
                additions["option_cost"].append(costs)
                additions["current_feasibility"].append(feasible)
                additions["checkpoint_value"].append(value)
        finally:
            sim.close()

    arrays = {"instruction_embedding": source["instruction_embedding"]}
    for key in ARRAY_KEYS - {"instruction_embedding"}:
        extension = additions[key]
        if extension:
            arrays[key] = np.concatenate((
                source[key], np.asarray(extension, dtype=source[key].dtype)
            ), axis=0)
        else:
            arrays[key] = source[key]
    steps = arrays["history_embeddings"].shape[0]
    expiry = np.zeros(steps, dtype=np.float32)
    if contract["expiry_observed"]:
        offset = contract["expiry_prefix"] - q_prefix
        expiry[offset] = 1.0
        expiry[offset + 1:] = -1.0
    arrays["expiry_hazard"] = expiry
    if set(arrays) != R3_ARRAY_KEYS:
        raise RuntimeError("R3 feature assembly schema mismatch")
    atomic_npz(output, arrays)
    result = existing_record(output, record, contract)
    if result is None:
        raise RuntimeError("new R3 shard failed post-write validation")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lane-result", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    lane_result = args.lane_result.resolve()
    if ROOT not in output_dir.parents or ROOT not in lane_result.parents:
        raise SystemExit("R3 output outside project")
    event_ids = json.loads(args.event_list.read_text())
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    records = {row["event_id"]: row for row in manifest["records"]}
    if any(
        event_id not in records or records[event_id]["split"] == "gold"
        for event_id in event_ids
    ):
        raise RuntimeError("R3 lane contains an unauthorized event")
    with gzip.open(RXR_TRAIN, "rt") as stream:
        wanted = {str(records[event_id]["event_id"].split("_ep", 1)[1]
                      .split("_", 1)[0]) for event_id in event_ids}
        episodes = {
            str(row["episode_id"]): row
            for row in json.load(stream)["episodes"]
            if str(row["episode_id"]) in wanted
        }
    if set(episodes) != wanted:
        raise RuntimeError("R3 episode closure failure")
    network = install_network_guard()
    os.chdir(ETPR1)
    torch.manual_seed(20260826)
    torch.cuda.manual_seed_all(20260826)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    policy, predictor, _ = build_models()
    outputs = []
    for event_id in event_ids:
        record = records[event_id]
        tx = json.loads(tx_path(record).read_text())["evidence"]
        episode_id = event_id.split("_ep", 1)[1].split("_", 1)[0]
        result = extend_event(
            policy, predictor, episodes[episode_id], record, tx,
            output_dir / f"{event_id}.npz", device,
        )
        outputs.append(result)
        print(event_id, "EXPIRY_R3_FEATURE_PASS", flush=True)
    if network["attempts"] != 0:
        raise RuntimeError("network attempt observed")
    value = {
        "schema_version": "revealnav-mf2-expiry-feature-lane/3",
        "physical_gpu": args.physical_gpu,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "records": outputs,
        "network_attempts": 0,
        "future_frames_used_for_online_input": 0,
        "gold_payload_read": False,
        "raw_images_written": 0,
    }
    lane_result.parent.mkdir(parents=True, exist_ok=True)
    part = lane_result.with_name(lane_result.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(part, lane_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
