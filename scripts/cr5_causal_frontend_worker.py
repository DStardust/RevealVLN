#!/usr/bin/env python3
"""Replay one RxR-train episode through the frozen 63-degree waypoint frontend.

This worker is deliberately episode-scoped so the six CR5 pilot trajectories
can run on separate GPUs.  It serializes only candidate angle/distance/endpoint
metadata; RGB, depth, features, logits, and tensor values are never written.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path


ROOT = Path("/mnt/daiyang/vla")
SCRIPTS = ROOT / "scripts"
ETPR1 = ROOT / "third_party/ETP-R1"
HABLAB = ROOT / "third_party/habitat-lab"
HABSIM = Path(os.environ.get(
    "VLA_HABITAT_SIM_ROOT", ROOT / "third_party/habitat-sim"))
for value in (ROOT, SCRIPTS, ETPR1, HABLAB, HABSIM):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from automatic_semantic_candidate_worker import (  # noqa: E402
    build_models,
    build_sim,
    candidate_position,
    install_network_guard,
    make_observations,
    set_state,
)
from phase0c_oracle_lowlevel_probe import build_lowlevel_trace  # noqa: E402
from revealnav_cr1.causal_frontend import (  # noqa: E402
    apply_raw_view_mask,
    filter_waypoint_outputs,
)


RXR_TRAIN = ETPR1 / (
    "data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
ALLOWED_EPISODES = {"34121", "41233", "43805", "46758", "56443", "7619"}
SEED = 20260825


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def stable_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def qpoint(value, places: int = 6):
    return [round(float(item), places) for item in value]


def scene_name(episode) -> str:
    parts = Path(episode["scene_id"]).parts
    if len(parts) != 3 or parts[0] != "mp3d" or parts[2] != parts[1] + ".glb":
        raise RuntimeError("unexpected RxR scene path")
    return parts[1]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(part, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episode_id = str(args.episode_id)
    if episode_id not in ALLOWED_EPISODES:
        raise SystemExit("episode not in CR5 pilot allowlist")
    output = args.output.resolve()
    if ROOT.resolve() not in output.parents or output.is_symlink():
        raise SystemExit("unsafe output path")
    if not RXR_TRAIN.is_file() or RXR_TRAIN.is_symlink():
        raise SystemExit("unsafe RxR-train payload")

    # The frozen ETP-R1 config loader resolves run_rxr/*.yaml relative to the
    # upstream repository root.  Make that precondition explicit while all
    # input/output paths remain resolved under this workspace.
    os.chdir(ETPR1)
    network = install_network_guard()
    with gzip.open(RXR_TRAIN, "rt") as handle:
        matches = [row for row in json.load(handle)["episodes"]
                   if str(row["episode_id"]) == episode_id]
    if len(matches) != 1:
        raise SystemExit("episode lookup did not resolve exactly once")
    episode = matches[0]
    scene = scene_name(episode)

    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    policy, predictor, config = build_models()
    device = torch.device("cuda:0")
    sim = build_sim(scene)
    try:
        trace = build_lowlevel_trace(sim.pathfinder, episode)
        if not trace:
            raise RuntimeError("empty low-level trace")
        records = []
        for prefix_index, state in enumerate(trace):
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
            candidates = []
            for local_index, (angle, distance) in enumerate(zip(
                    filtered["cand_angles"][0],
                    filtered["cand_distances"][0])):
                angle = float(angle)
                distance = float(distance)
                endpoint = candidate_position(
                    state["position"], state["heading"], angle, distance)
                signed = (angle + math.pi) % (2.0 * math.pi) - math.pi
                candidates.append({
                    "candidate_local_id": "C%02d" % local_index,
                    "relative_angle_rad": round(angle, 8),
                    "relative_angle_signed_deg": round(math.degrees(signed), 5),
                    "distance_m": round(distance, 6),
                    "endpoint_q": qpoint(endpoint),
                })
            records.append({
                "prefix_index": prefix_index,
                "action": state["action"],
                "position_q": qpoint(state["position"]),
                "heading_rad": round(float(state["heading"]), 8),
                "candidate_count": len(candidates),
                "candidates": candidates,
            })
            del observations, raw, filtered

        trace_contract = [{
            "prefix_index": row["prefix_index"],
            "action": row["action"],
            "position_q": row["position_q"],
            "heading_rad": row["heading_rad"],
        } for row in records]
        result = {
            "revision": "cr5-causal-frozen-frontend-worker/1",
            "episode_id": episode_id,
            "scene_id": scene,
            "source_scope": "RxR-train only",
            "rxr_train": {
                "path": str(RXR_TRAIN.relative_to(ROOT)),
                "bytes": RXR_TRAIN.stat().st_size,
                "sha256": sha256_file(RXR_TRAIN),
            },
            "prefix_count": len(records),
            "trace_pose_action_sha256": stable_sha(trace_contract),
            "prefix_records": records,
            "model_contract": {
                "policy_class": type(policy).__name__,
                "policy_net_class": type(policy.net).__name__,
                "task_type": config.MODEL.task_type,
                "final_checkpoint_strict": True,
                "waypoint_checkpoint_strict": True,
                "sensor_hfov_deg": 63,
                "causal_acquired_slots": [0],
                "hidden_panorama_slots_zeroed_before_encoder": True,
                "candidate_outputs_filtered_to_acquired_hfov": True,
            },
            "physical_gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_gpu": 0,
            "seed": SEED,
            "network_attempts": network["attempts"],
            "observation_tensors_written": 0,
            "checkpoint_tensors_written": 0,
        }
        if network["attempts"] != 0:
            raise RuntimeError("network attempt observed")
        atomic_json(output, result)
    finally:
        sim.close()
    print(json.dumps({
        "status": "PASS",
        "episode_id": episode_id,
        "prefix_count": result["prefix_count"],
        "candidate_prefixes": sum(
            row["candidate_count"] > 0 for row in result["prefix_records"]),
        "output": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
