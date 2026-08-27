#!/usr/bin/env python3
"""Extract frozen causal ETP features for one lane of MF2-CR6 events."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import socket
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
    build_models, build_sim, install_network_guard, make_observations, set_state,
)
from phase0c_oracle_lowlevel_probe import build_lowlevel_trace  # noqa: E402
from revealnav_cr1.causal_frontend import (  # noqa: E402
    apply_raw_view_mask, causal_vp_feature_variable, filter_waypoint_outputs,
)


BASE = ROOT / "artifacts/phase1/rxr_train_expansion"
V2 = BASE / "multibranch_v2"
INDEX = V2 / "RXR_MULTIBRANCH_TRAINING_INDEX_V2.json"
CAUSAL = V2 / "RXR_MULTIBRANCH_CAUSAL_CANDIDATE_ANALYSIS_V2.json"
TX_RUNS = V2 / "tx_runs/round1"
RXR_TRAIN = ETPR1 / (
    "data/datasets/RxR_VLNCE_v0_enc_xlmr/train/train_guide.json.gz"
)
BUDGETS = (1.5, 2.0, 3.0, 4.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(part, path)


def language_embedding(policy, episode, config, device):
    maximum = int(config.IL.max_text_len)
    tokens = list(episode["instruction"]["instruction_tokens"][:maximum])
    valid = len(tokens)
    tokens.extend([1] * (maximum - valid))
    task = [2] * valid + [0] * (maximum - valid)
    txt_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    txt_task = torch.tensor([task], dtype=torch.long, device=device)
    mask = txt_ids != 1
    with torch.no_grad():
        embeddings = policy.net(
            mode="language", txt_ids=txt_ids,
            txt_task_encoding=txt_task, txt_masks=mask,
        )
    pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
        dim=1, keepdim=True
    ).clamp_min(1)
    return pooled[0].detach().cpu().float().numpy()


def route_labels(tx, prefix: int, branch_ids, candidate_mask):
    costs = np.full(len(branch_ids), np.inf, dtype=np.float32)
    feasibility = np.full((len(branch_ids), len(BUDGETS)), -1.0,
                          dtype=np.float32)
    optionality = []
    for branch_index, branch_id in enumerate(branch_ids):
        controller = tx["branches"][branch_id]["controllers"][
            "frozen_shortest_path_compat"
        ]
        row = controller["prefix_costs"][prefix - tx["checkpoint"][
            "prefix_index"]]
        if row["prefix_index"] != prefix:
            raise RuntimeError("T_X prefix alignment failure")
        denominator = controller["normalization_denominator_actions"]
        if candidate_mask[branch_index] and row["cstar_action_count"] is not None:
            costs[branch_index] = float(row["cstar_action_count"]) / denominator
            for budget_index, budget in enumerate(BUDGETS):
                feasibility[branch_index, budget_index] = float(
                    row["cstar_action_count"] <= budget * denominator
                )
        branch_optionality = []
        for budget in BUDGETS:
            absolute = budget * denominator
            direct = row["direct"]
            saved = row["saved_via_checkpoint"]
            branch_optionality.append(
                bool(saved.get("success"))
                and saved.get("action_count", float("inf")) <= absolute
                and (not direct.get("success")
                     or direct.get("action_count", float("inf")) > absolute)
            )
        optionality.append(branch_optionality)
    checkpoint_value = float(np.asarray(optionality, dtype=np.bool_).any(
        axis=0
    ).mean())
    return costs, feasibility, checkpoint_value


def extract_event(policy, predictor, config, episode, index_row, causal_row,
                  tx, output: Path, device):
    branch_ids = index_row["candidate_branch_ids"]
    if tx["candidate_branch_ids"] != branch_ids:
        raise RuntimeError("T_X and feature candidate sets differ")
    q_prefix = index_row["Q_prefix"]
    d_prefix = index_row["D_prefix"]
    reveal_start = index_row["strict_reveal_interval"][0]
    established = index_row["branch_established_at_prefix"]
    prefix_map = {row["prefix_index"]: row for row in causal_row[
        "prefix_records"
    ]}
    instruction = language_embedding(policy, episode, config, device)
    sim = build_sim(index_row["scene_id"])
    histories = []
    candidate_steps = []
    masks = []
    target_indices = []
    target_in_set = []
    separation = []
    evidence_complete = []
    reveal_hazard = []
    option_cost = []
    feasibility = []
    checkpoint_values = []
    last_embedding = {}
    try:
        trace = build_lowlevel_trace(sim.pathfinder, episode)
        if d_prefix >= len(trace):
            raise RuntimeError("feature horizon exceeds the causal trace")
        for prefix in range(d_prefix + 1):
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
            causal_prefix = prefix_map[prefix]
            candidate_count = len(filtered["cand_angles"][0])
            if candidate_count != causal_prefix["candidate_count"]:
                raise RuntimeError("frozen candidate count drift")
            candidate_tokens = pano[0, :candidate_count]
            for branch_id in branch_ids:
                local_ids = causal_prefix["branch_candidate_local_ids"][branch_id]
                if local_ids:
                    indices = [int(value[1:]) for value in local_ids]
                    if any(index >= candidate_count for index in indices):
                        raise RuntimeError("branch candidate index drift")
                    last_embedding[branch_id] = candidate_tokens[
                        indices
                    ].mean(dim=0).detach().cpu().float().numpy()
            if prefix < q_prefix:
                continue
            history = (pano[0] * pano_mask[0].unsqueeze(-1)).sum(dim=0) \
                / pano_mask[0].sum().clamp_min(1)
            mask = np.asarray([
                prefix >= int(established[branch_id]) for branch_id in branch_ids
            ], dtype=np.bool_)
            if any(mask[index] and branch_id not in last_embedding
                   for index, branch_id in enumerate(branch_ids)):
                raise RuntimeError("persistent branch lacks a causal embedding")
            candidates = np.zeros((len(branch_ids), pano.shape[-1]),
                                  dtype=np.float32)
            for index, branch_id in enumerate(branch_ids):
                if mask[index]:
                    candidates[index] = last_embedding[branch_id]
            target_index = index_row["target_index"]
            closed = prefix >= reveal_start and bool(mask.all())
            costs, feasible, value = route_labels(
                tx, prefix, branch_ids, mask
            )
            histories.append(history.detach().cpu().float().numpy())
            candidate_steps.append(candidates)
            masks.append(mask)
            target_indices.append(target_index if closed else -1)
            target_in_set.append(float(mask[target_index]))
            separation.append(float(mask.all()))
            evidence_complete.append(float(closed))
            reveal_hazard.append(float(prefix == reveal_start))
            option_cost.append(costs)
            feasibility.append(feasible)
            checkpoint_values.append(value)
    finally:
        sim.close()
    arrays = {
        "instruction_embedding": instruction.astype(np.float32),
        "history_embeddings": np.asarray(histories, dtype=np.float32),
        "candidate_embeddings": np.asarray(candidate_steps, dtype=np.float32),
        "candidate_mask": np.asarray(masks, dtype=np.bool_),
        "target_index": np.asarray(target_indices, dtype=np.int64),
        "target_in_set": np.asarray(target_in_set, dtype=np.float32),
        "separation": np.asarray(separation, dtype=np.float32),
        "evidence_complete": np.asarray(evidence_complete, dtype=np.float32),
        "reveal_hazard": np.asarray(reveal_hazard, dtype=np.float32),
        "option_cost": np.asarray(option_cost, dtype=np.float32),
        "current_feasibility": np.asarray(feasibility, dtype=np.float32),
        "checkpoint_value": np.asarray(checkpoint_values, dtype=np.float32),
    }
    atomic_npz(output, arrays)
    return {
        "event_id": index_row["event_id"],
        "path": str(output.relative_to(ROOT)),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "steps": len(histories),
        "candidate_count": len(branch_ids),
        "feature_dim": arrays["history_embeddings"].shape[-1],
    }


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
        raise SystemExit("feature output outside project")
    event_ids = json.loads(args.event_list.read_text())
    index_doc = json.loads(INDEX.read_text())
    if index_doc.get("feature_generation_authorized") is not True:
        raise RuntimeError("feature generation is not authorized")
    index = {row["event_id"]: row for row in index_doc["records"]}
    causal = {row["event_id"]: row for row in json.loads(
        CAUSAL.read_text()
    )["events"]}
    with gzip.open(RXR_TRAIN, "rt") as stream:
        wanted_episodes = {index[event_id]["episode_id"] for event_id in event_ids}
        episodes = {str(row["episode_id"]): row for row in
                    json.load(stream)["episodes"]
                    if str(row["episode_id"]) in wanted_episodes}
    network = install_network_guard()
    os.chdir(ETPR1)
    torch.manual_seed(20260826)
    torch.cuda.manual_seed_all(20260826)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    policy, predictor, config = build_models()
    records = []
    for event_id in event_ids:
        tx_path = TX_RUNS / (event_id + ".json")
        tx = json.loads(tx_path.read_text())["evidence"]
        row = index[event_id]
        result = extract_event(
            policy, predictor, config, episodes[row["episode_id"]], row,
            causal[event_id], tx, output_dir / (event_id + ".npz"), device,
        )
        records.append(result)
        print(event_id, "FEATURE_PASS", flush=True)
    if network["attempts"] != 0:
        raise RuntimeError("network attempt observed")
    value = {
        "schema_version": "revealnav-mf2-feature-lane/1",
        "physical_gpu": args.physical_gpu,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "records": records,
        "network_attempts": 0,
        "future_frames_used": 0,
        "raw_images_written": 0,
    }
    lane_result.parent.mkdir(parents=True, exist_ok=True)
    lane_result.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
