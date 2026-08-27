#!/usr/bin/env python3
"""Stage 4: generate Reveal Prefix review units for the frozen 50-episode
RxR train queue (checkpoint inference/eval mode only; no training).

For each queue item (frozen order, no resampling):
  1. run scripts/run_reveal_collect.py -> scripts/collect_reveal_prefixes.py
     (skips re-running when a successful run directory already exists);
  2. build the review unit from the hash-chained prefix trace:
       - prefix hash-chain trace reference + root
       - candidate evolution summary
       - persistent candidate identity (node/ghost tiers)
       - policy-selected branch per prefix
       - reference-route target branch proposal (OFFLINE, proposal_only=true;
         GT/reference route never enters the model input)
       - replayable command/config
       - scene/episode/language/instruction hashes
       - contact-sheet media selection (<=12 JPEGs, 224x224, quality<=85)
  3. write artifacts/phase0/review_units/unit_<order>_<episode>.json

Target branch proposal method (reveal-target-proposal/v1):
  - progress index j = argmin Euclidean distance cur_pos -> reference path
    point (ties: lowest index);
  - forward window = subsequent reference points within 3.0 m cumulative
    arc distance from point j;
  - each candidate endpoint (chain candidate_positions_q) scores by its
    minimum Euclidean distance to the forward window;
  - proposal = lowest-score candidate; if margin to the second best is
    < 0.05 m the proposal is marked AMBIGUOUS;
  - every proposal carries proposal_only=true and full generation evidence.
  No proposal is a ground-truth RevealEvent until a human reviewer confirms
  it.

Writes artifacts/runtime/phase0_reveal_closure/STAGE4_TRACE_GENERATION_SUMMARY.json.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = "/mnt/daiyang/vla"
MAPPING_PATH = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                            "REVEAL_QUEUE_50_MAPPING.json")
RUNTIME_PAYLOAD = os.path.join(
    PROJECT_ROOT,
    "third_party/ETP-R1/data/datasets/RxR_VLNCE_v0_enc_xlmr/train/"
    "train_guide.json.gz")
COLLECT_ROOT = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                            "phase0_reveal_closure", "collect")
GATE_OUT = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                        "phase0_reveal_closure", "collect_gate")
UNITS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "phase0", "review_units")
MEDIA_DIR = os.path.join(PROJECT_ROOT, "artifacts", "phase0",
                         "review_packet_50", "private_media")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "artifacts", "runtime",
                            "phase0_reveal_closure",
                            "STAGE4_TRACE_GENERATION_SUMMARY.json")
PYBIN = os.path.join(PROJECT_ROOT, ".envs", "etpr1", "bin", "python")
RUNNER = os.path.join(PROJECT_ROOT, "scripts", "run_reveal_collect.py")

PROPOSAL_VERSION = "reveal-target-proposal/v1"
WINDOW_M = 3.0
AMBIGUITY_MARGIN_M = 0.05
MAX_MEDIA_PER_UNIT = 12


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def euclid(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2
                         for x, y in zip(a, b)))


def arc_distances(ref, start_idx):
    """Cumulative arc distance along the reference path from start_idx."""
    arcs = [0.0]
    for k in range(start_idx + 1, len(ref)):
        arcs.append(arcs[-1] + euclid(ref[k - 1], ref[k]))
    return arcs


def target_proposal(rec, ref_path):
    """Offline geometric target-branch proposal for one prefix record."""
    cur = rec["agent_pose"]["position_q"]
    cand_pos = rec.get("candidate_positions_q") or []
    mappings = rec["graph"]["mappings"]
    if not ref_path or not cand_pos:
        return {"proposal_status": "NO_DATA", "proposal_only": True,
                "version": PROPOSAL_VERSION}
    dists_to_ref = [euclid(cur, p) for p in ref_path]
    j = min(range(len(dists_to_ref)), key=lambda k: (dists_to_ref[k], k))
    arcs = arc_distances(ref_path, j)
    window_idx = [j + 1 + t for t, a in enumerate(arcs[1:])
                  if a <= WINDOW_M]
    if not window_idx:
        return {
            "proposal_status": "NO_FORWARD_SEGMENT",
            "proposal_only": True,
            "version": PROPOSAL_VERSION,
            "progress_index": j,
            "evidence": {"cur_to_ref_distance": dists_to_ref[j]},
        }
    window_pts = [ref_path[k] for k in window_idx]
    scored = []
    for i, e in enumerate(cand_pos):
        d = min(euclid(e, p) for p in window_pts)
        target_id = None
        for m in mappings:
            if m["cand_index"] == i:
                target_id = m["target"]
                break
        scored.append({"cand_index": i,
                       "persistent_id": target_id,
                       "endpoint_q": e,
                       "min_dist_to_window_m": round(d, 6)})
    scored.sort(key=lambda s: (s["min_dist_to_window_m"], s["cand_index"]))
    best = scored[0]
    margin = (scored[1]["min_dist_to_window_m"] - best["min_dist_to_window_m"]
              if len(scored) > 1 else None)
    ambiguous = margin is not None and margin < AMBIGUITY_MARGIN_M
    return {
        "proposal_status": "AMBIGUOUS" if ambiguous else "PROPOSED",
        "proposal_only": True,
        "version": PROPOSAL_VERSION,
        "target_branch_candidate_index": None if ambiguous
        else best["cand_index"],
        "target_branch_persistent_id": None if ambiguous
        else best["persistent_id"],
        "progress_index": j,
        "window_indices": window_idx,
        "margin_m": None if margin is None else round(margin, 6),
        "candidate_scores": scored,
        "evidence": {
            "method": "nearest_forward_reference_window; euclidean geometry "
                      "only; computed offline from the frozen GT reference "
                      "path and chain candidate endpoints; never fed to the "
                      "policy",
            "window_arc_limit_m": WINDOW_M,
            "ambiguity_margin_m": AMBIGUITY_MARGIN_M,
            "cur_to_ref_distance": round(dists_to_ref[j], 6),
            "reference_path_sha256": hashlib.sha256(
                json.dumps(ref_path).encode("utf-8")).hexdigest(),
            "prefix_record_hash": rec["current_record_hash"],
        },
    }


def candidate_evolution(chain):
    evolution = []
    for rec in chain:
        persistent = sorted(set(m["target"] for m in
                                rec["graph"]["mappings"]))
        evolution.append({
            "high_level_step": rec["high_level_step"],
            "candidate_count": rec["candidates"]["count"],
            "cur_vp": rec["cur_vp"],
            "mapped_persistent_ids": persistent,
            "post_node_ids": rec["graph"]["post_node_ids"],
            "post_ghost_ids": rec["graph"]["post_ghost_ids"],
            "ambiguous": rec["graph"].get("ambiguous", []),
        })
    return evolution


def select_media_indices(n, limit=MAX_MEDIA_PER_UNIT):
    if n <= 0:
        return []
    m = min(limit, n)
    if m == 1:
        return [0]
    return sorted({round(i * (n - 1) / (m - 1)) for i in range(m)})


def build_unit(item, order, run_dir, payload_eps, limit_media):
    with open(os.path.join(run_dir, "COLLECT_SUMMARY.json")) as fh:
        collect = json.load(fh)
    with open(os.path.join(run_dir, "ORCHESTRATOR_RESULT.json")) as fh:
        orch = json.load(fh)
    with open(os.path.join(run_dir, "LAUNCH_META.json")) as fh:
        launch = json.load(fh)
    chain = []
    chain_path = os.path.join(run_dir, "reveal_prefix_chain.jsonl")
    if os.path.isfile(chain_path):
        with open(chain_path) as fh:
            chain = [json.loads(ln) for ln in fh if ln.strip()]
    eid = str(item["episode_id"])
    ep = payload_eps.get(eid)
    ref_path = (ep or {}).get("reference_path") or []

    proposals = [target_proposal(rec, ref_path) for rec in chain]
    policy_branches = []
    for rec in chain:
        a = rec["action"]
        policy_branches.append({
            "high_level_step": rec["high_level_step"],
            "act": a.get("act"),
            "selected_vp": a.get("selected_vp"),
            "front_vp": a.get("front_vp"),
            "cur_vp": a.get("cur_vp"),
            "done": a.get("done"),
        })

    media = []
    staging = os.path.join(run_dir, "media_staging")
    if limit_media and os.path.isdir(staging):
        idxs = select_media_indices(len(chain))
        for out_k, k in enumerate(idxs):
            src = os.path.join(staging, "prefix_%03d.jpg" % k)
            if not os.path.isfile(src):
                continue
            dst_name = "%s_order%02d_p%02d.jpg" % (eid, order, out_k)
            dst = os.path.join(MEDIA_DIR, dst_name)
            shutil.copyfile(src, dst)
            media.append({
                "file": "private_media/" + dst_name,
                "sha256": sha256_file(dst),
                "bytes": os.path.getsize(dst),
                "source_prefix_index": k,
                "source_prefix_hash":
                    chain[k]["current_record_hash"] if k < len(chain)
                    else None,
                "content": "front RGB observation, 224x224 JPEG q85",
            })

    unit = {
        "unit_id": "unit_%02d_ep%s" % (order, eid),
        "schema_version": "reveal-review-unit/1",
        "queue_order": order,
        "episode_id": eid,
        "instruction_id": str(item.get("instruction_id")),
        "trajectory_id": str(item.get("trajectory_id")),
        "scene_id": item.get("scene_id"),
        "language": item.get("language"),
        "source_split": item.get("split"),
        "instruction_sha256": item.get("instruction_sha256_queue"),
        "note_no_instruction_text":
            "unit stores instruction hash only; instruction text remains in "
            "the frozen queue artifact",
        "run": {
            "exp_name": orch.get("exp_name"),
            "run_dir": os.path.relpath(run_dir, PROJECT_ROOT),
            "collect_status": collect.get("collect_status"),
            "collect_status_reason": collect.get("collect_status_reason"),
            "prefix_count": collect.get("prefix_count"),
            "chain_root": collect.get("chain_root"),
            "chain_sha256": sha256_file(chain_path)
            if os.path.isfile(chain_path) else None,
            "chain_file": os.path.relpath(chain_path, PROJECT_ROOT),
            "episode_meta": collect.get("episode_meta"),
            "network_attempts": orch.get("network_attempts_all_processes"),
            "gpu_evidence": {
                "gpu_index": orch.get("gpu_index"),
                "baseline_mib": orch.get("gpu_baseline_used_mib"),
                "post_exit_mib": orch.get("gpu_post_exit_used_mib"),
            },
            "replay_command": launch.get("cmd"),
            "replay_cwd": "third_party/ETP-R1",
            "replay_config_note":
                "run_rxr/iter_train.yaml + worker argv overrides; seed "
                "TASK_CONFIG.SEED=100; selection seed 20260824; see "
                "RXR_EN_EFFECTIVE_CONFIG.yaml",
        },
        "candidate_evolution": candidate_evolution(chain),
        "persistent_candidate_identity": {
            "final_node_ids": chain[-1]["graph"]["post_node_ids"]
            if chain else [],
            "final_ghost_ids": chain[-1]["graph"]["post_ghost_ids"]
            if chain else [],
            "loc_noise": 0.5,
            "merge_semantics":
                "upstream GraphMap loc_noise=0.5 node/ghost localization; "
                "mirror crosschecked; ambiguous candidates fail closed",
        },
        "policy_selected_branches": policy_branches,
        "target_branch_proposals": proposals,
        "target_branch_proposal_disclaimer":
            "proposal_only=true; derived offline from the GT reference path "
            "geometry; not a ground-truth RevealEvent until confirmed by a "
            "human reviewer; GT never enters model input",
        "media": media,
    }
    return unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no-media", action="store_true")
    ap.add_argument("--skip-runs", action="store_true",
                    help="only rebuild units from existing run dirs")
    args = ap.parse_args()

    with open(MAPPING_PATH) as fh:
        mapping = json.load(fh)
    items = mapping["items"]
    if not mapping.get("unique_mapping_50_of_50"):
        print(json.dumps({"status": "ABORT",
                          "reason": "queue mapping is not 50/50; batch runs "
                                    "are stopped per protocol"}))
        return 1

    import gzip
    with gzip.open(RUNTIME_PAYLOAD, "rt") as fh:
        payload_eps = {e["episode_id"]: e
                       for e in json.load(fh)["episodes"]}

    os.makedirs(UNITS_DIR, exist_ok=True)
    os.makedirs(MEDIA_DIR, exist_ok=True)

    summary_items = []
    counts = {"ok": 0, "ambiguous": 0, "failed": 0}
    limit_media = not args.no_media
    for order in range(args.start, min(args.limit, len(items))):
        item = items[order]
        eid = str(item["episode_id"])
        exp_name = "rpc50_%02d_ep%s" % (order, eid)
        run_dir = os.path.join(COLLECT_ROOT, exp_name)
        collect_path = os.path.join(run_dir, "COLLECT_SUMMARY.json")
        orch_path = os.path.join(run_dir, "ORCHESTRATOR_RESULT.json")
        need_run = not args.skip_runs and not (
            os.path.isfile(collect_path) and os.path.isfile(orch_path)
            and json.load(open(orch_path)).get("returncode") == 0)
        if need_run:
            staging = os.path.join(run_dir, "media_staging")
            cmd = [PYBIN, RUNNER,
                   "--episode-id", eid,
                   "--split", "train",
                   "--task", "rxr",
                   "--exp-name", exp_name,
                   "--run-dir", run_dir,
                   "--gate-out", GATE_OUT]
            if limit_media:
                cmd += ["--media-staging", staging]
            print("[%02d] running %s" % (order, exp_name), flush=True)
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
            if proc.returncode != 0:
                counts["failed"] += 1
                summary_items.append({
                    "queue_order": order,
                    "episode_id": eid,
                    "status": "RUN_FAILED",
                })
                continue

        with open(orch_path) as fh:
            orch = json.load(fh)
        with open(collect_path) as fh:
            collect = json.load(fh)
        unit = build_unit(item, order, run_dir, payload_eps, limit_media)
        unit_path = os.path.join(UNITS_DIR,
                                 "unit_%02d_ep%s.json" % (order, eid))
        with open(unit_path, "w") as fh:
            json.dump(unit, fh, indent=2)
        status = collect.get("collect_status")
        if status == "OK":
            counts["ok"] += 1
        elif status == "AMBIGUOUS":
            counts["ambiguous"] += 1
        else:
            counts["failed"] += 1
        summary_items.append({
            "queue_order": order,
            "episode_id": eid,
            "scene_id": item.get("scene_id"),
            "language": item.get("language"),
            "status": status,
            "prefix_count": collect.get("prefix_count"),
            "chain_root": collect.get("chain_root"),
            "network_attempts": orch.get("network_attempts_all_processes"),
            "unit_path": os.path.relpath(unit_path, PROJECT_ROOT),
        })
        print("[%02d] %s status=%s prefixes=%s root=%s" % (
            order, exp_name, status, collect.get("prefix_count"),
            str(collect.get("chain_root"))[:16]), flush=True)

    total_media = 0
    for root, _dirs, names in os.walk(MEDIA_DIR):
        for nm in names:
            total_media += os.path.getsize(os.path.join(root, nm))
    summary = {
        "stage": "stage4_review_unit_generation",
        "label": "rxr_train_engineering_only",
        "mapping_rule_version": mapping.get("mapping_rule_version"),
        "counts": counts,
        "total_units": len(summary_items),
        "media_total_bytes": total_media,
        "media_limit_bytes": 250 * 1024 * 1024,
        "items": summary_items,
    }
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({"counts": counts, "media_total_bytes": total_media,
                      "summary": os.path.relpath(SUMMARY_PATH,
                                                 PROJECT_ROOT)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
